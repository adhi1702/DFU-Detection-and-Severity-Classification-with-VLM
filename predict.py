import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json

# ─── CONFIG ───────────────────────────────────────────────
WEIGHTS_PATH = "./outputs/best_model.pth"
CLASSES      = ["non_ulcer", "ulcer"]
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Severity thresholds — tune these if needed
MILD_THRESHOLD     = 0.35
MODERATE_THRESHOLD = 0.60

# ─── Change this to your image path or folder ─────────────
SOURCE = "dataset/test/ulcer/79.jpg"


# ─── Model (same architecture as training) ────────────────
class DFUModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.efficientnet_v2_s(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ─── Transform ────────────────────────────────────────────
def get_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Load model ───────────────────────────────────────────
def load_model(weights_path):
    model = DFUModel(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {weights_path}")
    return model


# ─── Step 1: Detect ulcer ─────────────────────────────────
@torch.no_grad()
def detect_ulcer(model, image_path, transform):
    img    = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]
    pred   = probs.argmax().item()
    return CLASSES[pred], round(probs[1].item() * 100, 2)  # label, ulcer_confidence


# ─── Step 2: Isolate wound region ─────────────────────────
def isolate_wound_region(image_path):
    """
    Uses color-based segmentation to isolate the wound area.
    Targets reddish/dark/yellowish wound tones in HSV space.
    """
    img_bgr  = cv2.imread(str(image_path))
    img_bgr  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red tones (inflammation) — wraps around 0/180 in HSV
    mask_red1 = cv2.inRange(img_hsv, np.array([0,  40, 40]), np.array([15,  255, 255]))
    mask_red2 = cv2.inRange(img_hsv, np.array([160,40, 40]), np.array([180, 255, 255]))

    # Yellow/slough tones
    mask_yellow = cv2.inRange(img_hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))

    # Dark/necrotic tones (low value channel)
    mask_dark = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))

    # Combined wound mask
    wound_mask = cv2.bitwise_or(mask_red1, mask_red2)
    wound_mask = cv2.bitwise_or(wound_mask, mask_yellow)
    wound_mask = cv2.bitwise_or(wound_mask, mask_dark)

    # Morphological cleanup
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return img_rgb, img_hsv, wound_mask


# ─── Step 3: Extract clinical features ───────────────────
def extract_features(img_rgb, img_hsv, wound_mask):
    total_pixels = IMG_SIZE * IMG_SIZE
    wound_pixels = np.sum(wound_mask > 0)

    # ── Feature 1: Wound area ratio ───────────────────────
    area_ratio = wound_pixels / total_pixels

    if wound_pixels == 0:
        # fallback: analyse full image if mask is empty
        region_rgb = img_rgb
        region_hsv = img_hsv
    else:
        region_rgb = img_rgb[wound_mask > 0]
        region_hsv = img_hsv[wound_mask > 0]

    # ── Feature 2: Redness score ──────────────────────────
    # High saturation + red hue = active inflammation
    r_channel  = region_rgb[:, 0] if region_rgb.ndim == 2 else region_rgb[..., 0].flatten() if wound_pixels == 0 else region_rgb[:, 0]
    g_channel  = region_rgb[..., 1].flatten() if wound_pixels == 0 else region_rgb[:, 1]
    b_channel  = region_rgb[..., 2].flatten() if wound_pixels == 0 else region_rgb[:, 2]

    redness_score = float(np.mean(r_channel) - 0.5 * np.mean(g_channel) - 0.5 * np.mean(b_channel)) / 255.0
    redness_score = float(np.clip(redness_score, 0, 1))

    # ── Feature 3: Darkness / necrosis score ─────────────
    # Low brightness = necrotic/dead tissue
    v_channel      = img_hsv[..., 2].flatten() if wound_pixels == 0 else img_hsv[:, :, 2][wound_mask > 0]
    darkness_score = float(1.0 - np.mean(v_channel) / 255.0)
    darkness_score = float(np.clip(darkness_score, 0, 1))

    # ── Feature 4: Yellow / slough score ─────────────────
    # High R + High G + Low B = yellowish slough/pus
    yellow_score = float(np.mean(r_channel) + np.mean(g_channel) - 2 * np.mean(b_channel)) / 255.0 / 2.0
    yellow_score = float(np.clip(yellow_score, 0, 1))

    # ── Feature 5: Texture irregularity ──────────────────
    # High gradient variance = irregular/rough wound surface
    gray         = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x       = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y       = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_score = float(np.std(gradient_mag) / 128.0)
    texture_score = float(np.clip(texture_score, 0, 1))

    return {
        "area_ratio"    : round(area_ratio,     4),
        "redness"       : round(redness_score,  4),
        "darkness"      : round(darkness_score, 4),
        "yellow_slough" : round(yellow_score,   4),
        "texture"       : round(texture_score,  4),
    }


# ─── Step 4: Compute severity score & grade ───────────────
def compute_severity(features, ulcer_confidence):
    """
    Weighted combination of clinical features into a 0–1 severity score.
    Weights based on clinical significance:
      - Darkness (necrosis)  → most severe indicator
      - Area ratio           → larger wound = more severe
      - Redness              → moderate indicator
      - Yellow/slough        → moderate indicator
      - Texture              → mild indicator
    """
    score = (
        0.30 * features["darkness"]       +  # necrosis — most critical
        0.25 * features["area_ratio"]     +  # wound size
        0.20 * features["redness"]        +  # inflammation
        0.15 * features["yellow_slough"]  +  # slough/infection
        0.10 * features["texture"]           # surface irregularity
    )

    # Slightly boost score if model is very confident it's an ulcer
    confidence_boost = (ulcer_confidence / 100.0 - 0.5) * 0.1
    score = float(np.clip(score + confidence_boost, 0.0, 1.0))

    if score < MILD_THRESHOLD:
        grade = "Mild"
        description = "Superficial wound, minimal tissue involvement"
    elif score < MODERATE_THRESHOLD:
        grade = "Moderate"
        description = "Deeper wound with signs of inflammation or slough"
    else:
        grade = "Severe"
        description = "Significant necrosis, large wound area, or deep tissue damage"

    return round(score, 4), grade, description


# ─── Step 5: Full pipeline for one image ──────────────────
def analyse_image(model, image_path, transform):
    image_path = Path(image_path)

    # Detection
    label, ulcer_conf = detect_ulcer(model, image_path, transform)

    if label == "non_ulcer":
        return {
            "file"          : image_path.name,
            "detection"     : "Non-Ulcer",
            "ulcer_conf"    : f"{ulcer_conf}%",
            "severity_score": None,
            "severity_grade": "N/A",
            "description"   : "No ulcer detected — severity analysis skipped",
            "features"      : None,
        }

    # Wound isolation & feature extraction
    img_rgb, img_hsv, wound_mask = isolate_wound_region(image_path)
    features                     = extract_features(img_rgb, img_hsv, wound_mask)
    severity_score, grade, desc  = compute_severity(features, ulcer_conf)

    return {
        "file"          : image_path.name,
        "detection"     : "Ulcer",
        "ulcer_conf"    : f"{ulcer_conf}%",
        "severity_score": severity_score,
        "severity_grade": grade,
        "description"   : desc,
        "features"      : features,
    }


# ─── Main ─────────────────────────────────────────────────
def main():
    model     = load_model(WEIGHTS_PATH)
    transform = get_transform(IMG_SIZE)
    source    = Path(SOURCE)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = [p for p in source.iterdir() if p.suffix.lower() in valid_exts]
    else:
        raise FileNotFoundError(f"SOURCE not found: {SOURCE}")

    print(f"\nAnalysing {len(paths)} image(s)...\n")
    print(f"{'File':<30} {'Detection':<12} {'Conf':>6}  {'Grade':<10} {'Score':>6}  Description")
    print("─" * 95)

    all_results = []
    for path in sorted(paths):
        r = analyse_image(model, path, transform)
        print(f"{r['file']:<30} {r['detection']:<12} {r['ulcer_conf']:>6}  "
              f"{r['severity_grade']:<10} {str(r['severity_score']):>6}  {r['description']}")

        if r["features"]:
            f = r["features"]
            print(f"  └─ Features → Area: {f['area_ratio']:.3f}  "
                  f"Redness: {f['redness']:.3f}  "
                  f"Darkness: {f['darkness']:.3f}  "
                  f"Slough: {f['yellow_slough']:.3f}  "
                  f"Texture: {f['texture']:.3f}")
        print()
        all_results.append(r)

    # Save results to JSON
    out_path = "./outputs/severity_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()