from groq import Groq
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import mysql.connector
from mysql.connector import Error
import io
import base64
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
WEIGHTS_PATH       = "../outputs/best_model.pth"
CLASSES            = ["non_ulcer", "ulcer"]
IMG_SIZE           = 224
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MILD_THRESHOLD     = 0.25
MODERATE_THRESHOLD = 0.40

# ─── MySQL CONFIG ─────────────────────────────────────────
DB_CONFIG = {
    "host"    : os.getenv("DB_HOST"),
    "user"    : os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# ─── Groq CLIENT ──────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
groq_client   = Groq(api_key=GROQ_API_KEY)

# ─── Database ─────────────────────────────────────────────
def get_db():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    try:
        conn = mysql.connector.connect(
            host     = DB_CONFIG["host"],
            user     = DB_CONFIG["user"],
            password = DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        conn.commit()
        cursor.close()
        conn.close()

        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                filename        VARCHAR(255),
                image_base64    LONGTEXT,
                detection       VARCHAR(50),
                ulcer_conf      FLOAT,
                severity_grade  VARCHAR(50),
                severity_score  FLOAT,
                description     TEXT,
                area_ratio      FLOAT,
                redness         FLOAT,
                darkness        FLOAT,
                yellow_slough   FLOAT,
                texture         FLOAT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialised")
    except Error as e:
        print(f"❌ DB Error: {e}")


# ─── Model ────────────────────────────────────────────────
class DFUModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone    = models.efficientnet_v2_s(weights=None)
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


def load_model():
    model = DFUModel().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded on {DEVICE}")
    return model


def get_transform():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Detection ────────────────────────────────────────────
@torch.no_grad()
def detect_ulcer(model, img_array, transform):
    tensor = transform(image=img_array)["image"].unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]
    pred   = probs.argmax().item()
    return CLASSES[pred], round(probs[1].item() * 100, 2)


# ─── Wound isolation ──────────────────────────────────────
def isolate_wound_region(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_red1   = cv2.inRange(img_hsv, np.array([0,   40, 40]), np.array([15,  255, 255]))
    mask_red2   = cv2.inRange(img_hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    mask_yellow = cv2.inRange(img_hsv, np.array([15,  40, 40]), np.array([35,  255, 255]))
    mask_dark   = cv2.inRange(img_hsv, np.array([0,    0,  0]), np.array([180, 255,  80]))

    wound_mask = cv2.bitwise_or(mask_red1, mask_red2)
    wound_mask = cv2.bitwise_or(wound_mask, mask_yellow)
    wound_mask = cv2.bitwise_or(wound_mask, mask_dark)

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return img_rgb, img_hsv, wound_mask


# ─── Feature extraction ───────────────────────────────────
def extract_features(img_rgb, img_hsv, wound_mask):
    total_pixels = IMG_SIZE * IMG_SIZE
    wound_pixels = np.sum(wound_mask > 0)
    area_ratio   = wound_pixels / total_pixels

    if wound_pixels == 0:
        r = img_rgb[..., 0].flatten()
        g = img_rgb[..., 1].flatten()
        b = img_rgb[..., 2].flatten()
        v = img_hsv[..., 2].flatten()
    else:
        r = img_rgb[:, :, 0][wound_mask > 0]
        g = img_rgb[:, :, 1][wound_mask > 0]
        b = img_rgb[:, :, 2][wound_mask > 0]
        v = img_hsv[:, :, 2][wound_mask > 0]

    redness_score  = float(np.clip((np.mean(r) - 0.5*np.mean(g) - 0.5*np.mean(b)) / 255.0, 0, 1))
    darkness_score = float(np.clip(1.0 - np.mean(v) / 255.0, 0, 1))
    yellow_score   = float(np.clip((np.mean(r) + np.mean(g) - 2*np.mean(b)) / 255.0 / 2.0, 0, 1))

    gray          = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    grad_x        = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y        = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_score = float(np.clip(np.std(np.sqrt(grad_x**2 + grad_y**2)) / 128.0, 0, 1))

    return {
        "area_ratio"   : round(area_ratio,     4),
        "redness"      : round(redness_score,  4),
        "darkness"     : round(darkness_score, 4),
        "yellow_slough": round(yellow_score,   4),
        "texture"      : round(texture_score,  4),
    }


# ─── Severity scoring ─────────────────────────────────────
def compute_severity(features, ulcer_conf):
    score = (
        0.30 * features["darkness"]      +
        0.25 * features["area_ratio"]    +
        0.20 * features["redness"]       +
        0.15 * features["yellow_slough"] +
        0.10 * features["texture"]
    )
    score = float(np.clip(score + (ulcer_conf / 100.0 - 0.5) * 0.1, 0.0, 1.0))

    if score < MILD_THRESHOLD:
        grade = "Mild"
        desc  = "Superficial wound, minimal tissue involvement"
    elif score < MODERATE_THRESHOLD:
        grade = "Moderate"
        desc  = "Deeper wound with signs of inflammation or slough"
    else:
        grade = "Severe"
        desc  = "Significant necrosis, large wound area, or deep tissue damage"

    return round(score, 4), grade, desc


# ─── Save to MySQL ────────────────────────────────────────
def save_to_db(result: dict):
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                filename, image_base64, detection, ulcer_conf,
                severity_grade, severity_score, description,
                area_ratio, redness, darkness, yellow_slough, texture
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            result["filename"],
            result["image_base64"],
            result["detection"],
            result["ulcer_conf"],
            result.get("severity_grade"),
            result.get("severity_score"),
            result.get("description"),
            result.get("area_ratio"),
            result.get("redness"),
            result.get("darkness"),
            result.get("yellow_slough"),
            result.get("texture"),
        ))
        conn.commit()
        inserted_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return inserted_id
    except Error as e:
        print(f"❌ DB insert error: {e}")
        return None


# ─── Lifespan ─────────────────────────────────────────────
dfu_model = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dfu_model, transform
    init_db()
    dfu_model = load_model()
    transform = get_transform()
    yield

app = FastAPI(title="DFU Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://*.ngrok-free.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Predict ──────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_pil  = Image.open(io.BytesIO(contents)).convert("RGB")
    img_arr  = np.array(img_pil)
    img_arr  = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))

    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_b64  = base64.b64encode(buffered.getvalue()).decode("utf-8")

    label, ulcer_conf = detect_ulcer(dfu_model, img_arr, transform)

    result = {
        "filename"    : file.filename,
        "image_base64": img_b64,
        "detection"   : "Ulcer" if label == "ulcer" else "Non-Ulcer",
        "ulcer_conf"  : ulcer_conf,
    }

    if label == "ulcer":
        img_rgb, img_hsv, wound_mask = isolate_wound_region(img_arr)
        features                     = extract_features(img_rgb, img_hsv, wound_mask)
        score, grade, desc           = compute_severity(features, ulcer_conf)
        result.update({
            "severity_grade": grade,
            "severity_score": score,
            "description"   : desc,
            **features
        })
    else:
        result.update({
            "severity_grade": "N/A",
            "severity_score": None,
            "description"   : "No ulcer detected",
            "area_ratio"    : None,
            "redness"       : None,
            "darkness"      : None,
            "yellow_slough" : None,
            "texture"       : None,
        })

    inserted_id          = save_to_db(result)
    result["id"]         = inserted_id
    result["created_at"] = datetime.now().isoformat()
    return result


# ─── Get all predictions ──────────────────────────────────
@app.get("/predictions")
def get_all_predictions():
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, filename, detection, ulcer_conf,
                   severity_grade, severity_score, description,
                   area_ratio, redness, darkness, yellow_slough,
                   texture, created_at
            FROM predictions
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        for row in rows:
            if row.get("created_at"):
                row["created_at"] = row["created_at"].isoformat()
        return rows
    except Error as e:
        return {"error": str(e)}


# ─── Get single prediction ────────────────────────────────
@app.get("/predictions/{pred_id}")
def get_prediction(pred_id: int):
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE id = %s", (pred_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row and row.get("created_at"):
            row["created_at"] = row["created_at"].isoformat()
        return row or {"error": "Not found"}
    except Error as e:
        return {"error": str(e)}


# ─── Delete prediction ────────────────────────────────────
@app.delete("/predictions/{pred_id}")
def delete_prediction(pred_id: int):
    try:
        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = %s", (pred_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": f"Deleted prediction {pred_id}"}
    except Error as e:
        return {"error": str(e)}


# ─── Stats ────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    try:
        conn   = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                COUNT(*)                                       AS total,
                SUM(detection = 'Ulcer')                       AS total_ulcers,
                SUM(detection = 'Non-Ulcer')                   AS total_non_ulcers,
                SUM(severity_grade = 'Mild')                   AS mild,
                SUM(severity_grade = 'Moderate')               AS moderate,
                SUM(severity_grade = 'Severe')                 AS severe,
                ROUND(AVG(CASE WHEN severity_score IS NOT NULL
                          THEN severity_score END), 4)         AS avg_severity,
                ROUND(AVG(ulcer_conf), 2)                      AS avg_confidence
            FROM predictions
        """)
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        return stats
    except Error as e:
        return {"error": str(e)}


# ─── Chat (VLM) ───────────────────────────────────────────
class ChatMessage(BaseModel):
    role   : str
    content: str

class ChatRequest(BaseModel):
    messages        : list[ChatMessage]
    analysis_result : Optional[dict] = None
    image_base64    : Optional[str]  = None


@app.post("/chat")
async def chat(req: ChatRequest):
    system_prompt = """You are a clinical assistant specialising in diabetic foot ulcers (DFU).
You are helping a medical professional understand wound analysis results from an AI model.
You must never answer any questions other than medical reasoning and foot ulcers.
if the question is not related to foot ulcers, wound care, or severity analysis, politely decline to answer and remind them of your role.

Your role:
- Explain wound analysis results clearly and professionally
- Answer questions about diabetic foot ulcers, wound care, and severity
- Provide evidence-based information about treatment and management
- Be concise but thorough
- Always remind that your responses are informational and not a substitute for professional medical diagnosis

When analysis results are provided, reference them specifically in your responses."""

    if req.analysis_result:
        r = req.analysis_result
        system_prompt += f"""

Current wound analysis results from the AI model:
- Detection        : {r.get('detection', 'N/A')}
- Ulcer Confidence : {r.get('ulcer_conf', 'N/A')}%
- Severity Grade   : {r.get('severity_grade', 'N/A')}
- Severity Score   : {round(r.get('severity_score', 0) * 100, 1) if r.get('severity_score') else 'N/A'}%
- Description      : {r.get('description', 'N/A')}
- Wound Features   :
    • Redness (inflammation) : {round(r.get('redness', 0) * 100, 1) if r.get('redness') else 'N/A'}%
    • Darkness (necrosis)    : {round(r.get('darkness', 0) * 100, 1) if r.get('darkness') else 'N/A'}%
    • Yellow/Slough          : {round(r.get('yellow_slough', 0) * 100, 1) if r.get('yellow_slough') else 'N/A'}%
    • Wound Area Ratio       : {round(r.get('area_ratio', 0) * 100, 1) if r.get('area_ratio') else 'N/A'}%
    • Texture Irregularity   : {round(r.get('texture', 0) * 100, 1) if r.get('texture') else 'N/A'}%"""

    groq_messages = [{"role": "system", "content": system_prompt}]

    for msg in req.messages:
        groq_messages.append({
            "role"   : msg.role,
            "content": msg.content
        })

    def stream_response():
        stream = groq_client.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            messages   = groq_messages,
            max_tokens = 1024,
            stream     = True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(stream_response(), media_type="text/plain")