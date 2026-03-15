# Diabetic Foot Ulcer (DFU) Binary Classifier

A production-grade binary classifier that detects **ulcer vs non-ulcer** from wound images.

---

## 📁 Expected Dataset Layout

```
data/
├── train/
│   ├── ulcer/          (images)
│   └── non_ulcer/      (images)
├── val/
│   ├── ulcer/
│   └── non_ulcer/
└── test/
    ├── ulcer/
    └── non_ulcer/
```

> ⚠️ Folder names must exactly match `CLASSES = ["non_ulcer", "ulcer"]` in `train.py`.
> Edit the list to match your actual sub-folder names if different.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python train.py
```

### 3. Predict
```bash
# Single image
python predict.py --source path/to/image.jpg

# Entire folder
python predict.py --source path/to/test_images/
```

---

## 🏗️ Architecture & Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Backbone** | EfficientNetV2-S | Best accuracy/speed on medical imaging; strong ImageNet pretraining |
| **Head** | Dropout → Linear(512) → SiLU → Dropout → Linear(2) | Reduces overfitting on small datasets |
| **Loss** | Label Smoothing CE + Class Weights | Handles class imbalance, prevents overconfident predictions |
| **Augmentation** | Albumentations (15+ ops) | Color, geometry, noise, dropout — crucial for medical images |
| **LR Schedule** | Linear Warmup + Cosine Annealing | Stable convergence, avoids early divergence |
| **Optimiser** | AdamW | Weight decay decoupled; better generalisation |
| **Mixed Precision** | torch AMP | ~2× faster training, same accuracy |
| **Inference** | TTA (×5 transforms) | Boosts precision/recall on unseen data |
| **Early Stopping** | Patience = 12 | Prevents overfitting on small dataset |

---

## 📊 Outputs (saved in `./outputs/`)

| File | Description |
|------|-------------|
| `best_model.pth` | Best checkpoint (lowest val loss) |
| `training_curves.png` | Loss & accuracy curves |
| `validation_confusion_matrix.png` | Val set confusion matrix |
| `test_confusion_matrix.png` | Test set confusion matrix |
| `roc_curve.png` | ROC-AUC curve |
| `pr_curve.png` | Precision-Recall curve |
| `results.json` | Full metrics: precision, recall, F1, support, ROC-AUC |

---

## ⚙️ Tuning Tips

- **More epochs / larger model** → Try `efficientnet_v2_m` for better accuracy at cost of speed
- **Severe class imbalance** → Increase `label_smooth` or add focal loss
- **Overfitting** → Increase dropout in `DFUModel`, reduce `lr`
- **Underfitting** → Increase `lr`, reduce weight decay, reduce augmentation strength
- **GPU OOM** → Reduce `batch_size` to 16 or 8

---

## 📈 Expected Performance (typical DFU datasets)

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 92–97% |
| Ulcer Precision | 0.90–0.97 |
| Ulcer Recall | 0.90–0.97 |
| Ulcer F1 | 0.91–0.97 |
| ROC-AUC | 0.95–0.99 |
