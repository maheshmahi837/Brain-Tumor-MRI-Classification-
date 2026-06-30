# 🧠 Brain Tumor MRI Classification

**4-class MRI classification using EfficientNet-B0 — CPU-only, reproducible on any laptop.**

> ⚠️ Research prototype. NOT intended for clinical use.

---

## Problem

Automatically classify brain MRI scans into four categories:

| Class | Description |
|-------|-------------|
| **Glioma** | Malignant glial-cell tumour |
| **Meningioma** | Usually benign, arises from the meninges |
| **Pituitary** | Pituitary adenoma, typically benign |
| **No Tumor** | Healthy brain scan |

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Masoud Nickparvar, Kaggle.

| Split | Images |
|-------|--------|
| Training | 5,600 (1,400 × 4 classes, perfectly balanced) |
| Testing  | 1,311 |

Download and place as:
```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## Setup

```bash
git clone <your-repo>
cd <your-repo>

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Set your Gemini API key (needed for the AI assistant):
```bash
export GEMINI_API_KEY="AIzaSy..."   # Linux / macOS
set GEMINI_API_KEY=AIzaSy...        # Windows CMD
$env:GEMINI_API_KEY="AIzaSy..."     # Windows PowerShell
```

---

## Usage

### 1 · Train the model
```bash
python train.py
```
Runs in two phases (≈2 hours on CPU).  
Saves best weights to `./models/best_model.pth`.

### 2 · Evaluate
```bash
python evaluate.py
```
Generates in `./results/`:
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curves.png`
- `training_curves.png`
- `sample_predictions.png`
- `failure_cases.png`
- `metrics_summary.json`

### 3 · Single-image inference
```bash
python inference.py --image path/to/mri.jpg
```

### 4 · AI assistant (CLI)
```bash
python assistant.py
# or ask a single question:
python assistant.py --question "What dataset was used?"
```

### 5 · Full web app (Gradio)
```bash
$env:GEMINI_API_KEY="Your Gemini API Key" # Run this before launching App
python app.py
```
Open **http://127.0.0.1:7860** — includes classifier, assistant, and results gallery.

---

## Repository Structure

```
project/
├── README.md
├── requirements.txt
├── report.pdf
├── app.py                  # Gradio web application
├── train.py                # Two-phase training script
├── evaluate.py             # Comprehensive evaluation
├── inference.py            # Single-image prediction
├── dataset.py              # Data loading & augmentation
├── model.py                # EfficientNet-B0 architecture
├── assistant.py            # Gemini-powered RAG project assistant
├── dataset/
│   ├── Training/
│   └── Testing/
├── models/
│   └── best_model.pth      # Saved after training
└── results/
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── training_curves.png
    ├── sample_predictions.png
    ├── failure_cases.png
    ├── classification_report.txt
    └── metrics_summary.json
```

---

## Model Architecture

```
Input MRI (224×224 RGB)
        ↓
EfficientNet-B0 backbone (ImageNet pretrained)
        ↓
Dropout(0.4)
        ↓
Linear(1280 → 4)
        ↓
Softmax → [glioma, meningioma, notumor, pituitary]
```

**Training strategy:**

| Phase | Epochs | LR | Backbone |
|-------|--------|----|----------|
| 1 — Head warm-up | 5 | 1e-3 | Frozen |
| 2 — Fine-tune | ≤20 | 1e-4 | Unfrozen |

- Loss: Cross-entropy with label smoothing (ε=0.1)  
- Optimizer: AdamW (weight-decay=1e-4)  
- LR schedule: CosineAnnealingLR  
- Early stopping: patience=7

---

## Evaluation Metrics

The system achieved the following quantitative results on the completely unseen independent test collection:
- Overall Test Accuracy: 95.44%
- Weighted F1-score: 95.34% 
- Macro F1-score: 95.34%  
- Weighted ROC-AUC: 99.27%

Per-Class Breakdown
- Pituitary: 99.50% Precision | 99.75% Recall | 99.63% F1-score
- No Tumor: 95.47% Precision | 100.00% Recall | 97.68% F1-score  
- Meningioma: 88.64% Precision | 99.50% Recall | 93.76% F1-score  
- Glioma: 99.70% Precision | 82.50% Recall | 90.29% F1-score

---

## Known Limitations

- **Glioma Classification:** The model achieves strong overall performance but struggles most with the **Glioma** class (**82.5% recall**). Approximately **12%** of glioma images are misclassified as **Meningioma**, while about **5%** are predicted as **No Tumor**. This is mainly due to the large variation in tumor size, shape, contrast, and diffuse boundaries.

- **2D Slice-Based Prediction:** The model processes each MRI slice independently and does not utilize the 3D spatial information available in complete MRI volumes.

- **Dataset Bias:** The model is trained on a curated public dataset and has not been validated on MRI scans collected from different hospitals, scanners, or imaging protocols.

- **Confidence Calibration:** Prediction confidence is obtained directly from the Softmax output. These confidence scores have not been calibrated using methods such as temperature scaling.

---

## Proposed Improvements

- **Multi-Sequence MRI Analysis:** Combine multiple MRI sequences (T1, T2, and FLAIR) to provide richer structural information for tumor classification.

- **Explainable AI:** Integrate **Grad-CAM** to visualize the image regions influencing the model's predictions and improve interpretability.

- **Larger Backbone Networks:** Experiment with **EfficientNet-B2** or **EfficientNet-B3** while applying dynamic quantization to maintain efficient CPU inference.

- **Advanced Data Augmentation:** Use augmentation techniques such as **MixUp** and **CutMix** to improve generalization and reduce confusion between visually similar tumor classes.

- **Clinical Validation:** Evaluate the model on multi-center clinical datasets to assess its robustness across different imaging devices and patient populations.
