"""
evaluate.py
-----------
Comprehensive evaluation of the trained Brain Tumor MRI classifier.

Produces (all saved to ./results/):
  • classification_report.txt   – precision / recall / F1 / support per class
  • confusion_matrix.png        – normalised heatmap
  • roc_curves.png              – per-class one-vs-rest ROC + AUC
  • sample_predictions.png      – 12 random test images with GT vs Pred
  • failure_cases.png           – 12 mis-classified examples
  • metrics_summary.json        – overall numbers for the AI assistant

Usage:
    python evaluate.py
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import get_val_transform, BATCH_SIZE
from model   import build_model

# ── Config ───────────────────────────────────────────────────────────────────
TEST_DIR    = "./dataset/Testing"
MODEL_PATH  = "./models/best_model.pth"
RESULTS_DIR = "./results"
NUM_CLASSES = 4
DEVICE      = torch.device("cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(path: str, num_classes: int = 4) -> torch.nn.Module:
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ── Inference ────────────────────────────────────────────────────────────────
def predict(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.vstack(all_probs))


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, path):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalise

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix → {path}")


def plot_roc_curves(y_true, y_probs, class_names, path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Eval] ROC curves → {path}")


def _denorm(tensor):
    """Reverse ImageNet normalisation for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


def plot_sample_predictions(dataset, model, class_names, path,
                            n=12, correct_only=False, wrong_only=False):
    model.eval()
    collected = []
    idxs      = np.random.permutation(len(dataset))

    with torch.no_grad():
        for idx in idxs:
            img, label = dataset[idx]
            logit = model(img.unsqueeze(0))
            pred  = logit.argmax(dim=1).item()
            is_correct = (pred == label)
            if correct_only and not is_correct:
                continue
            if wrong_only and is_correct:
                continue
            collected.append((img, label, pred))
            if len(collected) == n:
                break

    cols = 4
    rows = (len(collected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for ax, (img, label, pred) in zip(axes, collected):
        ax.imshow(_denorm(img))
        colour = "green" if pred == label else "red"
        ax.set_title(f"GT: {class_names[label]}\nPred: {class_names[pred]}",
                     fontsize=9, color=colour)
        ax.axis("off")
    for ax in axes[len(collected):]:
        ax.axis("off")

    title = ("Failure Cases" if wrong_only else
             "Sample Predictions")
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] {title} → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Dataset
    test_dataset = datasets.ImageFolder(root=TEST_DIR,
                                        transform=get_val_transform())
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    class_names  = test_dataset.classes

    # Model
    model = load_model(MODEL_PATH, num_classes=len(class_names))
    print(f"[Eval] Loaded model from {MODEL_PATH}")

    # Predict
    y_true, y_pred, y_probs = predict(model, test_loader)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print("\n" + report)
    rpath = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(rpath, "w") as f:
        f.write(report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    plot_roc_curves(y_true, y_probs, class_names,
                    os.path.join(RESULTS_DIR, "roc_curves.png"))

    plot_sample_predictions(test_dataset, model, class_names,
                            os.path.join(RESULTS_DIR, "sample_predictions.png"))

    plot_sample_predictions(test_dataset, model, class_names,
                            os.path.join(RESULTS_DIR, "failure_cases.png"),
                            wrong_only=True)

    # ── Metrics summary JSON (for AI assistant) ───────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import label_binarize

    acc   = accuracy_score(y_true, y_pred)
    f1_w  = f1_score(y_true, y_pred, average="weighted")
    f1_m  = f1_score(y_true, y_pred, average="macro")
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_w = roc_auc_score(y_bin, y_probs, average="weighted",
                          multi_class="ovr")

    summary = {
        "accuracy":       round(acc,  4),
        "weighted_f1":    round(f1_w, 4),
        "macro_f1":       round(f1_m, 4),
        "weighted_auc":   round(auc_w, 4),
        "class_names":    class_names,
        "num_test_imgs":  len(test_dataset),
    }
    spath = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Eval] Metrics summary → {spath}")
    print(f"       Accuracy: {acc:.4f} | Weighted-F1: {f1_w:.4f} | AUC: {auc_w:.4f}")


if __name__ == "__main__":
    main()
