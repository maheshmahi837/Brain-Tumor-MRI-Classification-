"""
inference.py
------------
Run inference on a single MRI image.

Usage:
    python inference.py --image path/to/mri.jpg
    python inference.py --image path/to/mri.jpg --model ./models/best_model.pth
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import get_val_transform
from model   import build_model

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE      = torch.device("cpu")

CLINICAL_NOTES = {
    "glioma":     "Gliomas are tumours arising from glial cells. They vary widely "
                  "in grade and aggressiveness. Referral to neuro-oncology is indicated.",
    "meningioma": "Meningiomas arise from the meninges and are usually benign. "
                  "Many are managed with watchful waiting; some require surgery.",
    "notumor":    "No evidence of tumour detected in this scan. "
                  "Routine follow-up as clinically appropriate.",
    "pituitary":  "Pituitary adenomas are usually benign tumours of the pituitary gland. "
                  "Endocrine evaluation and neurosurgical review are recommended.",
}


def predict_image(image_path: str, model_path: str = "./models/best_model.pth"):
    # Load model
    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Preprocess
    transform = get_val_transform()
    img_pil   = Image.open(image_path).convert("RGB")
    tensor    = transform(img_pil).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
        pred   = int(probs.argmax())

    return pred, probs, img_pil


def visualise_prediction(image_path: str, model_path: str = "./models/best_model.pth",
                         save_path: str = None):
    pred, probs, img_pil = predict_image(image_path, model_path)
    pred_class = CLASS_NAMES[pred]
    confidence = probs[pred]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # MRI image
    ax1.imshow(img_pil, cmap="gray")
    ax1.set_title(f"Prediction: {pred_class.upper()}\n"
                  f"Confidence: {confidence*100:.1f}%", fontsize=13)
    ax1.axis("off")

    # Probability bar chart
    colors = ["#e41a1c" if i == pred else "#aec7e8" for i in range(len(CLASS_NAMES))]
    bars   = ax2.barh(CLASS_NAMES, probs * 100, color=colors)
    ax2.set_xlabel("Probability (%)", fontsize=11)
    ax2.set_title("Class Probabilities", fontsize=13)
    ax2.set_xlim(0, 100)
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{p*100:.1f}%", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved prediction plot → {save_path}")
    else:
        plt.savefig("inference_result.png", dpi=150)

    plt.close()

    print(f"\n{'='*50}")
    print(f" Predicted class : {pred_class.upper()}")
    print(f" Confidence      : {confidence*100:.2f}%")
    print(f"\n Clinical note   : {CLINICAL_NOTES[pred_class]}")
    print(f"{'='*50}")
    print("\n⚠  This is a research prototype. NOT for clinical use.")

    return pred_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Inference")
    parser.add_argument("--image", required=True, help="Path to MRI image")
    parser.add_argument("--model", default="./models/best_model.pth",
                        help="Path to model weights")
    parser.add_argument("--save",  default=None,
                        help="Optional path to save the prediction plot")
    args = parser.parse_args()

    visualise_prediction(args.image, args.model, args.save)
