"""
app.py
------
Gradio web application combining:
  1. MRI image classifier  – upload a scan, get a prediction + bar chart
  2. AI project assistant  – chat interface powered by the Gemini API

Usage:
    python app.py
Then open http://127.0.0.1:7860 in your browser.
"""

import os
import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

import torch
from dataset import get_val_transform
from model   import build_model
from assistant import ProjectAssistant, SYSTEM_PROMPT

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH  = "./models/best_model.pth"
DEVICE      = torch.device("cpu")

CLINICAL_NOTES = {
    "glioma":     "Gliomas arise from glial cells and range widely in grade. "
                  "Neuro-oncology referral is indicated.",
    "meningioma": "Meningiomas are usually benign (arise from meninges). "
                  "Many are managed with watchful waiting.",
    "notumor":    "No evidence of tumour detected. Routine follow-up as appropriate.",
    "pituitary":  "Pituitary adenomas are typically benign. "
                  "Endocrine evaluation and neurosurgical review are recommended.",
}

# ── Load model once at startup ────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _model = build_model(num_classes=len(CLASS_NAMES))
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model.eval()
    return _model


# ── Inference tab ─────────────────────────────────────────────────────────────
def classify_image(pil_image):
    if pil_image is None:
        return "Please upload an MRI image.", None

    model = get_model()
    if model is None:
        return ("Model weights not found. Run `python train.py` first.", None)

    transform  = get_val_transform()
    img_rgb    = pil_image.convert("RGB")
    tensor     = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
        pred   = int(probs.argmax())

    pred_class  = CLASS_NAMES[pred]
    confidence  = probs[pred]

    # Build bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    colors  = ["#e74c3c" if i == pred else "#aed6f1" for i in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, probs * 100, color=colors)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Class Probabilities")
    ax.set_xlim(0, 100)
    for i, p in enumerate(probs):
        ax.text(p * 100 + 0.5, i, f"{p*100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=120)
    plt.close()

    result_text = (
        f"**Predicted Class:** {pred_class.upper()}\n\n"
        f"**Confidence:** {confidence*100:.2f}%\n\n"
        f"**Clinical Note:** {CLINICAL_NOTES[pred_class]}\n\n"
        f"---\n⚠️ *Research prototype — NOT for clinical use.*"
    )

    return result_text, tmp.name


# ── Assistant tab ─────────────────────────────────────────────────────────────
_assistant = ProjectAssistant()

def chat_with_assistant(user_message, history):
    if not user_message.strip():
        return history, ""
    
    # Get the answer from Gemini
    answer = _assistant.ask(user_message)
    
    # Format the history exactly how the new Gradio version requires
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    return history, ""


def reset_chat():
    _assistant.reset()
    return [], ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
# Removed the theme parameter from Blocks per Gradio 6.0 requirements
with gr.Blocks(title="Brain Tumor MRI — AI System") as demo:

    gr.Markdown(
        "# 🧠 Brain Tumor MRI Classification\n"
        "**EfficientNet-B0 fine-tuned on 5,600 MRI scans · 4 classes**\n\n"
        "> ⚠️ Research prototype only — not for clinical use."
    )

    with gr.Tabs():

        # ── Tab 1: Classifier ──────────────────────────────────────────────
        with gr.TabItem("🔬 MRI Classifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input  = gr.Image(type="pil", label="Upload MRI Scan")
                    submit_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=1):
                    result_md  = gr.Markdown(label="Result")
                    chart_img  = gr.Image(label="Probability Chart",
                                         type="filepath")

            submit_btn.click(
                fn=classify_image,
                inputs=[img_input],
                outputs=[result_md, chart_img],
            )

        # ── Tab 2: AI Assistant ───────────────────────────────────────────
        with gr.TabItem("💬 Project Assistant"):
            gr.Markdown(
                "Ask me anything about the dataset, model, training, "
                "limitations, or how to interpret results."
            )
            # Standard chatbot component without the deprecated 'type' argument
            chatbot    = gr.Chatbot(height=420, label="Assistant")
            msg_input  = gr.Textbox(
                placeholder="e.g. Why was EfficientNet-B0 chosen?",
                label="Your question", lines=2
            )
            with gr.Row():
                send_btn  = gr.Button("Send",  variant="primary")
                reset_btn = gr.Button("Reset History")

            send_btn.click(
                fn=chat_with_assistant,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
            )
            msg_input.submit(
                fn=chat_with_assistant,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
            )
            reset_btn.click(
                fn=reset_chat,
                outputs=[chatbot, msg_input],
            )

        # ── Tab 3: Results ────────────────────────────────────────────────
        with gr.TabItem("📊 Evaluation Results"):
            gr.Markdown("### Evaluation artifacts (generated by `evaluate.py`)")
            with gr.Row():
                gr.Image(value="./results/confusion_matrix.png"
                         if os.path.exists("./results/confusion_matrix.png") else None,
                         label="Confusion Matrix")
                gr.Image(value="./results/roc_curves.png"
                         if os.path.exists("./results/roc_curves.png") else None,
                         label="ROC Curves")
            with gr.Row():
                gr.Image(value="./results/training_curves.png"
                         if os.path.exists("./results/training_curves.png") else None,
                         label="Training Curves")
                gr.Image(value="./results/sample_predictions.png"
                         if os.path.exists("./results/sample_predictions.png") else None,
                         label="Sample Predictions")


if __name__ == "__main__":
    # Theme is now passed directly to the launch method
    demo.launch(share=False, theme=gr.themes.Soft())