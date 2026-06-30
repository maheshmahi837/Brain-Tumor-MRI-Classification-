"""
train.py
--------
Two-phase training for Brain Tumor MRI Classification.

Phase 1  (epochs 1-5)  : frozen backbone, warm up the classifier head
Phase 2  (epochs 6-25) : unfreeze all, fine-tune end-to-end at 10× lower LR

Usage:
    python train.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from model   import build_model, freeze_backbone, unfreeze_all, count_params

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PHASE1_EPOCHS  = 5
PHASE2_EPOCHS  = 5#20
PHASE1_LR      = 1e-3
PHASE2_LR      = 1e-4
BATCH_SIZE     = 32
WEIGHT_DECAY   = 1e-4
PATIENCE       = 2          # early-stopping patience (phase 2 only)
TRAIN_DIR      = "./dataset/Training"
TEST_DIR       = "./dataset/Testing"
SAVE_DIR       = "./models"
RESULTS_DIR    = "./results"
os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cpu")


# ── Training / validation helpers ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="train" if train else "val ",
                                 leave=False, ncols=80):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def save_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.axvline(PHASE1_EPOCHS - 1, color="grey", linestyle="--",
                label="Phase2 start")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.axvline(PHASE1_EPOCHS - 1, color="grey", linestyle="--",
                label="Phase2 start")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Curves saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_loader, val_loader, _, class_names = build_dataloaders(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR, batch_size=BATCH_SIZE
    )

    model     = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc   = 0.0
    patience_count = 0

    # ── Phase 1: warm-up head ─────────────────────────────────────────────────
    print("\n══════════ PHASE 1 — Classifier warm-up ══════════")
    freeze_backbone(model)
    count_params(model)
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer1, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    train=False)
        scheduler1.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
              f"TrainLoss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"ValLoss {va_loss:.4f} Acc {va_acc:.4f} | "
              f"{time.time()-t0:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "best_model.pth"))

    # ── Phase 2: end-to-end fine-tune ────────────────────────────────────────
    print("\n══════════ PHASE 2 — End-to-end fine-tuning ══════════")
    unfreeze_all(model)
    count_params(model)
    optimizer2 = optim.AdamW(model.parameters(),
                             lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer2, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    train=False)
        scheduler2.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
              f"TrainLoss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"ValLoss {va_loss:.4f} Acc {va_acc:.4f} | "
              f"{time.time()-t0:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ✔ Best model saved (val_acc={va_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    # ── Save training history ─────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\n[Train] Best val accuracy: {best_val_acc:.4f}")
    print(f"[Train] Model weights  → {SAVE_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
