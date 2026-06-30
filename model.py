"""
model.py
--------
EfficientNet-B0 fine-tuned for 4-class Brain Tumor MRI classification.

Two-phase strategy:
  Phase 1 – freeze backbone, train only the custom classifier head  (fast)
  Phase 2 – unfreeze all layers, fine-tune end-to-end at low LR    (accurate)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 4, dropout: float = 0.4) -> nn.Module:
    """
    Returns an EfficientNet-B0 with a custom classification head.
    Backbone weights are ImageNet-pretrained; head is randomly initialised.
    """
    # Load pretrained backbone
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # Replace the default classifier
    in_features = model.classifier[1].in_features      # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the custom classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    print("[Model] Backbone FROZEN — training classifier head only.")


def unfreeze_all(model: nn.Module):
    """Unfreeze every layer for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("[Model] All layers UNFROZEN — end-to-end fine-tuning.")


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total:,} | Trainable: {trainable:,}")


if __name__ == "__main__":
    m = build_model()
    freeze_backbone(m)
    count_params(m)
    unfreeze_all(m)
    count_params(m)
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", m(x).shape)
