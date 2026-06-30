"""
dataset.py
----------
Handles all data loading, augmentation, and splitting for the
Brain Tumor MRI Classification project.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
NUM_WORKERS  = 0          # keep 0 for CPU / Windows compatibility
NUM_CLASSES  = 4
CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]

# ImageNet statistics work well for MRI after grayscale-to-RGB broadcast
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Dataset builder ──────────────────────────────────────────────────────────
def build_dataloaders(
    train_dir: str = "./dataset/Training",
    test_dir:  str = "./dataset/Testing",
    val_split: float = 0.15,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
):
    """
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Load full training set with augmentation transform first (labels only)
    full_train = datasets.ImageFolder(root=train_dir,
                                      transform=get_train_transform())

    # Stratified split → train / val
    labels = [label for _, label in full_train.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_subset = Subset(full_train, train_idx)

    # Val subset gets val-transform (no augmentation)
    val_base = datasets.ImageFolder(root=train_dir,
                                    transform=get_val_transform())
    val_subset = Subset(val_base, val_idx)

    # Test set
    test_dataset = datasets.ImageFolder(root=test_dir,
                                        transform=get_val_transform())

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=False)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"[Dataset] Train: {len(train_subset)} | "
          f"Val: {len(val_subset)} | Test: {len(test_dataset)}")
    print(f"[Dataset] Classes: {full_train.classes}")

    return train_loader, val_loader, test_loader, full_train.classes


# ── Class-distribution helper ────────────────────────────────────────────────
def class_distribution(data_dir: str):
    ds = datasets.ImageFolder(root=data_dir,
                              transform=transforms.ToTensor())
    counts = {c: 0 for c in ds.classes}
    for _, lbl in ds.samples:
        counts[ds.classes[lbl]] += 1
    return counts
