import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

# Detect native image size by sampling up to 10 images
_tmp = ImageFolder(DATA_PATH)
_sizes = [Image.open(p).size for p, _ in _tmp.imgs[:10]]  # (width, height)
(native_w, native_h) = Counter(_sizes).most_common(1)[0][0]
n_channels = len(Image.open(_tmp.imgs[0][0]).getbands())
print(f"[DATA] Native image size: {native_w}x{native_h}, channels={n_channels}")

# Handle grayscale vs RGB
if n_channels == 1:
    img_channels = 1
else:
    img_channels = n_channels

# Build full dataset (no transform yet — used for splitting)
dataset = ImageFolder(DATA_PATH)

# Class mapping
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"[DATA] Class mapping (name -> index): {class_to_idx}")

# Extract labels for stratified split
all_labels = [label for _, label in dataset.imgs]
all_indices = list(range(len(dataset)))

num_classes = len(dataset.classes)

# Stratified 80/20 split
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    stratify=all_labels
)

# Compute class distribution in train and val splits
train_labels_list = [all_labels[i] for i in train_idx]
val_labels_list   = [all_labels[i] for i in val_idx]

train_class_dist = Counter(train_labels_list)
val_class_dist   = Counter(val_labels_list)

train_class_dist_named = {idx_to_class[k]: v for k, v in sorted(train_class_dist.items())}
val_class_dist_named   = {idx_to_class[k]: v for k, v in sorted(val_class_dist.items())}
print(f"[DATA] Train class dist: {train_class_dist_named}")
print(f"[DATA] Val class dist:   {val_class_dist_named}")

# ── Model & Training ──────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import f1_score

# ImageNet normalization for ImageNet-pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# For grayscale images: convert to 3 channels to match ImageNet pretrained input
if img_channels == 1:
    _ch_pre = [transforms.Grayscale(num_output_channels=3)]
else:
    _ch_pre = []

# Training transform — augmentation including h-flip, v-flip, 90°-rotation, jitter, erasing
# 90°-multiple rotations are natural for nadir satellite imagery (4-fold symmetry)
train_transform = transforms.Compose(
    _ch_pre + [
        transforms.Resize((native_h, native_w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ])
        ], p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ]
)

val_transform = transforms.Compose(
    _ch_pre + [
        transforms.Resize((native_h, native_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Separate datasets with their respective transforms; use pre-computed splits
train_dataset = ImageFolder(DATA_PATH, transform=train_transform)
val_dataset   = ImageFolder(DATA_PATH, transform=val_transform)

BATCH_SIZE = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=SubsetRandomSampler(train_idx),
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=SubsetRandomSampler(val_idx),
    num_workers=0,
    pin_memory=True,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRAIN] Device: {device}")

# ConvNeXt V2-Tiny — 28M params, same scale as V1-Tiny but:
#   - Global Response Normalization (GRN) layers replace Channel LayerNorm → better feature diversity
#   - Pretrained with FCMAE (Fully Convolutional Masked Autoencoder) on ImageNet
#   - Achieves 82.9% IN1K top-1 vs ConvNeXt V1-Tiny 82.1%
# This is a direct architecture upgrade over the best exp 1 backbone at identical inference cost.
model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=num_classes)
model = model.to(device)
print(f"[MODEL] convnextv2_tiny | params={sum(p.numel() for p in model.parameters()):,}")

# Mixed precision for faster training
scaler = torch.cuda.amp.GradScaler()

# CutMix only (same as exp 1 which achieved 0.9848):
# switch_prob=1.0 → always use CutMix (not Mixup) when mixing fires
cutmix_fn = Mixup(
    mixup_alpha=0.0,
    cutmix_alpha=1.0,
    prob=0.5,
    switch_prob=1.0,
    mode='batch',
    label_smoothing=0.1,
    num_classes=num_classes,
)

# SoftTargetCrossEntropy handles the soft labels produced by CutMix
criterion = SoftTargetCrossEntropy()

# AdamW with linear warmup (5 epochs) + cosine decay
NUM_EPOCHS    = 40
WARMUP_EPOCHS = 5
LR            = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS],
)

# Training loop with CutMix + AMP
best_val_f1 = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Apply CutMix — converts integer labels to soft one-hot targets
        imgs, soft_labels = cutmix_fn(imgs, labels)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss    = criterion(outputs, soft_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    scheduler.step()
    avg_loss = running_loss / len(train_idx)

    # ── Validate with 5-view TTA ──
    # Views: original, h-flip, v-flip, rot90, rot270
    # All 4 rotational symmetries are natural for nadir satellite imagery
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                p0 = torch.softmax(model(imgs), dim=1)
                p1 = torch.softmax(model(torch.flip(imgs, dims=[-1])), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[-2])), dim=1)
                p3 = torch.softmax(model(torch.rot90(imgs, k=1, dims=[-2, -1])), dim=1)
                p4 = torch.softmax(model(torch.rot90(imgs, k=3, dims=[-2, -1])), dim=1)
            avg_probs = (p0 + p1 + p2 + p3 + p4) / 5.0
            preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    val_f1 = f1_score(all_true, all_preds, average="macro")
    lr_now = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"train_loss={avg_loss:.4f} | "
        f"val_macro_f1={val_f1:.6f} | "
        f"lr={lr_now:.2e}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

print(f"BEST_VAL_MACRO_F1: {best_val_f1:.6f}")
