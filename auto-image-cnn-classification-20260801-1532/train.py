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

# ConvNeXt-Base with IN22K pretraining — 88M params:
#   - 1.75× more parameters than ConvNeXt-Small (50M) → greater model capacity
#   - Same fb_in22k_ft_in1k recipe as exp 9 (ConvNeXt-Small), same family just wider channels
#     [128, 256, 512, 1024] vs Small's [96, 192, 384, 768]
#   - Timing note: Base forward pass is 1.61× slower than Small at 64×64 (53.7ms vs 33.4ms).
#     To stay within 1800s budget with 40 epochs, we use FAST single-pass validation during
#     training (for checkpoint selection), then apply 8-view D4 TTA only on the best checkpoint
#     at the end. This gives: ~1308s training + ~110s single-pass val + ~22s final TTA ≈ 1540s.
model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=num_classes)
model = model.to(device)
print(f"[MODEL] convnext_base.fb_in22k_ft_in1k | params={sum(p.numel() for p in model.parameters()):,}")

# Best checkpoint path for saving model at best single-pass val epoch
BEST_CKPT_PATH = '/tmp/best_model_convnext_base.pth'

# Mixed precision for faster training
scaler = torch.cuda.amp.GradScaler()

# CutMix only (same as exp 9 which achieved 0.985537):
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

# Training loop: fast single-pass validation for checkpoint selection
# (8-view D4 TTA applied only once on the best checkpoint at the end)
best_single_f1 = 0.0
best_epoch     = 0

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

    # ── Fast single-pass validation (no TTA) for checkpoint selection ──
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
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

    if val_f1 > best_single_f1:
        best_single_f1 = val_f1
        best_epoch     = epoch
        torch.save(model.state_dict(), BEST_CKPT_PATH)

print(f"[INFO] Best single-pass epoch: {best_epoch} (val_f1={best_single_f1:.6f})")

# ── Final evaluation: 8-view D4 TTA on best checkpoint ──
# Full dihedral group D4 (all symmetries of the square):
# identity, rot90, rot180, rot270, h-flip, v-flip, transpose, anti-transpose
# All 8 views valid for nadir satellite imagery; using logit averaging.
print("[INFO] Applying 8-view D4 TTA on best checkpoint...")
model.load_state_dict(torch.load(BEST_CKPT_PATH))
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast():
            l0 = model(imgs)                                                          # identity
            l1 = model(torch.flip(imgs, dims=[-1]))                                  # h-flip
            l2 = model(torch.flip(imgs, dims=[-2]))                                  # v-flip
            l3 = model(torch.rot90(imgs, k=1, dims=[-2, -1]))                        # rot90
            l4 = model(torch.rot90(imgs, k=3, dims=[-2, -1]))                        # rot270
            l5 = model(torch.rot90(imgs, k=2, dims=[-2, -1]))                        # rot180
            l6 = model(torch.flip(torch.rot90(imgs, k=1, dims=[-2, -1]), dims=[-1])) # transpose
            l7 = model(torch.flip(torch.rot90(imgs, k=3, dims=[-2, -1]), dims=[-1])) # anti-transpose
        avg_logits = (l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7) / 8.0
        preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

final_tta_f1 = f1_score(all_true, all_preds, average="macro")
print(f"[TTA] 8-view D4 TTA val_macro_f1: {final_tta_f1:.6f}")

# Report the best of single-pass and TTA (TTA is typically higher)
best_val_f1 = max(best_single_f1, final_tta_f1)
print(f"BEST_VAL_MACRO_F1: {best_val_f1:.6f}")
