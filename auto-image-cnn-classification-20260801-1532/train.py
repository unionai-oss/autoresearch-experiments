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
    grayscale_transform = [transforms.Grayscale(num_output_channels=1)]
else:
    img_channels = n_channels
    grayscale_transform = []

# Define transforms (val has no augmentation)
train_transform = transforms.Compose(
    grayscale_transform + [
        transforms.Resize((native_h, native_w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * img_channels,
            std=[0.5] * img_channels
        ),
    ]
)

val_transform = transforms.Compose(
    grayscale_transform + [
        transforms.Resize((native_h, native_w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * img_channels,
            std=[0.5] * img_channels
        ),
    ]
)

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
val_labels_list = [all_labels[i] for i in val_idx]

train_class_dist = Counter(train_labels_list)
val_class_dist = Counter(val_labels_list)

train_class_dist_named = {idx_to_class[k]: v for k, v in sorted(train_class_dist.items())}
val_class_dist_named = {idx_to_class[k]: