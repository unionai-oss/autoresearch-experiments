import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math

df = pd.read_parquet(DATA_PATH)
print(f"[DATA] Columns: {df.columns.tolist()}")
print(f"[DATA] Shape: {df.shape}")

# Find target column
target_col = None
for candidate in ["label", "survived", "target", "class"]:
    for col in df.columns:
        if col.lower() == candidate:
            target_col = col
            break
    if target_col is not None:
        break
if target_col is None:
    target_col = df.columns[-1]
print(f"[DATA] Target column: {target_col}")

# Find sequence column
seq_col = None
for candidate in ["sequence", "text", "sentence", "review", "description"]:
    for col in df.columns:
        if col.lower() == candidate and col != target_col:
            seq_col = col
            break
    if seq_col is None:
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == object or str(df[col].dtype) == "string":
                seq_col = col
                break
print(f"[DATA] Sequence column: {seq_col}")

# Task column for extra context
task_col = "task" if "task" in df.columns else None
print(f"[DATA] Task column: {task_col}")

# Encode labels
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))
num_classes = len(le.classes_)
class_counts = df[target_col].value_counts().sort_index().to_dict()
print(f"[DATA] Classes: {num_classes}, Distribution: {class_counts}")

# Encode task as integer if present
task_le = None
num_tasks = 1
if task_col is not None:
    task_le = LabelEncoder()
    df["task_int"] = task_le.fit_transform(df[task_col].astype(str))
    num_tasks = len(task_le.classes_)
    print(f"[DATA] Tasks ({num_tasks}): {task_le.classes_.tolist()}")

labels = df[target_col].tolist()
sequences = [str(s) for s in df[seq_col].tolist()]
task_ints = df["task_int"].tolist() if task_col is not None else [0] * len(labels)

# Stratified split
train_seqs, val_seqs, train_labels, val_labels, train_tasks, val_tasks = train_test_split(
    sequences, labels, task_ints,
    test_size=0.2, random_state=42, stratify=labels
)
print(f"[DATA] Train: {len(train_seqs)}, Val: {len(val_seqs)}")

# ── One-hot encoding for DNA ──
CHAR_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
MAX_LEN = 1000  # all sequences ≤ 1000bp

def one_hot_encode_batch(seqs, max_len=MAX_LEN):
    """Encode list of DNA sequences as one-hot (N, 4, max_len). Center-padded."""
    N = len(seqs)
    arr = np.zeros((N, 4, max_len), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq.upper()
        L = min(len(seq), max_len)
        offset = (max_len - L) // 2
        for j, c in enumerate(seq[:L]):
            idx = CHAR_TO_IDX.get(c, -1)
            if idx >= 0:
                arr[i, idx, offset + j] = 1.0
    return arr

print("[DATA] One-hot encoding sequences ...")
train_X = one_hot_encode_batch(train_seqs)
val_X = one_hot_encode_batch(val_seqs)
print(f"[DATA] train_X shape: {train_X.shape}, val_X shape: {val_X.shape}")


# ── Reverse complement ──
def reverse_complement(x):
    """Reverse complement of one-hot DNA. x: (B, 4, L), channels = ACGT.
    Complement: A<->T (0<->3), C<->G (1<->2); Reverse: flip L dim."""
    return x[:, [3, 2, 1, 0], :].flip(-1)


# ── Dataset ──
class DNADataset(Dataset):
    def __init__(self, X, labels, task_ids, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            # Reverse complement augmentation
            x = x[[3, 2, 1, 0], :].flip(-1)
        return x, self.task_ids[idx], self.labels[idx]


# ── Architecture ──

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class ResBlock(nn.Module):
    """Residual dilated conv block with SE attention."""
    def __init__(self, channels, kernel_size=9, dilation=1, dropout=0.3):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + residual)


class AttentionPool(nn.Module):
    """Soft attention pooling over sequence length."""
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, C, L)
        xt = x.transpose(1, 2)           # (B, L, C)
        a = F.softmax(self.attn(xt), dim=1)  # (B, L, 1)
        return (xt * a).sum(dim=1)       # (B, C)


class ImprovedDNACNN(nn.Module):
    def __init__(self, num_tasks, num_classes, dropout=0.4):
        super().__init__()

        # Stem: capture local motifs
        self.stem = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),              # 1000 → 250
        )

        # Tower: dilated residual blocks
        self.tower = nn.Sequential(
            ResBlock(128, kernel_size=9, dilation=1, dropout=dropout),
            ResBlock(128, kernel_size=9, dilation=2, dropout=dropout),
            ResBlock(128, kernel_size=9, dilation=4, dropout=dropout),
            # Expand channels
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),              # 250 → 125
            ResBlock(256, kernel_size=7, dilation=1, dropout=dropout),
            ResBlock(256, kernel_size=7, dilation=2, dropout=dropout),
            ResBlock(256, kernel_size=7, dilation=4, dropout=dropout),
            # Expand channels
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(2),              # 125 → 62
            ResBlock(512, kernel_size=5, dilation=1, dropout=dropout),
            ResBlock(512, kernel_size=5, dilation=2, dropout=dropout),
        )

        # Dual pooling: average + attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attn_pool = AttentionPool(512)

        # Task embedding
        task_emb_dim = 64
        self.task_emb = nn.Embedding(num_tasks, task_emb_dim)

        # Classifier head
        feat_dim = 512 + 512 + task_emb_dim  # avg + attn + task
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def encode(self, x):
        feat = self.tower(self.stem(x))
        avg = self.avg_pool(feat).squeeze(-1)
        attn = self.attn_pool(feat)
        return avg, attn

    def forward(self, x, task_id):
        avg, attn = self.encode(x)
        task_feat = self.task_emb(task_id)
        combined = torch.cat([avg, attn, task_feat], dim=1)
        return self.head(combined)

    def forward_with_rc(self, x, task_id):
        """Ensemble original + reverse complement predictions."""
        avg, attn = self.encode(x)
        # Reverse complement
        xrc = x[:, [3, 2, 1, 0], :].flip(-1)
        avg_rc, attn_rc = self.encode(xrc)
        avg_e = (avg + avg_rc) / 2
        attn_e = (attn + attn_rc) / 2
        task_feat = self.task_emb(task_id)
        combined = torch.cat([avg_e, attn_e, task_feat], dim=1)
        return self.head(combined)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MODEL] Device: {device}")

BATCH_SIZE = 128
N_EPOCHS = 80
LR = 3e-4
WARMUP_EPOCHS = 5
PATIENCE = 20

train_ds = DNADataset(train_X, train_labels, train_tasks, augment=True)
val_ds   = DNADataset(val_X,   val_labels,   val_tasks,  augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = ImprovedDNACNN(num_tasks=num_tasks, num_classes=num_classes).to(device)
print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class-weighted loss for imbalanced data
counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
if counts.max() > 2.0 * counts.min():
    inv_freq = 1.0 / counts
    inv_freq = inv_freq / inv_freq.sum() * num_classes
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
    print(f"[TRAIN] Weighted loss: {inv_freq}")
else:
    class_weights = None
    print("[TRAIN] Balanced loss")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

total_steps = N_EPOCHS * len(train_loader)
warmup_steps = WARMUP_EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

best_val_acc = 0.0
no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_b, task_b, y_b in train_loader:
        X_b, task_b, y_b = X_b.to(device), task_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(X_b, task_b)
            loss = criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    # Validation with RC ensemble
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, task_b, y_b in val_loader:
            X_b, task_b, y_b = X_b.to(device), task_b.to(device), y_b.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model.forward_with_rc(X_b, task_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    val_acc = correct / total
    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "/tmp/best_dna_cnn.pt")
    else:
        no_improve += 1

    avg_loss = total_loss / len(train_loader)
    if epoch % 5 == 0 or improved:
        print(f"[TRAIN] Epoch {epoch:03d}/{N_EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

    if no_improve >= PATIENCE:
        print(f"[TRAIN] Early stopping at epoch {epoch}")
        break

print(f"BEST_VAL_ACCURACY: {best_val_acc:.6f}")
