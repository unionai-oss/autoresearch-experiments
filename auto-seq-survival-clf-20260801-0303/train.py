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
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet(DATA_PATH)
print(f"[DATA] Columns: {df.columns.tolist()}")
print(f"[DATA] Shape: {df.shape}")

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

task_col = "task" if "task" in df.columns else None
print(f"[DATA] Task column: {task_col}")

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))
num_classes = len(le.classes_)
class_counts = df[target_col].value_counts().sort_index().to_dict()
print(f"[DATA] Classes: {num_classes}, Distribution: {class_counts}")

task_le = None
num_tasks = 1
if task_col is not None:
    task_le = LabelEncoder()
    df["task_int"] = task_le.fit_transform(df[task_col].astype(str))
    num_tasks = len(task_le.classes_)
    print(f"[DATA] Tasks ({num_tasks}): {task_le.classes_.tolist()}")

labels = df[target_col].tolist()
sequences = [str(s).upper() for s in df[seq_col].tolist()]
task_ints = df["task_int"].tolist() if task_col is not None else [0] * len(labels)

lens = [len(s) for s in sequences]
print(f"[DATA] Seq len: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.0f}")

train_seqs, val_seqs, train_labels, val_labels, train_tasks, val_tasks = train_test_split(
    sequences, labels, task_ints,
    test_size=0.2, random_state=42, stratify=labels
)
print(f"[DATA] Train: {len(train_seqs)}, Val: {len(val_seqs)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {device}")

# Fast vectorized one-hot encoding using lookup table
NUC2IDX = np.full(256, -1, dtype=np.int8)
NUC2IDX[ord('A')] = 0
NUC2IDX[ord('C')] = 1
NUC2IDX[ord('G')] = 2
NUC2IDX[ord('T')] = 3
MAX_LEN = 1000

def one_hot_np(seq, max_len=MAX_LEN):
    L = len(seq)
    if L > max_len:
        start = (L - max_len) // 2
        seq = seq[start:start + max_len]
        L = max_len
    arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    indices = NUC2IDX[arr]
    oh = np.zeros((4, max_len), dtype=np.float32)
    valid = indices >= 0
    pos = np.arange(L)
    oh[indices[valid], pos[valid]] = 1.0
    return oh

def one_hot_batch(seqs, max_len=MAX_LEN):
    return np.stack([one_hot_np(s, max_len) for s in seqs], axis=0)

print("[DATA] Pre-encoding validation sequences...")
val_X = one_hot_batch(val_seqs)
print(f"[DATA] val_X shape: {val_X.shape}")
print("[DATA] Pre-encoding training sequences...")
train_X = one_hot_batch(train_seqs)
print(f"[DATA] train_X shape: {train_X.shape}")


class DNADataset(Dataset):
    def __init__(self, X, labels, task_ids, augment=False):
        self.X = torch.from_numpy(X)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment and torch.rand(1).item() < 0.5:
            # Reverse complement augmentation: A<->T, C<->G, flip L
            x = x[[3, 2, 1, 0], :].flip(-1)
        return x, self.task_ids[idx], self.labels[idx]


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=9, dilation=1, dropout=0.3):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)
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
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x):
        xt = x.transpose(1, 2)              # (B, L, C)
        a = F.softmax(self.attn(xt), dim=1) # (B, L, 1)
        return (xt * a).sum(dim=1)          # (B, C)


class DNANet(nn.Module):
    def __init__(self, num_tasks, num_classes, dropout=0.4):
        super().__init__()
        # Stem: capture local motifs (k=15 covers typical TFBS width)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),   # 1000 -> 250
        )
        # Tower 1: dilated blocks at 128 channels
        self.tower1 = nn.Sequential(
            ResBlock(128, kernel_size=9, dilation=1, dropout=dropout),
            ResBlock(128, kernel_size=9, dilation=2, dropout=dropout),
            ResBlock(128, kernel_size=9, dilation=4, dropout=dropout),
        )
        self.expand1 = nn.Sequential(
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),   # 250 -> 125
        )
        # Tower 2: dilated blocks at 256 channels
        self.tower2 = nn.Sequential(
            ResBlock(256, kernel_size=7, dilation=1, dropout=dropout),
            ResBlock(256, kernel_size=7, dilation=2, dropout=dropout),
            ResBlock(256, kernel_size=7, dilation=4, dropout=dropout),
        )
        self.expand2 = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(2),   # 125 -> 62
        )
        # Tower 3: dilated blocks at 512 channels
        self.tower3 = nn.Sequential(
            ResBlock(512, kernel_size=5, dilation=1, dropout=dropout),
            ResBlock(512, kernel_size=5, dilation=2, dropout=dropout),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attn_pool = AttentionPool(512)

        task_emb_dim = 64
        self.task_emb = nn.Embedding(num_tasks, task_emb_dim)

        feat_dim = 512 + 512 + task_emb_dim
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
        x = self.stem(x)
        x = self.tower1(x)
        x = self.expand1(x)
        x = self.tower2(x)
        x = self.expand2(x)
        x = self.tower3(x)
        avg = self.avg_pool(x).squeeze(-1)
        attn = self.attn_pool(x)
        return avg, attn

    def forward(self, x, task_id):
        avg, attn = self.encode(x)
        t = self.task_emb(task_id)
        return self.head(torch.cat([avg, attn, t], dim=1))

    def forward_with_rc(self, x, task_id):
        avg, attn = self.encode(x)
        xrc = x[:, [3, 2, 1, 0], :].flip(-1)
        avg_rc, attn_rc = self.encode(xrc)
        avg_e = (avg + avg_rc) / 2.0
        attn_e = (attn + attn_rc) / 2.0
        t = self.task_emb(task_id)
        return self.head(torch.cat([avg_e, attn_e, t], dim=1))


def mixup_criterion(logits, y1, y2, lam, num_classes, label_smoothing=0.05):
    log_probs = F.log_softmax(logits, dim=1)
    y1_oh = F.one_hot(y1, num_classes).float()
    y2_oh = F.one_hot(y2, num_classes).float()
    soft = lam * y1_oh + (1.0 - lam) * y2_oh
    soft = soft * (1.0 - label_smoothing) + label_smoothing / num_classes
    return -(soft * log_probs).sum(dim=1).mean()


BATCH_SIZE = 128
N_EPOCHS = 90
LR = 3e-4
PATIENCE = 22
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MIXUP_ALPHA = 0.3

train_ds = DNADataset(train_X, train_labels, train_tasks, augment=True)
val_ds   = DNADataset(val_X,   val_labels,   val_tasks,  augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = DNANet(num_tasks=num_tasks, num_classes=num_classes).to(device)
print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
if counts.max() > 2.0 * counts.min():
    inv_freq = 1.0 / counts
    inv_freq = inv_freq / inv_freq.sum() * num_classes
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
    print(f"[TRAIN] Class weights: {np.round(inv_freq, 3).tolist()}")
else:
    class_weights = None
    print("[TRAIN] Balanced classes — no weighting")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = N_EPOCHS * len(train_loader)
warmup_steps = WARMUP_EPOCHS * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return max(1e-4, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

best_val_acc = 0.0
no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_b, task_b, y_b in train_loader:
        X_b = X_b.to(device, non_blocking=True)
        task_b = task_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)

        # Mixup
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        lam = max(lam, 1.0 - lam)
        idx = torch.randperm(X_b.size(0), device=device)
        X_mix = lam * X_b + (1.0 - lam) * X_b[idx]
        y_b2 = y_b[idx]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(X_mix, task_b)
            loss = mixup_criterion(logits, y_b, y_b2, lam, num_classes, label_smoothing=0.05)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, task_b, y_b in val_loader:
            X_b = X_b.to(device, non_blocking=True)
            task_b = task_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model.forward_with_rc(X_b, task_b)
            preds = logits.float().argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    val_acc = correct / total
    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "/tmp/best_dna_net.pt")
    else:
        no_improve += 1

    avg_loss = total_loss / len(train_loader)
    if epoch % 5 == 0 or improved:
        print(f"[TRAIN] Epoch {epoch:03d}/{N_EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

    if no_improve >= PATIENCE:
        print(f"[TRAIN] Early stopping at epoch {epoch}")
        break

print(f"BEST_VAL_ACCURACY: {best_val_acc:.6f}")
