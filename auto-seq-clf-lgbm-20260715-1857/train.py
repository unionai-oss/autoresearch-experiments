import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(42)
np.random.seed(42)

# ── Data loading ──────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)

target_col = "label"
seq_col = None
for col in df.columns:
    if col == target_col:
        continue
    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
        seq_col = col
        break

if seq_col is None:
    raise ValueError("Could not detect a sequence (string) column in the dataset.")

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
labels = df[target_col].tolist()
sequences = df[seq_col].tolist()

train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

num_classes = len(le.classes_)
class_dist = Counter(labels)
train_dist = Counter(train_labels)
print(f"[DATA] Total: {len(sequences)}, Train: {len(train_seqs)}, Val: {len(val_seqs)}, Classes: {num_classes}")
print(f"[DATA] Class distribution: {class_dist}")

# ── One-hot encoding (ATGC → 4 channels) ─────────────────────────────────────
MAX_LEN = 1024

def encode_sequences(seqs, max_len=MAX_LEN):
    N = len(seqs)
    X = np.zeros((N, 4, max_len), dtype=np.float32)
    for i, seq in enumerate(seqs):
        s = seq[:max_len].upper()
        b = np.frombuffer(s.encode(), dtype=np.uint8)
        L = len(b)
        X[i, 0, :L] = (b == 65)  # A
        X[i, 1, :L] = (b == 84)  # T
        X[i, 2, :L] = (b == 71)  # G
        X[i, 3, :L] = (b == 67)  # C
    return X

print("[FEAT] Encoding sequences...")
X_train = encode_sequences(train_seqs)
X_val   = encode_sequences(val_seqs)
print(f"[FEAT] X_train={X_train.shape}, X_val={X_val.shape}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 256
train_ds = DNADataset(X_train, train_labels)
val_ds   = DNADataset(X_val, val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model: 1D CNN ─────────────────────────────────────────────────────────────
class ResBlock1d(nn.Module):
    """Residual block for 1D sequences."""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)


class DNACNNClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        # Stem: (4, 1024) → (128, 512)
        self.stem = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        # Stage 1: (128, 512) → (128, 256)
        self.stage1 = nn.Sequential(
            ResBlock1d(128, kernel_size=7),
            nn.MaxPool1d(2),
        )
        # Stage 2: (128, 256) → (256, 128)
        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            ResBlock1d(256, kernel_size=5),
            nn.MaxPool1d(2),
        )
        # Stage 3: (256, 128) → (512, 64)
        self.stage3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            ResBlock1d(512, kernel_size=3),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MODEL] Device: {device}")

model = DNACNNClassifier(num_classes=num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Parameters: {n_params:,}")

# ── Focal Loss (alpha = 1/count, normalized) ──────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


counts = np.array([train_dist[c] for c in range(num_classes)], dtype=np.float32)
alpha_raw  = 1.0 / counts
alpha_norm = torch.tensor(alpha_raw / alpha_raw.sum(), dtype=torch.float32)
print(f"[MODEL] Focal alpha: {alpha_norm.tolist()}")
criterion = FocalLoss(alpha=alpha_norm, gamma=2.0).to(device)

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
N_EPOCHS  = 40
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)
scaler    = GradScaler()

# ── Training loop ─────────────────────────────────────────────────────────────
best_f1    = 0.0
best_state = None
patience   = 10
no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(X_b)
            loss   = criterion(logits, y_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    scheduler.step()

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            with autocast():
                logits = model(X_b.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())

    val_f1    = f1_score(val_labels, preds, average='macro')
    per_class = f1_score(val_labels, preds, average=None)
    avg_loss  = epoch_loss / len(train_loader)
    lr_now    = scheduler.get_last_lr()[0]

    print(f"[EPOCH {epoch:02d}/{N_EPOCHS}] loss={avg_loss:.4f} val_f1={val_f1:.4f} "
          f"per_class={[f'{v:.3f}' for v in per_class]} lr={lr_now:.2e}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"[EARLY STOP] No improvement for {patience} epochs, stopping at epoch {epoch}")
            break

# ── Final evaluation ──────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
preds = []
with torch.no_grad():
    for X_b, _ in val_loader:
        with autocast():
            logits = model(X_b.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())

macro_f1  = f1_score(val_labels, preds, average='macro')
per_class = f1_score(val_labels, preds, average=None)
print(f"[EVAL] Per-class F1: {per_class}")
print(f"[EVAL] Best epoch val macro_f1: {best_f1:.6f}")
print(f"BEST_VAL_MACRO_F1: {macro_f1:.6f}")
