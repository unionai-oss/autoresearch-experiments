import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

# Global seed for reproducible data splitting
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

RC_TABLE = str.maketrans('ATGCatgcNn', 'TACGtacgNn')

def reverse_complement(seq):
    return seq.translate(RC_TABLE)[::-1]

def random_mutate(seq, mut_rate=0.02):
    """Apply random base substitutions (SNP-like augmentation) to increase diversity."""
    bases = 'ATGC'
    seq = seq.upper()
    result = []
    for ch in seq:
        if ch in bases and random.random() < mut_rate:
            result.append(random.choice(bases))
        else:
            result.append(ch)
    return ''.join(result)

MAX_LEN = 800
BATCH_SIZE = 128


class DNADataset(Dataset):
    def __init__(self, seqs, y, max_len=MAX_LEN, augment=False):
        self.seqs = seqs
        self.y = torch.tensor(y, dtype=torch.long)
        self.max_len = max_len
        self.augment = augment

    def _encode(self, seq):
        x = np.zeros((4, self.max_len), dtype=np.float32)
        s = seq[:self.max_len].upper()
        b = np.frombuffer(s.encode(), dtype=np.uint8)
        L = len(b)
        x[0, :L] = (b == 65)   # A
        x[1, :L] = (b == 84)   # T
        x[2, :L] = (b == 71)   # G
        x[3, :L] = (b == 67)   # C
        return x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.augment:
            # Reverse complement (50% prob)
            if random.random() < 0.5:
                seq = reverse_complement(seq)
            # Random window selection for sequences longer than max_len
            if len(seq) > self.max_len:
                start = random.randint(0, len(seq) - self.max_len)
                seq = seq[start:start + self.max_len]
            # Random mutation (SNP-like, 30% prob, 2% rate)
            if random.random() < 0.3:
                seq = random_mutate(seq, mut_rate=0.02)
        return torch.from_numpy(self._encode(seq)), self.y[idx]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.net(x)) + x)


class DNACNNv3(nn.Module):
    """Multi-scale dilated CNN with SE attention and attention pooling."""
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        # Multi-scale stem: 3 kernel sizes to capture short/medium/long motifs
        self.stem_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 32, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(32),
                nn.GELU(),
            )
            for k in [3, 7, 15]
        ])
        # Project from 96 -> 128
        self.stem_proj = nn.Sequential(
            nn.Conv1d(96, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.stem_pool = nn.MaxPool1d(2)  # 800 -> 400

        self.stage1 = nn.Sequential(
            ResBlock(128, kernel_size=7, dilation=1, dropout=0.1),
            ResBlock(128, kernel_size=7, dilation=2, dropout=0.1),
            ResBlock(128, kernel_size=7, dilation=4, dropout=0.1),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(128, 192, 1, bias=False),
            nn.BatchNorm1d(192),
            nn.MaxPool1d(2),  # 400 -> 200
        )

        self.stage2 = nn.Sequential(
            ResBlock(192, kernel_size=5, dilation=1, dropout=0.1),
            ResBlock(192, kernel_size=5, dilation=2, dropout=0.1),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(192, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),  # 200 -> 100
        )

        self.stage3 = nn.Sequential(
            ResBlock(256, kernel_size=3, dilation=1, dropout=0.1),
            ResBlock(256, kernel_size=3, dilation=2, dropout=0.1),
        )

        # Attention pooling aggregates position-specific information
        self.attn_pool = nn.Conv1d(256, 1, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Multi-scale stem
        x = torch.cat([b(x) for b in self.stem_branches], dim=1)
        x = self.stem_proj(x)
        x = self.stem_pool(x)

        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        # Attention pooling
        w = torch.softmax(self.attn_pool(x), dim=-1)
        x = (x * w).sum(dim=-1)

        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MODEL] Device: {device}")
use_cuda = device.type == "cuda"

# Pre-compute WRS weights (fixed across models — only randomness is in sampling order)
counts = np.array([train_dist[c] for c in range(num_classes)], dtype=np.float64)
class_weights_sampler = 1.0 / counts
sample_weights = np.array([class_weights_sampler[l] for l in train_labels], dtype=np.float32)

# Pre-compute validation datasets + loaders (shared across ensemble members)
val_rc_seqs = [reverse_complement(s) for s in val_seqs]
val_ds     = DNADataset(val_seqs,    val_labels, augment=False)
val_ds_rc  = DNADataset(val_rc_seqs, val_labels, augment=False)
val_loader    = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
val_loader_rc = DataLoader(val_ds_rc, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

N_EPOCHS = 60
PATIENCE = 15
# Train 2 independent models with different seeds — averaging reduces variance
# from 21x class imbalance (high per-run variance for class 2 predictions)
ENSEMBLE_SEEDS = [42, 7]


def train_one_model(seed):
    """Train a single CNN model with the given seed. Returns best state dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Fresh dataset + sampler per model (augmentation randomness controlled by seed)
    train_ds_local = DNADataset(train_seqs, train_labels, augment=True)
    sampler_local = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_labels),
        replacement=True,
    )
    train_loader_local = DataLoader(
        train_ds_local, batch_size=BATCH_SIZE,
        sampler=sampler_local, num_workers=0, pin_memory=True
    )

    model = DNACNNv3(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [seed={seed}] Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    if use_cuda:
        scaler = GradScaler('cuda')
    else:
        scaler = None

    best_f1 = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader_local:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                with autocast('cuda'):
                    logits = model(X_b)
                    loss   = criterion(logits, y_b)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                if use_cuda:
                    with autocast('cuda'):
                        logits = model(X_b.to(device))
                else:
                    logits = model(X_b.to(device))
                preds.extend(logits.argmax(1).cpu().numpy())

        val_f1   = f1_score(val_labels, preds, average='macro')
        avg_loss = epoch_loss / len(train_loader_local)
        lr_now   = scheduler.get_last_lr()[0]

        print(f"  [seed={seed} EPOCH {epoch:02d}/{N_EPOCHS}] loss={avg_loss:.4f} val_f1={val_f1:.4f} lr={lr_now:.2e}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [EARLY STOP seed={seed}] No improvement for {PATIENCE} epochs at epoch {epoch}")
                break

    print(f"  [seed={seed}] Best val_f1: {best_f1:.4f}")
    return best_state


def get_softmax_probs(model, loader):
    """Get softmax probability matrix from model on a DataLoader."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            if use_cuda:
                with autocast('cuda'):
                    logits = model(X_b.to(device))
            else:
                logits = model(X_b.to(device))
            all_probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(all_probs, dim=0)


# ── Ensemble training ──────────────────────────────────────────────────────────
best_states = []
for seed in ENSEMBLE_SEEDS:
    print(f"\n[ENSEMBLE] Training model with seed={seed}")
    state = train_one_model(seed)
    best_states.append(state)

# ── Inference: ensemble (2 models) × TTA (fwd + RC) = 4 predictions averaged ─
inference_model = DNACNNv3(num_classes=num_classes).to(device)
all_probs = []

for i, state in enumerate(best_states):
    inference_model.load_state_dict(state)
    probs_fwd = get_softmax_probs(inference_model, val_loader)
    probs_rc  = get_softmax_probs(inference_model, val_loader_rc)
    model_probs = (probs_fwd + probs_rc) / 2
    all_probs.append(model_probs)
    print(f"[INFERENCE] Model {i+1}/{len(best_states)} done")

ensemble_probs = sum(all_probs) / len(all_probs)
preds = ensemble_probs.argmax(1).numpy()

macro_f1  = f1_score(val_labels, preds, average='macro')
per_class = f1_score(val_labels, preds, average=None)
print(f"[EVAL] Per-class F1 (ensemble+TTA): {per_class}")
print(f"BEST_VAL_MACRO_F1: {macro_f1:.6f}")
