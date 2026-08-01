import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)
print(f"[DATA] Columns: {df.columns.tolist()}")
print(f"[DATA] Shape: {df.shape}")

# Find target column: prefer 'label', then 'survived' (case-insensitive)
target_col = None
for candidate in ["label", "survived", "target", "class"]:
    for col in df.columns:
        if col.lower() == candidate:
            target_col = col
            break
    if target_col is not None:
        break

if target_col is None:
    # Last resort: pick last column
    target_col = df.columns[-1]
    print(f"[DATA] WARNING: could not detect target column; using last column: {target_col}")

print(f"[DATA] Target column: {target_col}")

# Find sequence column: prefer 'sequence', then first string/object non-target column
seq_col = None
for candidate in ["sequence", "text", "sentence", "review", "description"]:
    for col in df.columns:
        if col.lower() == candidate and col != target_col:
            seq_col = col
            break
    if seq_col is not None:
        break

if seq_col is None:
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "string":
            seq_col = col
            break

if seq_col is None:
    raise ValueError(f"Could not find sequence column. Columns: {df.columns.tolist()}")

print(f"[DATA] Sequence column: {seq_col}")

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"[DATA] Class mapping: {class_mapping}")

labels = df[target_col].tolist()
sequences = [str(s) for s in df[seq_col].tolist()]

num_classes = len(le.classes_)
class_counts = df[target_col].value_counts().sort_index().to_dict()

# Stratified 80/20 split
train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(
    f"[DATA] Total: {len(sequences)}, Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, Class distribution: {class_counts}"
)

# ── Frozen DistilBERT CLS embeddings + linear probe ──
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MODEL] Using device: {device}")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EMBED_BATCH = 128
CACHE_DIR = "/tmp/distilbert_embed_cache.npz"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def extract_embeddings(seqs, label=""):
    config = AutoConfig.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME, config=config)
    encoder = encoder.to(device).half()
    encoder.eval()

    all_cls = []
    for start in range(0, len(seqs), EMBED_BATCH):
        batch = seqs[start: start + EMBED_BATCH]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = encoder(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        all_cls.append(cls_emb)
        if (start // EMBED_BATCH) % 10 == 0:
            print(f"[EMBED] {label} {start}/{len(seqs)}")
    return np.vstack(all_cls)

# Cache embeddings to avoid re-extraction if script is re-run
if os.path.exists(CACHE_DIR):
    print("[EMBED] Loading cached embeddings …")
    cached = np.load(CACHE_DIR)
    train_emb = cached["train_emb"]
    val_emb = cached["val_emb"]
else:
    print("[EMBED] Extracting train embeddings …")
    train_emb = extract_embeddings(train_seqs, label="train")
    print("[EMBED] Extracting val embeddings …")
    val_emb = extract_embeddings(val_seqs, label="val")
    np.savez(CACHE_DIR, train_emb=train_emb, val_emb=val_emb)
    print("[EMBED] Embeddings cached.")

print(f"[EMBED] train_emb shape: {train_emb.shape}, val_emb shape: {val_emb.shape}")

# ── Class-imbalance: weighted CrossEntropyLoss if needed ──
counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
if counts.max() > 2.0 * counts.min():
    inv_freq = 1.0 / counts
    inv_freq = inv_freq / inv_freq.sum() * num_classes
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
    print(f"[TRAIN] Using weighted loss, weights: {inv_freq}")
else:
    class_weights = None
    print("[TRAIN] Classes are balanced — using unweighted CrossEntropyLoss")

# ── Linear probe ──
X_train_t = torch.tensor(train_emb, dtype=torch.float32)
y_train_t = torch.tensor(train_labels, dtype=torch.long)
X_val_t = torch.tensor(val_emb, dtype=torch.float32)
y_val_t = torch.tensor(val_labels, dtype=torch.long)

feat_dim = X_train_t.shape[1]
head = nn.Linear(feat_dim, num_classes).to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

N_EPOCHS = 100
best_val_acc = 0.0
patience = 15
no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    head.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = head(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

    head.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = head(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    val_acc = correct / total
    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or improved:
        print(f"[TRAIN] Epoch {epoch:03d}/{N_EPOCHS} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}")

    if no_improve >= patience:
        print(f"[TRAIN] Early stopping at epoch {epoch}")
        break

print(f"BEST_VAL_ACCURACY: {best_val_acc:.6f}")
