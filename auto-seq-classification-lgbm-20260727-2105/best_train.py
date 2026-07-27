import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)

# Try to find the target column case-insensitively
target_col = None
for col in df.columns:
    if col.lower() == "survived":
        target_col = col
        break

if target_col is None:
    # Fallback: use last column as target
    target_col = df.columns[-1]

print(f"Using target column: {target_col}")
print(f"Columns in dataframe: {list(df.columns)}")

# Detect the sequence column: non-target string/object column
seq_col = None
for col in df.columns:
    if col == target_col:
        continue
    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
        seq_col = col
        break

if seq_col is None:
    # Fallback: pick any non-target column
    for col in df.columns:
        if col != target_col:
            seq_col = col
            break

print(f"Using sequence column: {seq_col}")

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

labels = df[target_col].tolist()
sequences = df[seq_col].tolist()

# Ensure sequences are strings
sequences = [str(s) for s in sequences]

num_classes = len(set(labels))

# Stratified 80/20 split
train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Compute class distribution
from collections import Counter
class_dist = Counter(labels)
print(f"[DATA] Total samples: {len(sequences)}, Val samples: {len(val_seqs)}, "
      f"Classes: {num_classes}, Class distribution: {dict(class_dist)}")

# ---- Tier C: Frozen RoBERTa-base + Linear Probe ----

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 64

CACHE_TRAIN = "/tmp/emb_train_roberta.npy"
CACHE_VAL = "/tmp/emb_val_roberta.npy"


class SequenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def extract_embeddings(model, tokenizer, seqs, cache_path, batch_size=64):
    """Extract CLS embeddings at FP16, cache to disk, return float32 array."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    dataset = SequenceDataset(seqs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            encoded = tokenizer(
                list(batch),
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            outputs = model(**encoded)
            # CLS token = position 0 of last_hidden_state
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            # Cast to float32 for numpy
            all_embeddings.append(cls_emb.float().cpu().numpy())

            if (i + 1) % 20 == 0:
                done = min((i + 1) * batch_size, len(seqs))
                print(f"  Embedded {done}/{len(seqs)}")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(cache_path, embeddings)
    print(f"Cached {embeddings.shape} embeddings -> {cache_path}")
    return embeddings


# Load tokenizer and model
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Freeze all encoder parameters — frozen probe approach
for param in model.parameters():
    param.requires_grad = False

# Use FP16 on GPU for memory-efficient extraction
if DEVICE.type == "cuda":
    model = model.half().to(DEVICE)
else:
    model = model.to(DEVICE)

print("Extracting train embeddings ...")
train_emb = extract_embeddings(model, tokenizer, train_seqs, CACHE_TRAIN, BATCH_SIZE)
print(f"Train embeddings: {train_emb.shape}")

print("Extracting val embeddings ...")
val_emb = extract_embeddings(model, tokenizer, val_seqs, CACHE_VAL, BATCH_SIZE)
print(f"Val embeddings: {val_emb.shape}")

# Free GPU memory before sklearn training
del model
torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# Train linear probe (logistic regression)
print("Training logistic regression linear probe ...")
clf = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    multi_class="auto",
    random_state=42,
    n_jobs=-1,
)
clf.fit(train_emb, train_labels)

# Evaluate on validation set
val_preds = clf.predict(val_emb)
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Val accuracy: {val_accuracy:.6f}")

print(f"BEST_VAL_ACCURACY: {val_accuracy:.6f}")