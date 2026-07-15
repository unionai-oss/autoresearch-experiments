import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_parquet(DATA_PATH)

# Detect sequence column (non-target string column)
target_col = "label"
seq_col = None
for col in df.columns:
    if col != target_col and pd.api.types.is_string_dtype(df[col]):
        seq_col = col
        break
if seq_col is None:
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            seq_col = col
            break

print(f"[DATA] Detected sequence column: '{seq_col}'")

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"[DATA] Class mapping: {class_mapping}")

# Compute class distribution
class_dist = df[target_col].value_counts().sort_index().to_dict()
print(f"[DATA] Dataset summary: {len(df)} samples, class distribution: {class_dist}")

# Split into train/val (80/20, stratified)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target_col]
)

# Extract as plain Python lists
train_seqs = train_df[seq_col].tolist()
val_seqs = val_df[seq_col].tolist()
train_labels = train_df[target_col].tolist()
val_labels = val_df[target_col].tolist()

num_classes = len(le.classes_)

print(f"[DATA] Train samples: {len(train_seqs)}, Val samples: {len(val_seqs)}, Num classes: {num_classes}")

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: macro_f1
# Final output line must be: print(f"BEST_VAL_MACRO_F1: {value:.6f}")