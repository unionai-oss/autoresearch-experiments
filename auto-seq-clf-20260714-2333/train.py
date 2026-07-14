import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)

target_col = "label"

# Detect sequence column: non-target string/object column
seq_col = None
for col in df.columns:
    if col == target_col:
        continue
    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
        seq_col = col
        break

if seq_col is None:
    # Fallback: pick first non-target column
    seq_col = [c for c in df.columns if c != target_col][0]

print(f"[DATA] Detected sequence column: '{seq_col}'")

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"[DATA] Class mapping: {class_mapping}")

# Compute class distribution
class_dist = df[target_col].value_counts().sort_index().to_dict()

# Stratified 80/20 split
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[target_col]
)

train_seqs = train_df[seq_col].tolist()
val_seqs = val_df[seq_col].tolist()
train_labels = train_df[target_col].tolist()
val_labels = val_df[target_col].tolist()

num_classes = len(class_mapping)

print(
    f"[DATA] Total samples: {len(df)}, "
    f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, "
    f"Class distribution: {class_dist}"
)

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: macro_f1
# Final output line must be: print(f"BEST_VAL_MACRO_F1: {value:.6f}")