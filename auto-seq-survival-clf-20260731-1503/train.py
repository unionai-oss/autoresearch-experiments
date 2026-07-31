import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"

# Detect sequence column: non-target string/object column
seq_col = None
for col in df.columns:
    if col == target_col:
        continue
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        seq_col = col
        break

if seq_col is None:
    # Fallback: pick first non-target column
    for col in df.columns:
        if col != target_col:
            seq_col = col
            break

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"[DATA] Class mapping: {class_mapping}")

labels = df[target_col].tolist()
sequences = df[seq_col].astype(str).tolist()

num_classes = len(le.classes_)

# Compute class distribution
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
    f"[DATA] Total samples: {len(sequences)}, "
    f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, "
    f"Class distribution: {class_counts}"
)

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: accuracy
# Final output line must be: print(f"BEST_VAL_ACCURACY: {value:.6f}")