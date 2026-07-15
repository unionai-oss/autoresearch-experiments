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
    raise ValueError("Could not detect a sequence (string) column in the dataset.")

# Encode labels to 0-based integers
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

labels = df[target_col].tolist()
sequences = df[seq_col].tolist()

# Stratified 80/20 split
train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Compute class distribution in full dataset
from collections import Counter
class_dist = dict(Counter(labels))

num_classes = len(class_mapping)

print(
    f"[DATA] Total samples: {len(sequences)}, "
    f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, "
    f"Class distribution: {class_dist}"
)

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: macro_f1
# Final output line must be: print(f"BEST_VAL_MACRO_F1: {value:.6f}")