import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from collections import Counter
import lightgbm as lgb

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
class_dist = dict(Counter(labels))

num_classes = len(class_mapping)

print(
    f"[DATA] Total samples: {len(sequences)}, "
    f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, "
    f"Class distribution: {class_dist}"
)

# ── Step 2 of sequence ladder: k-mer TF-IDF → LightGBM ──────────────────────
# Sequences are DNA (ATCG), variable length 300–1000 chars.
# Not fixed-length so positional (step 1) is skipped.
# ngram_range=(3,6) captures 3-mers through 6-mers; sublinear_tf reduces
# effect of very frequent k-mers in long sequences.

print("[FEAT] Building k-mer TF-IDF features (ngram_range=(3,6))...")
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    sublinear_tf=True,
    lowercase=False,   # DNA is case-sensitive; preserve ATCG case
    min_df=2,          # drop k-mers appearing in only 1 sequence
)
X_train = vectorizer.fit_transform(train_seqs)
X_val   = vectorizer.transform(val_seqs)
print(f"[FEAT] Feature matrix: train={X_train.shape}, val={X_val.shape}")

# ── Class weights (largest/smallest ≈ 21×, must use weighted loss) ───────────
train_counts = Counter(train_labels)
n_train = len(train_labels)
# Balanced weights: n_samples / (n_classes * class_count)
class_weight = {
    cls: n_train / (num_classes * count)
    for cls, count in train_counts.items()
}
print(f"[MODEL] Class weights: {class_weight}")

# ── LightGBM classifier ───────────────────────────────────────────────────────
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight=class_weight,
    n_jobs=4,
    random_state=42,
    verbose=-1,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

model.fit(
    X_train, train_labels,
    eval_set=[(X_val, val_labels)],
    callbacks=callbacks,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
val_preds = model.predict(X_val)
macro_f1 = f1_score(val_labels, val_preds, average="macro")

per_class_f1 = f1_score(val_labels, val_preds, average=None)
print(f"[EVAL] Per-class F1: {per_class_f1}")
print(f"[EVAL] Best iteration: {model.best_iteration_}")

print(f"BEST_VAL_MACRO_F1: {macro_f1:.6f}")
