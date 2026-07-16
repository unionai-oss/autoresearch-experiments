import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
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
print(f"[DATA] Class mapping: {class_mapping}")

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

# Class distribution in full dataset
num_classes = len(le.classes_)
class_counts = df[target_col].value_counts().sort_index().to_dict()
print(
    f"[DATA] Total samples: {len(df)}, "
    f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, "
    f"Classes: {num_classes}, "
    f"Class distribution: {class_counts}"
)

# ── Feature extraction: char n-gram TF-IDF (3–4) ──────────────────────────────
print("[FEAT] Fitting TF-IDF vectorizer (char_wb ngram 3-4, max_features=50k)...")
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    max_features=50_000,
    sublinear_tf=True,
    min_df=2,
)
X_train = tfidf.fit_transform(train_seqs)
X_val = tfidf.transform(val_seqs)
print(f"[FEAT] Feature matrix: train={X_train.shape}, val={X_val.shape}")

y_train = np.array(train_labels)
y_val = np.array(val_labels)

# ── Class-balanced sample weights ─────────────────────────────────────────────
class_sample_counts = np.bincount(y_train)
class_weights = len(y_train) / (num_classes * class_sample_counts)
sample_weights = class_weights[y_train]
print(f"[MODEL] Class weights per class: {class_weights}")

# ── LightGBM ───────────────────────────────────────────────────────────────────
lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "seed": 42,
    "verbose": -1,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=20, verbose=True),
    lgb.log_evaluation(period=20),
]

print("[MODEL] Training LightGBM...")
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=300,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1)

macro_f1 = f1_score(y_val, y_pred, average="macro")
per_class_f1 = f1_score(y_val, y_pred, average=None)
print(f"[EVAL] Per-class F1: {per_class_f1}")
print(f"[EVAL] Val macro F1: {macro_f1:.6f}")

print(f"BEST_VAL_MACRO_F1: {macro_f1:.6f}")