import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import lightgbm as lgb

# =====================
# Load and prepare data
# =====================
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
    raise ValueError("Could not detect a sequence (string) column.")

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
print(f"[DATA] Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

labels = df[target_col].tolist()
sequences = df[seq_col].tolist()

train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

num_classes = len(le.classes_)
y_train = np.array(train_labels)
y_val = np.array(val_labels)
train_class_counts = np.bincount(y_train)
val_class_counts = np.bincount(y_val)
print(f"[DATA] Train={len(y_train)}, Val={len(y_val)}, Classes={num_classes}")
print(f"[DATA] Train class counts: {train_class_counts}")
print(f"[DATA] Val class counts: {val_class_counts}")

# =====================
# Feature Extraction: char_wb TF-IDF (3-5, 100k features) — reduced for speed
# =====================
print("\n[FEAT] Fitting char_wb TF-IDF (ngram 3-5, 100k features)...")
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=100000,
    sublinear_tf=True,
)
X_train = tfidf.fit_transform(train_seqs)
X_val = tfidf.transform(val_seqs)
print(f"[FEAT] Shape: train={X_train.shape}, val={X_val.shape}")

# =====================
# Inverse-frequency sample weights (for multiclass model)
# =====================
total_train = len(y_train)
inv_weights = np.array([
    total_train / (num_classes * train_class_counts[c]) for c in y_train
])
print(f"[MC] Sample weight range: [{inv_weights.min():.3f}, {inv_weights.max():.3f}]")


def macro_f1_eval(y_pred, dataset):
    y_true = dataset.get_label().astype(int)
    y_pred_class = np.argmax(y_pred.reshape(-1, num_classes), axis=1)
    return "macro_f1", f1_score(y_true, y_pred_class, average="macro"), True


# =====================
# Model A: Multiclass LightGBM — reduced complexity for speed
# =====================
print("\n[MODEL A] Multiclass LightGBM (num_leaves=63, lr=0.1, min_child_samples=20)...")

params_mc = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "None",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

train_mc = lgb.Dataset(X_train, label=y_train, weight=inv_weights)
val_mc = lgb.Dataset(X_val, label=y_val, reference=train_mc)

model_mc = lgb.train(
    params_mc, train_mc,
    num_boost_round=300,
    valid_sets=[val_mc],
    feval=macro_f1_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)
mc_proba = model_mc.predict(X_val)
y_pred_mc = np.argmax(mc_proba, axis=1)
f1_mc = f1_score(y_val, y_pred_mc, average="macro")
print(f"[MODEL A] macro_f1={f1_mc:.6f}, per-class={f1_score(y_val, y_pred_mc, average=None)}")

# =====================
# Model B: Binary class-2 vs rest
# =====================
y_s1_train = (y_train == 2).astype(float)
y_s1_val = (y_val == 2).astype(float)
class_ratio = float((y_train != 2).sum()) / float((y_train == 2).sum())
spw_boost = 2.0
boosted_spw = class_ratio * spw_boost
print(f"\n[BINARY] class_ratio={class_ratio:.2f}, scale_pos_weight={boosted_spw:.2f} (boost={spw_boost}x)")


def binary_f1_eval(y_pred, dataset):
    y_true = dataset.get_label().astype(int)
    y_pred_class = (y_pred > 0.5).astype(int)
    return "binary_f1", f1_score(y_true, y_pred_class, average="binary"), True


print("\n[MODEL B] Binary LightGBM class-2-vs-rest (num_leaves=63, lr=0.1, min_child_samples=20)...")
params_s1 = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": boosted_spw,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

train_s1 = lgb.Dataset(X_train, label=y_s1_train)
val_s1 = lgb.Dataset(X_val, label=y_s1_val, reference=train_s1)

model_s1 = lgb.train(
    params_s1, train_s1,
    num_boost_round=300,
    valid_sets=[val_s1],
    feval=binary_f1_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)
p2_val = model_s1.predict(X_val)
c2_f1 = f1_score(y_s1_val, (p2_val > 0.5).astype(int), average="binary")
print(f"[MODEL B] Class-2 binary F1={c2_f1:.6f}")
print(f"[MODEL B] Class-2 predicted positives: {(p2_val > 0.5).sum()} / {len(p2_val)}")

# =====================
# Model C: Binary class-0 vs class-1
# =====================
print("\n[MODEL C] Binary LightGBM (class 0 vs class 1 only)...")
mask_01_train = (y_train != 2)
X_train_01 = X_train[mask_01_train]
y_train_01 = y_train[mask_01_train].astype(float)

mask_01_val = (y_val != 2)
X_val_01 = X_val[mask_01_val]
y_val_01 = y_val[mask_01_val].astype(float)

print(f"[MODEL C] Train class01 shape={X_train_01.shape}, counts={np.bincount(y_train_01.astype(int))}")

params_s2 = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

train_s2 = lgb.Dataset(X_train_01, label=y_train_01)
val_s2 = lgb.Dataset(X_val_01, label=y_val_01, reference=train_s2)

model_s2 = lgb.train(
    params_s2, train_s2,
    num_boost_round=300,
    valid_sets=[val_s2],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)
p1_given_not2 = model_s2.predict(X_val)

y_pred_s2 = (model_s2.predict(X_val_01) > 0.5).astype(int)
acc_s2 = (y_pred_s2 == y_val_01.astype(int)).mean()
print(f"[MODEL C] class-0/1 accuracy={acc_s2:.6f}")

# =====================
# Hierarchical combination (two-stage decomposition)
# =====================
hier_proba = np.column_stack([
    (1.0 - p2_val) * (1.0 - p1_given_not2),  # P(class 0)
    (1.0 - p2_val) * p1_given_not2,            # P(class 1)
    p2_val,                                      # P(class 2)
])

y_pred_hier = np.argmax(hier_proba, axis=1)
f1_hier = f1_score(y_val, y_pred_hier, average="macro")
print(f"\n[HIER] macro_f1={f1_hier:.6f}, per-class={f1_score(y_val, y_pred_hier, average=None)}")

# =====================
# Blend: 50% multiclass (Model A) + 50% hierarchical (Models B+C)
# =====================
mc_norm = mc_proba / mc_proba.sum(axis=1, keepdims=True)
hier_norm = hier_proba / hier_proba.sum(axis=1, keepdims=True)

blended_proba = 0.5 * mc_norm + 0.5 * hier_norm
y_pred = np.argmax(blended_proba, axis=1)

macro_f1 = f1_score(y_val, y_pred, average="macro")
per_class_f1 = f1_score(y_val, y_pred, average=None)
print(f"\n[EVAL] Model A (multiclass) macro_f1: {f1_mc:.6f}")
print(f"[EVAL] Hierarchical (B+C) macro_f1: {f1_hier:.6f}")
print(f"[EVAL] Blend 50/50 per-class F1: {per_class_f1}")
print(f"[EVAL] Blend macro_f1: {macro_f1:.6f}")

print(f"BEST_VAL_MACRO_F1: {macro_f1:.6f}")