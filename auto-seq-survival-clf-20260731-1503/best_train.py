import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from collections import Counter

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"[DATA] Loaded: {df.shape}, columns: {df.columns.tolist()}")

target_col = "label"
seq_col    = "sequence"
task_col   = "task"

labels    = df[target_col].tolist()
sequences = df[seq_col].astype(str).tolist()
tasks     = df[task_col].astype(str).tolist()

num_classes  = len(set(labels))
class_counts = dict(pd.Series(labels).value_counts().sort_index())
print(f"[DATA] Classes: {num_classes}, Distribution: {class_counts}")

# Stratified 80/20 split
(train_seqs,   val_seqs,
 train_labels, val_labels,
 train_tasks,  val_tasks) = train_test_split(
    sequences, labels, tasks,
    test_size=0.2, random_state=42, stratify=labels
)
print(f"[DATA] Train: {len(train_seqs)}, Val: {len(val_seqs)}")

# ── K-mer features via char n-gram TF-IDF ─────────────────────────────────────
# For DNA: char 3-6 grams = k-mers of length 3-6
# ACGT alphabet → 64+256+1024+4096 ≈ 5440 unique features (manageable)
print("[FEAT] Extracting k-mer TF-IDF features (3-6 mers)...")
tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=None,   # DNA alphabet → small vocab, use all
    sublinear_tf=True,   # log(1+tf) dampens high-frequency k-mers
    min_df=2,
    lowercase=False,     # preserve ACGT case
)
X_train_tfidf = tfidf.fit_transform(train_seqs)
X_val_tfidf   = tfidf.transform(val_seqs)
print(f"[FEAT] TF-IDF shape: {X_train_tfidf.shape}")

# ── Task one-hot features ─────────────────────────────────────────────────────
all_task_names = sorted(set(tasks))
task_to_idx    = {t: i for i, t in enumerate(all_task_names)}
n_tasks        = len(all_task_names)
print(f"[FEAT] Tasks: {n_tasks} → {all_task_names}")


def tasks_to_onehot(task_list):
    arr = np.zeros((len(task_list), n_tasks), dtype=np.float32)
    for i, t in enumerate(task_list):
        if t in task_to_idx:
            arr[i, task_to_idx[t]] = 1.0
    return arr


train_task_emb = tasks_to_onehot(train_tasks)
val_task_emb   = tasks_to_onehot(val_tasks)

# Combine TF-IDF + task one-hot
X_train = hstack([X_train_tfidf, csr_matrix(train_task_emb)])
X_val   = hstack([X_val_tfidf,   csr_matrix(val_task_emb)])
print(f"[FEAT] Combined shape: {X_train.shape}")

# ── LightGBM classifier ───────────────────────────────────────────────────────
train_labels_arr = np.array(train_labels)
val_labels_arr   = np.array(val_labels)

counts = Counter(train_labels)
print(f"[TRAIN] Class distribution: {counts}")

clf = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    n_jobs=4,
    random_state=42,
    verbose=-1,
)

print("[TRAIN] Fitting LightGBM with early stopping...")
clf.fit(
    X_train, train_labels_arr,
    eval_set=[(X_val, val_labels_arr)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=200),
    ],
)

val_preds = clf.predict(X_val)
val_acc   = float((val_preds == val_labels_arr).mean())
print(f"\nBEST_VAL_ACCURACY: {val_acc:.6f}")
