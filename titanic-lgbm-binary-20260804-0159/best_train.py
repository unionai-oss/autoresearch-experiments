import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

import numpy as np
unique, counts = np.unique(y_encoded, return_counts=True)
class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"Dataset summary: total samples={len(df)}, class distribution={class_dist}, train={len(X_train)}, val={len(X_val)}")

# ---------- Feature Engineering ----------
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

# Make explicit copies to avoid SettingWithCopyWarning
X_train = X_train.copy()
X_val = X_val.copy()
y_train = np.array(y_train)
y_val = np.array(y_val)

# Identify column types
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical columns: {cat_cols}")
print(f"Numeric columns: {num_cols}")

# Add binary missingness-indicator flags for columns with >5% missing (computed on train)
miss_thresh = 0.05
for col in list(X_train.columns):
    miss_rate = X_train[col].isna().mean()
    if miss_rate > miss_thresh:
        miss_col = f"{col}_missing"
        X_train[miss_col] = X_train[col].isna().astype(int)
        X_val[miss_col] = X_val[col].isna().astype(int)
        print(f"  Added missingness indicator '{miss_col}' (train miss rate: {miss_rate:.1%})")

# Impute numeric columns with median (fit on train only)
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_val[num_cols] = num_imputer.transform(X_val[num_cols])

# Impute categorical columns with most-frequent, then label-encode for LightGBM
if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])

    for col in cat_cols:
        le_cat = LabelEncoder()
        X_train[col] = le_cat.fit_transform(X_train[col].astype(str))
        # Handle any unseen categories in val
        val_str = X_val[col].astype(str)
        unseen = ~val_str.isin(le_cat.classes_)
        if unseen.any():
            val_str = val_str.copy()
            val_str[unseen] = le_cat.classes_[0]
        X_val[col] = le_cat.transform(val_str)
        # Mark as categorical so LightGBM handles splits natively
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

# Class imbalance weight (549 vs 342 → ratio ~1.6, mild but worth setting)
scale_pos_weight = class_dist[0] / class_dist[1]

# ---------- Grid-based HPO: 5-fold stratified CV on training set ----------
import itertools

param_grid = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [10, 30],
    "feature_fraction": [0.7, 1.0],
    "bagging_fraction": [0.7, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.0, 0.1],
}

keys = list(param_grid.keys())
values = list(param_grid.values())
all_combinations = list(itertools.product(*values))

# Randomly sample up to 60 combinations
rng = np.random.RandomState(42)
indices = rng.choice(len(all_combinations), size=min(60, len(all_combinations)), replace=False)
sampled_combinations = [all_combinations[i] for i in indices]

print(f"Starting grid search ({len(sampled_combinations)} trials, 5-fold CV)...")

best_cv_score = -1.0
best_params = None

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for combo in sampled_combinations:
    params = dict(zip(keys, combo))
    params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 1,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 1000,
        "random_state": 42,
    })

    cv_scores = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        preds = model.predict_proba(X_va)[:, 1]
        cv_scores.append(roc_auc_score(y_va, preds))

    mean_score = float(np.mean(cv_scores))
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_params = params.copy()

print(f"Best CV ROC-AUC: {best_cv_score:.6f}")
print(f"Best params: {best_params}")

# ---------- Final model: best params, early-stopped on held-out val ----------
final_params = best_params.copy()
final_params["n_estimators"] = 2000

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

val_preds = final_model.predict_proba(X_val)[:, 1]
val_roc_auc = roc_auc_score(y_val, val_preds)

print(f"BEST_VAL_ROC_AUC: {val_roc_auc:.6f}")