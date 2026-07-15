import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

unique, counts = zip(*[(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()])
class_dist_str = ", ".join([f"{cls}: {cnt}" for cls, cnt in zip(unique, counts)])
print(f"Dataset summary: total_samples={len(df)}, train={len(X_train)}, val={len(X_val)}, class_distribution={{{class_dist_str}}}")

# ===================== Preprocessing =====================

def preprocess(X_tr, X_va):
    """Fit all preprocessing on X_tr, apply identically to X_va (no leakage)."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()

    # Missingness indicator flags for columns with >5% missing in the training fold
    for col in num_cols + cat_cols:
        if X_tr[col].isnull().mean() > 0.05:
            X_tr[f"{col}_missing"] = X_tr[col].isnull().astype(int)
            X_va[f"{col}_missing"] = X_va[col].isnull().astype(int)

    # Numeric: impute with training-fold median
    for col in num_cols:
        median = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(median)
        X_va[col] = X_va[col].fillna(median)

    # Categorical: fill missing then label-encode (fit on training fold only)
    encoded_cat_cols = []
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
        X_va[col] = X_va[col].fillna("MISSING").astype(str)
        le_col = LabelEncoder()
        le_col.fit(X_tr[col])
        known = set(le_col.classes_)
        # Map unseen val labels to first known class (rare with Titanic, but safe)
        X_va[col] = X_va[col].apply(lambda x: x if x in known else le_col.classes_[0])
        X_tr[col] = le_col.transform(X_tr[col])
        X_va[col] = le_col.transform(X_va[col])
        encoded_cat_cols.append(col)

    return X_tr, X_va, encoded_cat_cols


# ===================== LightGBM hyperparameters =====================

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

# ===================== 5-Fold Stratified CV (recommended for N<10k) =====================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_encoded))
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_encoded)):
    X_fold_tr = X.iloc[tr_idx]
    X_fold_va = X.iloc[va_idx]
    y_fold_tr = y_encoded[tr_idx]
    y_fold_va = y_encoded[va_idx]

    X_fold_tr_p, X_fold_va_p, cat_cols = preprocess(X_fold_tr, X_fold_va)

    ds_tr = lgb.Dataset(X_fold_tr_p, label=y_fold_tr, categorical_feature=cat_cols)
    ds_va = lgb.Dataset(X_fold_va_p, label=y_fold_va, reference=ds_tr)

    model = lgb.train(
        lgb_params,
        ds_tr,
        num_boost_round=1000,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[va_idx] = model.predict(X_fold_va_p)
    fold_auc = roc_auc_score(y_fold_va, oof_preds[va_idx])
    fold_aucs.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y_encoded, oof_preds)
print(
    f"OOF AUC: {oof_auc:.6f} | "
    f"Mean fold: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}"
)

# ===================== Final model on skeleton's 80/20 split =====================

X_train_p, X_val_p, cat_cols_holdout = preprocess(X_train, X_val)

ds_train = lgb.Dataset(X_train_p, label=y_train, categorical_feature=cat_cols_holdout)
ds_val = lgb.Dataset(X_val_p, label=y_val, reference=ds_train)

model_final = lgb.train(
    lgb_params,
    ds_train,
    num_boost_round=1000,
    valid_sets=[ds_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

val_preds = model_final.predict(X_val_p)
holdout_auc = roc_auc_score(y_val, val_preds)
print(f"Holdout val AUC: {holdout_auc:.6f}")

# Report OOF AUC as the primary metric (more reliable for N<10k per program.md)
best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
