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

unique, counts = zip(*sorted(
    zip(*[list(v) for v in zip(*[(cls, (y_encoded == cls).sum()) for cls in sorted(set(y_encoded))])])
))
class_dist_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in zip(unique, counts))
print(f"[DATA] Samples: {len(df)}, Classes: {len(set(y_encoded))}, Class distribution: {{{class_dist_str}}}, Train: {len(X_train)}, Val: {len(X_val)}")

# ── Feature engineering ─────────────────────────────────────────────────────

def engineer_features(X_ref, X_tr, X_te=None):
    """
    Engineer features using X_ref (training set) for computing statistics.
    Adds missingness indicator flags for high-missing columns and converts
    object/string columns to category dtype for LightGBM native handling.
    """
    X_tr = X_tr.copy()
    if X_te is not None:
        X_te = X_te.copy()

    # Missingness indicator flags for columns with >5% missing (measured on train ref)
    for col in X_ref.columns:
        miss_rate = X_ref[col].isna().mean()
        if miss_rate > 0.05:
            flag = f"{col}_missing"
            X_tr[flag] = X_tr[col].isna().astype(np.int8)
            if X_te is not None:
                X_te[flag] = X_te[col].isna().astype(np.int8)

    # Convert object/string columns to category dtype — LightGBM handles them
    # natively (including NaN) when the feature list is passed explicitly.
    cat_cols = X_ref.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        # Fit categories on reference (train) set only
        cats = pd.Categorical(X_ref[col]).categories
        X_tr[col] = pd.Categorical(X_tr[col], categories=cats)
        if X_te is not None:
            X_te[col] = pd.Categorical(X_te[col], categories=cats)

    if X_te is not None:
        return X_tr, X_te
    return X_tr


X_train_fe, X_val_fe = engineer_features(X_train, X_train, X_val)
cat_cols_fe = X_train_fe.select_dtypes(include=["category"]).columns.tolist()
print(f"Features after engineering: {X_train_fe.shape[1]}, categorical={cat_cols_fe}")

# ── LightGBM parameters ──────────────────────────────────────────────────────

lgb_base = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── 5-fold stratified CV to estimate optimal number of boosting rounds ────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
fold_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_fe, y_train)):
    X_f_tr = X_train_fe.iloc[tr_idx]
    X_f_va = X_train_fe.iloc[va_idx]
    y_f_tr = y_train[tr_idx]
    y_f_va = y_train[va_idx]

    clf = lgb.LGBMClassifier(**lgb_base)
    clf.fit(
        X_f_tr, y_f_tr,
        eval_set=[(X_f_va, y_f_va)],
        categorical_feature=cat_cols_fe,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds_va = clf.predict_proba(X_f_va)[:, 1]
    score = roc_auc_score(y_f_va, preds_va)
    fold_scores.append(score)
    fold_best_iters.append(clf.best_iteration_)
    print(f"Fold {fold + 1}: AUC={score:.4f}, best_iter={clf.best_iteration_}")

cv_mean = float(np.mean(fold_scores))
cv_std = float(np.std(fold_scores))
print(f"CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")

# ── Final model: train on full X_train using mean best iteration ─────────────

best_n_est = max(50, int(np.mean(fold_best_iters)))
print(f"Training final model with n_estimators={best_n_est}")

final_params = {**lgb_base, "n_estimators": best_n_est}
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train_fe, y_train,
    categorical_feature=cat_cols_fe,
)

val_preds = final_model.predict_proba(X_val_fe)[:, 1]
val_auc = float(roc_auc_score(y_val, val_preds))
print(f"Val ROC-AUC: {val_auc:.6f}")

print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")
