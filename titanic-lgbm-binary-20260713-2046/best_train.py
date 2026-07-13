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
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

unique, counts = zip(*[(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()])
class_dist = {str(cls): int(cnt) for cls, cnt in zip(unique, counts)}
print(f"[DATA] Samples: {len(df)}, Classes: {len(class_mapping)}, Class distribution: {class_dist}, Train: {len(X_train)}, Val: {len(X_val)}")

# ── Feature engineering ────────────────────────────────────────────────────────

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

X_tr = X_train.copy()
X_vl = X_val.copy()

# Add missingness indicator flags for columns with >5% missing (computed on train only)
miss_threshold = 0.05
for col in X_train.columns:
    miss_rate = X_train[col].isnull().mean()
    if miss_rate > miss_threshold:
        X_tr[f"{col}_missing"] = X_tr[col].isnull().astype(np.int8)
        X_vl[f"{col}_missing"] = X_vl[col].isnull().astype(np.int8)
        print(f"  Added missingness flag: {col}_missing (train miss rate: {miss_rate:.2%})")

# Impute numeric with median (computed on train)
for col in num_cols:
    median_val = X_train[col].median()
    X_tr[col] = X_tr[col].fillna(median_val)
    X_vl[col] = X_vl[col].fillna(median_val)

# Impute categoricals with mode (computed on train)
for col in cat_cols:
    mode_val = X_train[col].mode()
    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
    X_tr[col] = X_tr[col].fillna(fill_val)
    X_vl[col] = X_vl[col].fillna(fill_val)

# Label-encode categorical columns
# Fit on the union of train + val to handle any unseen values safely
cat_cols_fe = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()
le_dict = {}
for col in cat_cols_fe:
    le_col = LabelEncoder()
    combined = pd.concat([X_tr[col], X_vl[col]]).astype(str).unique()
    le_col.fit(combined)
    X_tr[col] = le_col.transform(X_tr[col].astype(str))
    X_vl[col] = le_col.transform(X_vl[col].astype(str))
    le_dict[col] = le_col

print(f"Feature matrix shape — train: {X_tr.shape}, val: {X_vl.shape}")

# ── 5-fold stratified CV on X_train (for OOF AUC insight) ─────────────────────

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_tr))

X_tr_np = X_tr.values
y_tr_np = y_train

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_np, y_tr_np)):
    fold_X_tr, fold_y_tr = X_tr_np[tr_idx], y_tr_np[tr_idx]
    fold_X_va, fold_y_va = X_tr_np[va_idx], y_tr_np[va_idx]

    dtrain = lgb.Dataset(fold_X_tr, label=fold_y_tr, feature_name=X_tr.columns.tolist())
    dval   = lgb.Dataset(fold_X_va, label=fold_y_va, reference=dtrain)

    cb = [
        lgb.early_stopping(stopping_rounds=60, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    fold_model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=cb,
    )
    oof_preds[va_idx] = fold_model.predict(fold_X_va)
    fold_auc = roc_auc_score(fold_y_va, oof_preds[va_idx])
    print(f"  Fold {fold+1} AUC: {fold_auc:.4f}  (best iter: {fold_model.best_iteration})")

oof_auc = roc_auc_score(y_tr_np, oof_preds)
print(f"OOF AUC (5-fold CV on train split): {oof_auc:.4f}")

# ── Final model: train on all X_train, evaluate on X_val ──────────────────────

dtrain_full = lgb.Dataset(X_tr, label=y_train, feature_name=X_tr.columns.tolist())
dval_final  = lgb.Dataset(X_vl, label=y_val,   reference=dtrain_full)

cb_final = [
    lgb.early_stopping(stopping_rounds=60, verbose=False),
    lgb.log_evaluation(period=50),
]
final_model = lgb.train(
    LGBM_PARAMS,
    dtrain_full,
    num_boost_round=1000,
    valid_sets=[dval_final],
    callbacks=cb_final,
)

val_preds = final_model.predict(X_vl)
best_val_roc_auc = roc_auc_score(y_val, val_preds)
print(f"Val AUC (final model on hold-out): {best_val_roc_auc:.4f}")
print(f"Best iteration: {final_model.best_iteration}")

# Feature importances (gain)
feat_imp = pd.Series(
    final_model.feature_importance(importance_type="gain"),
    index=X_tr.columns
).sort_values(ascending=False)
print(f"Top features:\n{feat_imp.head(10).to_string()}")

print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
