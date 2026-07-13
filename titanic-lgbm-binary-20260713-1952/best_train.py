import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
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
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

unique, counts = zip(*[(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()])
class_dist = {cls: int((y_encoded == le.transform([cls])[0]).sum()) for cls in le.classes_}

print(
    f"[DATA] Samples: {len(df)} total | "
    f"Train: {len(X_train)}, Val: {len(X_val)} | "
    f"Class distribution: {class_dist}"
)

# ── Feature engineering ──────────────────────────────────────────────────────

def engineer(X_df, ref_df=None):
    """
    Apply feature engineering.
    ref_df: used to compute fill statistics (pass X_train when transforming X_val).
    """
    X = X_df.copy()
    ref = ref_df if ref_df is not None else X_df

    num_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ref.select_dtypes(exclude=[np.number]).columns.tolist()

    # Missingness indicators for columns with >5% missing in ref
    for col in ref.columns:
        if ref[col].isnull().mean() > 0.05:
            X[f"{col}_missing"] = X[col].isnull().astype(np.int8)

    # Impute: median for numeric, mode for categorical
    for col in num_cols:
        fill = ref[col].median()
        X[col] = X[col].fillna(fill)

    for col in cat_cols:
        modes = ref[col].mode()
        fill = modes.iloc[0] if len(modes) > 0 else "Unknown"
        X[col] = X[col].fillna(fill)

    return X, num_cols, cat_cols


X_train_fe, num_cols, cat_cols = engineer(X_train)
X_val_fe, _, _ = engineer(X_val, ref_df=X_train)

# Ordinal-encode categoricals (LightGBM will treat them as numeric;
# we mark them as categorical in the Dataset so it handles splits natively)
if cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_fe[cat_cols] = oe.fit_transform(X_train_fe[cat_cols])
    X_val_fe[cat_cols] = oe.transform(X_val_fe[[c for c in cat_cols if c in X_val_fe.columns]])

# Align val columns to train
X_val_fe = X_val_fe.reindex(columns=X_train_fe.columns, fill_value=0)

feature_names = X_train_fe.columns.tolist()
cat_feature_idxs = [feature_names.index(c) for c in cat_cols if c in feature_names]

# ── LightGBM with 5-fold stratified CV (for robust AUC on tiny N=100) ───────

lgb_params = dict(
    objective="binary",
    metric="auc",
    verbosity=-1,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=4,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)

# OOF on the full dataset ──────────────────────────────────────────────────
# With N=100, CV gives a much more stable AUC than a single 20-sample hold-out.
X_full_fe, _, _ = engineer(X, ref_df=X)
if cat_cols:
    oe_full = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_full_fe[cat_cols] = oe_full.fit_transform(X_full_fe[cat_cols])
X_full_fe = X_full_fe.reindex(columns=feature_names, fill_value=0)

X_arr = X_full_fe.values.astype(np.float32)
y_arr = y_encoded.astype(np.int32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_arr))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr)):
    Xtr, Xva = X_arr[tr_idx], X_arr[va_idx]
    ytr, yva = y_arr[tr_idx], y_arr[va_idx]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        categorical_feature=cat_feature_idxs if cat_feature_idxs else "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_preds[va_idx] = clf.predict_proba(Xva)[:, 1]
    fold_auc = roc_auc_score(yva, oof_preds[va_idx])
    print(f"  Fold {fold + 1}/5  AUC={fold_auc:.4f}  best_iter={clf.best_iteration_}")

oof_auc = roc_auc_score(y_arr, oof_preds)
print(f"OOF AUC (5-fold, N={len(df)}): {oof_auc:.6f}")

# ── Val-set AUC using model trained on X_train ───────────────────────────────
clf_val = lgb.LGBMClassifier(**lgb_params)
clf_val.fit(
    X_train_fe.values.astype(np.float32), y_train,
    eval_set=[(X_val_fe.values.astype(np.float32), y_val)],
    categorical_feature=cat_feature_idxs if cat_feature_idxs else "auto",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)
val_preds = clf_val.predict_proba(X_val_fe.values.astype(np.float32))[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Val-set AUC (80/20 split, N_val={len(y_val)}): {val_auc:.6f}")

# Report the OOF AUC as the primary metric — far more reliable at N=100
best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
