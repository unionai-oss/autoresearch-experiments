import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
# Ensure user-installed packages (optuna, catboost) are on the path
_user_site = '/home/flyte/.local/lib/python3.12/site-packages'
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

train_class_dist = {int(cls): int((y_train == cls).sum()) for cls in sorted(set(y_train))}
val_class_dist = {int(cls): int((y_val == cls).sum()) for cls in sorted(set(y_val))}

print(
    f"[DATA] Total samples: {len(df)}, "
    f"Train: {len(X_train)}, Val: {len(X_val)}, "
    f"Train class distribution: {train_class_dist}, "
    f"Val class distribution: {val_class_dist}"
)

# ── Feature type detection ──────────────────────────────────────────────────
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype) == 'category']
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

# Columns with >5% missing in train — add binary missingness indicators
miss_rate_train = X_train.isnull().mean()
miss_cols = miss_rate_train[miss_rate_train > 0.05].index.tolist()
print(f"High-missing cols (>5%): {miss_cols}")


def preprocess(X_tr, X_v, num_cols, cat_cols, miss_cols):
    """
    Returns (X_tr_proc, X_v_proc) as DataFrames.
    - Adds binary missingness flags for high-missing columns (using train set rates).
    - Median-imputes numeric, mode-imputes categorical.
    - Label-encodes categorical columns (integers for LightGBM categorical_feature).
    """
    X_tr = X_tr.copy()
    X_v = X_v.copy()

    # Convert string/category columns to plain object dtype to avoid arrow string issues
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype(object)
        X_v[col] = X_v[col].astype(object)

    # Binary missingness indicators
    for col in miss_cols:
        X_tr[f'{col}_miss'] = X_tr[col].isnull().astype(np.int8)
        X_v[f'{col}_miss'] = X_v[col].isnull().astype(np.int8)

    # Numeric imputation (median from train)
    for col in num_cols:
        # Convert to numeric to avoid string dtype median error
        X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
        X_v[col] = pd.to_numeric(X_v[col], errors='coerce')
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_v[col] = X_v[col].fillna(med)

    # Categorical imputation + label encoding
    for col in cat_cols:
        mode_ser = X_tr[col].mode()
        mode_val = mode_ser.iloc[0] if len(mode_ser) > 0 else 'unknown'
        X_tr[col] = X_tr[col].fillna(mode_val)
        X_v[col] = X_v[col].fillna(mode_val)
        enc = LabelEncoder()
        X_tr[col] = enc.fit_transform(X_tr[col].astype(str))
        known = set(enc.classes_)
        fallback = enc.classes_[0]
        X_v[col] = enc.transform(
            X_v[col].astype(str).apply(lambda x: x if x in known else fallback)
        )

    return X_tr, X_v


X_train_proc, X_val_proc = preprocess(X_train, X_val, num_cols, cat_cols, miss_cols)
print(f"Processed feature shape — train: {X_train_proc.shape}, val: {X_val_proc.shape}")

# ── LightGBM + Optuna hyperparameter tuning ────────────────────────────────
def lgb_objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 4, 31),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    for tr_idx, vl_idx in skf.split(X_train_proc, y_train):
        X_tr = X_train_proc.iloc[tr_idx]
        X_vl = X_train_proc.iloc[vl_idx]
        y_tr = y_train[tr_idx]
        y_vl = y_train[vl_idx]

        if len(np.unique(y_vl)) < 2:
            continue  # skip fold if only one class present

        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, categorical_feature=cat_cols)
        prob = m.predict_proba(X_vl)[:, 1]
        fold_scores.append(roc_auc_score(y_vl, prob))

    return np.mean(fold_scores) if fold_scores else 0.5


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(lgb_objective, n_trials=100, timeout=65)
print(f"LightGBM Optuna best CV AUC: {study.best_value:.6f}  params: {study.best_params}")

# Final LightGBM trained on full train split
best_lgb_params = study.best_params.copy()
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
})
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
lgb_model.fit(X_train_proc, y_train, categorical_feature=cat_cols)
lgb_pred = lgb_model.predict_proba(X_val_proc)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_pred)
print(f"LightGBM Val ROC-AUC: {lgb_auc:.6f}")

# ── CatBoost — native categorical handling ────────────────────────────────
# cat_features accepts column indices into the processed DataFrame
cat_feature_indices = [X_train_proc.columns.get_loc(c) for c in cat_cols]

cb_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=5,
    min_data_in_leaf=3,
    random_seed=42,
    eval_metric='AUC',
    verbose=False,
    train_dir='/tmp/catboost_info',
)
cb_model.fit(X_train_proc, y_train, cat_features=cat_feature_indices)
cb_pred = cb_model.predict_proba(X_val_proc)[:, 1]
cb_auc = roc_auc_score(y_val, cb_pred)
print(f"CatBoost Val ROC-AUC: {cb_auc:.6f}")

# ── Ensemble: average predicted probabilities ──────────────────────────────
ensemble_pred = (lgb_pred + cb_pred) / 2.0
ensemble_auc = roc_auc_score(y_val, ensemble_pred)
print(f"Ensemble (LGB+CB avg) Val ROC-AUC: {ensemble_auc:.6f}")

best_auc = max(lgb_auc, cb_auc, ensemble_auc)
print(f"BEST_VAL_ROC_AUC: {best_auc:.6f}")