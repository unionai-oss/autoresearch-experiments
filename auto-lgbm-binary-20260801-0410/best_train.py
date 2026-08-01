import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(1, '/home/flyte/.local/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {orig: encoded for encoded, orig in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

class_counts = pd.Series(y_encoded).value_counts().sort_index().to_dict()
print(f"Dataset summary: total_samples={len(df)}, class_distribution={class_counts}, train_samples={len(X_train)}, val_samples={len(X_val)}")

# ── Feature preprocessing ──────────────────────────────────────────────────
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()
print(f"Categorical cols: {cat_cols}")
print(f"Numeric cols: {num_cols}")

X_train = X_train.copy()
X_val = X_val.copy()

# Add binary missingness-indicator flags for columns >5% missing in train
miss_thresh = 0.05
for col in X_train.columns:
    if X_train[col].isna().mean() > miss_thresh:
        X_train[f'{col}_missing'] = X_train[col].isna().astype(np.int8)
        X_val[f'{col}_missing'] = X_val[col].isna().astype(np.int8)

# Impute numeric columns with train median (single pass — no double normalisation)
medians = {}
for col in num_cols:
    med = X_train[col].median()
    medians[col] = med
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

# Ordinal-encode categorical columns; fill NaN as a separate '__NaN__' level
enc_map = {}
for col in cat_cols:
    train_vals = X_train[col].astype(str).fillna('__NaN__')
    val_vals = X_val[col].astype(str).fillna('__NaN__')
    enc = LabelEncoder()
    enc.fit(train_vals)
    class_to_idx = {cls: i for i, cls in enumerate(enc.classes_)}
    X_train[col] = enc.transform(train_vals)
    # Map val; unseen categories → -1 (LightGBM treats negatives as missing for cat features)
    X_val[col] = val_vals.map(lambda v, m=class_to_idx: m.get(v, -1)).values
    enc_map[col] = class_to_idx

# Convert cat cols to int (required by LightGBM categorical_feature)
for col in cat_cols:
    X_train[col] = X_train[col].astype(int)
    X_val[col] = X_val[col].astype(int)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

# ── Optuna hyperparameter search (5-fold CV on training set) ────────────────
LGBM_FIXED = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': 1,
}


def objective(trial):
    params = {
        **LGBM_FIXED,
        'num_leaves': trial.suggest_int('num_leaves', 8, 96),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for fold_tr_idx, fold_va_idx in skf.split(X_train, y_train):
        X_ft = X_train.iloc[fold_tr_idx]
        y_ft = y_train[fold_tr_idx]
        X_fv = X_train.iloc[fold_va_idx]
        y_fv = y_train[fold_va_idx]

        dtrain_fold = lgb.Dataset(X_ft, label=y_ft, categorical_feature=cat_cols, free_raw_data=False)
        dval_fold = lgb.Dataset(X_fv, label=y_fv, categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            dtrain_fold,
            num_boost_round=400,
            valid_sets=[dval_fold],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        preds = model.predict(X_fv)
        fold_aucs.append(roc_auc_score(y_fv, preds))

    return float(np.mean(fold_aucs))


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=60, show_progress_bar=False)

best_cv_auc = study.best_value
best_params = study.best_params
print(f"Optuna best CV AUC: {best_cv_auc:.6f}")
print(f"Best hyperparams: {best_params}")

# ── Train final model on full training set, evaluate on held-out val ────────
final_params = {
    **LGBM_FIXED,
    **best_params,
    'bagging_freq': 1,
}

dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval_full = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

final_model = lgb.train(
    final_params,
    dtrain_full,
    num_boost_round=1000,
    valid_sets=[dval_full],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

val_preds = final_model.predict(X_val)
val_auc = roc_auc_score(y_val, val_preds)
print(f"Val ROC-AUC (final model): {val_auc:.6f}")

print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")
