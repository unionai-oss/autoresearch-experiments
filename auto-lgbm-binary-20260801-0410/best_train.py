import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(1, '/home/flyte/.local/lib/python3.12/site-packages')

import re
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
y = df[target_col]
X_raw = df.drop(columns=[target_col])

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {orig: encoded for encoded, orig in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")
print(f"Raw columns: {X_raw.columns.tolist()}")

# ── Train/val split on raw data ─────────────────────────────────────────────
X_tr_raw, X_va_raw, y_train, y_val = train_test_split(
    X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
class_counts = pd.Series(y_encoded).value_counts().sort_index().to_dict()
print(f"Dataset: N={len(df)}, classes={class_counts}, train={len(X_tr_raw)}, val={len(X_va_raw)}")

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)


# ── Feature Engineering ──────────────────────────────────────────────────────
def get_title(name):
    m = re.search(r' ([A-Za-z]+)\.', str(name))
    title = m.group(1) if m else 'Unknown'
    rare = {'Dr', 'Rev', 'Col', 'Major', 'Countess', 'Lady',
            'Jonkheer', 'Don', 'Dona', 'Capt', 'Sir', 'Col'}
    alias = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
    if title in alias:
        return alias[title]
    if title in rare or title not in {'Mr', 'Mrs', 'Miss', 'Master'}:
        return 'Rare'
    return title


def engineer(df_tr: pd.DataFrame, df_va: pd.DataFrame):
    """
    Compute group statistics from df_tr, apply to both.
    Returns (X_train_eng, X_val_eng, cat_col_names, num_col_names)
    """
    frames = [df_tr.copy(), df_va.copy()]

    for i, df in enumerate(frames):
        # Drop PassengerId – it's just a sequential ID, not predictive
        if 'PassengerId' in df.columns:
            df = df.drop(columns=['PassengerId'])

        # Title from Name
        if 'Name' in df.columns:
            df['Title'] = df['Name'].apply(get_title)
            df = df.drop(columns=['Name'])

        # Family size
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(np.int8)

        # Fare features
        if 'Fare' in df.columns:
            df['FareLog'] = np.log1p(df['Fare'])
            if 'FamilySize' in df.columns:
                df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)

        # Age features
        if 'Age' in df.columns:
            df['IsChild'] = (df['Age'] < 14).astype(np.int8)
            if 'Pclass' in df.columns:
                df['AgePclass'] = df['Age'] * df['Pclass']

        # Cabin: single-char 'D' is the unknown-cabin fill
        if 'Cabin' in df.columns:
            df['HasCabin'] = (df['Cabin'].str.len() > 1).astype(np.int8)
            df['Deck'] = df.apply(
                lambda row: str(row['Cabin'])[0] if len(str(row['Cabin'])) > 1 else 'Unknown',
                axis=1
            )
            df = df.drop(columns=['Cabin'])

        # Sex × Pclass interaction
        if 'Sex' in df.columns and 'Pclass' in df.columns:
            df['SexPclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)

        frames[i] = df

    # Ticket frequency: computed from train, applied to both
    if 'Ticket' in frames[0].columns:
        ticket_freq = frames[0]['Ticket'].value_counts().to_dict()
        for i, df in enumerate(frames):
            frames[i]['TicketFreq'] = df['Ticket'].map(ticket_freq).fillna(1).astype(float)
            frames[i] = frames[i].drop(columns=['Ticket'])

    return frames[0], frames[1]


X_train, X_val = engineer(X_tr_raw, X_va_raw)
print(f"Engineered features ({len(X_train.columns)}): {X_train.columns.tolist()}")

# Identify column types from engineered train
# String columns detected robustly (covers both 'object' and 'string' dtypes)
cat_cols = [c for c in X_train.columns
            if X_train[c].dtype == object
            or pd.api.types.is_string_dtype(X_train[c])
            or str(X_train[c].dtype) == 'string']
num_cols = [c for c in X_train.columns if c not in cat_cols]
print(f"Cat cols: {cat_cols}")
print(f"Num cols: {num_cols}")

X_train = X_train.copy()
X_val = X_val.copy()

# Missingness indicators for columns with >5% missing in train
miss_thresh = 0.05
for col in X_train.columns:
    if X_train[col].isna().mean() > miss_thresh:
        X_train[f'{col}_missing'] = X_train[col].isna().astype(np.int8)
        X_val[f'{col}_missing'] = X_val[col].isna().astype(np.int8)

# Impute numeric with train median
for col in num_cols:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

# Ordinal-encode categorical columns
enc_map = {}
for col in cat_cols:
    train_vals = X_train[col].astype(str).fillna('__NaN__')
    val_vals = X_val[col].astype(str).fillna('__NaN__')
    enc = LabelEncoder()
    enc.fit(train_vals)
    class_to_idx = {cls: i for i, cls in enumerate(enc.classes_)}
    X_train[col] = enc.transform(train_vals)
    X_val[col] = val_vals.map(lambda v, m=class_to_idx: m.get(v, -1)).values
    enc_map[col] = class_to_idx

for col in cat_cols:
    X_train[col] = X_train[col].astype(int)
    X_val[col] = X_val[col].astype(int)

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

# ── Optuna hyperparameter search ─────────────────────────────────────────────
LGBM_FIXED = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': 1,
}


def objective(trial):
    params = {
        **LGBM_FIXED,
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for fold_tr_idx, fold_va_idx in skf.split(X_train, y_train):
        X_ft = X_train.iloc[fold_tr_idx]
        y_ft = y_train[fold_tr_idx]
        X_fv = X_train.iloc[fold_va_idx]
        y_fv = y_train[fold_va_idx]

        dtrain_fold = lgb.Dataset(X_ft, label=y_ft, categorical_feature=cat_cols,
                                  free_raw_data=False)
        dval_fold = lgb.Dataset(X_fv, label=y_fv, categorical_feature=cat_cols,
                                free_raw_data=False)

        model = lgb.train(
            params,
            dtrain_fold,
            num_boost_round=600,
            valid_sets=[dval_fold],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        preds = model.predict(X_fv)
        fold_aucs.append(roc_auc_score(y_fv, preds))

    return float(np.mean(fold_aucs))


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=80, show_progress_bar=False)

best_cv_auc = study.best_value
best_params = study.best_params
print(f"Optuna best CV AUC: {best_cv_auc:.6f}")
print(f"Best hyperparams: {best_params}")

# ── Final model on full training set ─────────────────────────────────────────
final_params = {**LGBM_FIXED, **best_params, 'bagging_freq': 1}

dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval_full = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

final_model = lgb.train(
    final_params,
    dtrain_full,
    num_boost_round=2000,
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
