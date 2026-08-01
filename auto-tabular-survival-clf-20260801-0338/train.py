import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
_user_site = '/home/flyte/.local/lib/python3.12/site-packages'
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
import xgboost as xgb

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
y_train = np.array(y_train)
y_val = np.array(y_val)

train_class_dist = {int(cls): int((y_train == cls).sum()) for cls in sorted(set(y_train))}
val_class_dist = {int(cls): int((y_val == cls).sum()) for cls in sorted(set(y_val))}
print(
    f"[DATA] Total samples: {len(df)}, Train: {len(X_train)}, Val: {len(X_val)}, "
    f"Train class distribution: {train_class_dist}, Val class distribution: {val_class_dist}"
)


def domain_feature_engineering(X_tr, X_v):
    """
    Titanic-specific feature engineering:
    - Extract Title from Name (Mr, Mrs, Miss, Master, Rare)
    - FamilySize = SibSp + Parch + 1
    - IsAlone = (FamilySize == 1)
    - CabinKnown = (Cabin != 'A6')  -- 'A6' is the fill value for missing cabins
    - Fare_log = log1p(Fare)
    - Drop high-cardinality/ID columns: PassengerId, Name, Ticket, Cabin
    """
    X_tr = X_tr.copy()
    X_v = X_v.copy()

    rare_titles = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}

    for df_ in [X_tr, X_v]:
        # Title extraction
        df_['Title'] = df_['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df_['Title'] = df_['Title'].fillna('Unknown')
        df_['Title'] = df_['Title'].apply(lambda x: 'Rare' if x in rare_titles else x)
        df_['Title'] = df_['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

        # Family size features
        df_['FamilySize'] = df_['SibSp'] + df_['Parch'] + 1
        df_['IsAlone'] = (df_['FamilySize'] == 1).astype(np.int8)

        # Cabin: distinguish known from fill value 'A6'
        # (76/100 of values are 'A6' fill for missing cabin)
        df_['CabinKnown'] = (df_['Cabin'] != 'A6').astype(np.int8)

        # Log-transform the highly skewed Fare (skew ~3.4)
        df_['Fare_log'] = np.log1p(df_['Fare'].clip(lower=0))

    # Drop ID / near-unique / already-processed columns
    drop_cols = [c for c in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cols)
    X_v = X_v.drop(columns=drop_cols)

    return X_tr, X_v


X_train_fe, X_val_fe = domain_feature_engineering(X_train, X_val)
print(f"After feature engineering — train: {X_train_fe.shape}, val: {X_val_fe.shape}")
print(f"Features: {list(X_train_fe.columns)}")

# Re-detect feature types on the engineered DataFrame
# Use is_string_dtype to also catch pandas StringDtype (from parquet), not just 'object'
def _is_categorical(series):
    return (series.dtype == 'object'
            or str(series.dtype) == 'category'
            or pd.api.types.is_string_dtype(series))

cat_cols = [c for c in X_train_fe.columns if _is_categorical(X_train_fe[c])]
num_cols = [c for c in X_train_fe.columns if c not in cat_cols]
print(f"Numeric cols ({len(num_cols)}): {num_cols}")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

miss_rate_train = X_train_fe.isnull().mean()
miss_cols = miss_rate_train[miss_rate_train > 0.05].index.tolist()
print(f"High-missing cols (>5%): {miss_cols}")


def preprocess(X_tr, X_v, y_tr_arr, num_cols, cat_cols, miss_cols):
    """
    Standard preprocessing + smoothed target encoding for categorical columns.
    - Binary missingness flags for high-missing cols
    - Median impute numerics, mode impute categoricals
    - Target encoding (5-fold CV, k=5 smoothing) for cat columns
    - Label-encode cat columns for LightGBM / CatBoost
    """
    X_tr = X_tr.copy().reset_index(drop=True)
    X_v = X_v.copy().reset_index(drop=True)

    # Convert categorical to object
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype(object)
        X_v[col] = X_v[col].astype(object)

    # Binary missingness indicators
    for col in miss_cols:
        X_tr[f'{col}_miss'] = X_tr[col].isnull().astype(np.int8)
        X_v[f'{col}_miss'] = X_v[col].isnull().astype(np.int8)

    # Numeric: convert + median impute from train
    for col in num_cols:
        X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce')
        X_v[col] = pd.to_numeric(X_v[col], errors='coerce')
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_v[col] = X_v[col].fillna(med)

    # Categorical: mode impute
    for col in cat_cols:
        mode_ser = X_tr[col].mode()
        mode_val = mode_ser.iloc[0] if len(mode_ser) > 0 else 'unknown'
        X_tr[col] = X_tr[col].fillna(mode_val)
        X_v[col] = X_v[col].fillna(mode_val)

    # Smoothed target encoding (5-fold CV on train to prevent leakage)
    global_mean = float(y_tr_arr.mean())
    smoothing_k = 5
    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for col in cat_cols:
        te_train = np.full(len(X_tr), global_mean, dtype=float)

        for fold_tr_idx, fold_vl_idx in skf_te.split(X_tr, y_tr_arr):
            fold_X = X_tr.iloc[fold_tr_idx]
            fold_y = y_tr_arr[fold_tr_idx]

            per_cat = {}
            for cat_val in fold_X[col].unique():
                mask = (fold_X[col] == cat_val).values
                n = mask.sum()
                c_mean = float(fold_y[mask].mean()) if n > 0 else global_mean
                per_cat[cat_val] = (n * c_mean + smoothing_k * global_mean) / (n + smoothing_k)

            for idx in fold_vl_idx:
                cat_val = X_tr.iloc[idx][col]
                te_train[idx] = per_cat.get(cat_val, global_mean)

        # Full-train encoding for the validation split
        full_means = {}
        for cat_val in X_tr[col].unique():
            mask = (X_tr[col] == cat_val).values
            n = mask.sum()
            c_mean = float(y_tr_arr[mask].mean()) if n > 0 else global_mean
            full_means[cat_val] = (n * c_mean + smoothing_k * global_mean) / (n + smoothing_k)

        X_tr[f'{col}_te'] = te_train
        X_v[f'{col}_te'] = X_v[col].map(full_means).fillna(global_mean).values

    # Label-encode categorical columns (for LightGBM categorical_feature + CatBoost)
    for col in cat_cols:
        enc = LabelEncoder()
        X_tr[col] = enc.fit_transform(X_tr[col].astype(str))
        known = set(enc.classes_)
        fallback = enc.classes_[0]
        X_v[col] = enc.transform(
            X_v[col].astype(str).apply(lambda x: x if x in known else fallback)
        )

    return X_tr, X_v


X_train_proc, X_val_proc = preprocess(X_train_fe, X_val_fe, y_train, num_cols, cat_cols, miss_cols)
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
            continue

        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, categorical_feature=cat_cols)
        prob = m.predict_proba(X_vl)[:, 1]
        fold_scores.append(roc_auc_score(y_vl, prob))

    return np.mean(fold_scores) if fold_scores else 0.5


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(lgb_objective, n_trials=100, timeout=50)
print(f"LightGBM Optuna best CV AUC: {study.best_value:.6f}  params: {study.best_params}")

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

# ── XGBoost ────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5.0,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train_proc, y_train)
xgb_pred = xgb_model.predict_proba(X_val_proc)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_pred)
print(f"XGBoost Val ROC-AUC: {xgb_auc:.6f}")

# ── Logistic Regression (scaled features) ─────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_proc)
X_val_scaled = scaler.transform(X_val_proc)

lr_model = LogisticRegression(C=0.05, solver='liblinear', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict_proba(X_val_scaled)[:, 1]
lr_auc = roc_auc_score(y_val, lr_pred)
print(f"Logistic Regression Val ROC-AUC: {lr_auc:.6f}")

# ── Ensemble: average of all four models ──────────────────────────────────
ensemble_pred = (lgb_pred + cb_pred + xgb_pred + lr_pred) / 4.0
ensemble_auc = roc_auc_score(y_val, ensemble_pred)
print(f"Ensemble (LGB+CB+XGB+LR avg) Val ROC-AUC: {ensemble_auc:.6f}")

# Also try 3-model ensemble without LR
ensemble3_pred = (lgb_pred + cb_pred + xgb_pred) / 3.0
ensemble3_auc = roc_auc_score(y_val, ensemble3_pred)
print(f"Ensemble (LGB+CB+XGB avg) Val ROC-AUC: {ensemble3_auc:.6f}")

best_auc = max(lgb_auc, cb_auc, xgb_auc, lr_auc, ensemble_auc, ensemble3_auc)
print(f"BEST_VAL_ROC_AUC: {best_auc:.6f}")
