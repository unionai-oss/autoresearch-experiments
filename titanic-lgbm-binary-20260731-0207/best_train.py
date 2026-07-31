import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)

# Convert Arrow/nullable extension dtypes to standard numpy types.
# Pandas 2+ uses StringDtype (Arrow-backed), Int64Dtype, Float64Dtype which
# LightGBM / XGBoost reject with "pandas dtypes must be int, float or bool."
def to_numpy_dtypes(frame):
    frame = frame.copy()
    for col in frame.columns:
        dtype = frame[col].dtype
        if isinstance(dtype, pd.StringDtype) or str(dtype) in ('str', 'string'):
            frame[col] = frame[col].astype(object)
        elif pd.api.types.is_extension_array_dtype(dtype):
            # Handles Int64Dtype, Float64Dtype, BooleanDtype, ArrowDtype…
            numpy_dtype = getattr(dtype, 'numpy_dtype', np.float64)
            frame[col] = frame[col].to_numpy(dtype=numpy_dtype, na_value=np.nan)
    return frame

df = to_numpy_dtypes(df)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int).values   # plain numpy int array

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Dataset: total={len(df)}, train={len(X_train)}, val={len(X_val)}")
print(f"Class dist — 0:{(y==0).sum()}, 1:{(y==1).sum()}")

# ── Feature engineering ──────────────────────────────────────────────────────

# Compute ticket-group sizes from the full X (structural feature, not label-derived)
ticket_sizes_global = X['Ticket'].value_counts().to_dict()

def compute_fill_values(df_in):
    fv = {}
    fv['Age_median']      = df_in['Age'].median()
    fv['Fare_median']     = df_in['Fare'].median()
    mode = df_in['Embarked'].mode()
    fv['Embarked_mode']   = mode.iloc[0] if len(mode) > 0 else 'S'
    return fv


def transform_features(df_in, fill_vals, ticket_sizes):
    d = df_in.copy()

    # PassengerId — random ID, no signal
    if 'PassengerId' in d.columns:
        d = d.drop(columns=['PassengerId'])

    # ── Name → Title ─────────────────────────────────────────────────────────
    if 'Name' in d.columns:
        d['Title'] = d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
            'Dr': 'Rare', 'Rev': 'Rare', 'Sir': 'Royalty', 'Lady': 'Royalty',
            'Don': 'Royalty', 'Countess': 'Royalty', 'Jonkheer': 'Royalty',
        }
        d['Title'] = d['Title'].map(title_map).fillna('Rare')
        d = d.drop(columns=['Name'])

    # ── Cabin → HasCabin + Deck ───────────────────────────────────────────────
    # In this dataset the NaN fill value for Cabin is 'B96 B98' (dominant value)
    if 'Cabin' in d.columns:
        CABIN_FILL = 'B96 B98'
        d['HasCabin'] = (d['Cabin'] != CABIN_FILL).astype(int)
        d['Deck'] = d['Cabin'].str[0]
        d.loc[d['Cabin'] == CABIN_FILL, 'Deck'] = 'U'
        d['Deck'] = d['Deck'].fillna('U')
        d = d.drop(columns=['Cabin'])

    # ── Ticket → group size + prefix ─────────────────────────────────────────
    if 'Ticket' in d.columns:
        d['TicketGroupSize'] = d['Ticket'].map(ticket_sizes).fillna(1).astype(float)
        d['Ticket_Prefix'] = (
            d['Ticket'].str.extract(r'^([A-Za-z./]+)', expand=False).fillna('NUM')
        )
        d = d.drop(columns=['Ticket'])

    # ── Age ───────────────────────────────────────────────────────────────────
    if 'Age' in d.columns:
        d['Age_Missing'] = d['Age'].isna().astype(int)
        d['Age'] = d['Age'].fillna(fill_vals['Age_median'])
        d['IsChild'] = (d['Age'] < 15).astype(int)
        d['IsElder'] = (d['Age'] > 60).astype(int)

    # ── Embarked ──────────────────────────────────────────────────────────────
    if 'Embarked' in d.columns:
        d['Embarked'] = d['Embarked'].fillna(fill_vals['Embarked_mode'])

    # ── Fare ──────────────────────────────────────────────────────────────────
    if 'Fare' in d.columns:
        d['Fare'] = d['Fare'].fillna(fill_vals['Fare_median'])
        d['Fare_Log'] = np.log1p(d['Fare'])
        if 'TicketGroupSize' in d.columns:
            d['Fare_PerPerson']     = d['Fare'] / d['TicketGroupSize'].clip(lower=1)
            d['Fare_PerPerson_Log'] = np.log1p(d['Fare_PerPerson'])

    # ── Family size ───────────────────────────────────────────────────────────
    if 'SibSp' in d.columns and 'Parch' in d.columns:
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
        d['IsAlone']    = (d['FamilySize'] == 1).astype(int)
        # Ordinal family category: 1=alone, 2-4=small, 5+=large
        d['FamilyBucket'] = d['FamilySize'].clip(upper=5)

    # ── Interaction features ──────────────────────────────────────────────────
    if 'Age' in d.columns and 'Pclass' in d.columns:
        d['Age_x_Pclass'] = d['Age'] * d['Pclass']

    if 'Sex' in d.columns and 'Pclass' in d.columns:
        # Encode before interaction so both sides are numeric later
        d['Sex_Pclass'] = (
            d['Sex'].astype(str) + '_' + d['Pclass'].astype(str)
        )

    return d


def label_encode(df_in, encoders=None, fit=False):
    """Encode every remaining object/string/category column to int."""
    df_in = df_in.copy()
    if encoders is None:
        encoders = {}
    for c in df_in.columns:
        if df_in[c].dtype == object or str(df_in[c].dtype) in ('category', 'str', 'string'):
            if fit:
                enc = LabelEncoder()
                df_in[c] = enc.fit_transform(df_in[c].astype(str))
                encoders[c] = enc
            else:
                enc = encoders.get(c)
                if enc is not None:
                    col_str = df_in[c].astype(str)
                    known   = set(enc.classes_)
                    col_str = col_str.map(lambda v: v if v in known else enc.classes_[0])
                    df_in[c] = enc.transform(col_str)
                else:
                    df_in[c] = LabelEncoder().fit_transform(df_in[c].astype(str))
    return df_in, encoders


fill_vals = compute_fill_values(X_train)

X_train_raw = transform_features(X_train, fill_vals, ticket_sizes_global)
X_val_raw   = transform_features(X_val,   fill_vals, ticket_sizes_global)

X_train_fe, cat_encs = label_encode(X_train_raw, fit=True)
X_val_fe,   _        = label_encode(X_val_raw, encoders=cat_encs, fit=False)

# Force all columns to float64 — guarantees compatibility with both LGBM and XGB
X_train_fe = X_train_fe.astype(np.float64)
X_val_fe   = X_val_fe.astype(np.float64)

print(f"Features ({len(X_train_fe.columns)}): {list(X_train_fe.columns)}")

# ── LightGBM Optuna HPO ───────────────────────────────────────────────────────

def objective_lgbm(trial):
    params = {
        'objective':         'binary',
        'metric':            'auc',
        'verbosity':         -1,
        'boosting_type':     'gbdt',
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 50),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq':      1,
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'n_estimators':      500,
        'random_state':      42,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xf_tr, yf_tr,
                eval_set=[(Xf_val, yf_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)])
        aucs.append(roc_auc_score(yf_val, clf.predict_proba(Xf_val)[:, 1]))
    return float(np.mean(aucs))


print("LightGBM HPO: 80 trials …")
study_lgbm = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_lgbm.optimize(objective_lgbm, n_trials=80, show_progress_bar=False)
print(f"Best LGBM CV AUC: {study_lgbm.best_value:.6f}")

best_lgbm = study_lgbm.best_params.copy()
best_lgbm.update({
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'boosting_type': 'gbdt', 'bagging_freq': 1,
    'n_estimators': 2000, 'random_state': 42,
})
final_lgbm = lgb.LGBMClassifier(**best_lgbm)
final_lgbm.fit(X_train_fe, y_train,
               eval_set=[(X_val_fe, y_val)],
               callbacks=[lgb.early_stopping(150, verbose=False),
                          lgb.log_evaluation(period=-1)])
lgbm_preds = final_lgbm.predict_proba(X_val_fe)[:, 1]
lgbm_auc   = roc_auc_score(y_val, lgbm_preds)
print(f"LGBM val AUC: {lgbm_auc:.6f}")

# ── XGBoost Optuna HPO ────────────────────────────────────────────────────────

def objective_xgb(trial):
    params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'max_depth':        trial.suggest_int('max_depth', 3, 9),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'n_estimators':     500,
        'subsample':        trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
        'random_state':     42,
        'verbosity':        0,
        'early_stopping_rounds': 50,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = xgb.XGBClassifier(**params)
        clf.fit(Xf_tr, yf_tr, eval_set=[(Xf_val, yf_val)], verbose=False)
        aucs.append(roc_auc_score(yf_val, clf.predict_proba(Xf_val)[:, 1]))
    return float(np.mean(aucs))


print("XGBoost HPO: 60 trials …")
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_xgb.optimize(objective_xgb, n_trials=60, show_progress_bar=False)
print(f"Best XGB CV AUC: {study_xgb.best_value:.6f}")

best_xgb = study_xgb.best_params.copy()
best_xgb.update({
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'n_estimators': 2000, 'random_state': 42, 'verbosity': 0,
    'early_stopping_rounds': 150,
})
final_xgb = xgb.XGBClassifier(**best_xgb)
final_xgb.fit(X_train_fe, y_train,
              eval_set=[(X_val_fe, y_val)],
              verbose=False)
xgb_preds = final_xgb.predict_proba(X_val_fe)[:, 1]
xgb_auc   = roc_auc_score(y_val, xgb_preds)
print(f"XGB  val AUC: {xgb_auc:.6f}")

# ── Ensemble ─────────────────────────────────────────────────────────────────
ens_preds = (lgbm_preds + xgb_preds) / 2.0
ens_auc   = roc_auc_score(y_val, ens_preds)
print(f"Ensemble AUC: {ens_auc:.6f}")

best_val_roc_auc = max(lgbm_auc, xgb_auc, ens_auc)
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
