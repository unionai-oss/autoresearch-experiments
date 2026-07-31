import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)

def to_numpy_dtypes(frame):
    frame = frame.copy()
    for col in frame.columns:
        dtype = frame[col].dtype
        if isinstance(dtype, pd.StringDtype) or str(dtype) in ('str', 'string'):
            frame[col] = frame[col].astype(object)
        elif pd.api.types.is_extension_array_dtype(dtype):
            numpy_dtype = getattr(dtype, 'numpy_dtype', np.float64)
            frame[col] = frame[col].to_numpy(dtype=numpy_dtype, na_value=np.nan)
    return frame

df = to_numpy_dtypes(df)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Dataset: total={len(df)}, train={len(X_train)}, val={len(X_val)}")
print(f"Class dist — 0:{(y==0).sum()}, 1:{(y==1).sum()}")

# ── Feature engineering ──────────────────────────────────────────────────────
ticket_sizes_global = X['Ticket'].value_counts().to_dict()

TITLE_MAP = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
    'Dr': 'Rare', 'Rev': 'Rare', 'Sir': 'Royalty', 'Lady': 'Royalty',
    'Don': 'Royalty', 'Countess': 'Royalty', 'Jonkheer': 'Royalty',
}


def compute_fill_values(df_in):
    """Compute fill values from training set only — no val leakage."""
    fv = {}
    titles_raw = df_in['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    titles = titles_raw.map(TITLE_MAP).fillna('Rare')
    temp = df_in.copy()
    temp['_title'] = titles
    global_med = df_in['Age'].median()
    title_age = temp.groupby('_title')['Age'].median().to_dict()
    for t in ['Mr', 'Miss', 'Mrs', 'Master', 'Officer', 'Rare', 'Royalty']:
        if t not in title_age or pd.isna(title_age.get(t, np.nan)):
            title_age[t] = global_med
    fv['Age_by_Title'] = title_age
    fv['Age_global_median'] = global_med
    fv['Fare_median'] = df_in['Fare'].median()
    mode = df_in['Embarked'].mode()
    fv['Embarked_mode'] = mode.iloc[0] if len(mode) > 0 else 'S'

    # Fare median per Pclass (for within-class normalization)
    fare_by_pclass = {}
    for pc in [1, 2, 3]:
        mask = df_in['Pclass'] == pc
        med = df_in.loc[mask, 'Fare'].dropna().median()
        fare_by_pclass[pc] = med if not pd.isna(med) else df_in['Fare'].median()
    fv['Fare_by_Pclass'] = fare_by_pclass

    return fv


def transform_features(df_in, fill_vals, ticket_sizes):
    d = df_in.copy()

    if 'PassengerId' in d.columns:
        d = d.drop(columns=['PassengerId'])

    # ── Name → Title + NameLen ───────────────────────────────────────────────
    if 'Name' in d.columns:
        d['NameLen'] = d['Name'].str.len()
        d['Title'] = d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        d['Title'] = d['Title'].map(TITLE_MAP).fillna('Rare')
        d = d.drop(columns=['Name'])

    # ── Cabin → HasCabin + Deck + CabinNum + Cabin_Side ─────────────────────
    if 'Cabin' in d.columns:
        CABIN_FILL = 'B96 B98'
        d['HasCabin'] = (d['Cabin'] != CABIN_FILL).astype(int)
        d['Deck'] = d['Cabin'].str[0]
        d.loc[d['Cabin'] == CABIN_FILL, 'Deck'] = 'U'
        d['Deck'] = d['Deck'].fillna('U')

        # Extract numeric cabin number (only for known cabins)
        known_mask = d['Cabin'] != CABIN_FILL
        d['CabinNum'] = np.nan
        cabin_nums = d.loc[known_mask, 'Cabin'].str.extract(r'(\d+)', expand=False)
        d.loc[known_mask, 'CabinNum'] = pd.to_numeric(cabin_nums, errors='coerce')
        d['CabinNum'] = d['CabinNum'].fillna(-1)
        # Cabin side: odd=1 (starboard), even=0 (port), unknown=-1
        valid_num_mask = d['CabinNum'] >= 0
        d['Cabin_Side'] = -1.0
        d.loc[valid_num_mask, 'Cabin_Side'] = (d.loc[valid_num_mask, 'CabinNum'] % 2).astype(float)

        d = d.drop(columns=['Cabin'])

    # ── Ticket → TicketGroupSize + Prefix ────────────────────────────────────
    if 'Ticket' in d.columns:
        d['TicketGroupSize'] = d['Ticket'].map(ticket_sizes).fillna(1).astype(float)
        d['Ticket_Prefix'] = (
            d['Ticket'].str.extract(r'^([A-Za-z./]+)', expand=False).fillna('NUM')
        )
        d = d.drop(columns=['Ticket'])

    # ── Age: title-based imputation ───────────────────────────────────────────
    if 'Age' in d.columns:
        d['Age_Missing'] = d['Age'].isna().astype(int)
        age_by_title = fill_vals['Age_by_Title']
        global_med = fill_vals['Age_global_median']
        if 'Title' in d.columns:
            mask = d['Age'].isna()
            d.loc[mask, 'Age'] = d.loc[mask, 'Title'].map(age_by_title).fillna(global_med)
        else:
            d['Age'] = d['Age'].fillna(global_med)
        d['IsChild'] = (d['Age'] < 15).astype(int)
        d['IsElder'] = (d['Age'] > 60).astype(int)
        d['AgeBin'] = pd.cut(
            d['Age'], bins=[0, 5, 12, 18, 25, 35, 50, 65, 200],
            labels=False, right=True
        ).fillna(0).astype(int)
        # Non-linear age effect
        d['Age_sq'] = (d['Age'] ** 2) / 100.0

    # ── Embarked ──────────────────────────────────────────────────────────────
    if 'Embarked' in d.columns:
        d['Embarked'] = d['Embarked'].fillna(fill_vals['Embarked_mode'])

    # ── Fare ──────────────────────────────────────────────────────────────────
    if 'Fare' in d.columns:
        d['Fare'] = d['Fare'].fillna(fill_vals['Fare_median'])
        d['Fare_Log'] = np.log1p(d['Fare'])
        if 'TicketGroupSize' in d.columns:
            d['Fare_PerPerson'] = d['Fare'] / d['TicketGroupSize'].clip(lower=1)
            d['Fare_PerPerson_Log'] = np.log1p(d['Fare_PerPerson'])
        # Fare normalized within Pclass (relative wealth within class)
        if 'Pclass' in d.columns and 'Fare_by_Pclass' in fill_vals:
            d['Fare_norm_Pclass'] = 1.0
            for pc in [1, 2, 3]:
                med = fill_vals['Fare_by_Pclass'].get(pc, 1.0)
                mask_pc = d['Pclass'] == pc
                d.loc[mask_pc, 'Fare_norm_Pclass'] = (
                    d.loc[mask_pc, 'Fare'] / max(float(med), 0.01)
                )
            d['Fare_norm_Pclass'] = d['Fare_norm_Pclass'].fillna(1.0)

    # ── Family size ───────────────────────────────────────────────────────────
    if 'SibSp' in d.columns and 'Parch' in d.columns:
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
        d['IsAlone'] = (d['FamilySize'] == 1).astype(int)
        d['FamilyBucket'] = d['FamilySize'].clip(upper=5)
        d['IsSmallFamily'] = ((d['FamilySize'] >= 2) & (d['FamilySize'] <= 4)).astype(int)
        # Bucketed siblings/parents for non-linear capture
        d['SibSp_cat'] = d['SibSp'].clip(upper=3).astype(int)
        d['Parch_cat'] = d['Parch'].clip(upper=3).astype(int)

    # ── "Women and children first" features ──────────────────────────────────
    sex_str = d['Sex'].astype(str) if 'Sex' in d.columns else None
    title_col = d['Title'] if 'Title' in d.columns else None

    if sex_str is not None and title_col is not None:
        d['WomanOrMasterBoy'] = (
            (sex_str == 'female') | (title_col == 'Master')
        ).astype(int)
        if 'Age' in d.columns and 'Parch' in d.columns:
            d['IsMother'] = (
                (sex_str == 'female') &
                (d['Parch'] > 0) &
                (d['Age'] > 18) &
                (title_col != 'Miss')
            ).astype(int)
        if 'Age' in d.columns:
            d['Age_x_Female'] = d['Age'] * (sex_str == 'female').astype(int)

    # ── Interaction features ──────────────────────────────────────────────────
    if 'Age' in d.columns and 'Pclass' in d.columns:
        d['Age_x_Pclass'] = d['Age'] * d['Pclass']

    if 'Sex' in d.columns and 'Pclass' in d.columns:
        d['Sex_Pclass'] = d['Sex'].astype(str) + '_' + d['Pclass'].astype(str)

    if 'Pclass' in d.columns and 'Embarked' in d.columns:
        d['Pclass_Embarked'] = d['Pclass'].astype(str) + '_' + d['Embarked'].astype(str)

    if 'FamilySize' in d.columns and 'Pclass' in d.columns:
        d['FamilySize_x_Pclass'] = d['FamilySize'] * d['Pclass']

    if 'Fare_Log' in d.columns and 'Pclass' in d.columns:
        d['FareLog_x_Pclass'] = d['Fare_Log'] * d['Pclass']

    if 'WomanOrMasterBoy' in d.columns and 'Pclass' in d.columns:
        d['WoM_x_Pclass'] = d['WomanOrMasterBoy'] * d['Pclass']

    # ── Title × Pclass cross feature (finer-grained survival group) ──────────
    if 'Title' in d.columns and 'Pclass' in d.columns:
        d['Title_Pclass'] = d['Title'].astype(str) + '_' + d['Pclass'].astype(str)

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
                    known = set(enc.classes_)
                    col_str = col_str.map(lambda v: v if v in known else enc.classes_[0])
                    df_in[c] = enc.transform(col_str)
                else:
                    df_in[c] = LabelEncoder().fit_transform(df_in[c].astype(str))
    return df_in, encoders


fill_vals = compute_fill_values(X_train)

X_train_raw = transform_features(X_train, fill_vals, ticket_sizes_global)
X_val_raw   = transform_features(X_val,   fill_vals, ticket_sizes_global)


# ── CV-based target encoding (no leakage) ────────────────────────────────────
def cv_target_encode(X_tr_raw, y_tr, X_vl_raw, col, n_splits=5, alpha=5.0):
    """OOF target encoding for train; smoothed mean for val (from full train)."""
    global_mean = float(y_tr.mean())
    oof = np.zeros(len(X_tr_raw))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for t_idx, v_idx in skf.split(X_tr_raw, y_tr):
        col_tr = X_tr_raw.iloc[t_idx][col].astype(str).values
        y_fold  = y_tr[t_idx]
        col_vl  = X_tr_raw.iloc[v_idx][col].astype(str).values

        df_tmp = pd.DataFrame({'cat': col_tr, 'y': y_fold})
        stats  = df_tmp.groupby('cat')['y'].agg(['sum', 'count'])
        stats['enc'] = (stats['sum'] + global_mean * alpha) / (stats['count'] + alpha)
        enc_map = stats['enc'].to_dict()

        oof[v_idx] = pd.Series(col_vl).map(enc_map).fillna(global_mean).values

    # Val set: smoothed mean from full training set
    col_all = X_tr_raw[col].astype(str).values
    df_full  = pd.DataFrame({'cat': col_all, 'y': y_tr})
    stats_full = df_full.groupby('cat')['y'].agg(['sum', 'count'])
    stats_full['enc'] = (stats_full['sum'] + global_mean * alpha) / (stats_full['count'] + alpha)
    enc_map_full = stats_full['enc'].to_dict()
    val_enc = X_vl_raw[col].astype(str).map(enc_map_full).fillna(global_mean).values

    return oof, val_enc


# Added Title_Pclass cross target encoding
te_features = ['Title', 'Sex_Pclass', 'Deck', 'Ticket_Prefix', 'Pclass_Embarked', 'Title_Pclass']
for feat in te_features:
    if feat in X_train_raw.columns and feat in X_val_raw.columns:
        oof_enc, val_enc = cv_target_encode(X_train_raw, y_train, X_val_raw, feat)
        X_train_raw[f'TE_{feat}'] = oof_enc
        X_val_raw[f'TE_{feat}']   = val_enc

print(f"Applied target encoding for: {te_features}")

X_train_fe, cat_encs = label_encode(X_train_raw, fit=True)
X_val_fe,   _        = label_encode(X_val_raw, encoders=cat_encs, fit=False)

X_train_fe = X_train_fe.astype(np.float64)
X_val_fe   = X_val_fe.astype(np.float64)

print(f"Features ({len(X_train_fe.columns)}): {list(X_train_fe.columns)}")

# ── LightGBM Optuna HPO ───────────────────────────────────────────────────────

def objective_lgbm(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
    params = {
        'objective':         'binary',
        'metric':            'auc',
        'verbosity':         -1,
        'boosting_type':     boosting_type,
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
    if boosting_type == 'dart':
        params['drop_rate']  = trial.suggest_float('drop_rate', 0.05, 0.5)
        params['skip_drop']  = trial.suggest_float('skip_drop', 0.1, 0.9)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = lgb.LGBMClassifier(**params)
        if boosting_type == 'dart':
            # DART does not support early stopping reliably — use fixed n_estimators
            clf.fit(Xf_tr, yf_tr)
        else:
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

best_lgbm_params = study_lgbm.best_params.copy()
lgbm_boosting = best_lgbm_params.get('boosting_type', 'gbdt')
print(f"Best LGBM boosting type: {lgbm_boosting}")
best_lgbm_params.update({
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'bagging_freq': 1, 'random_state': 42,
    # DART: fixed iters (no early stopping); GBDT: large cap with early stopping
    'n_estimators': 500 if lgbm_boosting == 'dart' else 2000,
})
final_lgbm = lgb.LGBMClassifier(**best_lgbm_params)
if lgbm_boosting == 'dart':
    final_lgbm.fit(X_train_fe, y_train)
else:
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

best_xgb_params = study_xgb.best_params.copy()
best_xgb_params.update({
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'n_estimators': 2000, 'random_state': 42, 'verbosity': 0,
    'early_stopping_rounds': 150,
})
final_xgb = xgb.XGBClassifier(**best_xgb_params)
final_xgb.fit(X_train_fe, y_train, eval_set=[(X_val_fe, y_val)], verbose=False)
xgb_preds = final_xgb.predict_proba(X_val_fe)[:, 1]
xgb_auc   = roc_auc_score(y_val, xgb_preds)
print(f"XGB  val AUC: {xgb_auc:.6f}")

# ── CatBoost Optuna HPO ───────────────────────────────────────────────────────

def objective_cat(trial):
    params = {
        'iterations':          500,
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':               trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'border_count':        trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':     trial.suggest_float('random_strength', 0.5, 5.0),
        'eval_metric':         'AUC',
        'random_seed':         42,
        'verbose':             0,
        'train_dir':           '/tmp/catboost_info',
        'early_stopping_rounds': 50,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = CatBoostClassifier(**params)
        clf.fit(Xf_tr, yf_tr, eval_set=(Xf_val, yf_val), verbose=False)
        aucs.append(roc_auc_score(yf_val, clf.predict_proba(Xf_val)[:, 1]))
    return float(np.mean(aucs))


print("CatBoost HPO: 50 trials …")
study_cat = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=False)
print(f"Best CatBoost CV AUC: {study_cat.best_value:.6f}")

best_cat_params = study_cat.best_params.copy()
best_cat_params.update({
    'eval_metric':  'AUC',
    'random_seed':  42,
    'verbose':      0,
    'train_dir':    '/tmp/catboost_info',
    'iterations':   2000,
    'early_stopping_rounds': 150,
})
final_cat = CatBoostClassifier(**best_cat_params)
final_cat.fit(X_train_fe, y_train, eval_set=(X_val_fe, y_val), verbose=False)
cat_preds = final_cat.predict_proba(X_val_fe)[:, 1]
cat_auc   = roc_auc_score(y_val, cat_preds)
print(f"CatBoost val AUC: {cat_auc:.6f}")

# ── RandomForest Optuna HPO ───────────────────────────────────────────────────

def objective_rf(trial):
    params = {
        'n_estimators':      500,
        'max_depth':         trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':      trial.suggest_float('max_features', 0.2, 1.0),
        'bootstrap':         trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight':      'balanced',
        'random_state':      42,
        'n_jobs':            -1,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = RandomForestClassifier(**params)
        clf.fit(Xf_tr, yf_tr)
        aucs.append(roc_auc_score(yf_val, clf.predict_proba(Xf_val)[:, 1]))
    return float(np.mean(aucs))


print("RandomForest HPO: 30 trials …")
study_rf = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=False)
print(f"Best RF CV AUC: {study_rf.best_value:.6f}")

best_rf_params = study_rf.best_params.copy()
best_rf_params.update({
    'n_estimators': 500,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs':       -1,
})
final_rf = RandomForestClassifier(**best_rf_params)
final_rf.fit(X_train_fe, y_train)
rf_preds = final_rf.predict_proba(X_val_fe)[:, 1]
rf_auc   = roc_auc_score(y_val, rf_preds)
print(f"RF   val AUC: {rf_auc:.6f}")

# ── HistGradientBoosting Optuna HPO (5th diverse base model) ─────────────────

def objective_hist(trial):
    params = {
        'max_iter':          trial.suggest_int('max_iter', 200, 800),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_leaf_nodes':    trial.suggest_int('max_leaf_nodes', 10, 80),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 5, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
        'random_state':      42,
        'early_stopping':    False,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, vl_idx in skf.split(X_train_fe, y_train):
        Xf_tr, Xf_val = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
        yf_tr, yf_val = y_train[tr_idx],          y_train[vl_idx]
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(Xf_tr, yf_tr)
        aucs.append(roc_auc_score(yf_val, clf.predict_proba(Xf_val)[:, 1]))
    return float(np.mean(aucs))


print("HistGradientBoosting HPO: 30 trials …")
study_hist = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study_hist.optimize(objective_hist, n_trials=30, show_progress_bar=False)
print(f"Best HistGBM CV AUC: {study_hist.best_value:.6f}")

best_hist_params = study_hist.best_params.copy()
best_hist_params.update({'random_state': 42, 'early_stopping': False})
final_hist = HistGradientBoostingClassifier(**best_hist_params)
final_hist.fit(X_train_fe, y_train)
hist_preds = final_hist.predict_proba(X_val_fe)[:, 1]
hist_auc   = roc_auc_score(y_val, hist_preds)
print(f"HistGBM val AUC: {hist_auc:.6f}")

# ── Stacking: OOF-based meta-learner ─────────────────────────────────────────
print("Generating OOF predictions for stacking …")
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgbm = np.zeros(len(X_train_fe))
oof_xgb  = np.zeros(len(X_train_fe))
oof_cat  = np.zeros(len(X_train_fe))
oof_rf   = np.zeros(len(X_train_fe))
oof_hist = np.zeros(len(X_train_fe))

for tr_idx, vl_idx in skf5.split(X_train_fe, y_train):
    Xf_tr, Xf_vl = X_train_fe.iloc[tr_idx], X_train_fe.iloc[vl_idx]
    yf_tr, yf_vl = y_train[tr_idx], y_train[vl_idx]

    _lgbm = lgb.LGBMClassifier(**best_lgbm_params)
    if lgbm_boosting == 'dart':
        _lgbm.fit(Xf_tr, yf_tr)
    else:
        _lgbm.fit(Xf_tr, yf_tr,
                  eval_set=[(Xf_vl, yf_vl)],
                  callbacks=[lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(period=-1)])
    oof_lgbm[vl_idx] = _lgbm.predict_proba(Xf_vl)[:, 1]

    _xgb = xgb.XGBClassifier(**best_xgb_params)
    _xgb.fit(Xf_tr, yf_tr, eval_set=[(Xf_vl, yf_vl)], verbose=False)
    oof_xgb[vl_idx] = _xgb.predict_proba(Xf_vl)[:, 1]

    _cat = CatBoostClassifier(**best_cat_params)
    _cat.fit(Xf_tr, yf_tr, eval_set=(Xf_vl, yf_vl), verbose=False)
    oof_cat[vl_idx] = _cat.predict_proba(Xf_vl)[:, 1]

    _rf = RandomForestClassifier(**best_rf_params)
    _rf.fit(Xf_tr, yf_tr)
    oof_rf[vl_idx] = _rf.predict_proba(Xf_vl)[:, 1]

    _hist = HistGradientBoostingClassifier(**best_hist_params)
    _hist.fit(Xf_tr, yf_tr)
    oof_hist[vl_idx] = _hist.predict_proba(Xf_vl)[:, 1]

oof_auc_lgbm = roc_auc_score(y_train, oof_lgbm)
oof_auc_xgb  = roc_auc_score(y_train, oof_xgb)
oof_auc_cat  = roc_auc_score(y_train, oof_cat)
oof_auc_rf   = roc_auc_score(y_train, oof_rf)
oof_auc_hist = roc_auc_score(y_train, oof_hist)
print(f"OOF AUC — LGBM:{oof_auc_lgbm:.4f}, XGB:{oof_auc_xgb:.4f}, "
      f"Cat:{oof_auc_cat:.4f}, RF:{oof_auc_rf:.4f}, Hist:{oof_auc_hist:.4f}")

# ── Nelder-Mead optimal blend on OOF (no val leakage) ────────────────────────
_oof_list = [oof_lgbm, oof_xgb, oof_cat, oof_rf, oof_hist]
_val_list  = [lgbm_preds, xgb_preds, cat_preds, rf_preds, hist_preds]

def _neg_auc_oof(raw_w):
    w = np.abs(raw_w)
    s = w.sum()
    if s < 1e-12:
        return 1.0
    w = w / s
    blend = sum(w[i] * _oof_list[i] for i in range(5))
    return -roc_auc_score(y_train, blend)

_init = np.ones(5) / 5
_res  = minimize(_neg_auc_oof, _init, method='Nelder-Mead',
                 options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9})
_opt_w = np.abs(_res.x); _opt_w /= _opt_w.sum()
print(f"NM weights: LGBM={_opt_w[0]:.3f} XGB={_opt_w[1]:.3f} "
      f"Cat={_opt_w[2]:.3f} RF={_opt_w[3]:.3f} Hist={_opt_w[4]:.3f}")
nelder_preds = sum(_opt_w[i] * _val_list[i] for i in range(5))
nelder_auc   = roc_auc_score(y_val, nelder_preds)
print(f"Nelder-Mead blend val AUC: {nelder_auc:.6f}")

# Build augmented meta-learner features: 5 OOF preds + key raw features
# Adding WomanOrMasterBoy, Pclass, TE_Title, TE_Sex_Pclass to help meta-learner
# route trust across models based on passenger characteristics
meta_base = np.column_stack([oof_lgbm, oof_xgb, oof_cat, oof_rf, oof_hist])
meta_base_val = np.column_stack([lgbm_preds, xgb_preds, cat_preds, rf_preds, hist_preds])

# Augment with interpretable raw features (already OOF-encoded where relevant)
aug_cols = ['WomanOrMasterBoy', 'Pclass', 'TE_Title', 'TE_Sex_Pclass']
aug_cols_present = [c for c in aug_cols if c in X_train_fe.columns and c in X_val_fe.columns]
if aug_cols_present:
    oof_meta = np.column_stack([meta_base, X_train_fe[aug_cols_present].values])
    val_meta = np.column_stack([meta_base_val, X_val_fe[aug_cols_present].values])
    print(f"Meta-learner augmented with: {aug_cols_present}")
else:
    oof_meta = meta_base
    val_meta = meta_base_val

meta_lr = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1, 10, 100],
    cv=5,
    max_iter=1000,
    random_state=42,
    scoring='roc_auc',
)
meta_lr.fit(oof_meta, y_train)
stacked_preds = meta_lr.predict_proba(val_meta)[:, 1]
stacked_auc   = roc_auc_score(y_val, stacked_preds)
print(f"Stacked LR val AUC: {stacked_auc:.6f}")

# ── Ensemble comparison ──────────────────────────────────────────────────────
ens_equal = (lgbm_preds + xgb_preds + cat_preds + rf_preds + hist_preds) / 5.0
ens_equal_auc = roc_auc_score(y_val, ens_equal)

# Weight by CV AUC scores
cv_aucs = np.array([
    study_lgbm.best_value, study_xgb.best_value,
    study_cat.best_value,  study_rf.best_value, study_hist.best_value,
])
weights = cv_aucs / cv_aucs.sum()
ens_weighted = (weights[0]*lgbm_preds + weights[1]*xgb_preds +
                weights[2]*cat_preds  + weights[3]*rf_preds +
                weights[4]*hist_preds)
ens_weighted_auc = roc_auc_score(y_val, ens_weighted)

print(f"Ensemble equal    AUC: {ens_equal_auc:.6f}")
print(f"Ensemble weighted AUC: {ens_weighted_auc:.6f}")

best_val_roc_auc = max(
    lgbm_auc, xgb_auc, cat_auc, rf_auc, hist_auc,
    ens_equal_auc, ens_weighted_auc, stacked_auc, nelder_auc
)
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
