import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(1, '/home/flyte/.local/lib/python3.12/site-packages')

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
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

X_tr_raw, X_va_raw, y_train, y_val = train_test_split(
    X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
class_counts = pd.Series(y_encoded).value_counts().sort_index().to_dict()
print(f"Dataset: N={len(df)}, classes={class_counts}, train={len(X_tr_raw)}, val={len(X_va_raw)}")

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)


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


def safe_logit(p, eps=1e-6):
    """Logit (log-odds) transform for meta-learner: maps probabilities to log-odds space.
    LR meta-learner is linear in log-odds space, so this makes the combination theoretically optimal."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def get_ticket_prefix(ticket):
    t = str(ticket).strip().upper()
    # Match leading alphabetic prefix (before any space or digit)
    # e.g. "SOTON/O2 3101294" -> "SOTON", "PC 17599" -> "PC", "113803" -> "NUMERIC"
    m = re.match(r'^([A-Z][A-Z./]*)', t)
    if m:
        prefix = re.sub(r'[^A-Z]', '', m.group(1))  # keep only letters
        return prefix if prefix else 'NUMERIC'
    return 'NUMERIC'


def engineer(df_tr: pd.DataFrame, df_va: pd.DataFrame):
    frames = [df_tr.copy(), df_va.copy()]

    for i, df in enumerate(frames):
        if 'PassengerId' in df.columns:
            df = df.drop(columns=['PassengerId'])
        if 'Name' in df.columns:
            df['Title'] = df['Name'].apply(get_title)
            df = df.drop(columns=['Name'])
        frames[i] = df

    if 'Age' in frames[0].columns:
        age_group_med = frames[0].groupby(['Title', 'Pclass'])['Age'].median()
        overall_age_med = frames[0]['Age'].median()

        def fill_age(row, grp_med=age_group_med, overall=overall_age_med):
            if pd.notna(row['Age']):
                return row['Age']
            key = (row['Title'], row['Pclass'])
            if key in grp_med.index and pd.notna(grp_med[key]):
                return grp_med[key]
            return overall

        for i, df in enumerate(frames):
            df['AgeMissing'] = df['Age'].isna().astype(np.int8)
            df['Age'] = df.apply(fill_age, axis=1)
            frames[i] = df

    for i, df in enumerate(frames):
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(np.int8)
            df['FamilyGroup'] = np.where(df['FamilySize'] == 1, 0,
                                 np.where(df['FamilySize'] <= 4, 1, 2)).astype(np.int8)

        if 'Fare' in df.columns:
            df['FareLog'] = np.log1p(df['Fare'])
            if 'FamilySize' in df.columns:
                df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
                df['FarePerPersonLog'] = np.log1p(df['FarePerPerson'])

        if 'Age' in df.columns:
            df['IsChild'] = (df['Age'] < 14).astype(np.int8)
            df['AgeBin'] = pd.cut(
                df['Age'], bins=[0, 12, 18, 35, 60, 150], labels=False
            ).fillna(0).astype(np.int8)
            if 'Pclass' in df.columns:
                df['AgePclass'] = df['Age'] * df['Pclass']

        if 'Sex' in df.columns and 'Age' in df.columns:
            df['IsAdultMale'] = (
                (df['Sex'] == 'male') & (df['Age'] >= 16)
            ).astype(np.int8)
            df['WomenChild'] = (
                (df['Sex'] == 'female') | (df['Age'] < 14)
            ).astype(np.int8)
            if 'Pclass' in df.columns:
                # Explicit class-stratified survival advantage: 1st class women/children ~97%, 3rd class ~50%
                df['WomenChildPclass'] = df['WomenChild'] * (4 - df['Pclass'])
            if 'Parch' in df.columns and 'Pclass' in df.columns:
                df['IsMother'] = (
                    (df['Sex'] == 'female') &
                    (df['Age'] > 18) &
                    (df['Parch'] > 0) &
                    (df['Pclass'] != 3)
                ).astype(np.int8)

        if 'Cabin' in df.columns:
            df['HasCabin'] = (df['Cabin'].str.len() > 1).astype(np.int8)
            df['Deck'] = df.apply(
                lambda row: str(row['Cabin'])[0] if len(str(row['Cabin'])) > 1 else 'Unknown',
                axis=1
            )
            df = df.drop(columns=['Cabin'])

        if 'Sex' in df.columns and 'Pclass' in df.columns:
            df['SexPclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)

        if 'Title' in df.columns and 'Pclass' in df.columns:
            df['TitlePclass'] = df['Title'].astype(str) + '_' + df['Pclass'].astype(str)

        frames[i] = df

    if 'Ticket' in frames[0].columns:
        ticket_freq = frames[0]['Ticket'].value_counts().to_dict()
        for i, df in enumerate(frames):
            frames[i]['TicketFreq'] = df['Ticket'].map(ticket_freq).fillna(1).astype(float)
            # TicketPrefix: alphabetic prefix of ticket captures booking class/agent correlations
            # e.g. "PC" -> 1st class Cherbourg, "SOTON" -> Southampton working class, "NUMERIC" -> often 3rd class
            frames[i]['TicketPrefix'] = df['Ticket'].apply(get_ticket_prefix)
            frames[i] = frames[i].drop(columns=['Ticket'])

    # FareRankByClass: within-Pclass fare percentile (computed from train only, applied to val)
    # Captures relative wealth within class — orthogonal to raw Fare and FarePerPerson
    if 'Fare' in frames[0].columns and 'Pclass' in frames[0].columns:
        frames[0]['FareRankByClass'] = 0.5  # default
        frames[1]['FareRankByClass'] = 0.5
        for pclass_val in sorted(frames[0]['Pclass'].dropna().unique()):
            tr_mask = frames[0]['Pclass'] == pclass_val
            va_mask = frames[1]['Pclass'] == pclass_val
            tr_fares = frames[0].loc[tr_mask, 'Fare'].fillna(0).values
            if len(tr_fares) == 0:
                continue
            # Training: standard percentile rank
            frames[0].loc[tr_mask, 'FareRankByClass'] = (
                pd.Series(tr_fares).rank(pct=True).values
            )
            # Val: rank each val fare against the training distribution
            va_fares = frames[1].loc[va_mask, 'Fare'].fillna(0).values
            if len(va_fares) > 0:
                frames[1].loc[va_mask, 'FareRankByClass'] = np.array([
                    float(np.mean(tr_fares <= f)) for f in va_fares
                ])

    return frames[0], frames[1]


# Save tickets before engineer() drops them — needed for OOF TicketGroupSurvival
tickets_tr = X_tr_raw['Ticket'].reset_index(drop=True).copy() if 'Ticket' in X_tr_raw.columns else None
tickets_va = X_va_raw['Ticket'].reset_index(drop=True).copy() if 'Ticket' in X_va_raw.columns else None

X_train_raw_eng, X_val_raw_eng = engineer(X_tr_raw, X_va_raw)
print(f"Engineered features ({len(X_train_raw_eng.columns)}): {X_train_raw_eng.columns.tolist()}")

cat_cols = [c for c in X_train_raw_eng.columns
            if X_train_raw_eng[c].dtype == object
            or pd.api.types.is_string_dtype(X_train_raw_eng[c])
            or str(X_train_raw_eng[c].dtype) == 'string']
num_cols = [c for c in X_train_raw_eng.columns if c not in cat_cols]
print(f"Cat cols: {cat_cols}")
print(f"Num cols: {num_cols}")

X_train = X_train_raw_eng.copy()
X_val = X_val_raw_eng.copy()

miss_thresh = 0.05
for col in X_train.columns:
    if X_train[col].isna().mean() > miss_thresh:
        X_train[f'{col}_missing'] = X_train[col].isna().astype(np.int8)
        X_val[f'{col}_missing'] = X_val[col].isna().astype(np.int8)

for col in num_cols:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

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

all_cols = X_train.columns.tolist()
cat_col_indices = [all_cols.index(c) for c in cat_cols]

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype(str)
    X_val_cat[col] = X_val_cat[col].astype(str)

X_train_np = X_train.values.astype(np.float64)
X_val_np = X_val.values.astype(np.float64)

# OOF TicketGroupSurvival: leak-free target encoding by ticket group
# Passengers sharing a ticket had correlated fates — this captures group survival signal
# different from TicketFreq (group size) which doesn't tell you WHO survived
if tickets_tr is not None:
    _tgs_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tgs_train_arr = np.full(len(X_train), y_train.mean())

    for _tr_idx, _va_idx in _tgs_skf.split(X_train, y_train):
        _tr_tix = tickets_tr.iloc[_tr_idx]
        _tr_y = y_train[_tr_idx]
        _surv_map = {}
        for _tix in _tr_tix.unique():
            _m = (_tr_tix == _tix).values
            if _m.sum() > 1:
                _surv_map[_tix] = float(_tr_y[_m].mean())
        _overall = float(_tr_y.mean())
        _va_tix = tickets_tr.iloc[_va_idx]
        tgs_train_arr[_va_idx] = np.array([_surv_map.get(t, _overall) for t in _va_tix])

    # Val set: use full training ticket survival rates (leak-free — val labels not used)
    _full_surv = {}
    for _tix in tickets_tr.unique():
        _m = (tickets_tr == _tix).values
        if _m.sum() > 1:
            _full_surv[_tix] = float(y_train[_m].mean())
    _overall_full = float(y_train.mean())
    if tickets_va is not None:
        tgs_val_arr = np.array([_full_surv.get(t, _overall_full) for t in tickets_va])
    else:
        tgs_val_arr = np.full(len(X_val), _overall_full)

    X_train['TicketGroupSurvival'] = tgs_train_arr
    X_val['TicketGroupSurvival'] = tgs_val_arr
    X_train_cat['TicketGroupSurvival'] = tgs_train_arr
    X_val_cat['TicketGroupSurvival'] = tgs_val_arr
    X_train_np = X_train.values.astype(np.float64)
    X_val_np = X_val.values.astype(np.float64)
    print(f"TicketGroupSurvival: train mean={tgs_train_arr.mean():.4f}, val mean={tgs_val_arr.mean():.4f}")

LGBM_FIXED = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': 1,
}


def lgbm_objective(trial):
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


lgbm_sampler = optuna.samplers.TPESampler(seed=42)
lgbm_study = optuna.create_study(direction='maximize', sampler=lgbm_sampler)
lgbm_study.optimize(lgbm_objective, n_trials=80, show_progress_bar=False)

lgbm_best_cv_auc = lgbm_study.best_value
lgbm_best_params = lgbm_study.best_params
print(f"LightGBM best CV AUC: {lgbm_best_cv_auc:.6f}")


def catboost_objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'iterations': 600,
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'od_wait': 40,
        'verbose': False,
        'task_type': 'CPU',
        'train_dir': '/tmp/catboost_info',
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for fold_tr_idx, fold_va_idx in skf.split(X_train_cat, y_train):
        X_ft = X_train_cat.iloc[fold_tr_idx]
        y_ft = y_train[fold_tr_idx]
        X_fv = X_train_cat.iloc[fold_va_idx]
        y_fv = y_train[fold_va_idx]

        train_pool = Pool(X_ft, label=y_ft, cat_features=cat_cols)
        val_pool = Pool(X_fv, label=y_fv, cat_features=cat_cols)

        model = CatBoostClassifier(**params, random_seed=42)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

        preds = model.predict_proba(X_fv)[:, 1]
        fold_aucs.append(roc_auc_score(y_fv, preds))

    return float(np.mean(fold_aucs))


cat_sampler = optuna.samplers.TPESampler(seed=123)
cat_study = optuna.create_study(direction='maximize', sampler=cat_sampler)
cat_study.optimize(catboost_objective, n_trials=40, show_progress_bar=False)

cat_best_cv_auc = cat_study.best_value
cat_best_params = cat_study.best_params
print(f"CatBoost best CV AUC: {cat_best_cv_auc:.6f}")

lgbm_final_params = {**LGBM_FIXED, **lgbm_best_params, 'bagging_freq': 1}

cat_final_params = {
    **cat_best_params,
    'iterations': 600,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'od_wait': 50,
    'verbose': False,
    'task_type': 'CPU',
    'train_dir': '/tmp/catboost_info',
}

skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgbm_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))

for fold_tr_idx, fold_va_idx in skf_oof.split(X_train, y_train):
    X_ft = X_train.iloc[fold_tr_idx]
    y_ft = y_train[fold_tr_idx]
    X_fv = X_train.iloc[fold_va_idx]
    y_fv = y_train[fold_va_idx]
    X_ft_cat = X_train_cat.iloc[fold_tr_idx]
    X_fv_cat = X_train_cat.iloc[fold_va_idx]

    dtrain_fold = lgb.Dataset(X_ft, label=y_ft, categorical_feature=cat_cols,
                              free_raw_data=False)
    dval_fold = lgb.Dataset(X_fv, label=y_fv, categorical_feature=cat_cols,
                            free_raw_data=False)
    m_lgbm = lgb.train(
        lgbm_final_params, dtrain_fold, num_boost_round=2000,
        valid_sets=[dval_fold],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    lgbm_oof[fold_va_idx] = m_lgbm.predict(X_fv)

    tr_pool = Pool(X_ft_cat, label=y_ft, cat_features=cat_cols)
    va_pool = Pool(X_fv_cat, label=y_fv, cat_features=cat_cols)
    m_cat = CatBoostClassifier(**cat_final_params, random_seed=42)
    m_cat.fit(tr_pool, eval_set=va_pool, use_best_model=True, verbose=False)
    cat_oof[fold_va_idx] = m_cat.predict_proba(X_fv_cat)[:, 1]

lgbm_oof_auc = roc_auc_score(y_train, lgbm_oof)
cat_oof_auc = roc_auc_score(y_train, cat_oof)
print(f"LightGBM OOF AUC: {lgbm_oof_auc:.6f}")
print(f"CatBoost OOF AUC: {cat_oof_auc:.6f}")

dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval_full = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

lgbm_preds_list = []
N_LGBM_SEEDS = 10
for seed in range(N_LGBM_SEEDS):
    seed_params = {**lgbm_final_params, 'seed': seed}
    model = lgb.train(
        seed_params,
        dtrain_full,
        num_boost_round=2000,
        valid_sets=[dval_full],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    lgbm_preds_list.append(model.predict(X_val))

lgbm_val_preds = np.mean(lgbm_preds_list, axis=0)
lgbm_val_auc = roc_auc_score(y_val, lgbm_val_preds)
print(f"LightGBM seed-ensemble Val AUC: {lgbm_val_auc:.6f}")

train_pool_full = Pool(X_train_cat, label=y_train, cat_features=cat_cols)
val_pool_full = Pool(X_val_cat, label=y_val, cat_features=cat_cols)

cat_preds_list = []
N_CAT_SEEDS = 5
for seed in range(N_CAT_SEEDS):
    model = CatBoostClassifier(**cat_final_params, random_seed=seed)
    model.fit(train_pool_full, eval_set=val_pool_full, use_best_model=True, verbose=False)
    cat_preds_list.append(model.predict_proba(X_val_cat)[:, 1])

cat_val_preds = np.mean(cat_preds_list, axis=0)
cat_val_auc = roc_auc_score(y_val, cat_val_preds)
print(f"CatBoost seed-ensemble Val AUC: {cat_val_auc:.6f}")

# Enrich meta-learner with nonlinear terms and domain anchors
# Use WomenChildPclass/3 instead of binary WomenChild: 4-level signal (0, 1/3, 2/3, 1)
# capturing the survival gradient across classes — 3rd class women/children ~50%, 1st class ~97%
# This allows the meta-learner to calibrate differently for high vs low class women/children
_wc_tr = X_train['WomenChildPclass'].values.astype(float) / 3.0
_wc_va = X_val['WomenChildPclass'].values.astype(float) / 3.0
_tgs_tr = X_train['TicketGroupSurvival'].values.astype(float) if 'TicketGroupSurvival' in X_train.columns else np.full(len(X_train), y_train.mean())
_tgs_va = X_val['TicketGroupSurvival'].values.astype(float) if 'TicketGroupSurvival' in X_val.columns else np.full(len(X_val), y_train.mean())
# Pclass_norm: direct class signal for ALL passengers (not just women/children like WCPclass)
# Maps 1st→1.0, 2nd→0.67, 3rd→0.33 — crucial for calibrating adult male predictions
# where 1st class (~37% survival) vs 3rd class (~15%) is not captured by WCPclass alone
_pclass_tr = (4.0 - X_train['Pclass'].values.astype(float)) / 3.0
_pclass_va = (4.0 - X_val['Pclass'].values.astype(float)) / 3.0

# Transform base model predictions to logit (log-odds) space before meta-learner.
# LR is a linear classifier in log-odds space — feeding logit(p) makes the relationship
# between base-model outputs and the ensemble output exactly linear (theoretically optimal).
# Interaction terms in logit space also capture model agreement/disagreement more sharply:
# logit(0.9)*logit(0.1) << 0 (strong disagreement), logit(0.9)*logit(0.9) >> 0 (both high).
lgbm_oof_l = safe_logit(lgbm_oof)
cat_oof_l = safe_logit(cat_oof)
lgbm_val_l = safe_logit(lgbm_val_preds)
cat_val_l = safe_logit(cat_val_preds)

# 14-feature meta-learner: extends exp 26's 13-feature logit-space design with
# TGS × Pclass_norm interaction.
# This captures class-stratified trust in the group survival signal:
# 1st class passengers with high TGS → very strong survival signal (high class + group survived)
# 3rd class passengers with same TGS → weaker signal (class drag reduces group priority effect)
# Currently missing: all TGS interactions are with model logits or WCPclass, but NOT with Pclass directly.
meta_X_train = np.column_stack([
    lgbm_oof_l, cat_oof_l, lgbm_oof_l * cat_oof_l,
    _wc_tr, _tgs_tr,
    lgbm_oof_l * _wc_tr, cat_oof_l * _wc_tr,
    lgbm_oof_l * _tgs_tr, cat_oof_l * _tgs_tr,
    _wc_tr * _tgs_tr,
    _pclass_tr,                          # direct Pclass signal for all passengers
    lgbm_oof_l * _pclass_tr,             # class-stratified LGBM logit calibration
    cat_oof_l * _pclass_tr,              # class-stratified CatBoost logit calibration
    _tgs_tr * _pclass_tr,                # NEW: class-stratified group survival trust
])
meta_X_val = np.column_stack([
    lgbm_val_l, cat_val_l, lgbm_val_l * cat_val_l,
    _wc_va, _tgs_va,
    lgbm_val_l * _wc_va, cat_val_l * _wc_va,
    lgbm_val_l * _tgs_va, cat_val_l * _tgs_va,
    _wc_va * _tgs_va,
    _pclass_va,
    lgbm_val_l * _pclass_va,
    cat_val_l * _pclass_va,
    _tgs_va * _pclass_va,                # NEW: class-stratified group survival trust
])

# Tune meta-learner C via inner 5-fold CV using a StandardScaler+LR pipeline.
# StandardScaler ensures L2 regularization (C) penalizes all 14 features equally,
# critical because logit features (~[-4,4]) and domain anchors ([0,1]) have very different scales.
# Pipeline fits scaler within each CV fold (no leakage).
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as MetaScaler

best_c, best_c_auc = 1.0, -1.0
for c_val in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    pipe_tmp = Pipeline([
        ('scaler', MetaScaler()),
        ('lr', LogisticRegression(C=c_val, random_state=42, max_iter=1000)),
    ])
    inner_scores = cross_val_score(pipe_tmp, meta_X_train, y_train, cv=5, scoring='roc_auc')
    c_auc = inner_scores.mean()
    if c_auc > best_c_auc:
        best_c_auc = c_auc
        best_c = c_val
print(f"Best meta C={best_c} (inner CV AUC={best_c_auc:.6f})")

# Final meta-learner: fit scaler on all training meta features, apply to val
meta_scaler = MetaScaler()
meta_X_train_s = meta_scaler.fit_transform(meta_X_train)
meta_X_val_s = meta_scaler.transform(meta_X_val)

meta = LogisticRegression(C=best_c, random_state=42, max_iter=1000)
meta.fit(meta_X_train_s, y_train)
meta_coef = meta.coef_[0]
_meta_feature_names = ['logit_LGBM', 'logit_Cat', 'logit_LGBM*logit_Cat', 'WCPclass', 'TGS',
                       'logitLGBM*WCPclass', 'logitCat*WCPclass', 'logitLGBM*TGS', 'logitCat*TGS',
                       'WCPclass*TGS', 'Pclass_norm', 'logitLGBM*Pclass', 'logitCat*Pclass',
                       'TGS*Pclass_norm']
_coef_str = ', '.join([f"{n}={v:.4f}" for n, v in zip(_meta_feature_names, meta_coef)])
print(f"Meta-learner weights: {_coef_str}")

val_preds = meta.predict_proba(meta_X_val_s)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Stacked Val AUC: {val_auc:.6f}")
print(f"Individual: LGBM={lgbm_val_auc:.6f}, CatBoost={cat_val_auc:.6f}")

print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")
