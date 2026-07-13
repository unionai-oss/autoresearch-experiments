import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(DATA_PATH)
target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col].values

print(f"[DATA] Samples: {len(df)}, Classes: {np.unique(y)}, "
      f"Distribution: {{0: {(y==0).sum()}, 1: {(y==1).sum()}}}")

# ── Save raw group identifiers BEFORE feature engineering ─────────────────────
# Needed for OOF group survival features (no leakage)
raw_surname = (X['Name'].str.split(',').str[0].str.strip().values
               if 'Name' in X.columns else None)
raw_ticket  = X['Ticket'].values if 'Ticket' in X.columns else None

# ── Feature engineering ──────────────────────────────────────────────────────
def extract_features(X):
    X = X.copy()

    if 'Name' in X.columns:
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare = {'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Capt',
                'Ms', 'Sir', 'Jonkheer', 'Lady', 'Mme', 'Don', 'Dona'}
        X['Title'] = X['Title'].apply(lambda t: 'Rare' if t in rare else t)
        X['Title'] = X['Title'].fillna('Unknown')
        X['Surname'] = X['Name'].str.split(',').str[0].str.strip()
        X['NameLength'] = X['Name'].str.len()
        X = X.drop(columns=['Name'])

    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isnull().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('Unknown')
        X['NumCabins'] = X['Cabin'].fillna('').apply(
            lambda c: len(c.split()) if c else 0)
        X['CabinNumber'] = X['Cabin'].str.extract(r'(\d+)', expand=False).astype(float)
        X = X.drop(columns=['Cabin'])

    if 'PassengerId' in X.columns:
        X = X.drop(columns=['PassengerId'])

    if 'Ticket' in X.columns:
        ticket_counts = X['Ticket'].map(X['Ticket'].value_counts())
        X['TicketGroup'] = ticket_counts.fillna(1).astype(int)
        def get_prefix(t):
            t_str = str(t).strip().upper().replace('.', '').replace('/', '').replace(' ', '')
            prefix = ''.join(c for c in t_str if not c.isdigit()).strip()
            return prefix if prefix else 'NONE'
        X['TicketPrefix'] = X['Ticket'].apply(get_prefix)
        X['TicketNum'] = X['Ticket'].str.extract(r'(\d+)$', expand=False).astype(float)
        X = X.drop(columns=['Ticket'])

    if 'Age' in X.columns:
        X['Age_missing'] = X['Age'].isnull().astype(int)

    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    if 'Fare' in X.columns:
        X['Fare_log'] = np.log1p(X['Fare'].fillna(0))
        if 'TicketGroup' in X.columns:
            X['Fare_per_person'] = X['Fare'].fillna(0) / X['TicketGroup'].clip(lower=1)
            X['Fare_per_person_log'] = np.log1p(X['Fare_per_person'])

    if 'Sex' in X.columns and 'Pclass' in X.columns:
        X['Sex_Pclass'] = X['Sex'].astype(str) + '_' + X['Pclass'].astype(str)

    return X


X_fe = extract_features(X)

# ── Age imputation: Title-based median ───────────────────────────────────────
if 'Age' in X_fe.columns and 'Title' in X_fe.columns:
    title_age_med = X_fe.groupby('Title')['Age'].median()
    global_age_med = X_fe['Age'].median()
    def impute_age(row):
        if pd.isna(row['Age']):
            return title_age_med.get(row['Title'], global_age_med)
        return row['Age']
    X_fe['Age'] = X_fe.apply(impute_age, axis=1)

# ── Other imputation ──────────────────────────────────────────────────────────
for col in ['Fare']:
    if col in X_fe.columns:
        X_fe[col] = X_fe[col].fillna(X_fe[col].median())

for col in ['Embarked']:
    if col in X_fe.columns:
        X_fe[col] = X_fe[col].fillna(X_fe[col].mode()[0])

if 'TicketNum' in X_fe.columns:
    X_fe['TicketNum'] = X_fe['TicketNum'].fillna(X_fe['TicketNum'].median())

# ── Post-imputation features ─────────────────────────────────────────────────
if 'Age' in X_fe.columns and 'Sex' in X_fe.columns:
    X_fe['WomanOrChild'] = (
        (X_fe['Sex'] == 'female') | (X_fe['Age'] < 15)).astype(int)
    X_fe['IsChild'] = (X_fe['Age'] < 15).astype(int)
    X_fe['AgeBin'] = pd.cut(
        X_fe['Age'], bins=[0, 12, 18, 35, 60, 100],
        labels=['child', 'teen', 'adult', 'middle', 'senior']
    ).astype(str)

if 'Fare' in X_fe.columns and 'Pclass' in X_fe.columns:
    fare_stats = X_fe.groupby('Pclass')['Fare'].agg(['mean', 'std'])
    X_fe['Fare_class_z'] = X_fe.apply(
        lambda r: (r['Fare'] - fare_stats.loc[r['Pclass'], 'mean']) /
                  (fare_stats.loc[r['Pclass'], 'std'] + 1e-6), axis=1
    )

if ('Sex' in X_fe.columns and 'Parch' in X_fe.columns
        and 'Age' in X_fe.columns and 'Title' in X_fe.columns):
    X_fe['IsMother'] = (
        (X_fe['Sex'] == 'female') &
        (X_fe['Parch'] > 0) &
        (X_fe['Age'] > 18) &
        (X_fe['Title'] != 'Miss')
    ).astype(int)

if 'Surname' in X_fe.columns:
    X_fe['SurnameGroup'] = X_fe['Surname'].map(
        X_fe['Surname'].value_counts()).fillna(1).astype(int)
    X_fe = X_fe.drop(columns=['Surname'])

if 'FamilySize' in X_fe.columns:
    X_fe['FamilySizeGroup'] = np.select(
        [X_fe['FamilySize'] == 1, X_fe['FamilySize'] <= 4],
        ['alone', 'small'], default='large')

# ── Interaction features ──────────────────────────────────────────────────────
if 'IsAlone' in X_fe.columns and 'Pclass' in X_fe.columns:
    X_fe['IsAlone_Pclass'] = X_fe['IsAlone'] * X_fe['Pclass']

if 'WomanOrChild' in X_fe.columns and 'Pclass' in X_fe.columns:
    X_fe['WomanOrChild_Pclass'] = X_fe['WomanOrChild'] * X_fe['Pclass']

if 'FamilySize' in X_fe.columns and 'Pclass' in X_fe.columns:
    X_fe['FamilySize_Pclass'] = X_fe['FamilySize'] * X_fe['Pclass']

if 'Age' in X_fe.columns and 'Pclass' in X_fe.columns:
    X_fe['Age_Pclass'] = X_fe['Age'] * X_fe['Pclass']

# ── OOF Group Survival Features ──────────────────────────────────────────────
# Core insight: families / groups on Titanic tended to survive or die together.
# We compute "what fraction of this passenger's group survived?" using OOF
# cross-validation so there is NO label leakage into any validation fold.
global_mean = float(y.mean())

# Use a fixed 5-fold split for OOF survival computation
skf_surv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def compute_group_oof_survival(groups, y, skf, fallback, k_smooth=5):
    """OOF mean survival rate for same-group passengers with Bayesian smoothing.
    Smoothing pulls small-group estimates toward global mean to reduce noise."""
    oof = np.full(len(y), fallback, dtype=np.float64)
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        g_sum, g_cnt = {}, {}
        for g, t in zip(groups[tr_idx], y[tr_idx]):
            if g not in g_sum:
                g_sum[g] = 0
                g_cnt[g] = 0
            g_sum[g] += t
            g_cnt[g] += 1
        # Bayesian smoothing: pull small groups toward global mean
        g_rate = {g: (g_sum[g] + k_smooth * fallback) / (g_cnt[g] + k_smooth)
                  for g in g_sum}
        oof[va_idx] = [g_rate.get(g, fallback) for g in groups[va_idx]]
    return oof

if raw_surname is not None:
    X_fe['Surname_survival_oof'] = compute_group_oof_survival(
        raw_surname, y, skf_surv, global_mean)
    print(f"[FE] Surname survival OOF: "
          f"mean={X_fe['Surname_survival_oof'].mean():.3f}  "
          f"range=[{X_fe['Surname_survival_oof'].min():.3f}, "
          f"{X_fe['Surname_survival_oof'].max():.3f}]")

if raw_ticket is not None:
    X_fe['Ticket_survival_oof'] = compute_group_oof_survival(
        raw_ticket, y, skf_surv, global_mean)
    print(f"[FE] Ticket  survival OOF: "
          f"mean={X_fe['Ticket_survival_oof'].mean():.3f}  "
          f"range=[{X_fe['Ticket_survival_oof'].min():.3f}, "
          f"{X_fe['Ticket_survival_oof'].max():.3f}]")

# Surname+Ticket combined: most specific family unit (same family, same booking)
if raw_surname is not None and raw_ticket is not None:
    surname_ticket = np.array([f"{s}__{t}" for s, t in zip(raw_surname, raw_ticket)])
    X_fe['SurnameTicket_survival_oof'] = compute_group_oof_survival(
        surname_ticket, y, skf_surv, global_mean)
    print(f"[FE] SurnameTicket OOF: "
          f"mean={X_fe['SurnameTicket_survival_oof'].mean():.3f}  "
          f"range=[{X_fe['SurnameTicket_survival_oof'].min():.3f}, "
          f"{X_fe['SurnameTicket_survival_oof'].max():.3f}]")

# ── Categorical columns (CatBoost handles natively) ──────────────────────────
cat_cols_cb = ['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass',
               'AgeBin', 'FamilySizeGroup', 'TicketPrefix']
cat_cols_cb = [c for c in cat_cols_cb if c in X_fe.columns]
for col in cat_cols_cb:
    X_fe[col] = X_fe[col].astype(str).fillna('Unknown')

print(f"[FE] Features ({len(X_fe.columns)}): {list(X_fe.columns)}")
print(f"[FE] CatBoost categoricals: {cat_cols_cb}")

# ── K-fold target encoding (no leakage) ──────────────────────────────────────
te_cols = [c for c in ['Title', 'Deck', 'Sex_Pclass', 'AgeBin',
                        'FamilySizeGroup', 'Embarked', 'TicketPrefix']
           if c in X_fe.columns]
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
for col in te_cols:
    encoded = np.zeros(len(X_fe))
    for tr_i, va_i in skf_te.split(X_fe, y):
        cat_tr = X_fe[col].iloc[tr_i].values
        y_tr = y[tr_i]
        means = {}
        for cat_val, target_val in zip(cat_tr, y_tr):
            if cat_val not in means:
                means[cat_val] = []
            means[cat_val].append(target_val)
        means = {k: float(np.mean(v)) for k, v in means.items()}
        encoded[va_i] = [means.get(v, global_mean)
                         for v in X_fe[col].iloc[va_i].values]
    X_fe[f'{col}_te'] = encoded

print(f"[FE] Target-encoded: {te_cols}")

# ── Label-encode categoricals for LightGBM / XGBoost / RF ────────────────────
X_fe_lgb = X_fe.copy()
for col in cat_cols_cb:
    le = LabelEncoder()
    X_fe_lgb[col] = le.fit_transform(X_fe_lgb[col].astype(str))

# ── Cross-validation setup ────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 1. CatBoost Optuna HPO
# ============================================================
def cb_objective(trial):
    params = dict(
        iterations=3000,
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        depth=trial.suggest_int('depth', 4, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        random_strength=trial.suggest_float('random_strength', 0.1, 3.0),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 2.0),
        border_count=trial.suggest_int('border_count', 32, 128),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 1, 20),
        cat_features=cat_cols_cb,
        eval_metric='AUC',
        early_stopping_rounds=150,
        use_best_model=True,
        random_seed=42,
        verbose=False,
        train_dir='/tmp/catboost_info',
    )
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe, y):
        m = CatBoostClassifier(**params)
        m.fit(X_fe.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
              eval_set=(X_fe.iloc[va_idx].reset_index(drop=True), y[va_idx]))
        oof[va_idx] = m.predict_proba(
            X_fe.iloc[va_idx].reset_index(drop=True))[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] CatBoost (60 trials) ...")
study_cb = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_cb.optimize(cb_objective, n_trials=60, show_progress_bar=False)
best_cb = study_cb.best_params
print(f"[HPO] CatBoost best AUC={study_cb.best_value:.4f}")

# ============================================================
# 2. LightGBM Optuna HPO
# ============================================================
def lgb_objective(trial):
    params = dict(
        n_estimators=3000,
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        num_leaves=trial.suggest_int('num_leaves', 15, 63),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe_lgb, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx],
              eval_set=[(X_fe_lgb.iloc[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] LightGBM (30 trials) ...")
study_lgb = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=123))
study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=False)
best_lgb = study_lgb.best_params
print(f"[HPO] LightGBM best AUC={study_lgb.best_value:.4f}")

# ============================================================
# 3. XGBoost Optuna HPO
# ============================================================
def xgb_objective(trial):
    params = dict(
        n_estimators=3000,
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        gamma=trial.suggest_float('gamma', 0.0, 1.0),
        eval_metric='auc',
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe_lgb, y):
        m = XGBClassifier(**params)
        m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx],
              eval_set=[(X_fe_lgb.iloc[va_idx], y[va_idx])],
              verbose=False)
        oof[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] XGBoost (20 trials) ...")
study_xgb = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=456))
study_xgb.optimize(xgb_objective, n_trials=20, show_progress_bar=False)
best_xgb = study_xgb.best_params
print(f"[HPO] XGBoost best AUC={study_xgb.best_value:.4f}")

# ============================================================
# 4. RandomForest Optuna HPO (bagging diversity vs boosting)
# ============================================================
def rf_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 300, 1000),
        max_depth=trial.suggest_int('max_depth', 4, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_float('max_features', 0.1, 0.9),
        random_state=42,
        n_jobs=-1,
    )
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe_lgb, y):
        m = RandomForestClassifier(**params)
        m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] RandomForest (25 trials) ...")
study_rf = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=789))
study_rf.optimize(rf_objective, n_trials=25, show_progress_bar=False)
best_rf = study_rf.best_params
print(f"[HPO] RandomForest best AUC={study_rf.best_value:.4f}")

# ============================================================
# 5. HistGradientBoosting Optuna HPO (sklearn histogram GBDT —
#    different binning & regularization from LGB/XGB)
# ============================================================
def hgb_objective(trial):
    params = dict(
        max_iter=500,
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 63),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 50),
        l2_regularization=trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        random_state=42,
    )
    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe_lgb, y):
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] HistGradientBoosting (20 trials) ...")
study_hgb = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=321))
study_hgb.optimize(hgb_objective, n_trials=20, show_progress_bar=False)
best_hgb = study_hgb.best_params
print(f"[HPO] HistGradientBoosting best AUC={study_hgb.best_value:.4f}")

# ============================================================
# Final OOF: multiple fold seeds × multiple model seeds
# ============================================================
FOLD_SEEDS = [42, 123, 456]

# ── CatBoost ──────────────────────────────────────────────────────────────────
final_cb = dict(
    iterations=3000,
    cat_features=cat_cols_cb,
    eval_metric='AUC',
    early_stopping_rounds=150,
    use_best_model=True,
    verbose=False,
    train_dir='/tmp/catboost_info',
)
final_cb.update(best_cb)

CB_SEEDS = [42, 123, 456, 789, 1000]
print(f"[TRAIN] CatBoost: {len(CB_SEEDS)} model seeds × {len(FOLD_SEEDS)} fold seeds × 5 folds")
oof_cb_all = []
for fold_seed in FOLD_SEEDS:
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for model_seed in CB_SEEDS:
        oof_s = np.zeros(len(y))
        m_params = {k: v for k, v in final_cb.items() if k != 'random_seed'}
        for tr_idx, va_idx in skf_fs.split(X_fe, y):
            m = CatBoostClassifier(random_seed=model_seed, **m_params)
            m.fit(X_fe.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
                  eval_set=(X_fe.iloc[va_idx].reset_index(drop=True), y[va_idx]))
            oof_s[va_idx] = m.predict_proba(
                X_fe.iloc[va_idx].reset_index(drop=True))[:, 1]
        oof_cb_all.append(oof_s)

oof_cb = np.mean(oof_cb_all, axis=0)
cb_auc = roc_auc_score(y, oof_cb)
print(f"[CV] CatBoost multi-seed/fold OOF AUC: {cb_auc:.6f}")

# ── LightGBM ──────────────────────────────────────────────────────────────────
final_lgb = dict(n_estimators=3000, n_jobs=1, verbose=-1)
final_lgb.update(best_lgb)

LGB_SEEDS = [42, 123, 456]
print(f"[TRAIN] LightGBM: {len(LGB_SEEDS)} model seeds × {len(FOLD_SEEDS)} fold seeds × 5 folds")
oof_lgb_all = []
for fold_seed in FOLD_SEEDS:
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for model_seed in LGB_SEEDS:
        oof_s = np.zeros(len(y))
        params_s = dict(**final_lgb, random_state=model_seed)
        for tr_idx, va_idx in skf_fs.split(X_fe_lgb, y):
            m = lgb.LGBMClassifier(**params_s)
            m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_fe_lgb.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(-1)])
            oof_s[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
        oof_lgb_all.append(oof_s)

oof_lgb = np.mean(oof_lgb_all, axis=0)
lgb_auc = roc_auc_score(y, oof_lgb)
print(f"[CV] LightGBM multi-seed/fold OOF AUC: {lgb_auc:.6f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
final_xgb = dict(
    n_estimators=3000, eval_metric='auc',
    early_stopping_rounds=100,
    n_jobs=1, verbosity=0)
final_xgb.update(best_xgb)

XGB_SEEDS = [42, 123, 456]
print(f"[TRAIN] XGBoost: {len(XGB_SEEDS)} model seeds × {len(FOLD_SEEDS)} fold seeds × 5 folds")
oof_xgb_all = []
for fold_seed in FOLD_SEEDS:
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for model_seed in XGB_SEEDS:
        oof_s = np.zeros(len(y))
        params_s = dict(**final_xgb, random_state=model_seed)
        for tr_idx, va_idx in skf_fs.split(X_fe_lgb, y):
            m = XGBClassifier(**params_s)
            m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_fe_lgb.iloc[va_idx], y[va_idx])],
                  verbose=False)
            oof_s[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
        oof_xgb_all.append(oof_s)

oof_xgb = np.mean(oof_xgb_all, axis=0)
xgb_auc = roc_auc_score(y, oof_xgb)
print(f"[CV] XGBoost multi-seed/fold OOF AUC: {xgb_auc:.6f}")

# ── RandomForest ──────────────────────────────────────────────────────────────
final_rf = dict(n_jobs=-1)
final_rf.update(best_rf)

RF_SEEDS = [42, 123, 456]
FOLD_SEEDS_RF = [42, 123, 456]
print(f"[TRAIN] RandomForest: {len(RF_SEEDS)} model seeds × {len(FOLD_SEEDS_RF)} fold seeds × 5 folds")
oof_rf_all = []
for fold_seed in FOLD_SEEDS_RF:
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for model_seed in RF_SEEDS:
        oof_s = np.zeros(len(y))
        params_s = dict(**final_rf, random_state=model_seed)
        for tr_idx, va_idx in skf_fs.split(X_fe_lgb, y):
            m = RandomForestClassifier(**params_s)
            m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx])
            oof_s[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
        oof_rf_all.append(oof_s)

oof_rf = np.mean(oof_rf_all, axis=0)
rf_auc = roc_auc_score(y, oof_rf)
print(f"[CV] RandomForest multi-seed/fold OOF AUC: {rf_auc:.6f}")

# ── HistGradientBoosting ───────────────────────────────────────────────────────
final_hgb = dict(max_iter=500)
final_hgb.update(best_hgb)

HGB_SEEDS = [42, 123, 456]
FOLD_SEEDS_HGB = [42, 123, 456]
print(f"[TRAIN] HistGBM: {len(HGB_SEEDS)} model seeds × {len(FOLD_SEEDS_HGB)} fold seeds × 5 folds")
oof_hgb_all = []
for fold_seed in FOLD_SEEDS_HGB:
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    for model_seed in HGB_SEEDS:
        oof_s = np.zeros(len(y))
        params_s = dict(**final_hgb, random_state=model_seed)
        for tr_idx, va_idx in skf_fs.split(X_fe_lgb, y):
            m = HistGradientBoostingClassifier(**params_s)
            m.fit(X_fe_lgb.iloc[tr_idx], y[tr_idx])
            oof_s[va_idx] = m.predict_proba(X_fe_lgb.iloc[va_idx])[:, 1]
        oof_hgb_all.append(oof_s)

oof_hgb = np.mean(oof_hgb_all, axis=0)
hgb_auc = roc_auc_score(y, oof_hgb)
print(f"[CV] HistGBM multi-seed/fold OOF AUC: {hgb_auc:.6f}")

# ============================================================
# Optimise blend weights via Optuna on OOF predictions
# ============================================================
def blend_objective(trial):
    w1 = trial.suggest_float('w_cb',  0.0, 1.0)
    w2 = trial.suggest_float('w_lgb', 0.0, 1.0)
    w3 = trial.suggest_float('w_xgb', 0.0, 1.0)
    w4 = trial.suggest_float('w_rf',  0.0, 1.0)
    w5 = trial.suggest_float('w_hgb', 0.0, 1.0)
    total = w1 + w2 + w3 + w4 + w5 + 1e-9
    blend = (w1 * oof_cb + w2 * oof_lgb + w3 * oof_xgb + w4 * oof_rf + w5 * oof_hgb) / total
    return roc_auc_score(y, blend)

study_blend = optuna.create_study(
    direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
study_blend.optimize(blend_objective, n_trials=300, show_progress_bar=False)
bp = study_blend.best_params
total_w = bp['w_cb'] + bp['w_lgb'] + bp['w_xgb'] + bp['w_rf'] + bp['w_hgb'] + 1e-9
w_cb  = bp['w_cb']  / total_w
w_lgb = bp['w_lgb'] / total_w
w_xgb = bp['w_xgb'] / total_w
w_rf  = bp['w_rf']  / total_w
w_hgb = bp['w_hgb'] / total_w
print(f"[BLEND] CB={w_cb:.3f}  LGB={w_lgb:.3f}  XGB={w_xgb:.3f}  RF={w_rf:.3f}  HGB={w_hgb:.3f}")

oof_blended = w_cb * oof_cb + w_lgb * oof_lgb + w_xgb * oof_xgb + w_rf * oof_rf + w_hgb * oof_hgb
blended_auc = roc_auc_score(y, oof_blended)

equal_auc = roc_auc_score(y, (oof_cb + oof_lgb + oof_xgb + oof_rf + oof_hgb) / 5)

print(f"[RESULT] CatBoost     : {cb_auc:.6f}")
print(f"[RESULT] LightGBM     : {lgb_auc:.6f}")
print(f"[RESULT] XGBoost      : {xgb_auc:.6f}")
print(f"[RESULT] RandomForest : {rf_auc:.6f}")
print(f"[RESULT] HistGBM      : {hgb_auc:.6f}")
print(f"[RESULT] EqualBlend   : {equal_auc:.6f}")
print(f"[RESULT] OptBlend     : {blended_auc:.6f}")

final_auc = max(blended_auc, equal_auc, cb_auc, lgb_auc, xgb_auc, rf_auc, hgb_auc)
print(f"BEST_VAL_ROC_AUC: {final_auc:.6f}")
