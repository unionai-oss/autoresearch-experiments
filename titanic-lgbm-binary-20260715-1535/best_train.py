import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Data loading ---
df = pd.read_parquet(DATA_PATH)
target_col = "Survived"
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]

le_target = LabelEncoder()
y_all = le_target.fit_transform(y_raw)
print(f"Class mapping: {dict(zip(le_target.classes_, range(len(le_target.classes_))))}")
print(f"Dataset: N={len(df)}, class distribution={dict(zip(*np.unique(y_all, return_counts=True)))}")


def engineer_features(X_raw):
    """Full feature engineering — no target info used, safe for full-dataset CV."""
    X = X_raw.copy()

    # Drop identifier
    X = X.drop(columns=['PassengerId'], errors='ignore')

    # ===== Name -> Title + Surname size =====
    if 'Name' in X.columns:
        title = X['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
        title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        rare_titles = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}
        X['Title'] = title.apply(lambda t: 'Rare' if t in rare_titles else t)
        # Surname group size — pure count feature, no leakage
        X['Surname'] = X['Name'].str.extract(r'^([^,]+),')[0].str.strip()
        surname_counts = X['Surname'].value_counts()
        X['Surname_size'] = X['Surname'].map(surname_counts).fillna(1).astype(int)
        X = X.drop(columns=['Name', 'Surname'])

    # ===== Ticket features =====
    if 'Ticket' in X.columns:
        ticket_counts = X['Ticket'].value_counts()
        X['Ticket_freq'] = X['Ticket'].map(ticket_counts).fillna(1).astype(int)
        X['Ticket_prefix'] = X['Ticket'].str.extract(r'^([A-Za-z/\.]+)')[0].fillna('NUM')
        X = X.drop(columns=['Ticket'])

    # ===== Age: missingness indicator + title-based imputation =====
    if 'Age' in X.columns:
        X['Age_missing'] = X['Age'].isna().astype(int)
        if 'Title' in X.columns:
            title_age_median = X.groupby('Title')['Age'].transform('median')
            global_age_median = X['Age'].median()
            X['Age'] = X['Age'].fillna(title_age_median).fillna(global_age_median)
        else:
            X['Age'] = X['Age'].fillna(X['Age'].median())

    # ===== Cabin -> Deck + missingness + count =====
    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isna().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('U')
        X['Cabin_count'] = X['Cabin'].apply(
            lambda c: len(str(c).split()) if pd.notna(c) else 0
        )
        X = X.drop(columns=['Cabin'])

    # ===== Fare: fill missing, per-person fare, log transform =====
    if 'Fare' in X.columns:
        X['Fare'] = X['Fare'].fillna(X['Fare'].median())
        if 'Ticket_freq' in X.columns:
            X['Fare_per_person'] = np.log1p(X['Fare'] / X['Ticket_freq'].clip(lower=1))
        if 'Pclass' in X.columns:
            X['Fare_Pclass_rank'] = X.groupby('Pclass')['Fare'].rank(pct=True)
        X['Fare'] = np.log1p(X['Fare'])

    # ===== Embarked =====
    if 'Embarked' in X.columns:
        X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

    # ===== Family features =====
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['FamilyGroup'] = X['FamilySize'].apply(
            lambda x: 0 if x == 1 else (1 if x <= 4 else 2)
        )

    # ===== Age-derived features =====
    if 'Age' in X.columns:
        X['IsChild'] = (X['Age'] < 12).astype(int)
        X['IsSenior'] = (X['Age'] > 60).astype(int)
        X['AgeBin'] = pd.cut(X['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False).fillna(0).astype(int)

    # ===== Sex-based interactions (core Titanic survival signal) =====
    if 'Sex' in X.columns and 'Pclass' in X.columns:
        sex_is_female = (X['Sex'] == 'female').astype(int)
        X['IsWoman'] = sex_is_female
        X['Woman_Pclass'] = sex_is_female * 4 - X['Pclass']
        X['WomanInHighClass'] = ((sex_is_female == 1) & (X['Pclass'] <= 2)).astype(int)
        X['ManInLowClass'] = ((sex_is_female == 0) & (X['Pclass'] >= 3)).astype(int)

        if 'IsChild' in X.columns:
            X['WomenOrChild'] = ((sex_is_female == 1) | (X['IsChild'] == 1)).astype(int)
            X['WomanChild_Pclass'] = X['WomenOrChild'] * 4 - X['Pclass']
            X['WomenOrChild_x_Class1'] = ((X['WomenOrChild'] == 1) & (X['Pclass'] == 1)).astype(int)

        if 'FamilySize' in X.columns:
            X['FamilySize_x_IsWoman'] = X['FamilySize'] * sex_is_female

        if 'Age' in X.columns:
            X['Age_x_IsWoman'] = X['Age'] * sex_is_female
            X['Age_x_Pclass'] = X['Age'] * X['Pclass']

    if 'IsAlone' in X.columns and 'Sex' in X.columns:
        sex_is_female2 = (X['Sex'] == 'female').astype(int)
        X['AloneWoman'] = (X['IsAlone'] * sex_is_female2).astype(int)
        X['AloneMan'] = (X['IsAlone'] * (1 - sex_is_female2)).astype(int)

    # Title + Pclass combined
    if 'Title' in X.columns and 'Pclass' in X.columns:
        X['Title_Pclass'] = X['Title'].astype(str) + '_' + X['Pclass'].astype(str)

    # FamilySize × Pclass
    if 'FamilySize' in X.columns and 'Pclass' in X.columns:
        X['FamilySize_x_Pclass'] = X['FamilySize'] * X['Pclass']

    # Surname_size × IsWoman (large named families, women survived more)
    if 'Surname_size' in X.columns and 'IsWoman' in X.columns:
        X['Surname_size_x_IsWoman'] = X['Surname_size'] * X['IsWoman']

    # ===== Fill any remaining numeric NaN =====
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(0)

    # Track categorical column names before encoding (for CatBoost)
    cat_col_names = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # ===== Label-encode all object/category columns =====
    for col in cat_col_names:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))

    return X, cat_col_names


X_all, cat_col_names = engineer_features(X_raw)
print(f"Engineered feature shape: {X_all.shape}")
print(f"Categorical columns: {cat_col_names}")

X_np = X_all.values
y_np = y_all

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


# =============================================================
# Helper: run OOF CV for each model family
# =============================================================

def run_lgb_oof(params, X, y, skf):
    oof = np.zeros(len(X))
    for tr_idx, vl_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[vl_idx], y[vl_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof[vl_idx] = m.predict_proba(X[vl_idx])[:, 1]
    return oof


def run_xgb_oof(params, X, y, skf):
    oof = np.zeros(len(X))
    for tr_idx, vl_idx in skf.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[vl_idx], y[vl_idx])], verbose=False)
        oof[vl_idx] = m.predict_proba(X[vl_idx])[:, 1]
    return oof


def run_cat_oof(params, X_df, cat_cols, y, skf):
    oof = np.zeros(len(X_df))
    for tr_idx, vl_idx in skf.split(X_df, y):
        X_tr = X_df.iloc[tr_idx].copy()
        X_vl = X_df.iloc[vl_idx].copy()
        for col in cat_cols:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype(str)
                X_vl[col] = X_vl[col].astype(str)
        train_pool = Pool(X_tr, y[tr_idx], cat_features=cat_cols)
        eval_pool = Pool(X_vl, y[vl_idx], cat_features=cat_cols)
        m = CatBoostClassifier(**params)
        m.fit(train_pool, eval_set=eval_pool)
        oof[vl_idx] = m.predict_proba(eval_pool)[:, 1]
    return oof


# =============================================================
# Optuna hyperparameter tuning
# =============================================================

# --- LightGBM tuning ---
def lgb_objective(trial):
    params = dict(
        objective='binary', metric='auc', boosting_type='gbdt', verbose=-1,
        num_leaves=trial.suggest_int('num_leaves', 15, 150),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0),
        bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 1.0),
        bagging_freq=trial.suggest_int('bagging_freq', 1, 10),
        min_child_samples=trial.suggest_int('min_child_samples', 3, 50),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        n_estimators=3000, random_state=42,
    )
    oof = run_lgb_oof(params, X_np, y_np, skf)
    return roc_auc_score(y_np, oof)


print("Tuning LightGBM (40 trials)...")
lgb_study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
print(f"Best LGB AUC={lgb_study.best_value:.6f}, params={lgb_study.best_params}")


# --- XGBoost tuning ---
def xgb_objective(trial):
    params = dict(
        n_estimators=3000, objective='binary:logistic', eval_metric='auc',
        random_state=42, verbosity=0, early_stopping_rounds=100,
        tree_method='hist',
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
    )
    oof = run_xgb_oof(params, X_np, y_np, skf)
    return roc_auc_score(y_np, oof)


print("Tuning XGBoost (30 trials)...")
xgb_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)
print(f"Best XGB AUC={xgb_study.best_value:.6f}, params={xgb_study.best_params}")


# --- CatBoost tuning ---
def cat_objective(trial):
    params = dict(
        iterations=3000, loss_function='Logloss', eval_metric='AUC',
        random_seed=42, verbose=0, train_dir='/tmp/catboost_info',
        early_stopping_rounds=100,
        depth=trial.suggest_int('depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
        border_count=trial.suggest_int('border_count', 32, 255),
    )
    oof = run_cat_oof(params, X_all, cat_col_names, y_np, skf)
    return roc_auc_score(y_np, oof)


print("Tuning CatBoost (20 trials)...")
cat_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_objective, n_trials=20, show_progress_bar=False)
print(f"Best CAT AUC={cat_study.best_value:.6f}, params={cat_study.best_params}")


# =============================================================
# Final OOF with best params
# =============================================================

# LightGBM
best_lgb_params = dict(
    objective='binary', metric='auc', boosting_type='gbdt', verbose=-1,
    n_estimators=3000, random_state=42,
    **lgb_study.best_params,
)
oof_lgb = run_lgb_oof(best_lgb_params, X_np, y_np, skf)
lgb_auc = roc_auc_score(y_np, oof_lgb)
print(f"\nFinal LGB OOF AUC: {lgb_auc:.6f}")

# XGBoost
best_xgb_params = dict(
    n_estimators=3000, objective='binary:logistic', eval_metric='auc',
    random_state=42, verbosity=0, early_stopping_rounds=100,
    tree_method='hist',
    **xgb_study.best_params,
)
oof_xgb = run_xgb_oof(best_xgb_params, X_np, y_np, skf)
xgb_auc = roc_auc_score(y_np, oof_xgb)
print(f"Final XGB OOF AUC: {xgb_auc:.6f}")

# CatBoost
best_cat_params = dict(
    iterations=3000, loss_function='Logloss', eval_metric='AUC',
    random_seed=42, verbose=0, train_dir='/tmp/catboost_info',
    early_stopping_rounds=100,
    **cat_study.best_params,
)
oof_cat = run_cat_oof(best_cat_params, X_all, cat_col_names, y_np, skf)
cat_auc = roc_auc_score(y_np, oof_cat)
print(f"Final CAT OOF AUC: {cat_auc:.6f}")

# =============================================================
# Rank-based ensemble (normalises score distributions)
# =============================================================
oof_lgb_r = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_r = rankdata(oof_xgb) / len(oof_xgb)
oof_cat_r = rankdata(oof_cat) / len(oof_cat)

oof_ensemble = (oof_lgb_r + oof_xgb_r + oof_cat_r) / 3.0
ensemble_auc = roc_auc_score(y_np, oof_ensemble)

print(f"\nIndividual OOF AUCs: LGB={lgb_auc:.6f}, XGB={xgb_auc:.6f}, CAT={cat_auc:.6f}")
print(f"Ensemble OOF AUC (rank-avg LGB+XGB+CAT): {ensemble_auc:.6f}")
print(f"BEST_VAL_ROC_AUC: {ensemble_auc:.6f}")
