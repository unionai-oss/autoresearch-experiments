import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
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

    # ===== Ticket group composition features (computed BEFORE dropping Ticket/Sex/Age) =====
    # These use only feature info (no survival target)
    if 'Ticket' in X.columns and 'Sex' in X.columns:
        ticket_n = X.groupby('Ticket')['Ticket'].transform('count')
        ticket_n_women = (X['Sex'] == 'female').groupby(X['Ticket']).transform('sum')
        X['TicketGroupAllWomen'] = (ticket_n_women == ticket_n).astype(int)
        X['TicketGroupWomenFrac'] = (ticket_n_women / ticket_n.clip(lower=1)).fillna(0.0)

    if 'Ticket' in X.columns and 'Age' in X.columns:
        # Only count confirmed children (known age < 12, no imputation needed for this flag)
        is_confirmed_child = (X['Age'] < 12).fillna(False)
        ticket_has_child = is_confirmed_child.groupby(X['Ticket']).transform('max')
        X['TicketGroupHasChild'] = ticket_has_child.astype(int)

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

    # ===== Cabin -> Deck + missingness + count + ordinal =====
    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isna().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('U')
        X['Cabin_count'] = X['Cabin'].apply(
            lambda c: len(str(c).split()) if pd.notna(c) else 0
        )
        # Ordinal deck: A=7 (top, closest to lifeboats), G=1 (bottom), U=0 (unknown)
        deck_order = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'U': 0}
        X['Deck_num'] = X['Deck'].map(deck_order).fillna(0).astype(int)
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
        X['Age_sq'] = X['Age'] ** 2

    # ===== Sex-based interactions (core Titanic survival signal) =====
    if 'Sex' in X.columns and 'Pclass' in X.columns:
        sex_is_female = (X['Sex'] == 'female').astype(int)
        X['IsWoman'] = sex_is_female
        X['SexPclass'] = sex_is_female * 4 - X['Pclass']  # numeric interaction
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

    # SibSp × IsWoman (married women = higher lifeboat priority)
    if 'SibSp' in X.columns and 'IsWoman' in X.columns:
        X['SibSp_x_IsWoman'] = X['SibSp'] * X['IsWoman']

    # Small family sweet spot (2-4 = best survival; alone or large family = worse)
    if 'FamilySize' in X.columns:
        X['SmallFamily'] = ((X['FamilySize'] >= 2) & (X['FamilySize'] <= 4)).astype(int)

    # Deck_num × Pclass
    if 'Deck_num' in X.columns and 'Pclass' in X.columns:
        X['Deck_x_Pclass'] = X['Deck_num'] * X['Pclass']

    # TicketGroupWomenFrac × Pclass (women-majority group in lower class = higher survival)
    if 'TicketGroupWomenFrac' in X.columns and 'Pclass' in X.columns:
        X['TicketWomenFrac_x_Pclass'] = X['TicketGroupWomenFrac'] * (4 - X['Pclass'])

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


def run_mlp_oof(X, y, skf, hidden_layer_sizes=(128, 64, 32), alpha=0.01, lr=0.001, max_iter=500):
    """MLPClassifier OOF — genuinely different model family from GBMs."""
    oof = np.zeros(len(X))
    for tr_idx, vl_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_vl = scaler.transform(X[vl_idx])
        m = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=32,
            learning_rate_init=lr,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
            tol=1e-5,
        )
        m.fit(X_tr, y[tr_idx])
        oof[vl_idx] = m.predict_proba(X_vl)[:, 1]
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


# --- MLP tuning ---
def mlp_objective(trial):
    n1 = trial.suggest_categorical('n1', [64, 128, 256])
    n2 = trial.suggest_categorical('n2', [32, 64, 128])
    n3 = trial.suggest_categorical('n3', [0, 16, 32, 64])
    alpha = trial.suggest_float('alpha', 1e-4, 0.1, log=True)
    lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    sizes = (n1, n2) if n3 == 0 else (n1, n2, n3)
    oof = run_mlp_oof(X_np, y_np, skf, hidden_layer_sizes=sizes, alpha=alpha, lr=lr, max_iter=800)
    return roc_auc_score(y_np, oof)


print("Tuning MLP (15 trials)...")
mlp_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
mlp_study.optimize(mlp_objective, n_trials=15, show_progress_bar=False)
best_mlp_p = mlp_study.best_params
print(f"Best MLP AUC={mlp_study.best_value:.6f}, params={best_mlp_p}")
_n3 = best_mlp_p['n3']
best_mlp_sizes = (best_mlp_p['n1'], best_mlp_p['n2']) if _n3 == 0 else (best_mlp_p['n1'], best_mlp_p['n2'], _n3)


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

# MLP — Optuna-tuned architecture
print("Running Optuna-tuned MLPClassifier OOF...")
oof_mlp = run_mlp_oof(X_np, y_np, skf,
                       hidden_layer_sizes=best_mlp_sizes,
                       alpha=best_mlp_p['alpha'],
                       lr=best_mlp_p['lr'],
                       max_iter=800)
mlp_auc = roc_auc_score(y_np, oof_mlp)
print(f"Final MLP OOF AUC: {mlp_auc:.6f} (architecture={best_mlp_sizes})")

# =============================================================
# Rank-normalize all OOF predictions
# =============================================================
oof_lgb_r = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_r = rankdata(oof_xgb) / len(oof_xgb)
oof_cat_r = rankdata(oof_cat) / len(oof_cat)
oof_mlp_r = rankdata(oof_mlp) / len(oof_mlp)

# Equal-weight blend across 4 models (baseline)
oof_ensemble = (oof_lgb_r + oof_xgb_r + oof_cat_r + oof_mlp_r) / 4.0
ensemble_auc = roc_auc_score(y_np, oof_ensemble)

# GBM-only 3-model ensemble
oof_gbm_only = (oof_lgb_r + oof_xgb_r + oof_cat_r) / 3.0
gbm_auc = roc_auc_score(y_np, oof_gbm_only)

print(f"\nIndividual OOF AUCs: LGB={lgb_auc:.6f}, XGB={xgb_auc:.6f}, CAT={cat_auc:.6f}, MLP={mlp_auc:.6f}")
print(f"GBM-only ensemble AUC: {gbm_auc:.6f}")
print(f"4-model equal-weight ensemble AUC: {ensemble_auc:.6f}")

# =============================================================
# Optuna-optimize ensemble weights (300 trials)
# Fit w_lgb, w_xgb, w_cat, w_mlp to maximize AUC on OOF preds
# OOF are honest (each sample predicted without seeing its label)
# =============================================================
print("\nOptimizing ensemble weights with Optuna (300 trials)...")

def ensemble_weight_objective(trial):
    w1 = trial.suggest_float('w_lgb', 0.1, 1.0)
    w2 = trial.suggest_float('w_xgb', 0.1, 1.0)
    w3 = trial.suggest_float('w_cat', 0.1, 1.0)
    w4 = trial.suggest_float('w_mlp', 0.0, 0.6)
    total = w1 + w2 + w3 + w4
    oof_blend = (w1 * oof_lgb_r + w2 * oof_xgb_r + w3 * oof_cat_r + w4 * oof_mlp_r) / total
    return roc_auc_score(y_np, oof_blend)

weight_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
weight_study.optimize(ensemble_weight_objective, n_trials=300, show_progress_bar=False)

bw = weight_study.best_params
total_w = bw['w_lgb'] + bw['w_xgb'] + bw['w_cat'] + bw['w_mlp']
oof_optuna_w = (
    bw['w_lgb'] * oof_lgb_r +
    bw['w_xgb'] * oof_xgb_r +
    bw['w_cat'] * oof_cat_r +
    bw['w_mlp'] * oof_mlp_r
) / total_w
optuna_weighted_auc = roc_auc_score(y_np, oof_optuna_w)
print(f"Optuna-weighted ensemble AUC: {optuna_weighted_auc:.6f}")
norm_weights = {k: round(v / total_w, 3) for k, v in bw.items()}
print(f"Normalized weights: {norm_weights}")

best_auc = max(ensemble_auc, gbm_auc, optuna_weighted_auc)
print(f"BEST_VAL_ROC_AUC: {best_auc:.6f}")
