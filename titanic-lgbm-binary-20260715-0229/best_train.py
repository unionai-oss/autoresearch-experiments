import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
# optuna/xgboost/catboost are in user site-packages
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
y = df[target_col].values
X_raw = df.drop(columns=[target_col])

print(f"Dataset: {df.shape}, classes: {np.bincount(y)}")

# ===================== Feature Engineering =====================
CABIN_NAN_PLACEHOLDER = "B96 B98"


def engineer_features(df):
    df = df.copy()

    # 0. Extract Surname before Name is dropped (family group size without target leakage)
    df['Surname'] = df['Name'].str.split(',').str[0].str.strip()
    df['SurnameSize'] = df.groupby('Surname')['Surname'].transform('count')

    # 1. Title from Name — very predictive on Titanic
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
    df['Title'] = df['Title'].replace(title_map)
    rare = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}
    df['Title'] = df['Title'].apply(lambda x: 'Rare' if x in rare else x)

    # 2. Age missing flag (before imputation)
    df['Age_missing'] = df['Age'].isna().astype(int)

    # 3. Age imputation by Title + Pclass group (vastly better than global median)
    age_medians = df.groupby(['Title', 'Pclass'])['Age'].median()
    global_age_median = df['Age'].median()
    mask = df['Age'].isna()
    df.loc[mask, 'Age'] = df.loc[mask].apply(
        lambda r: age_medians.get((r['Title'], r['Pclass']), global_age_median), axis=1
    )

    # 3.5. Age percentile rank within sex group (relative age — stronger signal than raw age)
    df['AgePctileInSex'] = df.groupby('Sex')['Age'].rank(pct=True)

    # 4. Fare: impute missing by Pclass median before computing per-person fare
    df['Fare_missing'] = df['Fare'].isna().astype(int)
    pclass_fare_median = df.groupby('Pclass')['Fare'].median()
    df['Fare'] = df['Fare'].fillna(df['Pclass'].map(pclass_fare_median))

    # 4.5. Fare percentile rank within Pclass (relative wealth within class)
    df['FarePctileInPclass'] = df.groupby('Pclass')['Fare'].rank(pct=True)

    # 5. Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)

    # 6. Fare features (right-skewed → log transform)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['LogFare'] = np.log1p(df['Fare'])
    df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')

    # 7. Cabin features
    df['HasRealCabin'] = (df['Cabin'] != CABIN_NAN_PLACEHOLDER).astype(int)
    df['Deck'] = df['Cabin'].str[0]
    df.loc[df['Cabin'] == CABIN_NAN_PLACEHOLDER, 'Deck'] = 'X'
    df['NumCabins'] = df.apply(
        lambda row: len(row['Cabin'].split()) if row['Cabin'] != CABIN_NAN_PLACEHOLDER else 0,
        axis=1
    )
    # DeckOrd: ordinal proximity to lifeboats (A deck = 7 = closest, G = 1, X = 0 = no cabin)
    deck_ord = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'X': 0}
    df['DeckOrd'] = df['Deck'].map(deck_ord).fillna(0).astype(int)

    # 8. Age features (age is now fully imputed)
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsInfant'] = (df['Age'] < 5).astype(int)  # infants had ~75% survival rate
    df['IsYouth'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
    df['IsSenior'] = (df['Age'] > 60).astype(int)
    df['AgeSq'] = df['Age'] ** 2
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # 9. "Women and children first" — the dominant survival rule
    df['WomenChild'] = ((df['Sex'] == 'female') | (df['Age'] < 15)).astype(int)
    df['WomenChildPclass'] = df['WomenChild'].astype(str) + '_' + df['Pclass'].astype(str)

    # 9.5. Three-way interaction: AgeGroup × Sex × Pclass
    age_cat = pd.cut(df['Age'], bins=[0, 5, 12, 18, 35, 60, 100],
                     labels=['infant', 'child', 'teen', 'young', 'adult', 'senior'])
    df['AgeSexPclass'] = age_cat.astype(str) + '_' + df['Sex'] + '_' + df['Pclass'].astype(str)

    # 9.6. Pclass + IsAlone (lone first-class vs lone third-class differ significantly)
    df['Pclass_IsAlone'] = df['Pclass'].astype(str) + '_' + df['IsAlone'].astype(str)

    # 9.7. IsMother: female, adult (≥18), traveling with child (Parch > 0), Mrs title
    df['IsMother'] = (
        (df['Sex'] == 'female') & (df['Age'] >= 18) &
        (df['Parch'] > 0) & (df['Title'] == 'Mrs')
    ).astype(int)

    # 9.8. MaleAdult3rd: male, adult, 3rd class — captures lowest-survival group
    df['MaleAdult3rd'] = (
        (df['Sex'] == 'male') & (df['Age'] >= 18) & (df['Pclass'] == 3)
    ).astype(int)

    # 10. Ticket features
    df['TicketPrefix'] = (
        df['Ticket'].str.extract(r'^([A-Za-z0-9./]+)\s', expand=False).fillna('NUMERIC')
    )
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    # Numeric part of ticket (lower numbers → forward decks, closer to boats)
    df['TicketNumeric'] = pd.to_numeric(
        df['Ticket'].str.extract(r'(\d+)$', expand=False), errors='coerce'
    ).fillna(-1)
    df['LogTicketNumeric'] = np.log1p(df['TicketNumeric'].clip(lower=0))

    # 11. Key interactions
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['Pclass_Title'] = df['Pclass'].astype(str) + '_' + df['Title']
    df['Title_Sex'] = df['Title'] + '_' + df['Sex']
    df['Age_x_Pclass'] = df['Age'] * df['Pclass']
    df['LogFare_x_Pclass'] = df['LogFare'] * df['Pclass']
    df['Age_x_Sex'] = df['Age'] * (df['Sex'] == 'male').astype(int)
    df['FamilySize_x_Pclass'] = df['FamilySize'] * df['Pclass']
    df['FareBin_x_Sex'] = df['FareBin'].astype(str) + '_' + df['Sex']

    # 12. Embarked interaction
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked_Pclass'] = df['Embarked'] + '_' + df['Pclass'].astype(str)

    # 13. Drop raw text columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Surname'])

    return df


X_eng = engineer_features(X_raw)
print(f"Features after engineering: {X_eng.shape[1]}")

# ===================== Preprocessing =====================

def preprocess_encoded(X_tr, X_va):
    """Fit on X_tr, apply to X_va — label encode categoricals."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_va[col] = X_va[col].fillna(med)

    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
        X_va[col] = X_va[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(X_tr[col])
        known = set(le.classes_)
        X_va[col] = X_va[col].apply(lambda x: x if x in known else le.classes_[0])
        X_tr[col] = le.transform(X_tr[col])
        X_va[col] = le.transform(X_va[col])

    return X_tr, X_va, cat_cols


def preprocess_catboost(X_tr, X_va):
    """For CatBoost: keep categoricals as strings (CatBoost handles natively)."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_va[col] = X_va[col].fillna(med)

    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
        X_va[col] = X_va[col].fillna("MISSING").astype(str)

    return X_tr, X_va, cat_cols


# ===================== Optuna for LightGBM =====================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
    }

    oof_preds = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_eng, y):
        X_tr_f, X_va_f, cat_cols = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        y_tr, y_va = y[tr_idx], y[va_idx]
        ds_tr = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols)
        ds_va = lgb.Dataset(X_va_f, label=y_va, reference=ds_tr)
        model = lgb.train(
            params, ds_tr, num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds[va_idx] = model.predict(X_va_f)

    return roc_auc_score(y, oof_preds)


print("Running LightGBM Optuna (200 trials)...")
lgb_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5),
)
lgb_study.optimize(lgb_objective, n_trials=200, timeout=330, show_progress_bar=False)
lgb_best = lgb_study.best_params
lgb_best.update({"objective": "binary", "metric": "auc", "verbosity": -1, "n_jobs": -1, "random_state": 42})
print(f"LightGBM best AUC={lgb_study.best_value:.6f} ({len(lgb_study.trials)} trials)")


# ===================== Optuna for XGBoost =====================

def xgb_objective(trial):
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        verbosity=0,
        nthread=4,
        seed=42,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        n_estimators=2000,
        early_stopping_rounds=50,
    )

    oof_preds = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_eng, y):
        X_tr_f, X_va_f, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_f, y_tr, eval_set=[(X_va_f, y_va)], verbose=False)
        oof_preds[va_idx] = model.predict_proba(X_va_f)[:, 1]

    return roc_auc_score(y, oof_preds)


print("Running XGBoost Optuna (100 trials)...")
xgb_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
xgb_study.optimize(xgb_objective, n_trials=100, timeout=250, show_progress_bar=False)
xgb_best = xgb_study.best_params
xgb_best.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "verbosity": 0,
    "nthread": 4,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
})
print(f"XGBoost best AUC={xgb_study.best_value:.6f} ({len(xgb_study.trials)} trials)")


# ===================== Optuna for CatBoost =====================

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def cat_objective(trial):
    params = dict(
        iterations=1500,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
        border_count=trial.suggest_int("border_count", 32, 255),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 20),
        random_seed=42,
        eval_metric='AUC',
        early_stopping_rounds=50,
        verbose=False,
        train_dir='/tmp/catboost_info',
    )
    oof_preds = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_eng, y):
        X_tr_c, X_va_c, cat_cols_c = preprocess_catboost(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        model = CatBoostClassifier(**params)
        model.fit(X_tr_c, y[tr_idx], cat_features=cat_cols_c, eval_set=(X_va_c, y[va_idx]))
        oof_preds[va_idx] = model.predict_proba(X_va_c)[:, 1]
    return roc_auc_score(y, oof_preds)


print("Running CatBoost Optuna (80 trials)...")
cat_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
cat_study.optimize(cat_objective, n_trials=80, timeout=220, show_progress_bar=False)
cat_best = cat_study.best_params
cat_best.update({
    "iterations": 2000,
    "random_seed": 42,
    "eval_metric": "AUC",
    "early_stopping_rounds": 100,
    "verbose": False,
    "train_dir": "/tmp/catboost_info",
})
print(f"CatBoost best AUC={cat_study.best_value:.6f} ({len(cat_study.trials)} trials)")


# ===================== Optuna for MLP =====================
# MLP provides genuinely different inductive bias (smooth boundaries vs piecewise trees)

def mlp_objective(trial):
    h_layers = trial.suggest_categorical('h_layers', [
        (32,), (64,), (128,), (32, 16), (64, 32), (128, 64),
        (128, 64, 32), (256, 128), (256, 128, 64)
    ])
    alpha = trial.suggest_float('alpha', 1e-5, 0.3, log=True)
    lr_init = trial.suggest_float('lr_init', 5e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    oof_preds = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_eng, y):
        X_tr_m, X_va_m, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_m)
        X_va_s = scaler.transform(X_va_m)
        y_tr = y[tr_idx]
        model = MLPClassifier(
            hidden_layer_sizes=h_layers,
            alpha=alpha,
            learning_rate_init=lr_init,
            activation='relu',
            solver='adam',
            batch_size=batch_size,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )
        model.fit(X_tr_s, y_tr)
        oof_preds[va_idx] = model.predict_proba(X_va_s)[:, 1]
    return roc_auc_score(y, oof_preds)


print("Running MLP Optuna (50 trials)...")
mlp_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
mlp_study.optimize(mlp_objective, n_trials=50, timeout=90, show_progress_bar=False)
mlp_best_params = mlp_study.best_params
print(f"MLP best AUC={mlp_study.best_value:.6f} ({len(mlp_study.trials)} trials)")


# ===================== Optuna for HistGradientBoosting =====================
# HGBC: sklearn's native fast GBM; different from LGB/XGB (histogram-based,
# native missing-value handling, different regularization path) — adds model diversity

def hgb_objective(trial):
    params = dict(
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_iter=1000,
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 63),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 60),
        l2_regularization=trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
        max_bins=trial.suggest_int('max_bins', 64, 255),
        random_state=42,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1,
    )
    oof_preds = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_eng, y):
        X_tr_h, X_va_h, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_tr_h, y[tr_idx])
        oof_preds[va_idx] = model.predict_proba(X_va_h)[:, 1]
    return roc_auc_score(y, oof_preds)


print("Running HistGradientBoosting Optuna (40 trials)...")
hgb_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
hgb_study.optimize(hgb_objective, n_trials=40, timeout=80, show_progress_bar=False)
hgb_best_params = hgb_study.best_params
hgb_best_params.update({
    "max_iter": 1000,
    "random_state": 42,
    "early_stopping": True,
    "n_iter_no_change": 30,
    "validation_fraction": 0.1,
})
print(f"HGBC best AUC={hgb_study.best_value:.6f} ({len(hgb_study.trials)} trials)")


# ===================== Final 5-Fold CV: All 7 Models =====================

oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))
oof_et = np.zeros(len(y))
oof_rf = np.zeros(len(y))
oof_mlp = np.zeros(len(y))
oof_hgb = np.zeros(len(y))
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_eng, y)):
    y_tr, y_va = y[tr_idx], y[va_idx]

    # --- LightGBM ---
    X_tr_l, X_va_l, cat_cols_l = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    ds_tr = lgb.Dataset(X_tr_l, label=y_tr, categorical_feature=cat_cols_l)
    ds_va = lgb.Dataset(X_va_l, label=y_va, reference=ds_tr)
    lgb_model = lgb.train(
        lgb_best, ds_tr, num_boost_round=3000,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va_l)

    # --- XGBoost ---
    X_tr_x, X_va_x, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    xgb_model = xgb.XGBClassifier(**xgb_best)
    xgb_model.fit(X_tr_x, y_tr, eval_set=[(X_va_x, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict_proba(X_va_x)[:, 1]

    # --- CatBoost (Optuna-tuned, handles categoricals natively) ---
    X_tr_c, X_va_c, cat_cols_c = preprocess_catboost(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    cat_model = CatBoostClassifier(**cat_best)
    cat_model.fit(X_tr_c, y_tr, cat_features=cat_cols_c, eval_set=(X_va_c, y_va))
    oof_cat[va_idx] = cat_model.predict_proba(X_va_c)[:, 1]

    # --- ExtraTrees (high variance, strong diversity vs GBMs) ---
    X_tr_e, X_va_e, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    et_model = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    et_model.fit(X_tr_e, y_tr)
    oof_et[va_idx] = et_model.predict_proba(X_va_e)[:, 1]

    # --- RandomForest (bagging complement to boosting) ---
    X_tr_r, X_va_r, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    rf_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_tr_r, y_tr)
    oof_rf[va_idx] = rf_model.predict_proba(X_va_r)[:, 1]

    # --- MLP (neural network — smooth boundaries, different inductive bias from trees) ---
    X_tr_m, X_va_m, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    scaler = StandardScaler()
    X_tr_ms = scaler.fit_transform(X_tr_m)
    X_va_ms = scaler.transform(X_va_m)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=mlp_best_params['h_layers'],
        alpha=mlp_best_params['alpha'],
        learning_rate_init=mlp_best_params['lr_init'],
        activation='relu',
        solver='adam',
        batch_size=mlp_best_params['batch_size'],
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )
    mlp_model.fit(X_tr_ms, y_tr)
    oof_mlp[va_idx] = mlp_model.predict_proba(X_va_ms)[:, 1]

    # --- HistGradientBoosting (sklearn native GBM — different regularization from LGB/XGB) ---
    X_tr_h, X_va_h, _ = preprocess_encoded(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    hgb_model = HistGradientBoostingClassifier(**hgb_best_params)
    hgb_model.fit(X_tr_h, y_tr)
    oof_hgb[va_idx] = hgb_model.predict_proba(X_va_h)[:, 1]

    # Fold-level ensemble stats
    fold_ens = (oof_lgb[va_idx] + oof_xgb[va_idx] + oof_cat[va_idx] +
                oof_et[va_idx] + oof_rf[va_idx] + oof_mlp[va_idx] + oof_hgb[va_idx]) / 7
    fold_auc = roc_auc_score(y_va, fold_ens)
    fold_aucs.append(fold_auc)
    lgb_f = roc_auc_score(y_va, oof_lgb[va_idx])
    xgb_f = roc_auc_score(y_va, oof_xgb[va_idx])
    cat_f = roc_auc_score(y_va, oof_cat[va_idx])
    et_f  = roc_auc_score(y_va, oof_et[va_idx])
    rf_f  = roc_auc_score(y_va, oof_rf[va_idx])
    mlp_f = roc_auc_score(y_va, oof_mlp[va_idx])
    hgb_f = roc_auc_score(y_va, oof_hgb[va_idx])
    print(f"Fold {fold+1}: LGB={lgb_f:.4f} XGB={xgb_f:.4f} CAT={cat_f:.4f} "
          f"ET={et_f:.4f} RF={rf_f:.4f} MLP={mlp_f:.4f} HGB={hgb_f:.4f} ENS={fold_auc:.4f}")

lgb_auc = roc_auc_score(y, oof_lgb)
xgb_auc = roc_auc_score(y, oof_xgb)
cat_auc = roc_auc_score(y, oof_cat)
et_auc  = roc_auc_score(y, oof_et)
rf_auc  = roc_auc_score(y, oof_rf)
mlp_auc = roc_auc_score(y, oof_mlp)
hgb_auc = roc_auc_score(y, oof_hgb)
print(f"Individual OOF: LGB={lgb_auc:.6f} XGB={xgb_auc:.6f} CAT={cat_auc:.6f} "
      f"ET={et_auc:.6f} RF={rf_auc:.6f} MLP={mlp_auc:.6f} HGB={hgb_auc:.6f}")

# Equal-weight ensemble (all 7 models)
oof_equal7 = (oof_lgb + oof_xgb + oof_cat + oof_et + oof_rf + oof_mlp + oof_hgb) / 7
equal7_auc = roc_auc_score(y, oof_equal7)

# Equal-weight (6 models, exclude HGBC — fallback if HGBC hurts)
oof_equal6 = (oof_lgb + oof_xgb + oof_cat + oof_et + oof_rf + oof_mlp) / 6
equal6_auc = roc_auc_score(y, oof_equal6)

# Equal-weight (5 tree models only — fallback if neural nets hurt)
oof_equal5 = (oof_lgb + oof_xgb + oof_cat + oof_et + oof_rf) / 5
equal5_auc = roc_auc_score(y, oof_equal5)

# AUC-proportional weighted ensemble across all 7 models
aucs7 = np.array([lgb_auc, xgb_auc, cat_auc, et_auc, rf_auc, mlp_auc, hgb_auc])
weights7 = aucs7 / aucs7.sum()
oof_weighted7 = (weights7[0]*oof_lgb + weights7[1]*oof_xgb + weights7[2]*oof_cat +
                 weights7[3]*oof_et  + weights7[4]*oof_rf  + weights7[5]*oof_mlp +
                 weights7[6]*oof_hgb)
weighted7_auc = roc_auc_score(y, oof_weighted7)

# GBM-only blend (LGB + XGB + CAT + HGB) — four complementary GBMs
oof_gbm4 = (oof_lgb + oof_xgb + oof_cat + oof_hgb) / 4
gbm4_auc = roc_auc_score(y, oof_gbm4)

# GBM-only blend (LGB + XGB + CAT) — used as baseline comparison
oof_gbm3 = (oof_lgb + oof_xgb + oof_cat) / 3
gbm3_auc = roc_auc_score(y, oof_gbm3)

all_blends = {
    "equal7": equal7_auc,
    "equal6": equal6_auc,
    "equal5": equal5_auc,
    "weighted7": weighted7_auc,
    "gbm4": gbm4_auc,
    "gbm3": gbm3_auc,
}
oof_auc = max(all_blends.values())
best_blend = max(all_blends, key=all_blends.get)

print(f"Equal-7 AUC={equal7_auc:.6f} | Equal-6 AUC={equal6_auc:.6f} | Equal-5 AUC={equal5_auc:.6f}")
print(f"Weighted-7 AUC={weighted7_auc:.6f} | GBM4 AUC={gbm4_auc:.6f} | GBM3 AUC={gbm3_auc:.6f}")
print(f"Best blend: {best_blend} | Mean fold: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")

best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
