import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
# optuna/xgboost/catboost are in user site-packages
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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

    # 4. Fare: impute missing by Pclass median before computing per-person fare
    df['Fare_missing'] = df['Fare'].isna().astype(int)
    pclass_fare_median = df.groupby('Pclass')['Fare'].median()
    df['Fare'] = df['Fare'].fillna(df['Pclass'].map(pclass_fare_median))

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

    # 8. Age features (age is now fully imputed)
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsYouth'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
    df['IsSenior'] = (df['Age'] > 60).astype(int)
    df['AgeSq'] = df['Age'] ** 2
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # 9. "Women and children first" — the dominant survival rule
    df['WomenChild'] = ((df['Sex'] == 'female') | (df['Age'] < 15)).astype(int)
    df['WomenChildPclass'] = df['WomenChild'].astype(str) + '_' + df['Pclass'].astype(str)

    # 10. Ticket features
    df['TicketPrefix'] = (
        df['Ticket'].str.extract(r'^([A-Za-z0-9./]+)\s', expand=False).fillna('NUMERIC')
    )
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')

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
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

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


# ===================== Final 5-Fold CV: LGB + XGB + CatBoost + ExtraTrees Ensemble =====================

oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))
oof_et = np.zeros(len(y))
oof_rf = np.zeros(len(y))
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

    fold_ens = (oof_lgb[va_idx] + oof_xgb[va_idx] + oof_cat[va_idx] + oof_et[va_idx] + oof_rf[va_idx]) / 5
    fold_auc = roc_auc_score(y_va, fold_ens)
    fold_aucs.append(fold_auc)
    lgb_f = roc_auc_score(y_va, oof_lgb[va_idx])
    xgb_f = roc_auc_score(y_va, oof_xgb[va_idx])
    cat_f = roc_auc_score(y_va, oof_cat[va_idx])
    et_f = roc_auc_score(y_va, oof_et[va_idx])
    rf_f = roc_auc_score(y_va, oof_rf[va_idx])
    print(f"Fold {fold+1}: LGB={lgb_f:.4f} XGB={xgb_f:.4f} CAT={cat_f:.4f} ET={et_f:.4f} RF={rf_f:.4f} ENS={fold_auc:.4f}")

lgb_auc = roc_auc_score(y, oof_lgb)
xgb_auc = roc_auc_score(y, oof_xgb)
cat_auc = roc_auc_score(y, oof_cat)
et_auc = roc_auc_score(y, oof_et)
rf_auc = roc_auc_score(y, oof_rf)
print(f"Individual OOF: LGB={lgb_auc:.6f} XGB={xgb_auc:.6f} CAT={cat_auc:.6f} ET={et_auc:.6f} RF={rf_auc:.6f}")

# Equal-weight ensemble
oof_equal = (oof_lgb + oof_xgb + oof_cat + oof_et + oof_rf) / 5
equal_auc = roc_auc_score(y, oof_equal)

# AUC-proportional weighted ensemble (models with higher OOF AUC get more weight)
aucs = np.array([lgb_auc, xgb_auc, cat_auc, et_auc, rf_auc])
weights = aucs / aucs.sum()
oof_weighted = (weights[0]*oof_lgb + weights[1]*oof_xgb + weights[2]*oof_cat +
                weights[3]*oof_et + weights[4]*oof_rf)
weighted_auc = roc_auc_score(y, oof_weighted)

# GBM-only blend (exclude tree bagging models if they hurt)
oof_gbm3 = (oof_lgb + oof_xgb + oof_cat) / 3
gbm3_auc = roc_auc_score(y, oof_gbm3)

oof_auc = max(equal_auc, weighted_auc, gbm3_auc)
best_blend = "equal" if equal_auc >= weighted_auc and equal_auc >= gbm3_auc else (
    "weighted" if weighted_auc >= gbm3_auc else "gbm3")
print(f"Equal-weight AUC={equal_auc:.6f} | Weighted AUC={weighted_auc:.6f} | GBM3 AUC={gbm3_auc:.6f}")
print(f"Best blend: {best_blend} | Mean fold: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")

best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
