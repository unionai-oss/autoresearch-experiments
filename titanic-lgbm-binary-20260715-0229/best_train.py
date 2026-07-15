import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
# optuna is installed in user site-packages, not the venv — add it to path
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
y = df[target_col].values
X_raw = df.drop(columns=[target_col])

print(f"Dataset: {df.shape}, classes: {np.bincount(y)}")

# ===================== Feature Engineering =====================
CABIN_NAN_PLACEHOLDER = "B96 B98"  # The cleaned dataset uses this for missing Cabin


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

    # 12. Drop raw text columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    return df


X_eng = engineer_features(X_raw)
print(f"Features after engineering: {X_eng.shape[1]}")
print(f"Feature list: {list(X_eng.columns)}")

# ===================== Preprocessing (per-fold, no leakage) =====================

def preprocess(X_tr, X_va):
    """Fit on X_tr, apply to X_va — no label leakage."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(include="number").columns.tolist()

    # Numeric: fill any residual NaN with fold-train median
    for col in num_cols:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_va[col] = X_va[col].fillna(med)

    # Categorical: label-encode (fit on train, apply to val)
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("MISSING").astype(str)
        X_va[col] = X_va[col].fillna("MISSING").astype(str)
        le_col = LabelEncoder()
        le_col.fit(X_tr[col])
        known = set(le_col.classes_)
        X_va[col] = X_va[col].apply(lambda x: x if x in known else le_col.classes_[0])
        X_tr[col] = le_col.transform(X_tr[col])
        X_va[col] = le_col.transform(X_va[col])

    return X_tr, X_va, cat_cols


# ===================== Optuna Hyperparameter Search =====================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
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
        X_tr_f, X_va_f, cat_cols = preprocess(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
        y_tr, y_va = y[tr_idx], y[va_idx]

        ds_tr = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols)
        ds_va = lgb.Dataset(X_va_f, label=y_va, reference=ds_tr)

        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_va],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds[va_idx] = model.predict(X_va_f)

    return roc_auc_score(y, oof_preds)


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5),
)

# Time budget: ~8 min for Optuna, then final CV
study.optimize(objective, n_trials=300, timeout=480, show_progress_bar=False)

print(f"Optuna: best AUC={study.best_value:.6f} after {len(study.trials)} trials")
best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
})

# ===================== Final 5-Fold CV with best params =====================

oof_preds = np.zeros(len(y))
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_eng, y)):
    X_tr_f, X_va_f, cat_cols = preprocess(X_eng.iloc[tr_idx], X_eng.iloc[va_idx])
    y_tr, y_va = y[tr_idx], y[va_idx]

    ds_tr = lgb.Dataset(X_tr_f, label=y_tr, categorical_feature=cat_cols)
    ds_va = lgb.Dataset(X_va_f, label=y_va, reference=ds_tr)

    model = lgb.train(
        best_params,
        ds_tr,
        num_boost_round=3000,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[va_idx] = model.predict(X_va_f)
    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    fold_aucs.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF AUC: {oof_auc:.6f} | Mean fold: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")

best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
