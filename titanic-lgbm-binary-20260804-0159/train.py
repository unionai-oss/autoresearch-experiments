import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.12/site-packages')

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
y = df[target_col].values.astype(int)
X_raw = df.drop(columns=[target_col]).copy()

print(f"Dataset: {len(df)} samples, {X_raw.shape[1]} raw features")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ================================================
# Feature Engineering (Titanic-specific)
# ================================================
CABIN_FILL = "B96 B98"  # fill value used for missing cabins in this dataset


def engineer_features(X):
    df = X.copy()

    # Title from Name — highly predictive for Titanic survival
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Jonkheer': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Sir': 'Rare',
        'Ms': 'Mrs'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # Age imputation by Title group median (Master ~ 4.5, Miss ~ 22, Mr ~ 30, etc.)
    # This is more accurate than global median imputation for Titanic
    df['AgeIsNull'] = df['Age'].isna().astype(int)
    title_age_medians = df.groupby('Title')['Age'].median()
    global_age_median = df['Age'].median()
    df['Age'] = df.apply(
        lambda r: title_age_medians.get(r['Title'], global_age_median)
        if pd.isna(r['Age']) else r['Age'],
        axis=1
    )

    # Family size features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
    df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)

    # Cabin features — "B96 B98" is the fill for missing cabin
    df['HasCabin'] = (df['Cabin'] != CABIN_FILL).astype(int)
    df['Deck'] = df.apply(
        lambda r: r['Cabin'][0] if r['Cabin'] != CABIN_FILL else 'U', axis=1
    )

    # Fare features
    df['LogFare'] = np.log1p(df['Fare'])
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])

    # Age bin categories
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 200],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    )

    # Interaction features
    df['AgeClass'] = df['Age'] * df['Pclass']
    df['LogFareClass'] = df['LogFare'] / (df['Pclass'] + 1e-6)
    df['SexMale'] = (df['Sex'] == 'male').astype(int)
    df['SexPclass'] = df['SexMale'] * df['Pclass']
    df['MasterOrMiss'] = (df['Title'].isin(['Master', 'Miss'])).astype(int)
    # Women and children first — strong survival signal
    df['WomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)
    # Age * Sex interaction: adult males far less likely to survive
    df['AgeSexMale'] = df['Age'] * df['SexMale']

    # Ticket prefix — some tickets have letter prefixes (PC, SOTON, etc.)
    df['TicketPrefix'] = (
        df['Ticket'].str.extract(r'^([A-Za-z/. ]+)', expand=False)
        .str.strip()
        .fillna('N')
    )

    # Drop raw text/ID columns
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

    return df


X_eng = engineer_features(X_raw)
print(f"Features after engineering: {X_eng.shape[1]}")
print(f"Columns: {list(X_eng.columns)}")

# Encode categoricals for LightGBM
cat_cols = X_eng.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical cols: {cat_cols}")
print(f"Numeric cols: {num_cols}")

X_final = X_eng.copy()
for col in cat_cols:
    X_final[col] = X_final[col].astype(str)
    le = LabelEncoder()
    X_final[col] = le.fit_transform(X_final[col])
    X_final[col] = X_final[col].astype('category')

print(f"Final feature matrix: {X_final.shape}")

# ================================================
# Optuna TPE HPO with LightGBM (5-fold CV)
# ================================================
class_counts = np.bincount(y)
scale_pos_weight = float(class_counts[0]) / float(class_counts[1])


def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "n_estimators": 1000,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for tr_idx, va_idx in skf.split(X_final, y):
        X_tr = X_final.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_final.iloc[va_idx]
        y_va = y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        preds = model.predict_proba(X_va)[:, 1]
        cv_scores.append(roc_auc_score(y_va, preds))

    return float(np.mean(cv_scores))


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=80, show_progress_bar=False)

best_cv_score = study.best_value
best_params = study.best_params
print(f"Best CV ROC-AUC (Optuna, 80 trials): {best_cv_score:.6f}")
print(f"Best params: {best_params}")

# ================================================
# Final OOF evaluation: multi-seed LightGBM ensemble
# Train 5 independent seeds of the best params, average OOF predictions
# to reduce variance — simple and reliable for small tabular datasets
# ================================================
base_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": best_params["num_leaves"],
    "learning_rate": best_params["learning_rate"],
    "min_child_samples": best_params["min_child_samples"],
    "feature_fraction": best_params["feature_fraction"],
    "bagging_fraction": best_params["bagging_fraction"],
    "bagging_freq": 1,
    "reg_alpha": best_params["reg_alpha"],
    "reg_lambda": best_params["reg_lambda"],
    "min_split_gain": best_params["min_split_gain"],
    "n_estimators": 2000,
    "scale_pos_weight": scale_pos_weight,
}

SEEDS = [42, 123, 456, 789, 1024]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_oof_preds = []

for seed_i, seed in enumerate(SEEDS):
    params = {**base_params, "random_state": seed}
    oof_preds_seed = np.zeros(len(y))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_final, y)):
        X_tr = X_final.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_final.iloc[va_idx]
        y_va = y[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_preds_seed[va_idx] = model.predict_proba(X_va)[:, 1]

    seed_auc = roc_auc_score(y, oof_preds_seed)
    print(f"  Seed {seed} OOF AUC: {seed_auc:.6f}")
    all_oof_preds.append(oof_preds_seed)

# Average across seeds for variance reduction
oof_preds = np.mean(all_oof_preds, axis=0)
val_roc_auc = roc_auc_score(y, oof_preds)
print(f"BEST_VAL_ROC_AUC: {val_roc_auc:.6f}")
