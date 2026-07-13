import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
# CatBoost and optuna are installed in user site-packages; add to path for the venv Python
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col].values

print(f"[DATA] Samples: {len(df)}, Classes: {np.unique(y)}, "
      f"Distribution: {{0: {(y==0).sum()}, 1: {(y==1).sum()}}}")

# ── Feature engineering ──────────────────────────────────────────────────────

def extract_features(X):
    """Feature engineering — core set + carefully selected additions."""
    X = X.copy()

    # Title from Name
    if 'Name' in X.columns:
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare = {'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Capt',
                'Ms', 'Sir', 'Jonkheer', 'Lady', 'Mme', 'Don', 'Dona'}
        X['Title'] = X['Title'].apply(lambda t: 'Rare' if t in rare else t)
        X['Title'] = X['Title'].fillna('Unknown')
        X = X.drop(columns=['Name'])

    # Deck from Cabin + missingness indicator
    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isnull().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('Unknown')
        X = X.drop(columns=['Cabin'])

    # Drop PassengerId
    if 'PassengerId' in X.columns:
        X = X.drop(columns=['PassengerId'])

    # Ticket group size: number of passengers sharing same ticket
    # (correlates with travelling group — groups had coordination advantage)
    if 'Ticket' in X.columns:
        ticket_counts = X['Ticket'].map(X['Ticket'].value_counts())
        X['TicketGroup'] = ticket_counts.fillna(1).astype(int)
        X = X.drop(columns=['Ticket'])

    # Age missingness indicator
    if 'Age' in X.columns:
        X['Age_missing'] = X['Age'].isnull().astype(int)

    # Family size features
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # Log-Fare
    if 'Fare' in X.columns:
        X['Fare_log'] = np.log1p(X['Fare'].fillna(0))
        # Fare per person in ticket group (corrects for shared ticket fares)
        if 'TicketGroup' in X.columns:
            X['Fare_per_person'] = X['Fare'].fillna(0) / X['TicketGroup'].clip(lower=1)
            X['Fare_per_person_log'] = np.log1p(X['Fare_per_person'])

    # Sex × Pclass interaction (most discriminative combination for Titanic)
    if 'Sex' in X.columns and 'Pclass' in X.columns:
        X['Sex_Pclass'] = X['Sex'].astype(str) + '_' + X['Pclass'].astype(str)

    return X


X_fe = extract_features(X)

# ── Imputation (on full data before OOF split) ───────────────────────────────
for col in ['Age', 'Fare']:
    if col in X_fe.columns:
        med = X_fe[col].median()
        X_fe[col] = X_fe[col].fillna(med)

for col in ['Embarked']:
    if col in X_fe.columns:
        mode_val = X_fe[col].mode()[0]
        X_fe[col] = X_fe[col].fillna(mode_val)

# WomanOrChild: "women and children first" — the actual rescue priority rule
# Computed after Age imputation so no NaN
if 'Age' in X_fe.columns and 'Sex' in X_fe.columns:
    X_fe['WomanOrChild'] = ((X_fe['Sex'] == 'female') | (X_fe['Age'] < 15)).astype(int)

# Ensure all categorical columns are strings (CatBoost requirement)
cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass']
cat_cols = [c for c in cat_cols if c in X_fe.columns]
for col in cat_cols:
    X_fe[col] = X_fe[col].astype(str).fillna('Unknown')

print(f"[FE] Features ({len(X_fe.columns)}): {list(X_fe.columns)}")
print(f"[FE] Categorical: {cat_cols}")

# ── Optuna hyperparameter search ─────────────────────────────────────────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = dict(
        iterations=3000,
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        depth=trial.suggest_int('depth', 4, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        random_strength=trial.suggest_float('random_strength', 0.1, 3.0),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 2.0),
        border_count=trial.suggest_int('border_count', 32, 128),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 1, 20),
        cat_features=cat_cols,
        eval_metric='AUC',
        early_stopping_rounds=150,
        use_best_model=True,
        random_seed=42,
        verbose=False,
        train_dir='/tmp/catboost_info',
    )

    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X_fe, y):
        X_tr = X_fe.iloc[tr_idx].reset_index(drop=True)
        X_va = X_fe.iloc[va_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        m = CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
    return roc_auc_score(y, oof)

print("[HPO] Running Optuna search (50 trials) ...")
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
print(f"[HPO] Best trial AUC: {study.best_value:.4f}")
print(f"[HPO] Best params: {best_params}")

# ── Final evaluation with best params ────────────────────────────────────────

final_params = dict(
    iterations=3000,
    cat_features=cat_cols,
    eval_metric='AUC',
    early_stopping_rounds=150,
    use_best_model=True,
    random_seed=42,
    verbose=False,
    train_dir='/tmp/catboost_info',
    **best_params,
)

oof_preds = np.zeros(len(y))
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y)):
    X_tr = X_fe.iloc[tr_idx].reset_index(drop=True)
    X_va = X_fe.iloc[va_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    model = CatBoostClassifier(random_seed=42 + fold, **{k: v for k, v in final_params.items()
                                                          if k != 'random_seed'})
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    oof_preds[va_idx] = model.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold+1}/5 — AUC: {fold_auc:.4f}, best_iter: {model.best_iteration_}")

oof_auc = roc_auc_score(y, oof_preds)
print(f"[CV] Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"[RESULT] OOF ROC-AUC (all {len(y)} samples): {oof_auc:.6f}")

print(f"BEST_VAL_ROC_AUC: {oof_auc:.6f}")
