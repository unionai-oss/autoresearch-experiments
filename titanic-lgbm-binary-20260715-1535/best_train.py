import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

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

    # ===== Ticket features =====
    if 'Ticket' in X.columns:
        ticket_counts = X['Ticket'].value_counts()
        X['Ticket_freq'] = X['Ticket'].map(ticket_counts).fillna(1).astype(int)
        X['Ticket_prefix'] = X['Ticket'].str.extract(r'^([A-Za-z/\.]+)')[0].fillna('NUM')
        X = X.drop(columns=['Ticket'])

    # ===== Name -> Title =====
    if 'Name' in X.columns:
        title = X['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
        title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        rare_titles = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}
        X['Title'] = title.apply(lambda t: 'Rare' if t in rare_titles else t)
        X = X.drop(columns=['Name'])

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
        # Fare percentile rank within Pclass (no target leakage — pure X info)
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

        # Women AND children first — the famous Titanic survival rule
        if 'IsChild' in X.columns:
            X['WomenOrChild'] = ((sex_is_female == 1) | (X['IsChild'] == 1)).astype(int)
            # Priority: WomenOrChild in high class survives most
            X['WomanChild_Pclass'] = X['WomenOrChild'] * 4 - X['Pclass']
            X['WomenOrChild_x_Class1'] = ((X['WomenOrChild'] == 1) & (X['Pclass'] == 1)).astype(int)

    if 'Age' in X.columns and 'Pclass' in X.columns:
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
print(f"Features: {list(X_all.columns)}")
print(f"Categorical columns: {cat_col_names}")

# Get categorical column indices for CatBoost
cat_col_indices = [list(X_all.columns).index(c) for c in cat_col_names if c in X_all.columns]

X_np = X_all.values
y_np = y_all

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# =============================================================
# Model 1: LightGBM
# =============================================================
lgb_params = dict(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    num_leaves=63,
    learning_rate=0.03,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=5,
    reg_alpha=0.05,
    reg_lambda=0.5,
    n_estimators=3000,
    random_state=42,
    verbose=-1,
)

oof_lgb = np.zeros(len(X_np))
lgb_iters = []
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_np, y_np)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_np[tr_idx], y_np[tr_idx],
        eval_set=[(X_np[vl_idx], y_np[vl_idx])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[vl_idx] = model.predict_proba(X_np[vl_idx])[:, 1]
    lgb_iters.append(model.best_iteration_)
    fold_auc = roc_auc_score(y_np[vl_idx], oof_lgb[vl_idx])
    print(f"  LGB Fold {fold+1}/{N_FOLDS}: AUC={fold_auc:.6f}, best_iter={model.best_iteration_}")

lgb_auc = roc_auc_score(y_np, oof_lgb)
print(f"LightGBM OOF AUC: {lgb_auc:.6f} (avg best_iter={np.mean(lgb_iters):.0f})")

# =============================================================
# Model 2: XGBoost
# =============================================================
oof_xgb = np.zeros(len(X_np))
xgb_iters = []
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_np, y_np)):
    model = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=0.5,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
    )
    model.fit(
        X_np[tr_idx], y_np[tr_idx],
        eval_set=[(X_np[vl_idx], y_np[vl_idx])],
        verbose=False,
    )
    oof_xgb[vl_idx] = model.predict_proba(X_np[vl_idx])[:, 1]
    xgb_iters.append(model.best_iteration)
    fold_auc = roc_auc_score(y_np[vl_idx], oof_xgb[vl_idx])
    print(f"  XGB Fold {fold+1}/{N_FOLDS}: AUC={fold_auc:.6f}, best_iter={model.best_iteration}")

xgb_auc = roc_auc_score(y_np, oof_xgb)
print(f"XGBoost OOF AUC: {xgb_auc:.6f} (avg best_iter={np.mean(xgb_iters):.0f})")

# =============================================================
# Model 3: CatBoost (with native categorical handling via Pool)
# =============================================================
oof_cat = np.zeros(len(X_np))
cat_iters = []
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_np, y_np)):
    # Use DataFrame-based Pool so CatBoost can handle cat_features by name
    # even after label encoding (integer-valued categoricals)
    X_tr_df = X_all.iloc[tr_idx].copy()
    X_vl_df = X_all.iloc[vl_idx].copy()

    # Convert cat columns back to string so CatBoost treats them as categorical
    for col in cat_col_names:
        if col in X_tr_df.columns:
            X_tr_df[col] = X_tr_df[col].astype(str)
            X_vl_df[col] = X_vl_df[col].astype(str)

    train_pool = Pool(X_tr_df, y_np[tr_idx], cat_features=cat_col_names)
    eval_pool = Pool(X_vl_df, y_np[vl_idx], cat_features=cat_col_names)

    model = CatBoostClassifier(
        iterations=3000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        train_dir='/tmp/catboost_info',
        early_stopping_rounds=100,
    )
    model.fit(train_pool, eval_set=eval_pool)

    oof_cat[vl_idx] = model.predict_proba(eval_pool)[:, 1]
    best_it = model.best_iteration_
    cat_iters.append(best_it)
    fold_auc = roc_auc_score(y_np[vl_idx], oof_cat[vl_idx])
    print(f"  CAT Fold {fold+1}/{N_FOLDS}: AUC={fold_auc:.6f}, best_iter={best_it}")

cat_auc = roc_auc_score(y_np, oof_cat)
print(f"CatBoost OOF AUC: {cat_auc:.6f} (avg best_iter={np.mean(cat_iters):.0f})")

# =============================================================
# Ensemble: simple average of OOF predictions
# =============================================================
oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3.0
ensemble_auc = roc_auc_score(y_np, oof_ensemble)

print(f"\nIndividual OOF AUCs: LGB={lgb_auc:.6f}, XGB={xgb_auc:.6f}, CAT={cat_auc:.6f}")
print(f"Ensemble OOF AUC (LGB+XGB+CAT avg): {ensemble_auc:.6f}")
print(f"BEST_VAL_ROC_AUC: {ensemble_auc:.6f}")