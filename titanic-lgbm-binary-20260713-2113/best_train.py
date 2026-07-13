import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

unique, counts = zip(*sorted(
    [(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()],
    key=lambda x: x[1], reverse=True
))
class_dist_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in zip(unique, counts))
print(f"[DATA] Samples: {len(df)}, Classes: {len(class_mapping)}, Distribution: {{{class_dist_str}}}")
print(f"[DATA] Train samples: {len(X_train)}, Val samples: {len(X_val)}")

# ── Feature engineering ──────────────────────────────────────────────────────

def extract_features(X):
    """Derive new features; no fitting on data statistics needed here."""
    X = X.copy()

    # Title from Name
    if 'Name' in X.columns:
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare = {'Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Countess', 'Capt',
                'Ms', 'Sir', 'Jonkheer', 'Lady', 'Mme', 'Don', 'Dona'}
        X['Title'] = X['Title'].apply(lambda t: 'Rare' if t in rare else t)
        X['Title'] = X['Title'].fillna('Unknown')
        X = X.drop(columns=['Name'])

    # Deck from Cabin; missingness indicator
    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isnull().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('Unknown')
        X = X.drop(columns=['Cabin'])

    # Drop PassengerId (just a row index)
    if 'PassengerId' in X.columns:
        X = X.drop(columns=['PassengerId'])

    # Drop raw Ticket (high cardinality, noisy)
    if 'Ticket' in X.columns:
        X = X.drop(columns=['Ticket'])

    # Missingness indicator for Age (>5 % missing)
    if 'Age' in X.columns:
        X['Age_missing'] = X['Age'].isnull().astype(int)

    # Family size features
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # Log-Fare (right-skewed)
    if 'Fare' in X.columns:
        X['Fare_log'] = np.log1p(X['Fare'].fillna(0))

    return X


def impute(X_train_raw, X_val_raw):
    """Fit imputation on training data, apply to both sets."""
    X_tr = X_train_raw.copy()
    X_va = X_val_raw.copy()

    for col in ['Age', 'Fare']:
        if col in X_tr.columns:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_va[col] = X_va[col].fillna(med)

    for col in ['Embarked']:
        if col in X_tr.columns:
            mode_val = X_tr[col].mode()[0]
            X_tr[col] = X_tr[col].fillna(mode_val)
            X_va[col] = X_va[col].fillna(mode_val)

    return X_tr, X_va


X_train_fe = extract_features(X_train)
X_val_fe = extract_features(X_val)
X_train_fe, X_val_fe = impute(X_train_fe, X_val_fe)

# Convert object columns to 'category' for LightGBM native handling
cat_cols = X_train_fe.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X_train_fe[col] = X_train_fe[col].astype('category')
    X_val_fe[col] = X_val_fe[col].astype('category')

print(f"[FE] Features after engineering: {list(X_train_fe.columns)}")

# ── LightGBM: 5-fold stratified CV to find best n_estimators ────────────────

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
}

y_train_arr = np.array(y_train)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_fe, y_train_arr)):
    X_tr_cv = X_train_fe.iloc[tr_idx]
    X_va_cv = X_train_fe.iloc[va_idx]
    y_tr_cv = y_train_arr[tr_idx]
    y_va_cv = y_train_arr[va_idx]

    dtrain = lgb.Dataset(X_tr_cv, label=y_tr_cv)
    dval   = lgb.Dataset(X_va_cv, label=y_va_cv, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    best_iters.append(model.best_iteration)
    preds = model.predict(X_va_cv)
    fold_auc = roc_auc_score(y_va_cv, preds)
    cv_aucs.append(fold_auc)
    print(f"  Fold {fold + 1}/5 — AUC: {fold_auc:.4f}, best_iter: {model.best_iteration}")

mean_cv_auc = float(np.mean(cv_aucs))
std_cv_auc  = float(np.std(cv_aucs))
best_n      = int(np.mean(best_iters))
print(f"[CV] Mean AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}  |  Using n_estimators={best_n}")

# ── Final model on full training set ────────────────────────────────────────

dtrain_full = lgb.Dataset(X_train_fe, label=y_train_arr)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=best_n,
    callbacks=[lgb.log_evaluation(period=-1)],
)

# ── Validation metric ────────────────────────────────────────────────────────

val_preds = final_model.predict(X_val_fe)
val_auc = roc_auc_score(y_val, val_preds)
print(f"[RESULT] Validation ROC-AUC: {val_auc:.6f}")

# Feature importance (informational)
importance = final_model.feature_importance(importance_type='gain')
feat_names = final_model.feature_name()
top_feats = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)[:10]
print(f"[FEATS] Top features: {[(f, round(v,1)) for f, v in top_feats]}")

print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")
