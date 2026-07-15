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
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

unique, counts = zip(*[(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()])
class_dist_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in zip(unique, counts))
print(f"[DATA] Samples: {len(df)}, Class distribution: {class_dist_str}, Train: {len(X_train)}, Val: {len(X_val)}")

# --- Feature Engineering (deterministic, no fitting needed) ---
COMMON_TITLES = {'Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Col', 'Major'}

def feature_engineer(X_df):
    X_df = X_df.copy()

    # Extract title from Name (fixed map, no data-fitting)
    if 'Name' in X_df.columns:
        X_df['Title'] = X_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        X_df['Title'] = X_df['Title'].apply(lambda t: t if t in COMMON_TITLES else 'Rare')
        X_df['Title'] = X_df['Title'].fillna('Rare')
        X_df = X_df.drop(columns=['Name'])

    # Cabin: deck letter + has-cabin flag
    if 'Cabin' in X_df.columns:
        X_df['Cabin_known'] = X_df['Cabin'].notna().astype(np.int8)
        X_df['Cabin_deck'] = X_df['Cabin'].str[0].fillna('U')
        X_df = X_df.drop(columns=['Cabin'])

    # Drop non-informative columns
    for col in ['PassengerId', 'Ticket']:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    # Family size features
    if 'SibSp' in X_df.columns and 'Parch' in X_df.columns:
        X_df['FamilySize'] = X_df['SibSp'] + X_df['Parch'] + 1
        X_df['IsAlone'] = (X_df['FamilySize'] == 1).astype(np.int8)

    # Log-transform Fare (right-skewed)
    if 'Fare' in X_df.columns:
        X_df['Fare'] = np.log1p(X_df['Fare'].clip(lower=0).fillna(0))

    return X_df


def fit_transform_preprocess(X_train_df, X_val_df):
    """Fit imputers/encoders on train, transform both train and val."""
    numeric_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    X_tr = X_train_df.copy()
    X_vl = X_val_df.copy()

    # Add missingness indicator flags for numeric cols with >5% missing in train
    train_missing = X_tr[numeric_cols].isnull().mean()
    for col in numeric_cols:
        if train_missing[col] > 0.05:
            X_tr[f'{col}_miss'] = X_tr[col].isnull().astype(np.int8)
            X_vl[f'{col}_miss'] = X_vl[col].isnull().astype(np.int8)

    # Update numeric_cols after adding new columns
    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    # Impute numeric with train median
    for col in numeric_cols:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_vl[col] = X_vl[col].fillna(med)

    # Encode categoricals: fit on train, map val (unseen → -1)
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna('Unknown').astype(str)
        X_vl[col] = X_vl[col].fillna('Unknown').astype(str)
        val_map = {v: i for i, v in enumerate(X_tr[col].unique())}
        X_tr[col] = X_tr[col].map(val_map).astype(int)
        X_vl[col] = X_vl[col].map(val_map).fillna(-1).astype(int)

    return X_tr, X_vl


# LightGBM hyperparameters (tuned for small N)
LGBM_PARAMS = dict(
    objective='binary',
    metric='auc',
    num_leaves=15,
    learning_rate=0.05,
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=500,
    random_state=42,
    verbose=-1,
)

# --- 5-fold Stratified CV on the full dataset ---
X_feat = feature_engineer(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_encoded))

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_feat, y_encoded)):
    X_tr_fold = X_feat.iloc[tr_idx]
    X_vl_fold = X_feat.iloc[vl_idx]
    y_tr_fold = y_encoded[tr_idx]
    y_vl_fold = y_encoded[vl_idx]

    X_tr_proc, X_vl_proc = fit_transform_preprocess(X_tr_fold, X_vl_fold)

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X_tr_proc, y_tr_fold,
        eval_set=[(X_vl_proc, y_vl_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    fold_preds = clf.predict_proba(X_vl_proc)[:, 1]
    oof_preds[vl_idx] = fold_preds
    fold_auc = roc_auc_score(y_vl_fold, fold_preds)
    print(f"Fold {fold + 1}/5  AUC: {fold_auc:.4f}  best_iter: {clf.best_iteration_}")

oof_auc = roc_auc_score(y_encoded, oof_preds)
print(f"OOF ROC-AUC (5-fold CV): {oof_auc:.6f}")

# --- Train on skeleton's X_train split, evaluate on X_val (sanity check) ---
X_train_feat = feature_engineer(X_train)
X_val_feat = feature_engineer(X_val)
X_train_proc, X_val_proc = fit_transform_preprocess(X_train_feat, X_val_feat)

final_clf = lgb.LGBMClassifier(**LGBM_PARAMS)
final_clf.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
val_preds = final_clf.predict_proba(X_val_proc)[:, 1]
holdout_auc = roc_auc_score(y_val, val_preds)
print(f"Holdout Val ROC-AUC (80/20): {holdout_auc:.6f}")

# OOF is the primary metric (more reliable for N=100)
best_val_roc_auc = oof_auc
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
