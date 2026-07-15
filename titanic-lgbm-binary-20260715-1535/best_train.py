import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# --- Data loading (skeleton) ---
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

unique, counts = zip(*[(cls, (y_encoded == idx).sum()) for cls, idx in class_mapping.items()])
class_dist = {str(cls): int((y_encoded == le.transform([cls])[0]).sum()) for cls in le.classes_}
print(f"Dataset summary: total_samples={len(df)}, train_samples={len(X_train)}, val_samples={len(X_val)}, class_distribution={class_dist}")

# --- Feature Engineering ---

def compute_train_stats(X):
    """Compute imputation statistics from training data only."""
    stats = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        stats[f'{col}_median'] = X[col].median()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        if X[col].notna().any():
            stats[f'{col}_mode'] = X[col].mode()[0]
        else:
            stats[f'{col}_mode'] = 'Unknown'
    return stats


def engineer_features(X, stats, label_encoders=None):
    """
    Engineer features. Returns (X_engineered, fitted_label_encoders).
    If label_encoders is None: fits new encoders on X (training mode).
    If label_encoders is provided: transforms X using existing encoders (inference mode).
    """
    X = X.copy()

    # Drop pure identifier
    X = X.drop(columns=['PassengerId'], errors='ignore')

    # Name -> Title
    if 'Name' in X.columns:
        title = X['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
        title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        rare_titles = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}
        X['Title'] = title.apply(lambda t: 'Rare' if t in rare_titles else t)
        X = X.drop(columns=['Name'])

    # Cabin -> missingness indicator + deck letter
    if 'Cabin' in X.columns:
        X['Cabin_missing'] = X['Cabin'].isna().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('U')
        X = X.drop(columns=['Cabin'])

    # Age -> missingness indicator + median impute from train stats
    if 'Age' in X.columns:
        X['Age_missing'] = X['Age'].isna().astype(int)
        X['Age'] = X['Age'].fillna(stats.get('Age_median', 28.0))

    # Fare -> fill missing + log transform (right-skewed distribution)
    if 'Fare' in X.columns:
        fare_fill = stats.get('Fare_median', 14.45)
        X['Fare'] = np.log1p(X['Fare'].fillna(fare_fill))

    # Ticket -> extract alphabetic prefix (if any)
    if 'Ticket' in X.columns:
        X['Ticket_prefix'] = X['Ticket'].str.extract(r'^([A-Za-z/\.]+)')[0].fillna('NUM')
        X = X.drop(columns=['Ticket'])

    # Embarked -> fill missing with train mode
    if 'Embarked' in X.columns:
        emb_fill = stats.get('Embarked_mode', 'S')
        X['Embarked'] = X['Embarked'].fillna(emb_fill)

    # Family-size derived features
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # Fill any remaining numeric NaN with train-computed median (safety net)
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(stats.get(f'{col}_median', 0))

    # Label-encode all object/category columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    fitted_encoders = {} if label_encoders is None else dict(label_encoders)

    for col in cat_cols:
        X[col] = X[col].astype(str)
        if label_encoders is None:
            # Fit mode: learn encoder from training data
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col])
            fitted_encoders[col] = enc
        else:
            # Inference mode: map unseen categories to first known class
            enc = fitted_encoders.get(col)
            if enc is None:
                X[col] = 0
            else:
                known = set(enc.classes_)
                X[col] = X[col].apply(lambda v: enc.classes_[0] if v not in known else v)
                X[col] = enc.transform(X[col])

    return X, fitted_encoders


# Compute statistics from raw training data (before any transformation)
train_stats = compute_train_stats(X_train)

# Apply feature engineering (fit on train, transform both)
X_train_fe, label_encoders = engineer_features(X_train, train_stats, label_encoders=None)
X_val_fe, _ = engineer_features(X_val, train_stats, label_encoders=label_encoders)

# Ensure identical column sets and order
X_val_fe = X_val_fe.reindex(columns=X_train_fe.columns, fill_value=0)

# Reset indices for clean positional (iloc) splits inside CV
X_train_fe = X_train_fe.reset_index(drop=True)
X_val_fe = X_val_fe.reset_index(drop=True)
y_train_arr = np.asarray(y_train)
y_val_arr = np.asarray(y_val)

print(f"Engineered feature shape: train={X_train_fe.shape}, val={X_val_fe.shape}")
print(f"Features: {list(X_train_fe.columns)}")

# --- LightGBM baseline with 5-fold stratified CV ---
lgb_params = dict(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_fe))
val_preds = np.zeros(len(X_val_fe))
cv_scores = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_train_fe, y_train_arr)):
    X_f_tr = X_train_fe.iloc[tr_idx]
    y_f_tr = y_train_arr[tr_idx]
    X_f_vl = X_train_fe.iloc[vl_idx]
    y_f_vl = y_train_arr[vl_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_f_tr, y_f_tr,
        eval_set=[(X_f_vl, y_f_vl)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[vl_idx] = model.predict_proba(X_f_vl)[:, 1]
    val_preds += model.predict_proba(X_val_fe)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_f_vl, oof_preds[vl_idx])
    cv_scores.append(fold_auc)
    print(f"  Fold {fold + 1}/{N_FOLDS}: AUC={fold_auc:.6f}, best_iter={model.best_iteration_}")

oof_auc = roc_auc_score(y_train_arr, oof_preds)
val_auc = roc_auc_score(y_val_arr, val_preds)

print(f"OOF AUC: {oof_auc:.6f}")
print(f"CV mean ± std: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"Val AUC (avg of {N_FOLDS} fold models): {val_auc:.6f}")
print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")
