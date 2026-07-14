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
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

unique, counts = zip(*sorted(
    [(cls, int((y_encoded == idx).sum())) for cls, idx in class_mapping.items()],
    key=lambda x: x[1], reverse=True
))
class_dist_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in zip(unique, counts))
print(f"[DATA] Samples: {len(df)}, Classes: {len(class_mapping)}, Distribution: {{{class_dist_str}}}, Train: {len(X_train)}, Val: {len(X_val)}")


def engineer_features(df_in, train_medians=None, train_modes=None, fit=False):
    df_e = df_in.copy()

    # --- Extract title from Name ---
    if "Name" in df_e.columns:
        df_e["Title"] = df_e["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                "Major", "Rev", "Sir", "Jonkheer", "Dona"]
        df_e["Title"] = df_e["Title"].replace(rare, "Rare")
        df_e["Title"] = df_e["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        df_e = df_e.drop(columns=["Name"])

    # --- Family features ---
    if "SibSp" in df_e.columns and "Parch" in df_e.columns:
        df_e["FamilySize"] = df_e["SibSp"] + df_e["Parch"] + 1
        df_e["IsAlone"] = (df_e["FamilySize"] == 1).astype(int)

    # --- Cabin: extract deck and has-cabin flag ---
    if "Cabin" in df_e.columns:
        df_e["HasCabin"] = df_e["Cabin"].notna().astype(int)
        df_e["Deck"] = df_e["Cabin"].str[0].fillna("U")
        df_e = df_e.drop(columns=["Cabin"])

    # --- Drop high-cardinality / ID columns ---
    for col in ["PassengerId", "Ticket"]:
        if col in df_e.columns:
            df_e = df_e.drop(columns=[col])

    # --- Missingness indicators for numeric cols with >5% missing ---
    num_cols = df_e.select_dtypes(include="number").columns.tolist()
    cat_cols = df_e.select_dtypes(include=["object", "category"]).columns.tolist()

    if fit:
        train_medians = {}
        train_modes = {}

    for col in num_cols:
        miss_rate = df_e[col].isna().mean()
        if miss_rate > 0.05:
            df_e[f"{col}_missing"] = df_e[col].isna().astype(int)
        if fit:
            train_medians[col] = df_e[col].median()
        df_e[col] = df_e[col].fillna(train_medians[col])

    for col in cat_cols:
        if fit:
            train_modes[col] = df_e[col].mode()[0] if not df_e[col].mode().empty else "Unknown"
        df_e[col] = df_e[col].fillna(train_modes[col])

    return df_e, train_medians, train_modes


# Engineer features (fit on train, transform both)
X_train, medians, modes = engineer_features(X_train, fit=True)
X_val, _, _ = engineer_features(X_val, train_medians=medians, train_modes=modes, fit=False)

# Label-encode all categorical columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le_col = LabelEncoder()
    X_train[col] = le_col.fit_transform(X_train[col].astype(str))
    known = set(le_col.classes_)
    X_val[col] = X_val[col].astype(str).apply(
        lambda x: le_col.transform([x])[0] if x in known else -1
    )
    label_encoders[col] = le_col

print(f"Features after engineering: {list(X_train.columns)}")

# Convert to numpy
X_train_arr = X_train.values.astype(np.float32)
X_val_arr = X_val.values.astype(np.float32)
y_train_arr = np.array(y_train)
y_val_arr = np.array(y_val)

# LightGBM hyperparameters
lgb_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=1000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
    random_state=42,
)

# 5-fold stratified CV on training set; ensemble predictions on val set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_preds = np.zeros(len(X_val_arr))
cv_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
    X_tr, X_va = X_train_arr[tr_idx], X_train_arr[va_idx]
    y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    fold_preds_val = model.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, fold_preds_val)
    cv_aucs.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}, best iter: {model.best_iteration_}")

    val_preds += model.predict_proba(X_val_arr)[:, 1] / 5

print(f"CV AUC mean: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

best_val_roc_auc = roc_auc_score(y_val_arr, val_preds)
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
