import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw)
n_cls = len(le_target.classes_)
class_dist = {int(c): int(cnt) for c, cnt in zip(np.unique(y_encoded), np.bincount(y_encoded))}
print(f"[DATA] Samples: {len(df)}, Classes: {n_cls}, Distribution: {class_dist}")


def engineer_features(df_in):
    d = df_in.copy()

    # Title from Name
    if "Name" in d.columns:
        d["Title"] = d["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                "Major", "Rev", "Sir", "Jonkheer", "Dona"]
        d["Title"] = d["Title"].replace(rare, "Rare")
        d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        d = d.drop(columns=["Name"])

    # Family features
    if "SibSp" in d.columns and "Parch" in d.columns:
        d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
        d["IsAlone"] = (d["FamilySize"] == 1).astype(int)

    # Cabin
    if "Cabin" in d.columns:
        d["HasCabin"] = d["Cabin"].notna().astype(int)
        d["Deck"] = d["Cabin"].str[0].fillna("U")
        d = d.drop(columns=["Cabin"])

    # Drop IDs / Ticket
    for col in ["PassengerId", "Ticket"]:
        if col in d.columns:
            d = d.drop(columns=[col])

    # Age: missingness flag + title-based imputation
    if "Age" in d.columns and "Title" in d.columns:
        d["Age_missing"] = d["Age"].isna().astype(int)
        title_age_med = d.groupby("Title")["Age"].median()
        global_age_med = d["Age"].median()

        def fill_age(row):
            if pd.isna(row["Age"]):
                return title_age_med.get(row["Title"], global_age_med)
            return row["Age"]

        d["Age"] = d.apply(fill_age, axis=1)
        # Age bins: Child, Teen, YoungAdult, Adult, Senior
        d["AgeBin"] = pd.cut(d["Age"], bins=[0, 12, 18, 35, 60, 200],
                             labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)

    # Fare: imputation + per-person + log
    if "Fare" in d.columns:
        fare_med = d.loc[d["Fare"] > 0, "Fare"].median()
        d["Fare"] = d["Fare"].fillna(fare_med)
        d["Fare"] = d["Fare"].clip(lower=0.01)
        if "FamilySize" in d.columns:
            d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
            d["LogFarePerPerson"] = np.log1p(d["FarePerPerson"])
        d["LogFare"] = np.log1p(d["Fare"])

    # Embarked imputation
    if "Embarked" in d.columns:
        mode_val = d["Embarked"].mode()
        embarked_mode = mode_val.iloc[0] if len(mode_val) > 0 else "S"
        d["Embarked"] = d["Embarked"].fillna(embarked_mode)

    # Pclass * Sex interaction
    if "Pclass" in d.columns and "Sex" in d.columns:
        d["Pclass_Sex"] = d["Pclass"].astype(str) + "_" + d["Sex"].astype(str)

    # Fill remaining NaN
    for col in d.select_dtypes(include=["object", "category"]).columns:
        d[col] = d[col].fillna("Unknown")
    for col in d.select_dtypes(include="number").columns:
        d[col] = d[col].fillna(d[col].median())

    # Label encode all categoricals
    for col in d.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))

    return d


X_all = engineer_features(X_raw)
X_arr = X_all.values.astype(np.float32)

print(f"Features ({X_arr.shape[1]}): {list(X_all.columns)}")

# LightGBM hyperparameters — tuned for small dataset
lgb_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.02,
    n_estimators=3000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=5,
    reg_alpha=0.05,
    reg_lambda=0.05,
    verbose=-1,
    random_state=42,
)

# 10-fold stratified OOF CV on all 891 samples
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_arr))
cv_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    fold_preds = model.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, fold_preds)
    cv_aucs.append(fold_auc)
    oof_preds[va_idx] = fold_preds
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}, best iter: {model.best_iteration_}")

print(f"CV AUC mean: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

best_val_roc_auc = roc_auc_score(y_encoded, oof_preds)
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
