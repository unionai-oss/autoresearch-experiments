import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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

    # Title from Name + extract last name + name length
    if "Name" in d.columns:
        d["Title"] = d["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                "Major", "Rev", "Sir", "Jonkheer", "Dona"]
        d["Title"] = d["Title"].replace(rare, "Rare")
        d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        d["NameLength"] = d["Name"].str.len()
        d["LastName"] = d["Name"].str.split(",").str[0].str.strip()
        d = d.drop(columns=["Name"])

    # Family features
    if "SibSp" in d.columns and "Parch" in d.columns:
        d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
        d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
        # Family size groups: alone(1), small(2-4), large(5+)
        d["FamilySizeGroup"] = pd.cut(d["FamilySize"], bins=[0, 1, 4, 20],
                                      labels=[0, 1, 2]).astype(float)

    # Family ID (last name + family size — collapses singletons)
    if "LastName" in d.columns and "FamilySize" in d.columns:
        d["FamilyID"] = d["LastName"] + "_" + d["FamilySize"].astype(str)
        family_counts = d["FamilyID"].value_counts()
        d["FamilyID"] = d["FamilyID"].map(
            lambda x: x if family_counts.get(x, 0) >= 2 else "Small"
        )
        d = d.drop(columns=["LastName"])

    # Cabin
    if "Cabin" in d.columns:
        d["HasCabin"] = d["Cabin"].notna().astype(int)
        d["Deck"] = d["Cabin"].str[0].fillna("U")
        d = d.drop(columns=["Cabin"])

    # Ticket: prefix + group size
    if "Ticket" in d.columns:
        # Group size = how many passengers share the same ticket
        d["TicketGroupSize"] = d["Ticket"].map(d["Ticket"].value_counts())
        # Extract alphabetic prefix; pure-number tickets → "NUM"
        cleaned = d["Ticket"].str.upper().str.replace(r'[\./\s]', '', regex=True)
        prefix = cleaned.str.extract(r'^([A-Z]+)', expand=False)
        d["TicketPrefix"] = prefix.fillna("NUM")
        # Compress rare prefixes (< 5 occurrences)
        prefix_counts = d["TicketPrefix"].value_counts()
        d["TicketPrefix"] = d["TicketPrefix"].map(
            lambda x: x if prefix_counts.get(x, 0) >= 5 else "RARE"
        )
        d = d.drop(columns=["Ticket"])

    # Drop IDs
    for col in ["PassengerId"]:
        if col in d.columns:
            d = d.drop(columns=[col])

    # Age: missingness flag + title-based imputation + bin
    if "Age" in d.columns and "Title" in d.columns:
        d["Age_missing"] = d["Age"].isna().astype(int)
        title_age_med = d.groupby("Title")["Age"].median()
        global_age_med = d["Age"].median()

        def fill_age(row):
            if pd.isna(row["Age"]):
                return title_age_med.get(row["Title"], global_age_med)
            return row["Age"]

        d["Age"] = d.apply(fill_age, axis=1)
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

    # Pclass × Sex interaction
    if "Pclass" in d.columns and "Sex" in d.columns:
        d["Pclass_Sex"] = d["Pclass"].astype(str) + "_" + d["Sex"].astype(str)

    # Binary sex for numeric interactions
    if "Sex" in d.columns:
        d["Sex_bin"] = (d["Sex"] == "male").astype(int)

    # Age × Pclass
    if "Age" in d.columns and "Pclass" in d.columns:
        d["Age_Pclass"] = d["Age"] * d["Pclass"]

    # Male × LogFare (female fare is much more predictive of survival)
    if "Sex_bin" in d.columns and "LogFare" in d.columns:
        d["Male_LogFare"] = d["Sex_bin"] * d["LogFare"]

    # Pclass × LogFarePerPerson
    if "Pclass" in d.columns and "LogFarePerPerson" in d.columns:
        d["Pclass_LogFPP"] = d["Pclass"] * d["LogFarePerPerson"]

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

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ── LightGBM ────────────────────────────────────────────────────────────────
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

oof_lgb = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[va_idx] = model.predict_proba(X_va)[:, 1]

auc_lgb = roc_auc_score(y_encoded, oof_lgb)
print(f"LGB OOF AUC: {auc_lgb:.4f}")

# ── XGBoost ─────────────────────────────────────────────────────────────────
oof_xgb = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[va_idx] = model.predict_proba(X_va)[:, 1]

auc_xgb = roc_auc_score(y_encoded, oof_xgb)
print(f"XGB OOF AUC: {auc_xgb:.4f}")

# ── CatBoost ─────────────────────────────────────────────────────────────────
oof_cat = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        train_dir='/tmp/catboost_info',
        early_stopping_rounds=100,
        eval_metric='AUC',
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[va_idx] = model.predict_proba(X_va)[:, 1]

auc_cat = roc_auc_score(y_encoded, oof_cat)
print(f"CAT OOF AUC: {auc_cat:.4f}")

# ── Ensemble (simple average) ────────────────────────────────────────────────
oof_blend = (oof_lgb + oof_xgb + oof_cat) / 3.0
best_val_roc_auc = roc_auc_score(y_encoded, oof_blend)
print(f"Ensemble OOF AUC: {best_val_roc_auc:.4f}")
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")