import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
sys.path.insert(0, '/home/flyte/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        # Extract cabin number (position along ship = survival advantage)
        d["CabinNum"] = d["Cabin"].str.extract(r'(\d+)', expand=False).fillna(-1).astype(float)
        d = d.drop(columns=["Cabin"])

    # Ticket: prefix + group size + keep full TicketID for group survival encoding
    if "Ticket" in d.columns:
        d["TicketGroupSize"] = d["Ticket"].map(d["Ticket"].value_counts())
        cleaned = d["Ticket"].str.upper().str.replace(r'[\./\s]', '', regex=True)
        prefix = cleaned.str.extract(r'^([A-Z]+)', expand=False)
        d["TicketPrefix"] = prefix.fillna("NUM")
        prefix_counts = d["TicketPrefix"].value_counts()
        d["TicketPrefix"] = d["TicketPrefix"].map(
            lambda x: x if prefix_counts.get(x, 0) >= 5 else "RARE"
        )
        # Keep full ticket number for OOF group survival encoding (ticket-mates' survival)
        d["TicketID"] = d["Ticket"].astype(str)
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

    # Fare: imputation + per-person + log + rank within Pclass
    if "Fare" in d.columns:
        fare_med = d.loc[d["Fare"] > 0, "Fare"].median()
        d["Fare"] = d["Fare"].fillna(fare_med)
        d["Fare"] = d["Fare"].clip(lower=0.01)
        if "FamilySize" in d.columns:
            d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
            d["LogFarePerPerson"] = np.log1p(d["FarePerPerson"])
        d["LogFare"] = np.log1p(d["Fare"])
        # Percentile rank of fare within Pclass — captures relative wealth within class
        if "Pclass" in d.columns:
            d["FareRank_Pclass"] = d.groupby("Pclass")["Fare"].rank(pct=True)

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

    # ── "Women and children first" semantic features ───────────────────────────
    if "Age" in d.columns and "Sex" in d.columns:
        d["IsChild"] = (d["Age"] < 15).astype(int)
        d["IsWomanOrChild"] = ((d["Sex"] == "female") | (d["Age"] < 15)).astype(int)
        d["IsAdultMale"] = ((d["Sex"] == "male") & (d["Age"] >= 18)).astype(int)

    if "Sex" in d.columns and "Parch" in d.columns and "Age" in d.columns:
        d["IsMother"] = ((d["Sex"] == "female") &
                         (d["Parch"] > 0) &
                         (d["Age"] > 18)).astype(int)

    # Age squared (nonlinear effect near age extremes)
    if "Age" in d.columns:
        d["Age_sq"] = d["Age"] ** 2

    # LogFare × WomanOrChild (women+children paid much more for 1st class)
    if "LogFare" in d.columns and "IsWomanOrChild" in d.columns:
        d["Fare_WomanChild"] = d["LogFare"] * d["IsWomanOrChild"]

    # Pclass × IsAdultMale (3rd class adult males had worst survival)
    if "Pclass" in d.columns and "IsAdultMale" in d.columns:
        d["Pclass_AdultMale"] = d["Pclass"] * d["IsAdultMale"]

    # ── Additional interaction features ───────────────────────────────────────
    # Pclass × IsAlone (being alone in 3rd class is particularly bad)
    if "Pclass" in d.columns and "IsAlone" in d.columns:
        d["Pclass_IsAlone"] = d["Pclass"] * d["IsAlone"]

    # Age × Sex_bin (captures that boys and girls have similar survival, men much lower)
    if "Age" in d.columns and "Sex_bin" in d.columns:
        d["Age_Sex"] = d["Age"] * d["Sex_bin"]

    # Title+Pclass combined category — captures survival by title within each class
    if "Title" in d.columns and "Pclass" in d.columns:
        d["Title_Pclass"] = d["Title"].astype(str) + "_" + d["Pclass"].astype(str)

    # TicketGroupSize × IsAlone interaction (ticket alone vs family on same ticket)
    if "TicketGroupSize" in d.columns and "IsAlone" in d.columns:
        d["TicketSize_IsAlone"] = d["TicketGroupSize"] * d["IsAlone"]

    # FamilySize × Pclass (large families in 3rd class fare worse)
    if "FamilySize" in d.columns and "Pclass" in d.columns:
        d["FamilySize_Pclass"] = d["FamilySize"] * d["Pclass"]

    # Fill remaining NaN
    for col in d.select_dtypes(include=["object", "category"]).columns:
        d[col] = d[col].fillna("Unknown")
    for col in d.select_dtypes(include="number").columns:
        d[col] = d[col].fillna(d[col].median())

    return d


X_all_raw = engineer_features(X_raw)

# Identify categorical columns for OOF target encoding
CAT_COLS_FOR_TE = [c for c in ["Title", "Pclass_Sex", "Deck", "TicketPrefix", "FamilyID",
                                "Embarked", "TicketID", "Title_Pclass"]
                   if c in X_all_raw.columns]
print(f"Categorical cols for target encoding: {CAT_COLS_FOR_TE}")

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


def add_oof_target_encoding(X_df, y_arr, skf, cat_cols, alpha=5):
    """
    OOF target encoding: for each fold, compute per-category survival rate
    using only the training fold, then apply to the validation fold.
    Prevents leakage while capturing survival rates per category.
    """
    global_mean = float(y_arr.mean())
    new_features = {}

    for col in cat_cols:
        if col not in X_df.columns:
            continue
        encoded = np.full(len(X_df), global_mean, dtype=np.float64)

        for tr_idx, va_idx in skf.split(np.zeros(len(X_df)), y_arr):
            train_vals = X_df[col].astype(str).values[tr_idx]
            train_y = y_arr[tr_idx]

            # Compute smoothed per-category mean on training fold
            cat_stats = {}
            for cat_val in np.unique(train_vals):
                mask = train_vals == cat_val
                n = int(mask.sum())
                mean = float(train_y[mask].mean())
                # Additive smoothing towards global mean
                cat_stats[cat_val] = (n * mean + alpha * global_mean) / (n + alpha)

            # Apply to validation fold (unseen categories fall back to global mean)
            val_vals = X_df[col].astype(str).values[va_idx]
            for i, va_i in enumerate(va_idx):
                encoded[va_i] = cat_stats.get(val_vals[i], global_mean)

        new_features[f"{col}_te"] = encoded

    return pd.DataFrame(new_features, index=X_df.index)


# Compute and concatenate OOF target-encoded features
te_df = add_oof_target_encoding(X_all_raw, y_encoded, skf, CAT_COLS_FOR_TE, alpha=5)
X_all = pd.concat([X_all_raw, te_df], axis=1)

# Label encode all remaining categoricals
for col in X_all.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X_all[col] = le.fit_transform(X_all[col].astype(str))

X_arr = X_all.values.astype(np.float32)
print(f"Features ({X_arr.shape[1]}): {list(X_all.columns)}")

# ── Optuna-tuned LightGBM ─────────────────────────────────────────────────────
print("Tuning LightGBM with Optuna...")

def lgb_objective(trial):
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=trial.suggest_int("num_leaves", 20, 127),
        max_depth=-1,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        n_estimators=3000,
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=5,
        min_child_samples=trial.suggest_int("min_child_samples", 3, 30),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 2.0, log=True),
        verbose=-1,
        random_state=42,
    )
    oof = np.zeros(len(X_arr))
    for tr_idx, va_idx in skf.split(X_arr, y_encoded):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_arr[tr_idx], y_encoded[tr_idx],
              eval_set=[(X_arr[va_idx], y_encoded[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict_proba(X_arr[va_idx])[:, 1]
    return roc_auc_score(y_encoded, oof)

study_lgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
best_lgb_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    verbose=-1,
    random_state=42,
    n_estimators=3000,
    bagging_freq=5,
    **study_lgb.best_params
)
print(f"Best LGB params: {study_lgb.best_params}")

# ── Multi-seed LightGBM ───────────────────────────────────────────────────────
lgb_seeds = [42, 123, 456]
oof_lgb_list = []
for seed in lgb_seeds:
    oof_seed = np.zeros(len(X_arr))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
        model = lgb.LGBMClassifier(**{**best_lgb_params, 'random_state': seed})
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_seed[va_idx] = model.predict_proba(X_va)[:, 1]
    oof_lgb_list.append(oof_seed)

oof_lgb = np.mean(oof_lgb_list, axis=0)
auc_lgb = roc_auc_score(y_encoded, oof_lgb)
print(f"LGB (3-seed, Optuna) OOF AUC: {auc_lgb:.4f}")

# ── XGBoost ──────────────────────────────────────────────────────────────────
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

# ── CatBoost ──────────────────────────────────────────────────────────────────
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

# ── ExtraTrees ────────────────────────────────────────────────────────────────
oof_et = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = ExtraTreesClassifier(
        n_estimators=1000,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    oof_et[va_idx] = model.predict_proba(X_va)[:, 1]

auc_et = roc_auc_score(y_encoded, oof_et)
print(f"ET OOF AUC: {auc_et:.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
oof_rf = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = RandomForestClassifier(
        n_estimators=1000,
        max_features='sqrt',
        min_samples_leaf=1,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    oof_rf[va_idx] = model.predict_proba(X_va)[:, 1]

auc_rf = roc_auc_score(y_encoded, oof_rf)
print(f"RF OOF AUC: {auc_rf:.4f}")

# ── SVM with RBF kernel ───────────────────────────────────────────────────────
oof_svm = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_tr_sc, y_tr)
    oof_svm[va_idx] = model.predict_proba(X_va_sc)[:, 1]

auc_svm = roc_auc_score(y_encoded, oof_svm)
print(f"SVM OOF AUC: {auc_svm:.4f}")

# ── HistGradientBoostingClassifier (sklearn) ──────────────────────────────────
# Different algorithm: native NaN handling, depth-wise growth
oof_hgb = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    model = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    model.fit(X_tr, y_tr)
    oof_hgb[va_idx] = model.predict_proba(X_va)[:, 1]

auc_hgb = roc_auc_score(y_encoded, oof_hgb)
print(f"HGB OOF AUC: {auc_hgb:.4f}")

# ── KNN (averaged over multiple k values) ────────────────────────────────────
# Instance-based: very different inductive bias from tree models
oof_knn = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    knn_preds = []
    for k in [5, 10, 15, 20, 30]:
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance', n_jobs=-1)
        model.fit(X_tr_sc, y_tr)
        knn_preds.append(model.predict_proba(X_va_sc)[:, 1])
    oof_knn[va_idx] = np.mean(knn_preds, axis=0)

auc_knn = roc_auc_score(y_encoded, oof_knn)
print(f"KNN OOF AUC: {auc_knn:.4f}")

# ── MLP (sklearn neural network) ─────────────────────────────────────────────
# Non-linear decision boundaries, different from tree-based models
oof_mlp = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
    y_tr, y_va = y_encoded[tr_idx], y_encoded[va_idx]
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        alpha=0.5,       # L2 regularization — crucial for small N
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_tr_sc, y_tr)
    oof_mlp[va_idx] = model.predict_proba(X_va_sc)[:, 1]

auc_mlp = roc_auc_score(y_encoded, oof_mlp)
print(f"MLP OOF AUC: {auc_mlp:.4f}")

# ── Equal-weight blend of all 9 models ────────────────────────────────────────
all_oofs = [oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf, oof_svm, oof_hgb, oof_knn, oof_mlp]
oof_blend9 = np.mean(all_oofs, axis=0)
auc_blend9 = roc_auc_score(y_encoded, oof_blend9)
print(f"Blend-9 OOF AUC: {auc_blend9:.4f}")

# ── Stacking: Logistic Regression meta-learner ────────────────────────────────
meta_9 = np.column_stack(all_oofs)

oof_meta_lr = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta_model.fit(meta_9[tr_idx], y_encoded[tr_idx])
    oof_meta_lr[va_idx] = meta_model.predict_proba(meta_9[va_idx])[:, 1]

auc_meta_lr = roc_auc_score(y_encoded, oof_meta_lr)
print(f"Stack-LR OOF AUC: {auc_meta_lr:.4f}")

# ── Stacking: LightGBM meta-learner with key original + all TE features ───────
all_te_cols = [c for c in X_all.columns if c.endswith('_te')]
meta_key_cols = [c for c in ['Pclass', 'Sex_bin', 'AgeBin', 'IsAdultMale',
                               'IsWomanOrChild', 'LogFare', 'HasCabin', 'FamilySize',
                               'FareRank_Pclass', 'IsAlone', 'TicketGroupSize'] + all_te_cols
                 if c in X_all.columns]
meta_orig = X_all[meta_key_cols].values.astype(np.float32)
meta_9_ext = np.column_stack([meta_9, meta_orig])

oof_meta_lgb = np.zeros(len(X_arr))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_encoded)):
    meta_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=7,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=300,
        feature_fraction=1.0,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=10,
        reg_alpha=1.0,
        reg_lambda=1.0,
        verbose=-1,
        random_state=42,
    )
    meta_model.fit(meta_9_ext[tr_idx], y_encoded[tr_idx])
    oof_meta_lgb[va_idx] = meta_model.predict_proba(meta_9_ext[va_idx])[:, 1]

auc_meta_lgb = roc_auc_score(y_encoded, oof_meta_lgb)
print(f"Stack-LGB OOF AUC: {auc_meta_lgb:.4f}")

# Report the best across all ensembling strategies
best_val_roc_auc = max(auc_blend9, auc_meta_lr, auc_meta_lgb)
print(f"BEST_VAL_ROC_AUC: {best_val_roc_auc:.6f}")
