import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
target_col = "Survived"
y_raw = df[target_col]
X_raw = df.drop(columns=[target_col])

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

print(f"Columns: {X_raw.columns.tolist()}")
unique, counts = np.unique(y_encoded, return_counts=True)
print(f"[DATA] N={len(df)}, classes={len(unique)}, dist={dict(zip(unique.tolist(), counts.tolist()))}")

# ── Train / val split ────────────────────────────────────────────────────────
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
y_train = np.array(y_train)
y_val = np.array(y_val)

# ── Feature engineering ──────────────────────────────────────────────────────

RARE_TITLES = frozenset({'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                          'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'})

RAW_DROP_COLS = ['Name', 'Ticket', 'Cabin', 'PassengerId']


def extract_title(name_series):
    titles = name_series.str.extract(r' ([A-Za-z]+)\.', expand=False)
    titles = titles.where(~titles.isin(RARE_TITLES), 'Rare')
    titles = titles.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    return titles.fillna('Unknown')


def ticket_prefix(t):
    if pd.isna(t):
        return 'UNK'
    parts = str(t).strip().split()
    if len(parts) > 1:
        pref = re.sub(r'[^A-Z]', '', parts[0].upper())
        return pref if pref else 'NUM'
    return 'NUM'


def engineer_features(df_in, ref):
    """
    Add Titanic-specific features and drop raw high-cardinality columns.
    ref: training set for computing statistics (prevents leakage).
    """
    df = df_in.copy()

    # 1. Title from Name (before dropping Name)
    if 'Name' in df.columns:
        df['Title'] = extract_title(df['Name'])

    # 2. Name length (as a proxy for social status)
    if 'Name' in df.columns:
        df['NameLength'] = df['Name'].str.len().fillna(0).astype(np.float32)

    # 3. Ticket prefix (before dropping Ticket)
    if 'Ticket' in df.columns:
        df['TicketPrefix'] = df['Ticket'].apply(ticket_prefix)

    # 4. Ticket group size — how many passengers share the same ticket number
    if 'Ticket' in df.columns and 'Ticket' in ref.columns:
        ticket_counts = ref['Ticket'].value_counts().to_dict()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts).fillna(1).astype(np.float32)

    # 5. Cabin → Deck + HasCabin (before dropping Cabin)
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(np.int8)
        df['Deck'] = df['Cabin'].fillna('U').str[0]
        df['Deck'] = df['Deck'].replace('n', 'U')

    # 6. Drop high-cardinality raw columns
    df = df.drop(columns=[c for c in RAW_DROP_COLS if c in df.columns])

    # 7. Family size
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(np.int8)
        # Family size bins
        df['FamilySizeBin'] = pd.cut(
            df['FamilySize'],
            bins=[0, 1, 4, 20],
            labels=['solo', 'small', 'large'],
        ).astype(str)

    # 8. Embarked imputation (from ref to avoid leakage)
    if 'Embarked' in df.columns:
        emb_mode = ref['Embarked'].mode()[0] if not ref['Embarked'].isna().all() else 'S'
        df['Embarked'] = df['Embarked'].fillna(emb_mode)

    # 9. Fare imputation + derived features
    if 'Fare' in df.columns:
        fare_median = ref['Fare'].median()
        df['Fare'] = df['Fare'].fillna(fare_median)
        if 'FamilySize' in df.columns:
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].clip(lower=1)
        df['LogFare'] = np.log1p(df['Fare'])
        if 'FarePerPerson' in df.columns:
            df['LogFarePerPerson'] = np.log1p(df['FarePerPerson'])
        # Fare quartile bin (computed from ref)
        fare_quartiles = ref['Fare'].fillna(ref['Fare'].median()).quantile([0.25, 0.5, 0.75]).values
        df['FareBin'] = pd.cut(
            df['Fare'],
            bins=[-np.inf, fare_quartiles[0], fare_quartiles[1], fare_quartiles[2], np.inf],
            labels=['q1', 'q2', 'q3', 'q4'],
        ).astype(str)

    # 10. Age imputation (group median by Pclass+Title from ref)
    if 'Age' in df.columns:
        ref2 = ref.copy()
        if 'Name' in ref2.columns:
            ref2['Title'] = extract_title(ref2['Name'])

        grp_cols = [c for c in ['Pclass', 'Title'] if c in ref2.columns and c in df.columns]
        global_age_med = ref['Age'].median()

        missing_mask = df['Age'].isna()
        if missing_mask.any() and grp_cols:
            age_medians = ref2.groupby(grp_cols)['Age'].median()
            for idx in df.index[missing_mask]:
                key = tuple(df.loc[idx, c] for c in grp_cols)
                val = age_medians.get(key, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = global_age_med
                df.loc[idx, 'Age'] = val
        elif missing_mask.any():
            df['Age'] = df['Age'].fillna(global_age_med)

    # 11. Age-derived features
    if 'Age' in df.columns:
        df['IsChild'] = (df['Age'] < 15).astype(np.int8)
        df['IsSenior'] = (df['Age'] >= 60).astype(np.int8)
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 55, 100],
            labels=['child', 'teen', 'young', 'adult', 'senior'],
        ).astype(str)

    # 12. Interaction features
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
    if 'Age' in df.columns and 'Pclass' in df.columns:
        df['Age_x_Pclass'] = df['Age'] * df['Pclass']
    if 'Age' in df.columns and 'Sex' in df.columns:
        sex_num = df['Sex'].map({'female': 0.0, 'male': 1.0}).fillna(0.5)
        df['Age_x_Sex'] = df['Age'] * sex_num
    if 'Fare' in df.columns and 'Pclass' in df.columns:
        df['Fare_x_Pclass'] = df['Fare'] * df['Pclass']
    # Women/children priority
    if 'Sex' in df.columns and 'Age' in df.columns:
        sex_female = (df['Sex'] == 'female').astype(np.int8)
        is_child = (df['Age'] < 15).astype(np.int8)
        df['WomanOrChild'] = ((sex_female == 1) | (is_child == 1)).astype(np.int8)

    # 13. Missingness indicators
    for col in ref.columns:
        if col in RAW_DROP_COLS:
            continue
        if ref[col].isna().mean() > 0.05:
            df[f'{col}_miss'] = df[col].isna().astype(np.int8)

    return df


# ── Prepare engineered datasets ───────────────────────────────────────────────
X_train_fe = engineer_features(X_train_raw, ref=X_train_raw)
X_val_fe = engineer_features(X_val_raw, ref=X_train_raw)

# Identify categorical columns
str_cols = X_train_fe.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
extra_cats = ['Title', 'Deck', 'AgeGroup', 'TicketPrefix', 'Sex_Pclass', 'FamilySizeBin', 'FareBin']
cat_cols_all = list(dict.fromkeys(str_cols + [c for c in extra_cats if c in X_train_fe.columns]))

print(f"Features after engineering: {X_train_fe.shape[1]}")
print(f"Categorical cols: {cat_cols_all}")


# ── Model-specific data preparers ─────────────────────────────────────────────

def make_lgbm_data(X_tr_ref, X_tr, X_te):
    """Convert cat cols to category dtype. Categories fit from X_tr_ref."""
    res = []
    for df in [X_tr, X_te]:
        df = df.copy()
        for col in cat_cols_all:
            if col not in df.columns:
                continue
            cats = pd.Categorical(X_tr_ref[col].astype(str)).categories
            df[col] = pd.Categorical(df[col].astype(str), categories=cats)
        res.append(df)
    return res[0], res[1]


def make_xgb_data(X_tr_ref, X_tr, X_te):
    """Ordinal-encode categoricals. Encoder fit from X_tr_ref."""
    encoders = {}
    for col in cat_cols_all:
        if col not in X_tr_ref.columns:
            continue
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                             encoded_missing_value=-2)
        # Fix for ArrowStringArray: use tolist() → numpy conversion to avoid reshape issue
        train_vals = np.array(X_tr_ref[col].astype(str).tolist()).reshape(-1, 1)
        enc.fit(train_vals)
        encoders[col] = enc

    res = []
    for df in [X_tr, X_te]:
        df = df.copy()
        for col, enc in encoders.items():
            if col in df.columns:
                vals = np.array(df[col].astype(str).tolist()).reshape(-1, 1)
                df[col] = enc.transform(vals).ravel()
        obj_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        df = df.drop(columns=obj_cols, errors='ignore')
        res.append(df.values.astype(np.float32))
    return res[0], res[1]


def make_hgb_data(X_tr_ref, X_tr, X_te):
    """
    Ordinal-encode categoricals for HistGradientBoosting.
    HGB supports native categorical handling with integer-encoded features.
    Returns (X_tr_arr, X_te_arr, categorical_feature_indices).
    """
    encoders = {}
    for col in cat_cols_all:
        if col not in X_tr_ref.columns:
            continue
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                             encoded_missing_value=-1)
        train_vals = np.array(X_tr_ref[col].astype(str).tolist()).reshape(-1, 1)
        enc.fit(train_vals)
        encoders[col] = enc

    # Get final column order (drop raw string cols, keep encoded)
    sample = X_tr.copy()
    for col, enc in encoders.items():
        if col in sample.columns:
            vals = np.array(sample[col].astype(str).tolist()).reshape(-1, 1)
            sample[col] = enc.transform(vals).ravel()
    obj_cols = sample.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    sample = sample.drop(columns=obj_cols, errors='ignore')
    final_cols = sample.columns.tolist()
    cat_indices = [i for i, c in enumerate(final_cols) if c in encoders]

    res = []
    for df in [X_tr, X_te]:
        df = df.copy()
        for col, enc in encoders.items():
            if col in df.columns:
                vals = np.array(df[col].astype(str).tolist()).reshape(-1, 1)
                df[col] = enc.transform(vals).ravel()
        obj_cols2 = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        df = df.drop(columns=obj_cols2, errors='ignore')
        df = df[final_cols]  # ensure consistent column order
        res.append(df.values.astype(np.float64))
    return res[0], res[1], cat_indices


# ── Pre-prepare data ──────────────────────────────────────────────────────────
X_train_lgb, X_val_lgb = make_lgbm_data(X_train_fe, X_train_fe, X_val_fe)
lgbm_cat_feats = [c for c in cat_cols_all if c in X_train_lgb.columns]

X_train_xgb, X_val_xgb = make_xgb_data(X_train_fe, X_train_fe, X_val_fe)

X_train_hgb, X_val_hgb, hgb_cat_indices = make_hgb_data(X_train_fe, X_train_fe, X_val_fe)

print(f"HGB categorical feature indices: {hgb_cat_indices}")

# ── Hyperparameters ───────────────────────────────────────────────────────────

lgbm_params = dict(
    objective='binary', metric='auc',
    n_estimators=3000, learning_rate=0.01,
    num_leaves=31, min_child_samples=10,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1,
)

xgb_params = dict(
    objective='binary:logistic', eval_metric='auc',
    n_estimators=3000, learning_rate=0.01,
    max_depth=5, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0,
    tree_method='hist',
    early_stopping_rounds=150,
)

hgb_params = dict(
    loss='log_loss',
    learning_rate=0.05,
    max_iter=1000,
    max_leaf_nodes=31,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    random_state=42,
)

# ── 5-fold stratified CV ─────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_tr = len(y_train)

lgbm_oof = np.zeros(N_tr)
xgb_oof = np.zeros(N_tr)
hgb_oof = np.zeros(N_tr)

lgbm_val_preds = np.zeros(len(y_val))
xgb_val_preds = np.zeros(len(y_val))
hgb_val_preds = np.zeros(len(y_val))

lgbm_best_iters, xgb_best_iters = [], []
pos_indices = np.arange(N_tr)

for fold, (tr_idx, va_idx) in enumerate(skf.split(pos_indices, y_train)):
    y_f_tr = y_train[tr_idx]
    y_f_va = y_train[va_idx]

    # ── LightGBM ──
    X_f_tr_lgb = X_train_lgb.iloc[tr_idx]
    X_f_va_lgb = X_train_lgb.iloc[va_idx]
    fold_cats_lgb = []
    X_f_tr_lgb2 = X_f_tr_lgb.copy()
    X_f_va_lgb2 = X_f_va_lgb.copy()
    X_val_lgb2 = X_val_lgb.copy()
    for col in lgbm_cat_feats:
        if col not in X_f_tr_lgb.columns:
            continue
        cats = pd.Categorical(X_f_tr_lgb2[col].astype(str)).categories
        X_f_tr_lgb2[col] = pd.Categorical(X_f_tr_lgb2[col].astype(str), categories=cats)
        X_f_va_lgb2[col] = pd.Categorical(X_f_va_lgb2[col].astype(str), categories=cats)
        X_val_lgb2[col] = pd.Categorical(X_val_lgb2[col].astype(str), categories=cats)
        fold_cats_lgb.append(col)

    lgb_clf = lgb.LGBMClassifier(**lgbm_params)
    lgb_clf.fit(
        X_f_tr_lgb2, y_f_tr,
        eval_set=[(X_f_va_lgb2, y_f_va)],
        categorical_feature=fold_cats_lgb,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    lgbm_oof[va_idx] = lgb_clf.predict_proba(X_f_va_lgb2)[:, 1]
    lgbm_val_preds += lgb_clf.predict_proba(X_val_lgb2)[:, 1] / skf.n_splits
    lgbm_best_iters.append(lgb_clf.best_iteration_)

    # ── XGBoost ──
    X_f_tr_xgb = X_train_xgb[tr_idx]
    X_f_va_xgb = X_train_xgb[va_idx]

    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(
        X_f_tr_xgb, y_f_tr,
        eval_set=[(X_f_va_xgb, y_f_va)],
        verbose=False,
    )
    xgb_oof[va_idx] = xgb_clf.predict_proba(X_f_va_xgb)[:, 1]
    xgb_val_preds += xgb_clf.predict_proba(X_val_xgb)[:, 1] / skf.n_splits
    xgb_best_iters.append(xgb_clf.best_iteration)

    # ── HistGradientBoosting ──
    X_f_tr_hgb = X_train_hgb[tr_idx]
    X_f_va_hgb = X_train_hgb[va_idx]

    hgb_clf = HistGradientBoostingClassifier(
        **hgb_params,
        categorical_features=hgb_cat_indices if hgb_cat_indices else None,
    )
    hgb_clf.fit(X_f_tr_hgb, y_f_tr)
    hgb_oof[va_idx] = hgb_clf.predict_proba(X_f_va_hgb)[:, 1]
    hgb_val_preds += hgb_clf.predict_proba(X_val_hgb)[:, 1] / skf.n_splits

    lgbm_f = roc_auc_score(y_f_va, lgbm_oof[va_idx])
    xgb_f = roc_auc_score(y_f_va, xgb_oof[va_idx])
    hgb_f = roc_auc_score(y_f_va, hgb_oof[va_idx])
    print(f"Fold {fold+1}: LGB={lgbm_f:.4f}, XGB={xgb_f:.4f}, HGB={hgb_f:.4f}")

# ── OOF scores ───────────────────────────────────────────────────────────────
lgbm_cv = roc_auc_score(y_train, lgbm_oof)
xgb_cv = roc_auc_score(y_train, xgb_oof)
hgb_cv = roc_auc_score(y_train, hgb_oof)
print(f"OOF AUC → LGB={lgbm_cv:.4f}, XGB={xgb_cv:.4f}, HGB={hgb_cv:.4f}")
print(f"Best iters → LGB={int(np.mean(lgbm_best_iters))}, XGB={int(np.mean(xgb_best_iters))}")

# ── Weighted blend (by OOF AUC) ──────────────────────────────────────────────
total = lgbm_cv + xgb_cv + hgb_cv
w_lgb = lgbm_cv / total
w_xgb = xgb_cv / total
w_hgb = hgb_cv / total
print(f"Blend weights → LGB={w_lgb:.3f}, XGB={w_xgb:.3f}, HGB={w_hgb:.3f}")

blend_val = w_lgb * lgbm_val_preds + w_xgb * xgb_val_preds + w_hgb * hgb_val_preds
val_auc = float(roc_auc_score(y_val, blend_val))

lgbm_val_auc = float(roc_auc_score(y_val, lgbm_val_preds))
xgb_val_auc = float(roc_auc_score(y_val, xgb_val_preds))
hgb_val_auc = float(roc_auc_score(y_val, hgb_val_preds))
print(f"Individual Val AUC → LGB={lgbm_val_auc:.4f}, XGB={xgb_val_auc:.4f}, HGB={hgb_val_auc:.4f}")
print(f"Blended Val ROC-AUC: {val_auc:.6f}")

print(f"BEST_VAL_ROC_AUC: {val_auc:.6f}")