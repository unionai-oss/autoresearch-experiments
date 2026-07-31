import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import sys
import site as _site_mod
# Make CatBoost available from user site-packages if not already in venv
_user_site = _site_mod.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
    print("[INFO] CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARNING] CatBoost not available — skipping")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
target_col = "Survived"
X_raw = df.drop(columns=[target_col])
y = df[target_col].values.astype(int)

unique_vals, counts = np.unique(y, return_counts=True)
print(f"[DATA] Samples={len(df)}  Classes={len(unique_vals)}  "
      f"Dist={dict(zip(unique_vals.tolist(), counts.tolist()))}")

# ── Feature engineering ────────────────────────────────────────────────────────
COMMON_TITLES = {'Mr', 'Mrs', 'Miss', 'Master'}
TITLE_PRIORITY = {'Mr': 0, 'Rare': 0, 'Master': 1, 'Miss': 2, 'Mrs': 2}


def engineer(df_in):
    """Pure function – returns new DataFrame with engineered features."""
    X = df_in.copy()

    # Title + surname from Name
    if 'Name' in X.columns:
        X['Title'] = (X['Name']
                      .str.extract(r' ([A-Za-z]+)\.', expand=False)
                      .apply(lambda t: t if t in COMMON_TITLES else 'Rare')
                      .fillna('Rare'))
        X['TitlePriority'] = X['Title'].map(TITLE_PRIORITY).fillna(0).astype(int)
        X['Surname'] = X['Name'].str.split(',').str[0].str.strip()
        X = X.drop(columns=['Name'])

    if 'Cabin' in X.columns:
        X['CabinKnown'] = X['Cabin'].notna().astype(int)
        X['Deck'] = X['Cabin'].str[0].fillna('U')
        X = X.drop(columns=['Cabin'])

    # Ticket
    if 'Ticket' in X.columns:
        def _tprefix(t):
            if pd.isna(t):
                return 'Unknown'
            parts = str(t).split()
            return parts[0] if len(parts) > 1 else 'Numeric'
        X['TicketPrefix'] = X['Ticket'].apply(_tprefix)
        # Ticket group size (same ticket = traveling together)
        ticket_counts = X['Ticket'].map(X['Ticket'].value_counts())
        X['TicketGroupSize'] = ticket_counts.fillna(1).astype(int)
        X = X.drop(columns=['Ticket'])

    X = X.drop(columns=[c for c in ['PassengerId'] if c in X.columns])

    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['SmallFamily'] = (X['FamilySize'].between(2, 4)).astype(int)
        X['LargeFamily'] = (X['FamilySize'] > 4).astype(int)
        X['FamilySize_sq'] = X['FamilySize'] ** 2

    if 'Fare' in X.columns:
        fam = X.get('FamilySize', pd.Series(1, index=X.index))
        X['FarePerPerson'] = X['Fare'] / fam.clip(lower=1)
        X['Fare_log'] = np.log1p(X['Fare'].clip(lower=0).fillna(0))
        X['FarePerPerson_log'] = np.log1p(X['FarePerPerson'].clip(lower=0).fillna(0))
        if 'TicketGroupSize' in X.columns:
            X['FarePerTicketGroup'] = X['Fare'] / X['TicketGroupSize'].clip(lower=1)
            X['FarePerTicketGroup_log'] = np.log1p(X['FarePerTicketGroup'].clip(lower=0).fillna(0))
        # TicketGroupSize minus FamilySize → non-family companions
        if 'FamilySize' in X.columns and 'TicketGroupSize' in X.columns:
            X['NonFamilyCompanions'] = (X['TicketGroupSize'] - X['FamilySize']).clip(lower=0)

    return X


# ── Preprocessing: fit on train, apply to val ──────────────────────────────────

def preprocess_fit_transform(X_tr_in, X_vl_in, y_tr_in=None):
    X_tr = X_tr_in.copy()
    X_vl = X_vl_in.copy()

    # ── Better Age imputation: group median by (Title, Pclass) ──────────────────
    # Done BEFORE missing indicators so the imputed values are used in derived feats
    if 'Age' in X_tr.columns and 'Title' in X_tr.columns and 'Pclass' in X_tr.columns:
        age_group_medians = X_tr.groupby(['Title', 'Pclass'])['Age'].median()
        age_title_medians = X_tr.groupby('Title')['Age'].median()
        global_age_median = X_tr['Age'].median()

        for Xdf in [X_tr, X_vl]:
            mask = Xdf['Age'].isna()
            if mask.any():
                keys = list(zip(Xdf.loc[mask, 'Title'].values,
                                Xdf.loc[mask, 'Pclass'].values))
                filled = []
                for k in keys:
                    if k in age_group_medians.index:
                        filled.append(age_group_medians[k])
                    elif k[0] in age_title_medians.index:
                        filled.append(age_title_medians[k[0]])
                    else:
                        filled.append(global_age_median)
                Xdf.loc[mask, 'Age'] = filled

    # Missing indicators for numerics with >5% missing (fit on train)
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    miss_cols = [c for c in num_cols if X_tr[c].isnull().mean() > 0.05]
    for c in miss_cols:
        X_tr[f'{c}_miss'] = X_tr[c].isnull().astype(int)
        X_vl[f'{c}_miss'] = X_vl[c].isnull().astype(int)

    # Numeric imputation with train medians (Age mostly already imputed above)
    num_cols2 = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    medians = {c: X_tr[c].median() for c in num_cols2}
    for c, m in medians.items():
        X_tr[c] = X_tr[c].fillna(m)
        X_vl[c] = X_vl[c].fillna(m)

    # Age-derived bins (after imputation)
    if 'Age' in X_tr.columns:
        for Xdf in [X_tr, X_vl]:
            Xdf['IsChild'] = (Xdf['Age'] < 12).astype(int)
            Xdf['IsSenior'] = (Xdf['Age'] > 60).astype(int)
            Xdf['AgeBand'] = (pd.cut(Xdf['Age'],
                                     bins=[0, 12, 18, 35, 60, 100],
                                     labels=False)
                              .fillna(2).astype(int))

    # Fare quartile bin (fit on train)
    if 'Fare' in X_tr.columns:
        fare_qs = X_tr['Fare'].quantile([0.25, 0.5, 0.75]).values
        for Xdf in [X_tr, X_vl]:
            Xdf['FareBin'] = pd.cut(
                Xdf['Fare'],
                bins=[-np.inf, fare_qs[0], fare_qs[1], fare_qs[2], np.inf],
                labels=False
            ).fillna(0).astype(int)

    # Categorical encoding (fit categories on train)
    cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    # target-encode high-cardinality columns (Surname, TicketPrefix) using train labels
    if y_tr_in is not None:
        for c in list(cat_cols):
            n_unique = X_tr[c].fillna('Unknown').nunique()
            if n_unique > 15:
                X_tr[c] = X_tr[c].fillna('Unknown').astype(str)
                X_vl[c] = X_vl[c].fillna('Unknown').astype(str)
                global_mean = float(np.mean(y_tr_in))
                tmp = pd.DataFrame({'cat': X_tr[c].values, 'target': y_tr_in})
                means = tmp.groupby('cat')['target'].mean()
                X_tr[f'{c}_te'] = X_tr[c].map(means).fillna(global_mean).values
                X_vl[f'{c}_te'] = X_vl[c].map(means).fillna(global_mean).values
                # drop original high-card column
                X_tr = X_tr.drop(columns=[c])
                X_vl = X_vl.drop(columns=[c])
                cat_cols = [x for x in cat_cols if x != c]

    cat_cols2 = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols2:
        X_tr[c] = X_tr[c].fillna('Unknown').astype(str)
        X_vl[c] = X_vl[c].fillna('Unknown').astype(str)
        cats = {v: i for i, v in enumerate(sorted(X_tr[c].unique()))}
        X_tr[c] = X_tr[c].map(cats).astype(int)
        X_vl[c] = X_vl[c].map(cats).fillna(-1).astype(int)

    # Interaction features (after encoding)
    for Xdf in [X_tr, X_vl]:
        if 'Pclass' in Xdf.columns and 'Sex' in Xdf.columns:
            Xdf['Pclass_Sex'] = Xdf['Pclass'] * 10 + Xdf['Sex']
        if 'Pclass' in Xdf.columns and 'IsAlone' in Xdf.columns:
            Xdf['Pclass_IsAlone'] = Xdf['Pclass'] * Xdf['IsAlone']
        if 'Age' in Xdf.columns and 'Pclass' in Xdf.columns:
            Xdf['Age_x_Pclass'] = Xdf['Age'] * Xdf['Pclass']
        if 'Age' in Xdf.columns and 'Sex' in Xdf.columns:
            Xdf['Age_x_Sex'] = Xdf['Age'] * Xdf['Sex']
        if 'Sex' in Xdf.columns and 'IsChild' in Xdf.columns:
            Xdf['WomenOrChild'] = ((Xdf['Sex'] == 0) | (Xdf['IsChild'] == 1)).astype(int)
        if 'WomenOrChild' in Xdf.columns and 'Pclass' in Xdf.columns:
            # Women/children in 1st class had near 100% survival; 3rd class much lower
            Xdf['WomenOrChild_x_Pclass'] = Xdf['WomenOrChild'] * Xdf['Pclass']
        if 'IsChild' in Xdf.columns and 'Pclass' in Xdf.columns:
            Xdf['IsChild_x_Pclass'] = Xdf['IsChild'] * Xdf['Pclass']
        if 'TitlePriority' in Xdf.columns and 'Pclass' in Xdf.columns:
            Xdf['TitlePriority_x_Pclass'] = Xdf['TitlePriority'] * Xdf['Pclass']
        if 'FamilySize' in Xdf.columns and 'Pclass' in Xdf.columns:
            Xdf['FamilySize_x_Pclass'] = Xdf['FamilySize'] * Xdf['Pclass']
        if 'Fare' in Xdf.columns and 'Sex' in Xdf.columns:
            Xdf['Fare_x_Sex'] = Xdf['Fare'] * Xdf['Sex']
        if 'Fare_log' in Xdf.columns and 'Pclass' in Xdf.columns:
            Xdf['FareLog_x_Pclass'] = Xdf['Fare_log'] * Xdf['Pclass']
        if 'AgeBand' in Xdf.columns and 'Sex' in Xdf.columns:
            Xdf['AgeBand_x_Sex'] = Xdf['AgeBand'] * (Xdf['Sex'] + 1)

    # Align columns (val might miss a column if a category didn't appear)
    cols = X_tr.columns.tolist()
    X_vl = X_vl.reindex(columns=cols, fill_value=0)

    return X_tr, X_vl


# ── Setup ──────────────────────────────────────────────────────────────────────
X_feat = engineer(X_raw)
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(X_feat, y))

# class ratio for scale_pos_weight
neg_count, pos_count = counts[0], counts[1]
spw = float(neg_count) / float(pos_count)
print(f"[INFO] scale_pos_weight={spw:.4f}")

rng = np.random.RandomState(42)


# ── Helper: run HPO trial ─────────────────────────────────────────────────────

def lgbm_cv(params):
    scores = []
    for tr_idx, vl_idx in splits:
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        X_tr, X_vl = preprocess_fit_transform(X_feat.iloc[tr_idx], X_feat.iloc[vl_idx], y_tr_in=y_tr)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr,
                eval_X=X_vl, eval_y=y_vl,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        scores.append(roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1]))
    return float(np.mean(scores))


def xgb_cv(params):
    scores = []
    for tr_idx, vl_idx in splits:
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        X_tr, X_vl = preprocess_fit_transform(X_feat.iloc[tr_idx], X_feat.iloc[vl_idx], y_tr_in=y_tr)
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        scores.append(roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1]))
    return float(np.mean(scores))


def cb_cv(params):
    scores = []
    for tr_idx, vl_idx in splits:
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        X_tr, X_vl = preprocess_fit_transform(X_feat.iloc[tr_idx], X_feat.iloc[vl_idx], y_tr_in=y_tr)
        pool_tr = Pool(X_tr, y_tr)
        pool_vl = Pool(X_vl, y_vl)
        clf = CatBoostClassifier(**params)
        clf.fit(pool_tr, eval_set=pool_vl, use_best_model=True, verbose=False)
        scores.append(roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1]))
    return float(np.mean(scores))


# ── LightGBM HPO ──────────────────────────────────────────────────────────────
print("\n[LGBM] Running HPO (80 trials)...")
N_LGBM = 80
best_lgbm_score = -1.0
best_lgbm_params = None

for trial_idx in range(N_LGBM):
    params = dict(
        objective='binary',
        metric='auc',
        n_estimators=2000,
        num_leaves=int(rng.randint(8, 129)),
        learning_rate=float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        min_child_samples=int(rng.randint(3, 61)),
        feature_fraction=float(rng.uniform(0.4, 1.0)),
        bagging_fraction=float(rng.uniform(0.4, 1.0)),
        bagging_freq=int(rng.randint(1, 11)),
        reg_alpha=float(np.exp(rng.uniform(np.log(1e-8), np.log(10.0)))),
        reg_lambda=float(np.exp(rng.uniform(np.log(1e-8), np.log(10.0)))),
        max_depth=int(rng.randint(3, 13)),
        min_split_gain=float(rng.uniform(0.0, 1.0)),
        scale_pos_weight=float(rng.uniform(0.5, spw * 2.0)),
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )
    score = lgbm_cv(params)
    if score > best_lgbm_score:
        best_lgbm_score = score
        best_lgbm_params = params.copy()
        print(f"  [LGBM Trial {trial_idx:3d}] New best: {best_lgbm_score:.6f}")

print(f"[LGBM] Best CV AUC: {best_lgbm_score:.6f}")


# ── XGBoost HPO ───────────────────────────────────────────────────────────────
print("\n[XGB] Running HPO (50 trials)...")
N_XGB = 50
best_xgb_score = -1.0
best_xgb_params = None

for trial_idx in range(N_XGB):
    params = dict(
        n_estimators=2000,
        max_depth=int(rng.randint(3, 10)),
        learning_rate=float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        min_child_weight=int(rng.randint(1, 20)),
        subsample=float(rng.uniform(0.5, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        colsample_bylevel=float(rng.uniform(0.4, 1.0)),
        reg_alpha=float(np.exp(rng.uniform(np.log(1e-8), np.log(10.0)))),
        reg_lambda=float(np.exp(rng.uniform(np.log(1e-8), np.log(10.0)))),
        scale_pos_weight=float(rng.uniform(0.5, spw * 2.0)),
        eval_metric='auc',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    score = xgb_cv(params)
    if score > best_xgb_score:
        best_xgb_score = score
        best_xgb_params = params.copy()
        print(f"  [XGB Trial {trial_idx:3d}] New best: {best_xgb_score:.6f}")

print(f"[XGB] Best CV AUC: {best_xgb_score:.6f}")


# ── CatBoost HPO ──────────────────────────────────────────────────────────────
best_cb_score = -1.0
best_cb_params = None

if CATBOOST_AVAILABLE:
    print("\n[CB] Running HPO (40 trials)...")
    N_CB = 40
    for trial_idx in range(N_CB):
        params = dict(
            iterations=2000,
            learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
            depth=int(rng.randint(4, 11)),
            l2_leaf_reg=float(np.exp(rng.uniform(np.log(1.0), np.log(30.0)))),
            border_count=int(rng.choice([32, 64, 128, 254])),
            bagging_temperature=float(rng.uniform(0.0, 1.5)),
            random_strength=float(rng.uniform(0.5, 5.0)),
            early_stopping_rounds=50,
            eval_metric='AUC',
            loss_function='Logloss',
            random_state=42,
            train_dir='/tmp/catboost_info',
        )
        try:
            score = cb_cv(params)
            if score > best_cb_score:
                best_cb_score = score
                best_cb_params = params.copy()
                print(f"  [CB Trial {trial_idx:3d}] New best: {best_cb_score:.6f}")
        except Exception as e:
            print(f"  [CB Trial {trial_idx:3d}] Error: {e}")
    print(f"[CB] Best CV AUC: {best_cb_score:.6f}")
else:
    print("\n[CB] Skipped (not available)")


# ── Final OOF with best params from each model ─────────────────────────────────
print("\n[FINAL] Building OOF predictions from best models...")

final_lgbm_params = dict(**best_lgbm_params)
final_lgbm_params['n_estimators'] = 3000

final_xgb_params = dict(**best_xgb_params)
final_xgb_params['n_estimators'] = 3000

oof_lgbm = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cb = np.zeros(len(y))

for fold, (tr_idx, vl_idx) in enumerate(splits):
    y_tr, y_vl = y[tr_idx], y[vl_idx]
    X_tr, X_vl = preprocess_fit_transform(X_feat.iloc[tr_idx], X_feat.iloc[vl_idx], y_tr_in=y_tr)

    # LightGBM
    clf_lgbm = lgb.LGBMClassifier(**final_lgbm_params)
    clf_lgbm.fit(X_tr, y_tr,
                 eval_X=X_vl, eval_y=y_vl,
                 callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)])
    oof_lgbm[vl_idx] = clf_lgbm.predict_proba(X_vl)[:, 1]

    # XGBoost
    clf_xgb = xgb.XGBClassifier(**final_xgb_params)
    clf_xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    oof_xgb[vl_idx] = clf_xgb.predict_proba(X_vl)[:, 1]

    # CatBoost
    if CATBOOST_AVAILABLE and best_cb_params is not None:
        final_cb_params = dict(**best_cb_params)
        final_cb_params['iterations'] = 3000
        pool_tr = Pool(X_tr, y_tr)
        pool_vl = Pool(X_vl, y_vl)
        clf_cb = CatBoostClassifier(**final_cb_params)
        clf_cb.fit(pool_tr, eval_set=pool_vl, use_best_model=True, verbose=False)
        oof_cb[vl_idx] = clf_cb.predict_proba(X_vl)[:, 1]

    fold_auc_lgbm = roc_auc_score(y_vl, oof_lgbm[vl_idx])
    fold_auc_xgb = roc_auc_score(y_vl, oof_xgb[vl_idx])
    msg = f"  Fold {fold+1}/{N_SPLITS}: LGBM={fold_auc_lgbm:.4f}, XGB={fold_auc_xgb:.4f}"
    if CATBOOST_AVAILABLE and best_cb_params is not None:
        fold_auc_cb = roc_auc_score(y_vl, oof_cb[vl_idx])
        msg += f", CB={fold_auc_cb:.4f}"
    print(msg)

lgbm_auc = roc_auc_score(y, oof_lgbm)
xgb_auc = roc_auc_score(y, oof_xgb)
print(f"\n[OOF] LGBM={lgbm_auc:.6f}, XGB={xgb_auc:.6f}")

# Build candidate predictions list
candidates = [oof_lgbm, oof_xgb]
candidate_aucs = [lgbm_auc, xgb_auc]
candidate_names = ['LGBM', 'XGB']

if CATBOOST_AVAILABLE and best_cb_params is not None:
    cb_auc = roc_auc_score(y, oof_cb)
    print(f"[OOF] CB={cb_auc:.6f}")
    candidates.append(oof_cb)
    candidate_aucs.append(cb_auc)
    candidate_names.append('CB')

# Simple average ensemble
avg_preds = np.mean(candidates, axis=0)
avg_auc = roc_auc_score(y, avg_preds)
print(f"[OOF] Simple average ensemble: {avg_auc:.6f}")

# Weighted average by individual AUC
total_auc = sum(candidate_aucs)
weighted_preds = sum(w * p for w, p in zip([a / total_auc for a in candidate_aucs], candidates))
weighted_auc = roc_auc_score(y, weighted_preds)
weights_str = ", ".join(f"{n}={a/total_auc:.3f}" for n, a in zip(candidate_names, candidate_aucs))
print(f"[OOF] Weighted average ({weights_str}): {weighted_auc:.6f}")

# Stacking meta-learner (cross-validated to avoid overfitting)
stack_X = np.column_stack(candidates)
stack_oof = np.zeros(len(y))
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
for meta_tr_idx, meta_vl_idx in meta_skf.split(stack_X, y):
    meta_clf = LogisticRegression(C=1.0, random_state=42, max_iter=500)
    meta_clf.fit(stack_X[meta_tr_idx], y[meta_tr_idx])
    stack_oof[meta_vl_idx] = meta_clf.predict_proba(stack_X[meta_vl_idx])[:, 1]
stack_auc = roc_auc_score(y, stack_oof)
print(f"[OOF] Stacking (CV meta-learner): {stack_auc:.6f}")

# Pick best
all_preds = candidates + [avg_preds, weighted_preds, stack_oof]
all_aucs = candidate_aucs + [avg_auc, weighted_auc, stack_auc]
all_names = candidate_names + ['SimpleAvg', 'WeightedAvg', 'Stack']
final_auc = max(all_aucs)
method = all_names[all_aucs.index(final_auc)]
print(f"\n[RESULT] Best OOF AUC: {final_auc:.6f} ({method})")
print(f"BEST_VAL_ROC_AUC: {final_auc:.6f}")
