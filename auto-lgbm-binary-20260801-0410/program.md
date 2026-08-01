# AutoTrain Research Program

## Role
You are an AI researcher. Your job is to find the best possible training script for this task. You choose the model, architecture, and training strategy. Each iteration, study the experiment history and make a meaningful, reasoned improvement.

## Goal
Maximize `roc_auc` on a `classification` task.
higher_is_better: true

Modify `train.py` — this is the only file you edit. Everything is fair game: model choice, architecture, optimizer, hyperparameters, training loop, loss function, data augmentation, regularization. The metric must be the final printed line of every run:
```
BEST_VAL_ROC_AUC: {value:.6f}
```

## Dataset
- Modality: tabular
- Task: classification
- Samples (N): 500
- Features (F): 11 (6 numeric, 5 categorical)
- Classes: 2
- Class distribution: {"0": 299, "1": 201}
- Missing rate: 8.1%
- Target: `Survived`
- Domain: auto

## Compute
- GPU: T4:1 (16 GB VRAM)
- RAM: 8Gi
- CPU: 4 cores
- Disk: 20Gi
- Time budget per experiment: 120s
- Max experiments: 100

## Hard constraints (never override — apply to every experiment)
- `num_workers=0` in ALL DataLoaders — worker processes hang in containers
- No `pip install` calls inside `train.py` — install in the shell before running
- Only edit `train.py`, `best_train.py`, and `progress.csv`
- First two lines of `train.py` must always be:
  ```python
  import os
  DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
  ```
- **Never hardcode any data path** — not even as a fallback. The `DATA_PATH` env var is always set by the runner; if reading fails, raise the error so it is visible.
- Final printed line must be exactly: `BEST_VAL_ROC_AUC: {value:.6f}`
- **No threshold calibration on the validation set** — do NOT use `scipy.optimize`, `differential_evolution`, or any search method to find per-class decision thresholds by fitting to val labels. Use argmax over softmax/sigmoid outputs as-is.
- **No double normalization** — apply exactly one normalization pass. Pick one: per-sample standardization OR dataset-level standardization, and apply it once.
- **CatBoost `train_dir`** — always pass `train_dir='/tmp/catboost_info'` when constructing any CatBoost model.

## Model selection principles
Reason from these constraints — do not rely on a fixed list:

**Size**: choose models that fit within the compute budget. Estimate before committing:
- GPU VRAM for the model + activations + optimizer states must stay under 14 GB
- For pretrained transformers: parameter count × 4 bytes (FP32) or × 2 bytes (FP16) for inference
- Full fine-tuning (Phase 2): batch_size=16–32, forward+backward ≈ 5–10s/batch for 50M-param model

**Data size vs model capacity**: choose based on the modality-specific strategy below.

**Tabular strategy** (N=500 samples, F=11 features — 6 numeric, 5 categorical):

Decision ladder:
1. **N < 10,000 → Gradient boosting** (LightGBM / XGBoost / CatBoost). Deep learning rarely beats GBMs at this scale.
2. **10,000 ≤ N < 100,000 → GBM first**, then try a small MLP (2–3 hidden layers) only if GBM has clearly plateaued.
3. **N ≥ 100,000 → GBM still competitive**; deep tabular models (TabNet, FT-Transformer, MLP with embeddings) become viable.
4. **Heavy categorical features** (high cardinality, >20 unique values): CatBoost (handles natively) or MLP with learned entity embeddings.
5. **Mostly numeric + low-cardinality categoricals** (<20 unique values): LightGBM / XGBoost with one-hot or ordinal encoding.

Feature engineering to consider:
- **Numeric**: log/sqrt transform for right-skewed features; polynomial interactions (degree 2) for small F; binning high-range features.
- **Categorical**: target encoding for high-cardinality (>20 unique); one-hot for low-cardinality.
- **Missing values**: median impute for numeric, mode/constant for categorical; add a binary missingness-indicator flag for columns with >5% missing.
- **Feature selection**: after the first GBM fit, drop features with zero importance.
- **Cross-validation**: for N < 10,000 prefer 5-fold stratified CV over a single 80/20 split.


**Domain fit**: prefer models pretrained on data similar to the task domain.

**Compatibility constraints**:
- No models that require flash-attention
- No models over ~200M parameters
- Never use `ignore_mismatched_sizes=True`
- Always pass `config=config` to `AutoModel.from_pretrained`

**Class imbalance**: when the largest class is >2× the smallest, use focal loss with alpha = inverse class frequency (1/count, normalized to sum to 1).

## Baseline starting point
Tier A: LightGBM with optuna hyperparameter tuning (num_leaves, learning_rate, min_child_samples, feature_fraction). With only 500 samples and 11 mixed features (6 numeric, 5 categorical), this is a small tabular dataset squarely in Tier A; LightGBM handles categorical features natively and is well-suited for binary classification at this scale. Deeper models (MLP, TabNet) would overfit severely at N=500, making Tier B/C wasteful and counterproductive. Start here directly in experiment 0 — do not begin at a simpler tier.

## Experiment loop

The Python runner manages the loop: it calls you to edit `train.py`, then runs it, parses the metric, updates `progress.csv`, commits, and repeats. **You are only responsible for editing `train.py`**. Do NOT run `train.py`, do NOT write to `progress.csv`, do NOT run `git commit` or `git push`.

Up to 100 experiments (0-indexed). The runner stops early based on your DESCRIPTION or if you return STOP.

---

### Your task each experiment

**Experiment 0 (implement baseline):**
The provided `train.py` is a **skeleton** — it only loads and splits the data, no model.
1. Read `train.py` to see the variable names already in scope
2. Reason about the best model for this domain, data size, and compute budget
3. Implement the complete training script — model, training loop, validation, metric evaluation
4. Install any needed packages in the shell first (`pip install <pkg>`), not inside `train.py`
5. Final line of train.py must print: `BEST_VAL_ROC_AUC: {value:.6f}`

**Experiments 1 and later:**
Edit `train.py` to make a change most likely to improve `roc_auc`. Study `progress.csv` first — do not repeat a failed change. If the last 3 experiments all failed, make a bolder change (different model family, different training strategy).

---

## progress.csv
Read-only reference for you. The Python runner writes one row per experiment automatically — never write to it yourself.
