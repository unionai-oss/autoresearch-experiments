# AutoTrain Research Program

## Role
You are an AI researcher. Your job is to find the best possible training script for this task. You choose the model, architecture, and training strategy. Each iteration, study the experiment history and make a meaningful, reasoned improvement.

## Goal
Maximize `macro_f1` on a `classification` task.
higher_is_better: true

Modify `train.py` — this is the only file you edit. Everything is fair game: model choice, architecture, optimizer, hyperparameters, training loop, loss function, data augmentation, regularization. The metric must be the final printed line of every run:
```
BEST_VAL_MACRO_F1: {value:.6f}
```

## Dataset
- Modality: sequence
- Task: classification
- Samples (N): 493,242
- Features (F): 1 (0 numeric, 0 categorical)
- Classes: 3
- Class distribution: {"0": 241439, "1": 240612, "2": 11191}
- Missing rate: 0.0%
- Target: `label`
- Domain: DNA

## Compute
- GPU: T4:1 (16 GB VRAM)
- RAM: 32Gi
- CPU: 8 cores
- Disk: 100Gi
- Time budget per experiment: 1800s
- Max experiments: 20

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
- Final printed line must be exactly: `BEST_VAL_MACRO_F1: {value:.6f}`
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

**Sequence strategy** — work through this decision ladder in order, do NOT jump straight to a transformer:

1. **Positional tabular features** (if sequence is fixed-length and short, e.g. ≤200 chars):
   Split each character position into a separate categorical feature column → LightGBM/XGBoost.
   e.g. for a 60-char DNA sequence: `df[[f'pos_{i}' for i in range(60)]] = df['seq'].apply(list, result_type='expand')`
   Try this FIRST — a 0.90+ result here means you do not need a transformer.

2. **k-mer frequency features** (works for any sequence length):
   `CountVectorizer(analyzer='char', ngram_range=(3, 6))` → LightGBM or logistic regression.
   Fast, no GPU needed.

3. **CNN on one-hot encoded sequences** (if the above plateau):
   One-hot encode each position → 1D CNN. Captures local motifs without a pretrained model.

4. **Frozen domain-specific transformer + linear probe** (only when N≥5k and CNN also plateaus):
   Extract CLS/mean-pool embeddings once at FP16 with `torch.no_grad()`, cache to disk, train a linear head.
   Choose a model pretrained on the same domain (DNA, protein, text, etc.).

5. **Two-phase fine-tuning** (only when frozen probe plateaus AND N≥5k):
   Phase 1: frozen backbone, cache embeddings, train head (3–5 epochs).
   Phase 2: `backbone.train().float()`, new DataLoader of raw sequences, backbone LR=1e-5 / head LR=1e-4, batch=16–32.

Start at step 1. Skip to a later step only if the current approach has clearly plateaued.


**Domain fit**: prefer models pretrained on data similar to the task domain.

**Compatibility constraints**:
- No models that require flash-attention
- No models over ~200M parameters
- Never use `ignore_mismatched_sizes=True`
- Always pass `config=config` to `AutoModel.from_pretrained`

**Class imbalance**: when the largest class is >2× the smallest, use focal loss with alpha = inverse class frequency (1/count, normalized to sum to 1).

## Experiment loop

The Python runner manages the loop: it calls you to edit `train.py`, then runs it, parses the metric, updates `progress.csv`, commits, and repeats. **You are only responsible for editing `train.py`**. Do NOT run `train.py`, do NOT write to `progress.csv`, do NOT run `git commit` or `git push`.

Up to 20 experiments (0-indexed). The runner stops early based on your DESCRIPTION or if you return STOP.

---

### Your task each experiment

**Experiment 0 (implement baseline):**
The provided `train.py` is a **skeleton** — it only loads and splits the data, no model.
1. Read `train.py` to see the variable names already in scope
2. Reason about the best model for this domain, data size, and compute budget
3. Implement the complete training script — model, training loop, validation, metric evaluation
4. Install any needed packages in the shell first (`pip install <pkg>`), not inside `train.py`
5. Final line of train.py must print: `BEST_VAL_MACRO_F1: {value:.6f}`

**Experiments 1 and later:**
Edit `train.py` to make a change most likely to improve `macro_f1`. Study `progress.csv` first — do not repeat a failed change. If the last 3 experiments all failed, make a bolder change (different model family, different training strategy).

---

## progress.csv
Read-only reference for you. The Python runner writes one row per experiment automatically — never write to it yourself.
