# AutoTrain Research Program

## Role
You are an AI researcher. Your job is to find the best possible training script for this task. You choose the model, architecture, and training strategy. Each iteration, study the experiment history and make a meaningful, reasoned improvement.

## Goal
Maximize `accuracy` on a `classification` task.
higher_is_better: true

Modify `train.py` — this is the only file you edit. Everything is fair game: model choice, architecture, optimizer, hyperparameters, training loop, loss function, data augmentation, regularization. The metric must be the final printed line of every run:
```
BEST_VAL_ACCURACY: {value:.6f}
```

## Dataset
- Modality: sequence
- Task: classification
- Samples (N): 20,000
- Features (F): 1 (0 numeric, 0 categorical)
- Classes: 0
- Class distribution: {}
- Missing rate: 0.0%
- Target: `Survived`
- Domain: auto

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
- Final printed line must be exactly: `BEST_VAL_ACCURACY: {value:.6f}`
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

**Sequence strategy** — two sub-types; pick the right ladder based on the sequence type:

---

### Biological sequences (DNA, RNA, protein — characters are ACGT / amino acids)

Work up this ladder, but skip straight to the tier specified in **"Baseline starting point"**:

1. **k-mer TF-IDF** (ngram_range=(3,6)) → LightGBM: fast baseline for N<5k or short fixed-length sequences.
2. **1D-CNN on one-hot**: only if k-mer clearly underperforms and N<5k.
3. **Frozen domain-specific language model + linear probe** (N≥5k — start here if N≥5k):
   - DNA/nucleotide → `zhihan1996/DNABERT-2-117M` or `InstaDeepAI/nucleotide-transformer-v2-100m`
   - Protein → `facebook/esm2_t6_8M_UR50D`
   - Extract CLS or mean-pool embeddings at FP16 with `torch.no_grad()`, cache to disk, train a linear head.
   - **Never use a general NLP model (DistilBERT, RoBERTa) on biological sequences** — they have no biological pretraining and will produce random embeddings.
4. **Two-phase fine-tuning** (frozen probe plateau AND N≥5k): `backbone.train().float()`, backbone LR=1e-5 / head LR=1e-4, batch=16–32.

---

### Natural language / NLP text sequences (human-readable text — product descriptions, reviews, reports, etc.)

1. **TF-IDF + LightGBM**: reasonable for N<5k; use `analyzer='word'` for normal text, `analyzer='char_wb'` for noisy/short text.
2. **Frozen pre-trained text transformer + linear probe** (N≥5k — start here if N≥5k):
   - General text → `distilbert-base-uncased` or `roberta-base`
   - Domain-specific text → check for a domain-pretrained BERT (e.g. `ProsusAI/finbert` for finance, `allenai/scibert_scivocab_uncased` for science)
   - Extract CLS embeddings at FP16 with `torch.no_grad()`, max_length=128–256, cache to disk, train a linear head.
3. **Full fine-tuning** (frozen probe plateau AND N≥5k): unfreeze all layers, LR=2e-5, warmup 10%, cosine decay.

---

**Key rule**: never use a text NLP model (DistilBERT, BERT, RoBERTa) on biological sequences, and never use a biological model (DNABERT-2, ESM-2) on natural language text. The pretraining domain must match the data.


**Domain fit**: prefer models pretrained on data similar to the task domain.

**Compatibility constraints**:
- No models that require flash-attention
- No models over ~200M parameters
- Never use `ignore_mismatched_sizes=True`
- Always pass `config=config` to `AutoModel.from_pretrained`

**Class imbalance**: when the largest class is >2× the smallest, use focal loss with alpha = inverse class frequency (1/count, normalized to sum to 1).

## Baseline starting point
Tier C: frozen bert-base-uncased or roberta-base + linear probe. With N=20,000 samples and avg sequence length of 762 chars, the dataset comfortably meets the Tier C threshold (N≥5k) for NLP text, making TF-IDF approaches (Tiers A/B) suboptimal for capturing contextual semantics at this scale. The 'auto' domain lacks a well-established domain-pretrained BERT, so a general-purpose transformer (roberta-base) with a frozen encoder and linear probe is the strongest starting baseline before committing to full fine-tuning. Start here directly in experiment 0 — do not begin at a simpler tier.

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
5. Final line of train.py must print: `BEST_VAL_ACCURACY: {value:.6f}`

**Experiments 1 and later:**
Edit `train.py` to make a change most likely to improve `accuracy`. Study `progress.csv` first — do not repeat a failed change. If the last 3 experiments all failed, make a bolder change (different model family, different training strategy).

---

## progress.csv
Read-only reference for you. The Python runner writes one row per experiment automatically — never write to it yourself.
