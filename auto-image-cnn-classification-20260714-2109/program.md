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
- Modality: image
- Task: classification
- Samples (N): 16,200
- Features (F): 0 (0 numeric, 0 categorical)
- Classes: 10
- Class distribution: {"Annual Crop": 1791, "Forest": 1787, "Herbaceous Vegetation": 1799, "Highway": 1505, "Industrial Buildings": 1492, "Pasture": 1195, "Permanent Crop": 1481, "Residential Buildings": 1863, "River": 1460, "SeaLake": 1827}
- Missing rate: 0.0%
- Target: `label`
- Domain: auto

## Compute
- GPU: T4:1 (16 GB VRAM)
- RAM: 32Gi
- CPU: 8 cores
- Disk: 100Gi
- Time budget per experiment: 1800s
- Max experiments: 10

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

**Image strategy** (N=16,200 samples, native resolution and channels detected at runtime by the data skeleton):

Variables already in scope from the skeleton: `dataset` (ImageFolder), `train_idx`, `val_idx`, `native_h`, `native_w`, `img_channels`, `num_classes`, `train_transform`, `val_transform`. Build DataLoaders with `SubsetRandomSampler(train_idx)` / `SubsetRandomSampler(val_idx)` — do NOT re-split.

Scale-based strategy:
- **N < 1,000** — freeze backbone completely. Extract embeddings once (FP16, no_grad, batch=256), cache to disk. Train a sklearn LogisticRegression or SVC on the cached embeddings.
- **1,000 ≤ N < 10,000** — lightweight ImageNet-pretrained backbone (EfficientNet-B0, MobileNetV3-Small, ResNet18). Two-phase: Phase 1 — freeze backbone, train linear head (10–15 epochs, LR=1e-3). Phase 2 — unfreeze all, cosine LR schedule (backbone LR=1e-4, head LR=1e-3, 20–30 epochs).
- **10,000 ≤ N < 50,000** — lightweight to medium backbone (EfficientNet-B0/B2, ResNet34). Full fine-tuning or short 5-epoch frozen warm-up. Aggressive augmentation (random crop, flip, color jitter, random erasing).
- **N ≥ 50,000** — heavier backbone (EfficientNet-B4, ResNet50). Full fine-tuning with mixed precision (`torch.cuda.amp.autocast`).

Backbone selection:
- Load from `timm` (`timm.create_model(name, pretrained=True, num_classes=num_classes)`). Prefer small backbones for small N.
- For domain-specific images (medical, satellite, histology): check for domain-pretrained models (BiT, RadImageNet) — they transfer better than ImageNet weights.
- Do NOT use backbones >50M parameters with N<10k.
- Always use `native_h` and `native_w` from the skeleton in ALL transforms — never hardcode a resolution.

Resolution and channels:
- Use `transforms.Resize((native_h, native_w))` or `transforms.Resize(min(native_h, native_w))` + `CenterCrop`.
- If `img_channels == 1` (grayscale): add `transforms.Grayscale(num_output_channels=1)` and set the backbone's first conv `in_channels=1` (or use `Grayscale(num_output_channels=3)` with standard weights).
- Normalize with ImageNet mean/std for ImageNet-pretrained backbones: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

Augmentation (training only — val_transform must be resize + normalize only):
- Standard: `RandomHorizontalFlip`, `RandomCrop` with padding, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.
- For small N: also add `RandomRotation(15)` and `RandomErasing(p=0.25)`.

Class imbalance: use `WeightedRandomSampler` in the training DataLoader (weight = inverse class frequency) rather than SMOTE.


**Domain fit**: prefer models pretrained on data similar to the task domain.

**Compatibility constraints**:
- No models that require flash-attention
- No models over ~200M parameters
- Never use `ignore_mismatched_sizes=True`
- Always pass `config=config` to `AutoModel.from_pretrained`

**Class imbalance**: when the largest class is >2× the smallest, use focal loss with alpha = inverse class frequency (1/count, normalized to sum to 1).

## Experiment loop

The Python runner manages the loop: it calls you to edit `train.py`, then runs it, parses the metric, updates `progress.csv`, commits, and repeats. **You are only responsible for editing `train.py`**. Do NOT run `train.py`, do NOT write to `progress.csv`, do NOT run `git commit` or `git push`.

Up to 10 experiments (0-indexed). The runner stops early based on your DESCRIPTION or if you return STOP.

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
