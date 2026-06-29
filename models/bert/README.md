# BERT Profiling Models

BERT is used exclusively for **profiling**: training party, gender, age, and
political-axis classifiers on a speaker-disjoint profiling split, then
extracting probability signals for use in downstream SVM attribution models.

Two training approaches are available:
- **STL** (`bert_profiling.py`) — one independent BERT per target
- **MTL** (`bert_profiling_multitask.py`) — one shared BERT encoder with per-target heads trained jointly

The pipeline has four stages: dev search → final training → signal extraction
→ evaluation.

---

## Files

| File | Role |
|---|---|
| `bert_profiling.py` | STL: train one BERT per target (dev search + final) |
| `bert_profiling_multitask.py` | MTL: train one shared BERT for all targets jointly (dev search + final) |
| `bert_profiling_extractor.py` | Apply saved profiling models to attribution speeches → probability matrices |
| `bert_profiling_eval.py` | Score the extracted probability matrices against ground-truth labels |
| `bert_utils.py` | Shared: dataset (head+tail truncation, `SpeechDataset`), training loop, label derivation, author sampling |

---

## Runtime dependencies

The repository's base `requirements.txt` supports the data and classical SVM
pipelines. The BERT modules also require PyTorch and Hugging Face Transformers:

```bash
python -m pip install torch transformers
```

Use a PyTorch build appropriate for the machine where the models will train.
The checked-in configs use `ltg/norbert3-base`, so the environment must be able
to download that model or already have it available in the Hugging Face cache.

---

## Prerequisites

### Profiling split

```
data/splits/<profiling_split>/
    corpus/train.csv         outer-train speeches with demographic labels (used by dev search + final training)
    memberships/folds.csv    author-disjoint fold assignments (folds are carved out of train.csv only)
```

### Attribution split (signal extraction + eval)

```
data/splits/<attribution_split>/
    corpus/train.csv         outer-train speeches (extractor: fold train/val + final_test's train)
    corpus/test.csv          held-out test speeches (extractor + eval: final_test's eval role)
    corpus/all.csv           all speeches (eval only: ground-truth lookup for train/val splits)
    memberships/folds.csv    fold assignments
```

`train.csv` and `test.csv` must never overlap — the extractor builds every
unit's training data from `train.csv` alone so the held-out test speeches are
never seen during training or used as context for a fold's validation rows.

---

## STL — Single-task training

A single config lists every target in `[source].targets`; each target is
trained as an independent model so runs can be parallelized or rerun per
target without affecting the others.

### Step 1 — Dev epoch search

```bash
python -m models.bert.bert_profiling \
    --config models/configs/profiling/bokmal_profiling_bert.toml \
    --mode dev
```

Completed `(fold, epoch)` pairs are skipped automatically on re-run.

Output at `results/models/<profiling_split>/<experiment>/seed_<seed>/<target>/`:

```
best_epoch.json          best epoch + mean val metrics + majority baseline
epoch_summary.csv        one row per epoch: mean/std across folds
<config>.toml            config copy
fold_01/ … fold_05/
    epoch_metrics.csv    per-epoch val metrics for this fold
    params.json          hyperparameter snapshot (used for resume detection)
```

### Step 2 — Final training

Reads each target's `best_epoch.json` from the dev run to determine how many
epochs to train for, then trains on the full `train.csv` (no held-out split).

```bash
python -m models.bert.bert_profiling \
    --config models/configs/profiling/bokmal_profiling_bert.toml \
    --mode final
```

Saved models:
```
models/artifacts/profiling/<profiling_split>/<experiment>/seed_<seed>_final/final/
    <target>/saved_model/              HuggingFace model weights + tokenizer
    <target>/saved_model/label_map.json    {"0": "female", "1": "male", …}
```

---

## MTL — Multi-task training

All targets are trained jointly on one shared encoder. A separate subdirectory
is created per target combination so different combinations can coexist.

### Step 1 — Dev epoch search

```bash
python -m models.bert.bert_profiling_multitask \
    --config models/configs/profiling/bokmal_profiling_bert_multitask.toml \
    --mode dev
```

Output at `results/models/<profiling_split>/<experiment>/seed_<seed>/<targets_key>/`:

```
best_epoch.json          shared best epoch + per-task metrics + majority baselines
epoch_summary.csv        one row per epoch: per-task mean/std + combined mean F1
<config>.toml
fold_01/ … fold_05/
    epoch_metrics.csv    per-epoch metrics for all tasks in this fold
    params.json
```

`<targets_key>` is the sorted target names joined by `+`, e.g. `age_bin+female+left_right`.
The best epoch is chosen by the combined mean F1 across all tasks and folds.

### Step 2 — Final training

```bash
python -m models.bert.bert_profiling_multitask \
    --config models/configs/profiling/bokmal_profiling_bert_multitask.toml \
    --mode final
```

Reads `best_epoch.json` from the dev run. Each task head is exported as a
standalone `AutoModelForSequenceClassification`:

```
models/artifacts/profiling/<profiling_split>/<experiment>/seed_<seed>_final/<targets_key>/final/
    <target>/saved_model/
    <target>/saved_model/label_map.json
```

> **Caveat:** the extractor (Step 3 below) always looks for final models at
> `<artifacts_dir>/<profiling_split>/<profiling_experiment_name>/seed_<seed>_final/final/<target>/saved_model`
> — it has no `<targets_key>` path segment. MTL's final output nests an extra
> `<targets_key>` directory that the extractor does not account for, so
> pointing an extraction config at an MTL experiment will currently fail to
> find the models unless `profiling_experiment_name` is set to
> `"<experiment_name>/<targets_key>"` to absorb the extra segment.

---

## Step 3 — Signal extraction

Applies saved profiling models to attribution speeches. Works the same way
regardless of whether models came from STL or MTL training (modulo the path
caveat above). A single run always produces both the CV fold units (train +
val per fold) and the final unit (full train + test) — there is no
`--mode` flag.

```bash
python -m models.bert.bert_profiling_extractor \
    --config models/configs/profiling/bokmal_profiling_bert_signal_extraction.toml
```

Output at `data/splits/<attribution_split>/materialized_features/<materialization_name>/`:

```
manifest.json                         targets, column_names, per-unit row/dim counts
fold_01/ … fold_05/
    row_order/
        train_rows.csv
        val_rows.csv
    matrices/
        X_{train|val}_profiling.npz                combined across all targets (soft probs)
        X_{train|val}_profiling_hard.npz            combined across all targets (one-hot argmax)
        X_{train|val}_profiling_<target>.npz        per-target soft probs
        X_{train|val}_profiling_hard_<target>.npz   per-target one-hot argmax
final_test/
    row_order/
        train_rows.csv
        test_rows.csv
    matrices/                         same layout as above, with {train|test} roles
```

`--no-progress` suppresses the per-target/per-unit progress lines.

---

## Step 4 — Evaluation

Scores the extracted probability matrices against ground-truth demographic
labels for both the CV val folds and the final held-out test set. Takes the
same config as the extractor.

```bash
python -m models.bert.bert_profiling_eval \
    --config models/configs/profiling/bokmal_profiling_bert_signal_extraction.toml \
    --output results/profiling_eval/bert
```

Output at `<output>/`:

```
per_split_metrics.csv     one row per (target, split): accuracy, macro F1, per-class F1, majority baselines
summary_metrics.csv       aggregated: val_mean, train_fold_mean (across folds), train_final, test
```

Splits reported: `train_fold_XX` / `val_fold_XX` per CV fold, `train_final`
(full outer-train set), and `test` (held-out outer-test set). A target is
silently skipped for a split if its label map or ground-truth column isn't
found — check the printed `[split] target: ...` lines to confirm all expected
targets were scored.

---

## Configs

| Config | Use |
|---|---|
| `bokmal_profiling_bert.toml` | STL — one independent NorBERT3 model per target listed in `[source].targets` |
| `bokmal_profiling_bert_multitask.toml` | MTL — one shared NorBERT3 encoder across all listed targets |
| `bokmal_profiling_bert_signal_extraction.toml` | Signal extraction (Step 3) and evaluation (Step 4) |

---

## Truncation

Speeches longer than `max_length` tokens are truncated to:

```
[CLS]  first head_tokens  …  last tail_tokens  [SEP]
```

Padding is appended after `[CLS]`/`[SEP]` are added, up to `max_length`.
Configured via `[truncation] pairs = [[head, tail]]` in the config. The code
assumes `head + tail == max_length - 2`.
