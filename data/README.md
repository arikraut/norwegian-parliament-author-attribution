# Data

This folder contains machine-readable data artifacts used by the pipeline and modeling code. If a later stage needs to load it as an input, it belongs here. Human-facing summaries and figures belong under `../results/` instead.

## Source Dataset

The raw dataset is not distributed with this repository. Place it at
`data/NPD_v1.csv` before running preprocessing. The source dataset is described
in Fiva, Nedregård, and Øien (2025), [The Norwegian Parliamentary Debates
Dataset](https://doi.org/10.1038/s41597-024-04142-x). A reusable BibTeX entry is
provided in [`../CITATION.bib`](../CITATION.bib). The repository's MIT License
applies to the software, not to the source dataset.

## Responsibility

`data/` is the storage layer for:

- the raw source corpus
- cleaned corpora derived from that source
- split bundles that define the author universe and row memberships
- reusable row-level feature bundles
- materialized train/eval matrices and labels consumed by models

The folder is organized so later stages can load explicit artifacts rather than reconstructing state from notebook output or in-memory variables.

## Top-Level Structure

- `NPD_v1.csv`
    - The raw source corpus.
    - This file is treated as immutable input.
- `clean/`
    - Cleaned corpora produced by `data_pipeline/00_preprocessing.ipynb`.
    - These are reusable starting points for split creation.
- `splits/`
    - One generated split bundle per split config version.
    - This is the main working area for experiment-specific data artifacts.

## Cleaned Corpora

`clean/` stores corpus variants after preprocessing, for example:

- Bokmal-only
- Nynorsk-only
- majority-language-only or combined filtered variants

Why this layer exists:

- raw cleaning and language filtering should happen once
- later split experiments should reuse the same cleaned source instead of reimplementing preprocessing choices

These cleaned files are consumed by `data_pipeline.split.creation` and the repo-root pipeline runners described in [../data_pipeline/README.md](../data_pipeline/README.md).

## Split Bundles

Each directory under `splits/<split_name>/` is a complete data bundle for one split definition. The `<split_name>` comes from the split config and is part of the experiment contract.

Typical structure:

```text
splits/<split_name>/
  split_config.toml
  manifest.json
  authors.csv
  corpus/
  memberships/
  row_features/
  materialized_features/
```

### Split root files

- `split_config.toml`
    - Copy of the config that created the split.
    - Used for provenance and reproducibility.
- `manifest.json`
    - High-level summary of the split, counts, kept and dropped folds, support-policy thresholds, and output paths.
    - Used to inspect the split without loading all CSVs.
- `authors.csv`
    - The final selected author universe for the split, with selection metadata and merged author-level stats.
    - This file defines the closed-set classification task used downstream.

In current code, `authors.csv` is not just an ID list. It also carries fields such as:

- `selection_metric_value` and `rank_in_party`
- author metadata such as `name`, `female`, `partyname`, `language_main`, and age summaries
- outer-role support totals such as `train_chars`, `test_chars`, and the corresponding speech counts
- full-corpus totals such as `total_chars_all` and `total_speeches_all`

### What `outer_role` means

`outer_role` is the top-level split assignment for a speech inside one split bundle.

It answers the question:

- is this speech part of the outer `train` or outer `test` partition?

Each speech gets exactly one `outer_role`.

In current code, the only supported outer roles are:

- `train`
- `test`

There is no outer `val` role in the split contract. Validation is represented only by `fold_role = val` inside `memberships/folds.csv`.

This is different from `fold_role`, which is the speech's role inside an inner validation fold. A speech can have:

- one `outer_role`
- zero, one, or several fold rows in `memberships/folds.csv`, depending on the split strategy

Concrete example:

- `id_speech = 613330.0` has `outer_role = train`
    - it appears in `corpus/train.csv`
    - it also appears in `memberships/outer.csv` with `outer_role = train`
    - it appears in `memberships/folds.csv` as `fold_01, fold_role = train`
- the same speech could also appear as `fold_02, fold_role = val`
    - that would still not change its `outer_role`
    - `fold_role` is reassigned per fold; `outer_role` is fixed once per split bundle
- `id_speech = 703316.0` has `outer_role = test`
    - it appears in `corpus/test.csv`
    - it appears in `memberships/outer.csv` with `outer_role = test`
    - it does not appear in `memberships/folds.csv`, because inner folds are used for development-time train/validation work, not for the held-out outer test set

### `corpus/`

`corpus/` stores the actual speech rows for the selected authors:

- `all.csv`: all selected speeches in one file
- `train.csv`, `test.csv`: outer split partitions

Here, "outer split" means the first-level experiment partition defined by the split config. In other words, these files are the speech rows grouped by `outer_role`.

Why these files exist:

- feature generation needs the speech text and metadata directly
- the explicit outer-role files make inspection easier and remove ambiguity about what was included where at the top split level

### `memberships/`

`memberships/` stores authoritative row-to-role mappings:

- `outer.csv`: one row per speech with its top-level `outer_role`
- `folds.csv`: one row per speech per inner fold with fold-level `train` or `val` assignment

Why this directory exists:

- downstream code should not infer membership from filenames or notebook logs
- fold coverage and dropped-fold decisions need to be explicit and auditable

### `row_features/<feature_set_name>/`

This directory is created by feature generation and holds reusable per-speech outputs:

- `feature_config.toml`
- `manifest.json`
- `row_meta.csv`
- `targets.csv`

When stylometry is enabled in the feature config, it also contains:

- `stylometry_raw.csv.gz`
- `stylometry_quality_report.csv`
- `stylometry_low_variance_report.csv`

What these files are for:

- `row_meta.csv`: traceability and analysis metadata for each speech
- `targets.csv`: labels for attribution and later profiling tasks
- `stylometry_raw.csv.gz`: reusable document-level stylometry vectors, when generated
- quality and variance reports: auditability before stylometry is trusted in models, when generated

Why this bundle exists:

- these outputs do not depend on fold-specific corpus statistics
- they can therefore be reused across multiple materialization experiments

### `materialized_features/<materialization_name>/`

This directory is created by fold materialization and contains train-correct model inputs.

At the materialization root:

- `materialization_config.toml`
- `manifest.json`, including target class-count metadata
- `stylometry_drift_summary.csv` when the `stylo` block is enabled
- derived block metadata after model-side signal extraction or oracle injection,
  such as `profiling`, `profiling_hard`, and `profiling_oracle` blocks

Per materialization unit in `<unit_id>/`:

- `preprocessors/`: fitted vectorizers and, when the `stylo` block is enabled, the stylometry scaler
- `matrices/`: `X_train_*` and `X_val_*` or `X_test_*` feature blocks
- `labels/`: `y_train_*` and `y_val_*` or `y_test_*` arrays for each target
- `row_order/`: `train_rows.csv` and `val_rows.csv` or `test_rows.csv` for exact matrix traceability
- `feature_columns.json`: ordered feature names for each block
- `stylometry_column_report.csv`: keep or drop decisions for stylometry columns, when the `stylo` block is enabled
- `manifest.json`

Why this bundle exists:

- TF-IDF vocabularies and scalers must be fit on fold-train only
- each train/eval unit therefore needs its own fitted preprocessing and its own saved matrices
- later model code should be able to load a fold directory without recomputing anything

Phase 2 signal extraction and oracle injection can add model-side derived
matrices to these same unit directories. Predicted profiling blocks contain
calibrated probability vectors, hard-label profiling blocks contain one-hot
argmax labels derived from those probabilities, and oracle blocks contain
ground-truth profile labels for upper-bound experiments.

## What Later Stages Use From `data/`

- split creation produces the initial split root, `corpus/`, and `memberships/`
- feature generation adds `row_features/`
- materialization adds `materialized_features/`
- model training reads `materialized_features/` matrices, row-order files, raw
  label arrays, and any registered profiling-derived blocks requested by the
  model config

The flow is cumulative: each stage extends the split bundle rather than creating a completely separate data tree.

## Key Conventions

- Split names, feature-set names, and materialization names come from versioned configs and should be treated as stable identifiers.
- The selected author universe is fixed at split-creation time. Later stages must respect it rather than silently redefining the task.
- Row-order files are part of the contract. They are what let you map model rows back to speeches.
- The outer test partition may exist in the data bundle during development, but it should not be used to drive model or preprocessing choices.
- Model selection is done with fold-level validation only. Outer `test` remains untouched, and there is no separate outer `val`.

## Relationship To Other Folders

- The code that writes these artifacts lives in [../data_pipeline/README.md](../data_pipeline/README.md).
- The model code that consumes the materialized outputs lives in [../models/README.md](../models/README.md).
- Human-facing summaries and figures derived from these artifacts live under `../results/`.

## Tracking

This folder mainly contains raw or generated artifacts and is usually not tracked in git in full. Lightweight manifests or copied configs may still appear in version control when they are useful for review or reporting, but the folder should be understood primarily as the local experiment data store.
