# Data Pipeline

This folder contains the preprocessing, split creation, row-level feature
generation, and materialization code for the project.

The data pipeline turns the raw Norwegian Parliamentary Debates Dataset into
cleaned corpora, split bundles, reusable row-level features, and fold-correct
model matrices. Model training lives outside this folder.

## Stages

| Stage | Main module | Purpose |
| --- | --- | --- |
| Preprocessing | `preprocessing.py` | Clean the raw corpus and write `data/clean/*.csv` |
| Dataset summaries | `dataset_stats.py` | Write corpus-level summary tables |
| Split creation | `split/creation.py` | Build attribution, profiling, and temporal split bundles |
| Row features | `row_features/extraction.py` | Orchestrate row metadata, targets, and optional raw stylometry |
| Materialization | `materialization/core.py` | Orchestrate train-only TF-IDF/scalers and sparse matrix writing |

The notebook `00_preprocessing.ipynb` is a wrapper around the preprocessing
module. The module code is the source of truth.

## Entrypoints

Preprocessing is run through `preprocessing.py` or the thin notebook wrapper.
The data-only root runner starts after cleaned corpora already exist and runs
split creation, row-feature generation, and materialization:

```bash
python data_pipeline/preprocessing.py --config data_pipeline/configs/preprocessing/default.toml
python run_data_pipeline.py --preset authorwise --stage all
```

Available presets are:

| Preset | Split/config family |
| --- | --- |
| `authorwise` | Main 50-author Bokmal attribution setup |
| `profiling` | Background-author profiling setup |
| `temporal` | Temporal robustness setup |

Most project runners create or reuse data prerequisites automatically. Use
`--rebuild` only when you intentionally want to regenerate existing generated
artifacts.

## Configs

| Config folder | Contents |
| --- | --- |
| `configs/preprocessing/` | Raw-corpus cleaning choices, language views, redaction lists, and party-code fixes |
| `configs/splits/` | Split definitions for author-wise, profiling, temporal, and smoke runs |
| `configs/features/` | Row-level feature and target definitions |
| `configs/materializations/` | TF-IDF, stylometry scaling, and matrix materialization definitions |

Each config folder has a `_template.toml` with the detailed config contract.

## Outputs

| Location | Contents |
| --- | --- |
| `data/clean/` | Cleaned Bokmal, Nynorsk, and majority-language corpora |
| `data/splits/<split_name>/` | Split bundles, row features, materialized matrices, labels, row order, and manifests |
| `results/dataset/` | Recombined cleaned-corpus summaries |
| `results/dataset_bokmal/` | Bokmal-only corpus summaries |
| `results/splits/<split_name>/` | Split diagnostics and author/fold summaries |
| `results/features/<split_name>/` | Feature-generation diagnostics |

## Methodological Contracts

- Preprocessing is the only stage that reads `data/NPD_v1.csv`.
- Split creation consumes cleaned corpora from `data/clean/`.
- Row-level feature generation may compute per-speech metadata, targets, and raw
  stylometry once per speech.
- TF-IDF vocabularies, IDF weights, zero-variance decisions, and stylometry
  scalers are fit during materialization on train rows only.
- Materialization writes separate feature-block matrices rather than one
  permanent combined matrix.
- Row-order files are part of the artifact contract and preserve traceability
  from matrix rows back to speeches and authors.

## Important Files

| File | Use |
| --- | --- |
| `preprocessing.py` | Raw corpus cleaning and output writing |
| `split/context.py` | Shared split input loading and validation |
| `split/authorwise.py` | Main author-wise attribution split logic |
| `split/profiling.py` | Background-author profiling split logic |
| `split/temporal.py` | Election-based temporal split logic |
| `split/selection.py` | Author eligibility and selection helpers |
| `split/writer.py` | Split bundle writing |
| `split/reports.py` | Human-readable split summary tables |
| `split/reliability.py` | Per-author/fold support and reliability diagnostics |
| `split/imbalance.py` | Split class-imbalance diagnostics |
| `row_features/extraction.py` | Row-feature runner and manifest writing |
| `row_features/targets.py` | Row metadata, target, age-bin, and party-axis helpers |
| `row_features/stylometry.py` | spaCy/TextDescriptives stylometry extraction |
| `row_features/quality.py` | Stylometry quality and low-variance reports |
| `materialization/core.py` | Materialization runner |
| `materialization/config.py` | Stage and config resolution |
| `materialization/inputs.py` | Split, row-feature, and target loading/alignment |
| `materialization/units.py` | Fold/final materialization unit selection |
| `materialization/blocks.py` | Matrix, label, row-order, and feature-column writers |
| `materialization/reports.py` | Stylometry column/drift and target summaries |

## Tests

Related tests live under `tests/`, including preprocessing, split creation,
feature extraction, and materialization coverage.
