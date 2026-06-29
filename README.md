# Author Attribution and Profiling on Norwegian Parliamentary Debates

This repository contains the data pipeline, model training code, experiment
orchestration, and results support for a master's thesis on authorship
attribution in Norwegian parliamentary debates.

The core task is closed-set authorship attribution: given a parliamentary
speech, predict which selected parliamentarian delivered it. The thesis also
tests whether automatically predicted author-profile signals, such as party,
gender, age group, and broad political bloc, can improve attribution when used
as additional model features. Profiling signals can be produced by the
classical SVM profiling pipeline or by the BERT/NorBERT3 profiling modules.

## Relationship to the Pre-project

This repository builds on code developed during the preceding pre-project. The
pre-project provided the conceptual and experimental starting point for the
thesis, including initial implementations of the preprocessing workflow,
author-selection logic, sparse text and stylometric features, a linear SVM
attribution baseline, and an oracle experiment using ground-truth
author-profile information. Parts of that code were reused or adapted in this
repository.

For the thesis, this foundation was substantially redesigned and extended into
a modular, configuration-driven experimental system. The thesis implementation
adds new split and cross-validation methodologies, leakage and
author-disjointness safeguards, fold-specific feature materialization,
development-time model selection followed by frozen final evaluation, trained
and calibrated author-profiling classifiers, transfer of predicted profiling
signals, direct and stacked attribution architectures, and tooling for
provenance, diagnostics, feature importance, and statistical analysis.

## Dataset and Citation

The raw dataset is not distributed with this repository. The preprocessing
pipeline expects it at `data/NPD_v1.csv`. The data are described in Fiva,
Nedregård, and Øien (2025), [The Norwegian Parliamentary Debates
Dataset](https://doi.org/10.1038/s41597-024-04142-x). The corresponding BibTeX
entry is available in [`CITATION.bib`](CITATION.bib).

## Current Scope

Implemented:

- data cleaning, split creation, row-level feature generation, and
  train/evaluation materialization;
- direct `LinearSVC` attribution models;
- stacked late-fusion attribution models;
- auxiliary profiling classifiers trained on background authors;
- BERT/NorBERT3 profiling classifiers, with single-task and multitask training
  paths plus signal extraction and evaluation;
- predicted probability, predicted hard-label, and oracle profiling variants;
- diagnostics, final evaluation outputs, and significance-test utilities.

Not implemented:

- a dynamic feature-selection phase.

## Experiment Surface

The active experiments use the cleaned Bokmal-only view of the Norwegian
Parliamentary Debates Dataset from the 2001 election onward.

| Surface                   | Current setup                                                    |
| ------------------------- | ---------------------------------------------------------------- |
| Corpus                    | 145,203 speeches, 893 authors, 7 retained parties                |
| Main attribution split    | 50 authors; each author's 100 latest speeches form the test set  |
| Profiling split           | Eligible background authors excluding the 50 attribution authors |
| Temporal robustness split | Train on 2005-2017 election periods; test on 2021                |

The main attribution test set contains each selected author's 100 latest
speeches. Profiling classifiers are trained on background authors, not on the
50 attribution authors.

## Phases

| Phase          | System                                       | Purpose                                           |
| -------------- | -------------------------------------------- | ------------------------------------------------- |
| Phase 1A       | Direct `LinearSVC` attribution               | Main sparse-feature baseline                      |
| Phase 1B       | Stacked attribution without profiling        | Tests learned feature-family fusion               |
| Phase 2        | Profiling classifiers                        | Predicts party, gender, age bin, and party bloc   |
| Phase 3A       | Direct attribution with predicted profiling  | Tests profile signals as ordinary feature blocks  |
| Phase 3B       | Stacked attribution with predicted profiling | Tests profile signals in the stacked architecture |
| Oracle Phase 3 | Attribution with ground-truth profile labels | Upper-bound diagnostic only                       |

Phase 3 uses predicted profile signals by default. Oracle runs use true profile
labels only to estimate an upper bound; they are not the realistic prediction
setting. The root phase runners use the classical profiling workflow by
default. The BERT profiling workflow is run through `models.bert` entry points
and writes compatible profiling matrices for downstream attribution
experiments.

## Setup

The project requires Python 3.12 or newer. Core pipeline dependencies are
declared in `requirements.txt` and `pyproject.toml`.

Create a Python 3.12+ environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download nb_core_news_sm
```

BERT/NorBERT3 profiling additionally requires PyTorch and Hugging Face
Transformers in the active environment. Install the PyTorch build appropriate
for the target machine, then add Transformers, for example:

```bash
python -m pip install torch transformers
```

Run commands from the repository root.

## Common Commands

Run the main phase tracks:

```bash
python run_phase1a_pipeline.py --stage all
python run_phase1b_pipeline.py --stage all
python run_phase2_pipeline.py
python run_phase3a_pipeline.py --stage all
python run_phase3b_pipeline.py --stage all
```

Run hard-label profiling variants:

```bash
python run_phase3a_pipeline.py --stage all --profiling-representation hard
python run_phase3b_pipeline.py --stage all --profiling-representation hard
```

Run oracle profiling variants:

```bash
python run_phase3a_oracle_pipeline.py --stage all
python run_phase3b_oracle_pipeline.py --stage all
```

Run single-signal Phase 3 follow-ups:

```bash
python run_phase3a_pipeline.py --stage all --profiling-scope single_signal
python run_phase3b_pipeline.py --stage all --profiling-scope single_signal
python run_phase3a_pipeline.py --stage all --profiling-representation hard --profiling-scope single_signal
python run_phase3b_pipeline.py --stage all --profiling-representation hard --profiling-scope single_signal
python run_phase3a_oracle_pipeline.py --stage all --profiling-scope single_signal
python run_phase3b_oracle_pipeline.py --stage all --profiling-scope single_signal
```

Run the single-task BERT profiling path after the profiling and attribution
splits exist:

```bash
python -m models.bert.bert_profiling \
    --config models/configs/profiling/bokmal_profiling_bert.toml \
    --mode dev
python -m models.bert.bert_profiling \
    --config models/configs/profiling/bokmal_profiling_bert.toml \
    --mode final
python -m models.bert.bert_profiling_extractor \
    --config models/configs/profiling/bokmal_profiling_bert_signal_extraction.toml
python -m models.bert.bert_profiling_eval \
    --config models/configs/profiling/bokmal_profiling_bert_signal_extraction.toml \
    --output results/profiling_eval/bert
```

For multitask BERT training, use `models.bert.bert_profiling_multitask` with
`models/configs/profiling/bokmal_profiling_bert_multitask.toml`.

Most runners create or reuse split, feature, and materialization prerequisites.
Use `--rebuild` only when you intentionally want to regenerate prerequisite
artifacts.

## Repository Layout

```text
.
|-- data/                     # Raw, cleaned, split, row-feature, and materialized data
|-- data_pipeline/            # Preprocessing plus split, row-feature, materialization packages
|-- models/                   # Model code, configs, artifacts, and profiling modules
|-- pipelines/                # Shared orchestration used by root runners
|-- results/                  # Metrics, summaries, diagnostics, figures, and predictions
|-- scripts/                  # Utilities for significance tests and result reports
|-- thesis_reporting/         # Reusable research-result analysis and rendering
|-- tests/                    # Unit and integration tests
|-- run_data_pipeline.py      # Data-only runner
|-- run_phase1a_pipeline.py   # Direct SVM attribution
|-- run_phase1b_pipeline.py   # Stacked attribution without profiling
|-- run_phase2_pipeline.py    # Profiling workflow and transfer diagnostics
|-- run_phase3a_pipeline.py   # Direct SVM attribution with profiling
|-- run_phase3b_pipeline.py   # Stacked attribution with profiling
|-- run_phase3a_oracle_pipeline.py
`-- run_phase3b_oracle_pipeline.py
```

Machine-readable artifacts live mainly under `data/` and `models/artifacts/`.
Human-facing summaries, diagnostics, predictions, and figures live under
`results/`.

## Key Outputs

| Location                     | Contents                                                  |
| ---------------------------- | --------------------------------------------------------- |
| `data/clean/`                | Cleaned Bokmal, Nynorsk, and majority-language corpora    |
| `data/splits/<split_name>/`  | Split bundles, row features, and materialized matrices    |
| `models/artifacts/`          | Serialized models, preprocessors, and selected candidates |
| `results/models/`            | Development and final model metrics and predictions       |
| `results/profiling_eval/`    | BERT profiling extraction evaluation summaries            |
| `results/profiling_quality/` | Profiling transfer diagnostics and target decisions       |
| `results/splits/`            | Split summaries and reliability diagnostics               |

## Methodological Notes

- TF-IDF vectorizers and stylometry scalers are fit on train rows only.
- Attribution author selection uses train-side support only.
- Final evaluation uses frozen development-selected candidates.
- Profiling authors are disjoint from attribution authors.
- Profiling target decisions do not use attribution test metrics.
- BERT profiling extraction builds fold training rows from attribution
  `train.csv` only, keeping the held-out attribution test set separate until
  final evaluation.
- Row-order files are saved so matrix rows and predictions can be traced back to
  speeches.

## Tests

Run the test suite with:

```bash
python -m pytest tests
```

## More Documentation

| Document                                        | Use                                                 |
| ----------------------------------------------- | --------------------------------------------------- |
| `data/README.md`                                | Data artifact layout and source-data expectations   |
| `data_pipeline/README.md`                       | Detailed data pipeline documentation                |
| `models/README.md`                              | Detailed model phase and artifact documentation     |
| `models/SVM/README.md`                          | Active classical-model implementation               |
| `models/bert/README.md`                         | BERT/NorBERT3 profiling workflow                    |
| `pipelines/README.md`                           | Orchestration layer documentation                   |
| `scripts/README.md`                             | Result-analysis and reporting utilities             |

## License

This project is licensed under the [MIT License](LICENSE). The source dataset
is not covered by this software license; consult the dataset publication and
its distribution terms before obtaining or using the data.
