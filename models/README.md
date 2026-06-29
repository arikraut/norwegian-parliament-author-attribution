# Models

This folder contains model configs, the classical-model implementation, the
BERT/NorBERT3 profiling implementation, and serialized model artifacts.

Model code starts after the data pipeline has produced split bundles,
row-level features, and materialized train/evaluation matrices. Raw data
cleaning, split creation, feature extraction, and materialization happen under
`data_pipeline/` and `data/`.

## Folder Layout

```text
models/
|-- README.md
|-- SVM/                 # Classical-model package; see SVM/README.md
|-- artifacts/           # Serialized model artifacts, usually not tracked
|-- bert/                # BERT/NorBERT3 profiling trainers, extractor, and eval
`-- configs/             # Human-written model configs
```

Human-facing metrics, predictions, diagnostics, and reports are written under
`results/`. Machine-facing serialized models and preprocessors are written under
`models/artifacts/`.

## Key Configs

Primary config families:

- `configs/attribution/bokmal_authorwise_linear_svm*.toml`: direct attribution
  config family.
- `configs/attribution/stacked/bokmal_authorwise_stacked*.toml`: stacked
  attribution config family.
- `configs/profiling/bokmal_profiling_linear_svm.toml`: profiling classifier
  config.
- `configs/profiling/bokmal_profiling_signal_extraction.toml`: profiling signal
  extraction config.
- `configs/profiling/bokmal_ground_truth_signal_injection.toml`: oracle signal
  injection config.
- `configs/profiling/bokmal_profiling_bert.toml`: single-task BERT/NorBERT3
  profiling config.
- `configs/profiling/bokmal_profiling_bert_multitask.toml`: multitask
  BERT/NorBERT3 profiling config.
- `configs/profiling/bokmal_profiling_bert_signal_extraction.toml`: BERT
  profiling signal extraction and evaluation config.

See [configs/attribution/_template.toml](configs/attribution/_template.toml)
and [configs/profiling/_template.toml](configs/profiling/_template.toml) for
config conventions, stage-specific fields, and parallel-worker notes.

## Inputs

Model code consumes materialized feature roots from `data/splits/`:

```text
data/splits/<split_name>/materialized_features/<materialization_name>/
|-- manifest.json
`-- <unit_id>/
    |-- labels/
    |-- matrices/
    |-- preprocessors/
    `-- row_order/
```

Row-order files, labels, sparse matrices, and fitted preprocessors are produced
by materialization and read by the model package.

## Outputs

Development attribution runs usually write:

```text
results/models/<split_name>/<experiment_name>/seed_<seed>/
|-- candidate_summary.csv, condition_summary.csv, fold_metrics.csv
|-- selected_candidates.json, resolved_attribution_run_spec.json, manifest.json
`-- diagnostics/
```

Final attribution runs usually write:

```text
results/models/<split_name>/<experiment_name>/seed_<seed>/
|-- final_condition_summary.csv, selected_candidates.json, manifest.json
|-- final_by_condition/<condition_id>/
|   `-- final_test_metrics.json, final_test_predictions.csv, resolved_candidate.json
`-- diagnostics/
```

Serialized model artifacts are written under:

```text
models/artifacts/attribution/<split_name>/<experiment_name>/seed_<seed>/
models/artifacts/profiling/<split_name>/<experiment_name>/seed_<seed>/
```

BERT profiling dev and final runs write metrics under
`results/models/<profiling_split>/<experiment_name>/seed_<seed>/` and saved
Hugging Face models under
`models/artifacts/profiling/<profiling_split>/<experiment_name>/seed_<seed>_final/`.
BERT signal extraction writes compatible profiling matrices under
`data/splits/<attribution_split>/materialized_features/<materialization_name>/`.

Profiling-transfer reports are written under `results/profiling_quality/`.
BERT extraction quality summaries are written wherever
`models.bert.bert_profiling_eval --output` points, commonly
`results/profiling_eval/bert`.
