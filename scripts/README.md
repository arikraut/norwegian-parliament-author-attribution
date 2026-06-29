# scripts/

Small utility scripts live here. Root pipeline execution does not.

Reusable result analysis and rendering live in `../thesis_reporting/`; the
files in this directory only resolve CLI paths and orchestrate those modules.

Use the repo-root phase runners for split/feature/materialization/model workflows:

- `../run_data_pipeline.py`
- `../run_phase1a_pipeline.py`
- `../run_phase1b_pipeline.py`
- `../run_phase2_pipeline.py`
- `../run_phase3a_pipeline.py`
- `../run_phase3b_pipeline.py`

See [../README.md](../README.md) for the phase order and dependency graph.

## Utilities

### `build_results_additions.py`

Build research-result analysis tables from local artifacts. The script
reads existing CSV diagnostics from `results/` and writes
derived outputs under `results/result_additions/`. It does not run any
preprocessing, split creation, feature generation, materialization, or model
training.

```bash
python -m scripts.build_results_additions \
  --sections all \
  --significance-bootstrap 10000
```

Implemented sections:

- `author_performance`, which writes per-author rankings, best/worst author
  tables, and error-concentration summaries.
- `confusions`, which writes directed author confusions, symmetric author-pair
  confusions, party-level confusion summaries, normalized confusion matrices,
  and curated heatmap copies.
- `profiling_effects`, which writes per-author deltas for baseline,
  predicted-profile, oracle-profile, and direct-versus-stacked comparisons.
- `topk_confidence`, which writes top-k rescue summaries, score-margin
  summaries, confident-error rows, and uncertain correct rows.
- `profile_quality`, which writes profiling-target metrics, target confusion
  pairs, confusion/confidence summaries, confident profile errors, and
  attribution-versus-profile correctness tables.
- `significance`, which writes pairwise bootstrap macro-F1 and McNemar
  summaries for configured final-system comparisons.
- `feature_importance`, which optionally reads saved model artifacts and writes
  feature-importance copies outside the original model directories.

Use `--sections feature_importance` or `--sections all_with_feature_importance`
when feature importance should be included.

### `build_results_additions_feature_importance.py`

Build feature-importance reports from saved final model artifacts.
This is also available through `build_results_additions.py`, but the standalone
entry point remains useful because it loads model artifacts through the existing
feature-importance helpers.

```bash
python -m scripts.build_results_additions_feature_importance
```

The script reads local final manifests from `results/`, follows their canonical
paths into `results/` and `data/`, runs the condition-level importance helpers,
and copies outputs into `results/result_additions/feature_importance/`.

### `plot_authorwise_training_timeline.py`

Render the author-wise training support figures from the configured split
corpus. By default it reads
`data_pipeline/configs/splits/bokmal_authorwise.toml`, then uses that config's
`splits_dir` and split name to find `corpus/train.csv`. Figures are written to
`results/figures/splits/`.

```bash
python -m scripts.plot_authorwise_training_timeline
```

### `plot_temporal_training_timeline.py`

Render the temporal training and retained-test support figures from the
configured split corpus. By default it reads
`data_pipeline/configs/splits/bokmal_temporal.toml`, then uses that config's
`splits_dir`, split name, and `source_dataset` paths. Figures are written to
`results/figures/splits/`.

```bash
python -m scripts.plot_temporal_training_timeline
```

### `run_significance_tests.py`

Pairwise final-system comparison using bootstrap macro F1 intervals and McNemar's test.
Use it after final prediction files exist for two systems evaluated on the same held-out
speeches.

```bash
python -m scripts.run_significance_tests \
  --system-a results/models/.../final_by_condition/<condition_a>/final_test_predictions.csv \
  --system-b results/models/.../final_by_condition/<condition_b>/final_test_predictions.csv \
  --label-a "Phase 1A" \
  --label-b "Phase 3A"
```

The script verifies that both files contain the same `id_speech` set and the same `y_true`
labels before computing statistics.

### `report_profiling_fold_balance.py`

Create the profiling validation-fold balance report from existing split and
target artifacts. It reads existing inputs and does not recreate the split or
feature files. By default it writes
`results/reports/profiling_fold_balance/<split>.md`.

```bash
python -m scripts.report_profiling_fold_balance
```
