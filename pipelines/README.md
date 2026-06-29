# pipelines/

Importable orchestration for the repo-root phase runners lives here.

The root scripts are the user-facing entry points:

- `../run_data_pipeline.py`
- `../run_phase1a_pipeline.py`
- `../run_phase1b_pipeline.py`
- `../run_phase2_pipeline.py`
- `../run_phase3a_pipeline.py`
- `../run_phase3b_pipeline.py`
- `../run_phase3a_oracle_pipeline.py`
- `../run_phase3b_oracle_pipeline.py`

`pipelines/tracks.py` is the shared, testable layer those scripts call. It creates or
reuses phase prerequisites, delegates work to the data/model modules, and writes a
top-level manifest under `results/pipelines/<pipeline_name>/manifest.json`.
Attribution phase manifests expose selected dev candidates and final summaries
under regular artifact keys: `selected_candidates_path` and
`final_condition_summary_path`.

## Public Functions

- `run_data_pipeline(...)`
  - Data-only split creation, row-feature generation, and materialization.
- `run_phase1a_track(...)`
  - Phase 1A baseline attribution. Supports the `authorwise` and `temporal`
    presets.
- `run_phase1b_track(...)`
  - Phase 1B stacked attribution.
- `run_phase2_track(...)`
  - Complete Phase 2 profiling workflow: profiling training, final profilers, attribution signal extraction, and transfer diagnostics.
- `run_phase3a_track(...)`
  - Phase 3A baseline attribution with predicted profiling signals. Use `profiling_representation="hard"` for hard-label signals.
- `run_phase3b_track(...)`
  - Phase 3B stacked attribution with predicted profiling signals. Use `profiling_representation="hard"` for hard-label signals.
- `run_phase3a_oracle_track(...)`
  - Phase 3A baseline attribution with ground-truth oracle profiling signals.
- `run_phase3b_oracle_track(...)`
  - Phase 3B stacked attribution with ground-truth oracle profiling signals.

These are exported from `pipelines/__init__.py`.

## Runner Flags

The repo-root attribution runners share these common controls:

- `--stage {all,dev,final}` runs development search, final evaluation, or both.
- `--config <path>` overrides the staged model config.
- `--selected-candidates-path <path>` supplies frozen dev selections for final
  evaluation.
- `--rebuild` reruns stages even when manifests already exist.
- `--skip-diagnostics` skips attribution diagnostics after model stages.
- `--top-confusions <n>` controls how many confusion pairs final diagnostics keep.
- `--smoke` runs small dev-only smoke configs where supported.

Phase 3 predicted-profile runners also accept
`--profiling-representation {probability,hard}`.

`run_data_pipeline.py` has data-specific controls for preset selection, config
overrides, materialization stage selection, rebuilds, and the pipeline manifest
name.

## Scope

This package handles orchestration:

- resolve preset config paths
- create or reuse split and row-feature prerequisites
- create or reuse materialization stages needed by orchestration-only steps
- call staged model runners and profiling utilities
- run phase-level diagnostics, including dev selection diagnostics and final
  prediction diagnostics for attribution phases
- write top-level phase manifests

Implementation details live in the packages that perform the work:

- split creation, row-feature generation, and materialization: `../data_pipeline/`
- attribution model staging: `../models/SVM/training/attribution_stages.py`
- direct SVM training: `../models/SVM/training/train_svm_attribution.py`
- stacked attribution training: `../models/SVM/training/train_stacked_attribution.py`
- profiling classifier training: `../models/SVM/training/train_profiling_classifiers.py`
- profiling signal extraction: `../models/SVM/signals/profiling_signal_extractor.py`
- diagnostics internals: `../models/SVM/diagnostics/`

Do not put model-training logic, split logic, feature extraction internals, or large
analysis routines in this package.

## Smoke Runs

Smoke attribution configs are staged configs and are run through the same phase functions
with `smoke=True`.

Phase 2 smoke remains a profiling-only development utility. Phase 3 predicted
profile smoke runs first create/reuse those small profiling models, then the
staged attribution runner extracts only the profiling targets requested by the
model config.

Oracle Phase 3 smoke runs do not use Phase 2 profilers. They inject
ground-truth profiling blocks from labels already present in the attribution
materialization.
