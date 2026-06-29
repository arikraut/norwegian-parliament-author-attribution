"""Shared orchestration for the top-level data and phase pipeline runners."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from data_pipeline.materialization import (
    resolve_materialization_stage,
    run_materialization,
)
from data_pipeline.row_features import run_feature_generation
from data_pipeline.split.creation import run_split_creation
from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.SVM.diagnostics.attribution_diagnostics import (
    run_dev_attribution_selection_diagnostics,
    run_final_attribution_diagnostics,
)
from models.SVM.signals.profiling_signal_extractor import (
    resolve_extraction_stage_source,
    run_profiling_signal_extraction,
)
from models.SVM.diagnostics.profiling_transfer_diagnostics import (
    profiling_quality_manifest_path,
    run_profiling_transfer_diagnostics,
)
from models.SVM.training.train_profiling_classifiers import (
    run_final_profiling_training,
    run_profiling_experiment,
)
from models.SVM.training.attribution_stages import run_attribution_model


JsonDict = dict[str, Any]
StageRunner = Callable[[], JsonDict]
ReuseValidator = Callable[[JsonDict], bool]


ATTRIBUTION_SMOKE_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_authorwise_smoke.toml")
ATTRIBUTION_SMOKE_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_authorwise_smoke.toml")
ATTRIBUTION_SMOKE_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_authorwise_smoke.toml"
)

ATTRIBUTION_DEV_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_authorwise.toml")
ATTRIBUTION_DEV_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_authorwise.toml")
ATTRIBUTION_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_authorwise_char_word_stylo.toml"
)

BASELINE_SMOKE_CONFIG = Path("models/configs/attribution/bokmal_authorwise_smoke_linear_svm.toml")
BASELINE_SMOKE_WITH_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_smoke_linear_svm_with_profiling.toml"
)
BASELINE_CONFIG = Path("models/configs/attribution/bokmal_authorwise_linear_svm.toml")
BASELINE_WITH_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_profiling.toml"
)
BASELINE_WITH_SINGLE_SIGNAL_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_single_signal_profiling.toml"
)
BASELINE_WITH_HARD_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_hard_profiling.toml"
)
BASELINE_WITH_SINGLE_SIGNAL_HARD_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_single_signal_hard_profiling.toml"
)
TEMPORAL_BASELINE_CONFIG = Path("models/configs/attribution/bokmal_temporal_linear_svm.toml")
TEMPORAL_SMOKE_BASELINE_CONFIG = Path("models/configs/attribution/bokmal_temporal_smoke_linear_svm.toml")

STACKED_SMOKE_BASELINE_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_smoke_stacked.toml"
)
STACKED_SMOKE_WITH_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_smoke_stacked_with_profiling.toml"
)
STACKED_BASELINE_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked.toml"
)
STACKED_WITH_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_profiling.toml"
)
STACKED_WITH_SINGLE_SIGNAL_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_single_signal_profiling.toml"
)
STACKED_WITH_HARD_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_hard_profiling.toml"
)
STACKED_WITH_SINGLE_SIGNAL_HARD_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_single_signal_hard_profiling.toml"
)

PROFILING_SMOKE_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_profiling_smoke.toml")
PROFILING_SMOKE_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_profiling_smoke.toml")
PROFILING_SMOKE_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_profiling_smoke.toml"
)
PROFILING_SMOKE_MODEL_CONFIG = Path("models/configs/profiling/bokmal_profiling_smoke_linear_svm.toml")

PROFILING_DEV_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_profiling.toml")
PROFILING_DEV_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_profiling.toml")
PROFILING_DEV_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_profiling.toml"
)
PROFILING_DEV_MODEL_CONFIG = Path("models/configs/profiling/bokmal_profiling_linear_svm.toml")
PROFILING_EXTRACTION_CONFIG = Path(
    "models/configs/profiling/bokmal_profiling_signal_extraction.toml"
)

BASELINE_WITH_ORACLE_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_oracle_profiling.toml"
)
BASELINE_WITH_SINGLE_SIGNAL_ORACLE_PROFILING_CONFIG = Path(
    "models/configs/attribution/bokmal_authorwise_linear_svm_with_single_signal_oracle_profiling.toml"
)
STACKED_WITH_ORACLE_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_oracle_profiling.toml"
)
STACKED_WITH_SINGLE_SIGNAL_ORACLE_PROFILING_CONFIG = Path(
    "models/configs/attribution/stacked/bokmal_authorwise_stacked_with_single_signal_oracle_profiling.toml"
)

TEMPORAL_DEV_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_temporal.toml")
TEMPORAL_DEV_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_temporal.toml")
TEMPORAL_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_temporal_char_word_stylo.toml"
)
TEMPORAL_SMOKE_SPLIT_CONFIG = Path("data_pipeline/configs/splits/bokmal_temporal_smoke.toml")
TEMPORAL_SMOKE_FEATURE_CONFIG = Path("data_pipeline/configs/features/bokmal_temporal_smoke.toml")
TEMPORAL_SMOKE_MATERIALIZATION_CONFIG = Path(
    "data_pipeline/configs/materializations/bokmal_temporal_smoke.toml"
)

DATA_PIPELINE_PRESETS = {
    "authorwise": {
        "split_config": ATTRIBUTION_DEV_SPLIT_CONFIG,
        "feature_config": ATTRIBUTION_DEV_FEATURE_CONFIG,
        "materialization_config": ATTRIBUTION_MATERIALIZATION_CONFIG,
    },
    "temporal": {
        "split_config": TEMPORAL_DEV_SPLIT_CONFIG,
        "feature_config": TEMPORAL_DEV_FEATURE_CONFIG,
        "materialization_config": TEMPORAL_MATERIALIZATION_CONFIG,
    },
    "profiling": {
        "split_config": PROFILING_DEV_SPLIT_CONFIG,
        "feature_config": PROFILING_DEV_FEATURE_CONFIG,
        "materialization_config": PROFILING_DEV_MATERIALIZATION_CONFIG,
    },
    "authorwise-smoke": {
        "split_config": ATTRIBUTION_SMOKE_SPLIT_CONFIG,
        "feature_config": ATTRIBUTION_SMOKE_FEATURE_CONFIG,
        "materialization_config": ATTRIBUTION_SMOKE_MATERIALIZATION_CONFIG,
    },
    "temporal-smoke": {
        "split_config": TEMPORAL_SMOKE_SPLIT_CONFIG,
        "feature_config": TEMPORAL_SMOKE_FEATURE_CONFIG,
        "materialization_config": TEMPORAL_SMOKE_MATERIALIZATION_CONFIG,
    },
    "profiling-smoke": {
        "split_config": PROFILING_SMOKE_SPLIT_CONFIG,
        "feature_config": PROFILING_SMOKE_FEATURE_CONFIG,
        "materialization_config": PROFILING_SMOKE_MATERIALIZATION_CONFIG,
    },
}


@dataclass(frozen=True)
class AttributionPhaseDefinition:
    pipeline_name: str
    phase_label: str
    model_config: Path
    split_config: Path
    feature_config: Path
    dev_only: bool = False
    profiling_smoke_prerequisite: bool = False


ATTRIBUTION_PHASES: dict[str, AttributionPhaseDefinition] = {
    "phase1a": AttributionPhaseDefinition(
        pipeline_name="phase1a_baseline",
        phase_label="Phase 1A",
        model_config=BASELINE_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase1a_temporal": AttributionPhaseDefinition(
        pipeline_name="phase1a_temporal",
        phase_label="Phase 1A temporal",
        model_config=TEMPORAL_BASELINE_CONFIG,
        split_config=TEMPORAL_DEV_SPLIT_CONFIG,
        feature_config=TEMPORAL_DEV_FEATURE_CONFIG,
    ),
    "phase1a_smoke": AttributionPhaseDefinition(
        pipeline_name="phase1a_smoke",
        phase_label="Phase 1A smoke",
        model_config=BASELINE_SMOKE_CONFIG,
        split_config=ATTRIBUTION_SMOKE_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_SMOKE_FEATURE_CONFIG,
        dev_only=True,
    ),
    "phase1a_temporal_smoke": AttributionPhaseDefinition(
        pipeline_name="phase1a_temporal_smoke",
        phase_label="Phase 1A smoke",
        model_config=TEMPORAL_SMOKE_BASELINE_CONFIG,
        split_config=TEMPORAL_SMOKE_SPLIT_CONFIG,
        feature_config=TEMPORAL_SMOKE_FEATURE_CONFIG,
        dev_only=True,
    ),
    "phase1b": AttributionPhaseDefinition(
        pipeline_name="phase1b_stacked",
        phase_label="Phase 1B",
        model_config=STACKED_BASELINE_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase1b_smoke": AttributionPhaseDefinition(
        pipeline_name="phase1b_smoke",
        phase_label="Phase 1B smoke",
        model_config=STACKED_SMOKE_BASELINE_CONFIG,
        split_config=ATTRIBUTION_SMOKE_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_SMOKE_FEATURE_CONFIG,
        dev_only=True,
    ),
    "phase3a_probability": AttributionPhaseDefinition(
        pipeline_name="phase3a_baseline_with_profiling",
        phase_label="Phase 3A",
        model_config=BASELINE_WITH_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3a_probability_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3a_baseline_with_single_signal_profiling",
        phase_label="Phase 3A single-signal",
        model_config=BASELINE_WITH_SINGLE_SIGNAL_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3a_hard": AttributionPhaseDefinition(
        pipeline_name="phase3a_baseline_with_hard_profiling",
        phase_label="Phase 3A hard",
        model_config=BASELINE_WITH_HARD_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3a_hard_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3a_baseline_with_single_signal_hard_profiling",
        phase_label="Phase 3A hard single-signal",
        model_config=BASELINE_WITH_SINGLE_SIGNAL_HARD_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3a_probability_smoke": AttributionPhaseDefinition(
        pipeline_name="phase3a_smoke",
        phase_label="Phase 3A smoke",
        model_config=BASELINE_SMOKE_WITH_PROFILING_CONFIG,
        split_config=ATTRIBUTION_SMOKE_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_SMOKE_FEATURE_CONFIG,
        dev_only=True,
        profiling_smoke_prerequisite=True,
    ),
    "phase3b_probability": AttributionPhaseDefinition(
        pipeline_name="phase3b_stacked_with_profiling",
        phase_label="Phase 3B",
        model_config=STACKED_WITH_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_probability_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3b_stacked_with_single_signal_profiling",
        phase_label="Phase 3B single-signal",
        model_config=STACKED_WITH_SINGLE_SIGNAL_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_hard": AttributionPhaseDefinition(
        pipeline_name="phase3b_stacked_with_hard_profiling",
        phase_label="Phase 3B hard",
        model_config=STACKED_WITH_HARD_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_hard_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3b_stacked_with_single_signal_hard_profiling",
        phase_label="Phase 3B hard single-signal",
        model_config=STACKED_WITH_SINGLE_SIGNAL_HARD_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_probability_smoke": AttributionPhaseDefinition(
        pipeline_name="phase3b_smoke",
        phase_label="Phase 3B smoke",
        model_config=STACKED_SMOKE_WITH_PROFILING_CONFIG,
        split_config=ATTRIBUTION_SMOKE_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_SMOKE_FEATURE_CONFIG,
        dev_only=True,
        profiling_smoke_prerequisite=True,
    ),
    "phase3a_oracle": AttributionPhaseDefinition(
        pipeline_name="phase3a_oracle_baseline_with_oracle_profiling",
        phase_label="Phase 3A oracle",
        model_config=BASELINE_WITH_ORACLE_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3a_oracle_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3a_oracle_baseline_with_single_signal_oracle_profiling",
        phase_label="Phase 3A oracle single-signal",
        model_config=BASELINE_WITH_SINGLE_SIGNAL_ORACLE_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_oracle": AttributionPhaseDefinition(
        pipeline_name="phase3b_oracle_stacked_with_oracle_profiling",
        phase_label="Phase 3B oracle",
        model_config=STACKED_WITH_ORACLE_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
    "phase3b_oracle_single_signal": AttributionPhaseDefinition(
        pipeline_name="phase3b_oracle_stacked_with_single_signal_oracle_profiling",
        phase_label="Phase 3B oracle single-signal",
        model_config=STACKED_WITH_SINGLE_SIGNAL_ORACLE_PROFILING_CONFIG,
        split_config=ATTRIBUTION_DEV_SPLIT_CONFIG,
        feature_config=ATTRIBUTION_DEV_FEATURE_CONFIG,
    ),
}


def _load_toml(config_path: Path) -> JsonDict:
    """Load a TOML config file into the orchestration dictionary format."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _read_json(path: Path) -> JsonDict:
    """Load a JSON manifest written by a pipeline stage."""
    return json.loads(path.read_text(encoding="utf-8"))


def _split_manifest_path(project_root: Path, config_path: Path) -> Path:
    """Return the expected manifest path for a split config."""
    config = _load_toml(config_path)
    split_name = str(config["split"]["name"])
    splits_dir = resolve_project_path(
        project_root,
        config.get("data", {}).get("splits_dir", "data/splits"),
    )
    return splits_dir / split_name / "manifest.json"


def _feature_manifest_path(project_root: Path, config_path: Path) -> Path:
    """Return the expected manifest path for a row-feature config."""
    config = _load_toml(config_path)
    feature_cfg = config["feature"]
    split_name = str(feature_cfg["split_name"])
    feature_name = str(feature_cfg["name"])
    splits_dir = resolve_project_path(
        project_root,
        config.get("data", {}).get("splits_dir", "data/splits"),
    )
    return splits_dir / split_name / "row_features" / feature_name / "manifest.json"


def _materialization_manifest_path(project_root: Path, config_path: Path, *, stage: str) -> Path:
    """Return the expected manifest path for one materialization config stage."""
    resolved = resolve_materialization_stage(project_root, config_path, stage=stage)
    return Path(resolved["materialized_root"]) / "manifest.json"


def _materialization_stages_for_request(config_path: Path, requested_stage: str) -> list[str]:
    """Resolve a user materialization stage request to concrete configured stages."""
    stage_name = str(requested_stage).strip().lower()
    if stage_name not in {"all", "dev", "final"}:
        raise ValueError("Data pipeline stage must be one of: all, dev, final.")

    config = _load_toml(config_path)
    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("Materialization config must define [stages.dev] and/or [stages.final].")

    available = [stage for stage in ("dev", "final") if stage in stages]
    if stage_name == "all":
        return available
    if stage_name not in available:
        raise ValueError(
            f"Materialization config does not define stage {stage_name!r}. "
            f"Available stages: {available}"
        )
    return [stage_name]


def _model_manifest_path(project_root: Path, config_path: Path) -> Path:
    """Return the expected manifest path for a direct profiling model config."""
    config = _load_toml(config_path)
    experiment_name = str(config.get("experiment", {}).get("name", config_path.stem))
    seed = int(config.get("experiment", {}).get("seed", 42))
    split_name = str(config["source"]["split_name"])
    results_dir = resolve_project_path(
        project_root,
        config.get("data", {}).get("results_dir", "results/models"),
    )
    return results_dir / split_name / experiment_name / f"seed_{seed}" / "manifest.json"


def _profiling_final_manifest_path(project_root: Path, config_path: Path) -> Path:
    """Return the expected final-training manifest path for a profiling config."""
    return _model_manifest_path(project_root, config_path).with_name("final_manifest.json")


def _extraction_manifest_path(
    project_root: Path,
    config_path: Path,
    *,
    stage: str = "dev",
) -> Path:
    """Return the expected profiling-signal extraction manifest path."""
    config = _load_toml(config_path)
    source_cfg = resolve_extraction_stage_source(config, stage=stage)
    splits_dir = resolve_project_path(
        project_root,
        config.get("data", {}).get("splits_dir", "data/splits"),
    )
    return (
        splits_dir
        / str(source_cfg["attribution_split_name"])
        / "materialized_features"
        / str(source_cfg["attribution_materialization_name"])
        / "profiling_extraction_manifest.json"
    )


def _stage_result(project_root: Path, status: str, manifest_path: Path, summary: JsonDict) -> JsonDict:
    """Wrap a stage manifest summary with reuse/execution status for pipeline manifests."""
    return {
        "status": status,
        "manifest_path": relative_to_project(project_root, manifest_path),
        "summary": summary,
    }


def _run_json_stage(
    *,
    project_root: Path,
    label: str,
    config_path: Path,
    manifest_path: Path,
    runner: StageRunner,
    rebuild: bool,
    reuse_validator: ReuseValidator | None = None,
) -> JsonDict:
    """Run or reuse a filesystem-manifest stage in a root pipeline."""
    if not rebuild and manifest_path.exists():
        existing = _read_json(manifest_path)
        if reuse_validator is None or reuse_validator(existing):
            print(f"== {label}: reused {manifest_path.relative_to(project_root)} ==")
            return _stage_result(project_root, "reused", manifest_path, existing)

    print(f"== {label}: {config_path.relative_to(project_root)} ==")
    runner()
    payload = _read_json(manifest_path) if manifest_path.exists() else {}
    return _stage_result(project_root, "executed", manifest_path, payload)


def _split_manifest_fold_count(manifest: JsonDict) -> int:
    """Return the current split-manifest fold count."""
    return int(manifest.get("fold_count") or 0)


def _require_split_folds(
    project_root: Path,
    split_config: Path,
    split_stage: JsonDict,
) -> None:
    """Fail early when a dev attribution phase has no fold materialization units."""
    summary = split_stage.get("summary", {})
    manifest = summary if isinstance(summary, dict) else {}
    if _split_manifest_fold_count(manifest) > 0:
        return

    dropped_folds = manifest.get("dropped_folds", [])
    dropped_fold_ids = []
    if isinstance(dropped_folds, list):
        dropped_fold_ids = [
            str(record.get("fold_id"))
            for record in dropped_folds
            if isinstance(record, dict) and record.get("fold_id")
        ]
    dropped_detail = (
        f" Dropped folds: {', '.join(dropped_fold_ids)}."
        if dropped_fold_ids
        else ""
    )
    raise ValueError(
        f"{relative_to_project(project_root, split_config)} produced zero dev folds; "
        "cannot run a dev attribution stage. Adjust the split [folds] settings so "
        "memberships/folds.csv contains at least one fold."
        f"{dropped_detail}"
    )


def _run_inline_stage(label: str, runner: StageRunner) -> JsonDict:
    """Run an in-process stage that returns its summary directly."""
    print(f"== {label} ==")
    return {
        "status": "executed",
        "summary": runner(),
    }


def _results_dir_from_manifest(project_root: Path, manifest: JsonDict) -> Path:
    """Resolve the results directory recorded in a model manifest."""
    results_dir = manifest.get("results_dir")
    if not results_dir:
        raise KeyError("Manifest is missing results_dir.")
    return resolve_project_path(project_root, str(results_dir))


def _selected_candidates_path_from_manifest(project_root: Path, manifest: JsonDict) -> Path:
    """Return the condition-aware selected-candidates path for a dev model manifest."""
    selected_path = manifest.get("selected_candidates_path")
    if selected_path:
        return resolve_project_path(project_root, str(selected_path))
    raise KeyError("Dev attribution manifest is missing selected_candidates_path.")


def _manifest_relative_path_exists(project_root: Path, path_value: Any) -> bool:
    """Check whether a manifest path field points to an existing project file."""
    if not path_value:
        return False
    return resolve_project_path(project_root, str(path_value)).exists()


def _profiling_dev_manifest_complete(project_root: Path, manifest: JsonDict) -> bool:
    """Return whether a reused profiling dev manifest has all required model artifacts."""
    results_dir = _results_dir_from_manifest(project_root, manifest)
    targets = [str(target) for target in manifest.get("targets", [])]
    targets_summary = manifest.get("targets_summary", {})
    if not targets or not isinstance(targets_summary, dict):
        return False

    for target in targets:
        if not (results_dir / target / "best_candidate.json").exists():
            return False

        target_summary = targets_summary.get(target, {})
        saved_units = target_summary.get("saved_units", [])
        if not saved_units:
            return False
        for saved_unit in saved_units:
            if not _manifest_relative_path_exists(project_root, saved_unit.get("model_path")):
                return False

    return True


def _profiling_final_manifest_complete(project_root: Path, manifest: JsonDict) -> bool:
    """Return whether a reused profiling final manifest has all required model artifacts."""
    artifacts_dir = resolve_project_path(project_root, manifest.get("artifacts_dir", ""))
    final_dir = artifacts_dir / "final"
    if not (final_dir / "feature_build_meta.json").exists():
        return False

    saved_targets = manifest.get("saved_targets", {})
    if not saved_targets or not isinstance(saved_targets, dict):
        return False
    for target_payload in saved_targets.values():
        if not _manifest_relative_path_exists(project_root, target_payload.get("model_path")):
            return False

    return True


def _write_pipeline_manifest(project_root: Path, pipeline_name: str, payload: JsonDict) -> JsonDict:
    """Write a top-level pipeline manifest under results/pipelines."""
    manifest_path = project_root / "results" / "pipelines" / pipeline_name / "manifest.json"
    payload = dict(payload)
    payload["pipeline_name"] = pipeline_name
    payload["pipeline_manifest_path"] = relative_to_project(project_root, manifest_path)
    write_json(manifest_path, payload)
    return payload


def run_data_pipeline(
    *,
    split_config: Path,
    feature_config: Path,
    materialization_config: Path,
    materialization_stage: str = "all",
    project_root: Path | None = None,
    rebuild: bool = False,
    pipeline_name: str = "data_pipeline",
) -> JsonDict:
    """Run split creation, row-feature generation, and materialization only."""

    project_root = project_root or find_project_root()
    split_config = resolve_project_path(project_root, split_config)
    feature_config = resolve_project_path(project_root, feature_config)
    materialization_config = resolve_project_path(project_root, materialization_config)
    materialization_stages = _materialization_stages_for_request(
        materialization_config,
        materialization_stage,
    )

    stages: JsonDict = {
        "split": _run_json_stage(
            project_root=project_root,
            label="Data split creation",
            config_path=split_config,
            manifest_path=_split_manifest_path(project_root, split_config),
            runner=lambda: run_split_creation(split_config),
            rebuild=rebuild,
        ),
        "features": _run_json_stage(
            project_root=project_root,
            label="Data feature generation",
            config_path=feature_config,
            manifest_path=_feature_manifest_path(project_root, feature_config),
            runner=lambda: run_feature_generation(feature_config),
            rebuild=rebuild,
        ),
        "materializations": {},
    }

    for stage in materialization_stages:
        stages["materializations"][stage] = _run_json_stage(
            project_root=project_root,
            label=f"Data {stage} materialization",
            config_path=materialization_config,
            manifest_path=_materialization_manifest_path(project_root, materialization_config, stage=stage),
            runner=lambda stage=stage: run_materialization(
                materialization_config,
                stage=stage,
                rebuild=rebuild,
                show_progress=True,
            ),
            rebuild=rebuild,
        )

    return _write_pipeline_manifest(
        project_root,
        pipeline_name,
        {
            "kind": "data_pipeline",
            "requested_materialization_stage": str(materialization_stage).strip().lower(),
            "materialization_stages": materialization_stages,
            "configs": {
                "split": relative_to_project(project_root, split_config),
                "features": relative_to_project(project_root, feature_config),
                "materialization": relative_to_project(project_root, materialization_config),
            },
            "stages": stages,
        },
    )


def _phase_stage_name(stage: str) -> str:
    """Normalize a phase stage requested by a root runner."""
    stage_name = str(stage).strip().lower()
    if stage_name not in {"all", "dev", "final"}:
        raise ValueError("Phase stage must be one of: all, dev, final.")
    return stage_name


def _ensure_split_and_features(
    project_root: Path,
    *,
    split_config: Path,
    feature_config: Path,
    label: str,
    rebuild: bool,
    require_folds: bool = False,
) -> JsonDict:
    """Create or reuse the split and row-feature prerequisites for one phase."""
    split_config = resolve_project_path(project_root, split_config)
    feature_config = resolve_project_path(project_root, feature_config)

    split_stage = _run_json_stage(
        project_root=project_root,
        label=f"{label} split creation",
        config_path=split_config,
        manifest_path=_split_manifest_path(project_root, split_config),
        runner=lambda: run_split_creation(split_config),
        rebuild=rebuild,
    )
    if require_folds:
        _require_split_folds(project_root, split_config, split_stage)

    feature_stage = _run_json_stage(
        project_root=project_root,
        label=f"{label} feature generation",
        config_path=feature_config,
        manifest_path=_feature_manifest_path(project_root, feature_config),
        runner=lambda: run_feature_generation(feature_config),
        rebuild=rebuild,
    )

    return {
        "split": split_stage,
        "features": feature_stage,
    }


def _ensure_materialization(
    project_root: Path,
    *,
    materialization_config: Path,
    stage: str,
    label: str,
    rebuild: bool,
) -> JsonDict:
    """Create or reuse one named materialization stage for orchestration-only stages."""
    materialization_config = resolve_project_path(project_root, materialization_config)
    return _run_json_stage(
        project_root=project_root,
        label=label,
        config_path=materialization_config,
        manifest_path=_materialization_manifest_path(project_root, materialization_config, stage=stage),
        runner=lambda: run_materialization(
            materialization_config,
            stage=stage,
            rebuild=rebuild,
            show_progress=True,
        ),
        rebuild=rebuild,
    )


def _run_attribution_phase_diagnostics(
    project_root: Path,
    *,
    phase_label: str,
    model_manifest: JsonDict,
    skip_diagnostics: bool,
    top_confusions: int,
) -> JsonDict:
    """Run diagnostics for any dev and final attribution stages present in a phase manifest."""
    if skip_diagnostics:
        return {}

    diagnostics: JsonDict = {}
    model_stages = model_manifest.get("stages", {})
    if "dev" in model_stages:
        diagnostics["dev_selection_diagnostics"] = _run_inline_stage(
            f"{phase_label} dev selection diagnostics",
            lambda: run_dev_attribution_selection_diagnostics(
                _results_dir_from_manifest(project_root, model_stages["dev"]),
            ),
        )
    if "final" in model_stages:
        diagnostics["final_diagnostics"] = _run_inline_stage(
            f"{phase_label} final diagnostics",
            lambda: run_final_attribution_diagnostics(
                _results_dir_from_manifest(project_root, model_stages["final"]),
                top_confusions=top_confusions,
            ),
        )
    return diagnostics


def _selected_candidates_artifact(project_root: Path, model_manifest: JsonDict) -> str | None:
    """Return the dev selected-candidates path for a phase manifest when dev ran."""
    dev_manifest = model_manifest.get("stages", {}).get("dev")
    if not dev_manifest:
        return None
    return relative_to_project(project_root, _selected_candidates_path_from_manifest(project_root, dev_manifest))


def _run_attribution_phase_track(
    *,
    project_root: Path,
    pipeline_name: str,
    phase_label: str,
    model_config: Path,
    split_config: Path,
    feature_config: Path,
    stage: str,
    rebuild: bool,
    skip_diagnostics: bool,
    top_confusions: int,
    selected_candidates_path_override: Path | None = None,
    prerequisite_stages: JsonDict | None = None,
) -> JsonDict:
    """Run one staged attribution phase with its data prerequisites and diagnostics."""
    stage_name = _phase_stage_name(stage)
    model_config = resolve_project_path(project_root, model_config)
    split_config = resolve_project_path(project_root, split_config)
    feature_config = resolve_project_path(project_root, feature_config)

    stages: JsonDict = {
        "data": _ensure_split_and_features(
            project_root,
            split_config=split_config,
            feature_config=feature_config,
            label=phase_label,
            rebuild=rebuild,
            require_folds=stage_name in {"all", "dev"},
        ),
    }
    if prerequisite_stages:
        stages.update(prerequisite_stages)

    model_manifest = run_attribution_model(
        model_config,
        stage=stage_name,
        rebuild=rebuild,
        show_progress=True,
        selected_candidates_path_override=selected_candidates_path_override,
    )
    stages["model"] = model_manifest
    stages.update(
        _run_attribution_phase_diagnostics(
            project_root,
            phase_label=phase_label,
            model_manifest=model_manifest,
            skip_diagnostics=skip_diagnostics,
            top_confusions=top_confusions,
        )
    )

    artifacts: JsonDict = {}
    selected_candidates_path = _selected_candidates_artifact(project_root, model_manifest)
    if selected_candidates_path is not None:
        artifacts["selected_candidates_path"] = selected_candidates_path
    final_summary_path = model_manifest.get("stages", {}).get("final", {}).get(
        "final_condition_summary_path"
    )
    if final_summary_path:
        artifacts["final_condition_summary_path"] = final_summary_path

    return _write_pipeline_manifest(
        project_root,
        pipeline_name,
        {
            "kind": pipeline_name,
            "phase_label": phase_label,
            "requested_stage": stage_name,
            "configs": {
                "model": relative_to_project(project_root, model_config),
                "split": relative_to_project(project_root, split_config),
                "features": relative_to_project(project_root, feature_config),
            },
            "stages": stages,
            "artifacts": artifacts,
        },
    )


def _profiling_representation_name(profiling_representation: str) -> str:
    """Normalize the Phase 3 profiling representation selector."""
    representation = str(profiling_representation).strip().lower()
    if representation not in {"probability", "hard"}:
        raise ValueError("Profiling representation must be one of: probability, hard.")
    return representation


def _profiling_scope_name(profiling_scope: str) -> str:
    """Normalize the Phase 3 profiling signal scope selector."""
    scope = str(profiling_scope).strip().lower()
    if scope not in {"all", "single_signal"}:
        raise ValueError("Profiling scope must be one of: all, single_signal.")
    return scope


def _run_registered_attribution_phase(
    phase_definition_key: str,
    *,
    project_root: Path,
    stage: str,
    config_path: Path | None,
    rebuild: bool,
    skip_diagnostics: bool,
    top_confusions: int,
    selected_candidates_path_override: Path | None,
) -> JsonDict:
    """Run one attribution phase from the phase registry."""
    definition = ATTRIBUTION_PHASES[phase_definition_key]
    stage_name = _phase_stage_name(stage)
    if definition.dev_only:
        stage_name = "dev"

    prerequisite_stages: JsonDict | None = None
    if definition.profiling_smoke_prerequisite:
        prerequisite_stages = {
            "profiling_smoke": run_profiling_smoke(
                project_root=project_root,
                rebuild=rebuild,
            )
        }

    return _run_attribution_phase_track(
        project_root=project_root,
        pipeline_name=definition.pipeline_name,
        phase_label=definition.phase_label,
        model_config=config_path or definition.model_config,
        split_config=definition.split_config,
        feature_config=definition.feature_config,
        stage=stage_name,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
        prerequisite_stages=prerequisite_stages,
    )


def run_profiling_smoke(
    *,
    project_root: Path | None = None,
    rebuild: bool = False,
) -> JsonDict:
    """Run the small profiling-only smoke path used by Phase 3 smoke checks."""
    project_root = project_root or find_project_root()

    split_config = resolve_project_path(project_root, PROFILING_SMOKE_SPLIT_CONFIG)
    feature_config = resolve_project_path(project_root, PROFILING_SMOKE_FEATURE_CONFIG)
    materialization_config = resolve_project_path(project_root, PROFILING_SMOKE_MATERIALIZATION_CONFIG)
    model_config = resolve_project_path(project_root, PROFILING_SMOKE_MODEL_CONFIG)

    stages: JsonDict = {
        "data": _ensure_split_and_features(
            project_root,
            split_config=split_config,
            feature_config=feature_config,
            label="Phase 2 smoke profiling",
            rebuild=rebuild,
        ),
        "materialization": _ensure_materialization(
            project_root,
            materialization_config=materialization_config,
            stage="dev",
            label="Phase 2 smoke profiling materialization",
            rebuild=rebuild,
        ),
        "training": _run_json_stage(
            project_root=project_root,
            label="Phase 2 smoke profiling training",
            config_path=model_config,
            manifest_path=_model_manifest_path(project_root, model_config),
            runner=lambda: run_profiling_experiment(model_config, show_progress=True),
            rebuild=rebuild,
        ),
    }

    return _write_pipeline_manifest(
        project_root,
        "phase2_smoke",
        {
            "kind": "phase2_smoke",
            "requested_stage": "dev",
            "stages": stages,
        },
    )


def _run_phase2_track_full(project_root: Path, *, rebuild: bool) -> JsonDict:
    """Run the complete Phase 2 profiling workflow."""
    profiling_split_config = resolve_project_path(project_root, PROFILING_DEV_SPLIT_CONFIG)
    profiling_feature_config = resolve_project_path(project_root, PROFILING_DEV_FEATURE_CONFIG)
    profiling_materialization_config = resolve_project_path(
        project_root, PROFILING_DEV_MATERIALIZATION_CONFIG
    )
    profiling_model_config = resolve_project_path(project_root, PROFILING_DEV_MODEL_CONFIG)
    attribution_split_config = resolve_project_path(project_root, ATTRIBUTION_DEV_SPLIT_CONFIG)
    attribution_feature_config = resolve_project_path(project_root, ATTRIBUTION_DEV_FEATURE_CONFIG)
    extraction_config = resolve_project_path(project_root, PROFILING_EXTRACTION_CONFIG)
    attribution_materialization_config = resolve_project_path(
        project_root, ATTRIBUTION_MATERIALIZATION_CONFIG
    )

    stages: JsonDict = {
        "profiling_data": _ensure_split_and_features(
            project_root,
            split_config=profiling_split_config,
            feature_config=profiling_feature_config,
            label="Phase 2 profiling",
            rebuild=rebuild,
        ),
        "profiling_materialization": _ensure_materialization(
            project_root,
            materialization_config=profiling_materialization_config,
            stage="dev",
            label="Phase 2 profiling materialization",
            rebuild=rebuild,
        ),
    }
    stages["profiling_dev_training"] = _run_json_stage(
        project_root=project_root,
        label="Phase 2 profiling dev training",
        config_path=profiling_model_config,
        manifest_path=_model_manifest_path(project_root, profiling_model_config),
        runner=lambda: run_profiling_experiment(profiling_model_config, show_progress=True),
        rebuild=rebuild,
        reuse_validator=lambda manifest: _profiling_dev_manifest_complete(project_root, manifest),
    )
    stages["profiling_final_training"] = _run_json_stage(
        project_root=project_root,
        label="Phase 2 profiling final training",
        config_path=profiling_model_config,
        manifest_path=_profiling_final_manifest_path(project_root, profiling_model_config),
        runner=lambda: run_final_profiling_training(profiling_model_config, show_progress=True),
        rebuild=rebuild,
        reuse_validator=lambda manifest: _profiling_final_manifest_complete(project_root, manifest),
    )
    stages["attribution_data"] = _ensure_split_and_features(
        project_root,
        split_config=attribution_split_config,
        feature_config=attribution_feature_config,
        label="Phase 2 attribution",
        rebuild=rebuild,
    )
    stages["attribution_dev_materialization"] = _ensure_materialization(
        project_root,
        materialization_config=attribution_materialization_config,
        stage="dev",
        label="Phase 2 attribution dev materialization",
        rebuild=rebuild,
    )
    stages["attribution_final_materialization"] = _ensure_materialization(
        project_root,
        materialization_config=attribution_materialization_config,
        stage="final",
        label="Phase 2 attribution final materialization",
        rebuild=rebuild,
    )
    stages["profiling_extraction_dev"] = _run_json_stage(
        project_root=project_root,
        label="Phase 2 dev signal extraction",
        config_path=extraction_config,
        manifest_path=_extraction_manifest_path(project_root, extraction_config, stage="dev"),
        runner=lambda: run_profiling_signal_extraction(
            extraction_config,
            stage="dev",
            show_progress=True,
        ),
        rebuild=rebuild,
    )
    stages["profiling_extraction_final"] = _run_json_stage(
        project_root=project_root,
        label="Phase 2 final signal extraction",
        config_path=extraction_config,
        manifest_path=_extraction_manifest_path(project_root, extraction_config, stage="final"),
        runner=lambda: run_profiling_signal_extraction(
            extraction_config,
            stage="final",
            show_progress=True,
        ),
        rebuild=rebuild,
    )
    stages["profiling_transfer_diagnostics"] = _run_json_stage(
        project_root=project_root,
        label="Phase 2 profiling transfer diagnostics",
        config_path=extraction_config,
        manifest_path=profiling_quality_manifest_path(
            extraction_config,
            project_root=project_root,
        ),
        runner=lambda: run_profiling_transfer_diagnostics(
            extraction_config,
            show_progress=True,
        ),
        rebuild=rebuild,
    )

    diagnostic_summary = stages["profiling_transfer_diagnostics"]["summary"]
    decision = diagnostic_summary.get("decision", {})
    artifacts = {
        "profiling_signal_decision_path": diagnostic_summary.get("outputs", {}).get(
            "profiling_signal_decision"
        ),
        "selected_targets": decision.get("selected_targets", []),
        "excluded_targets": decision.get("excluded_targets", []),
        "decision_basis": decision.get("decision_basis"),
    }
    return _write_pipeline_manifest(
        project_root,
        "phase2_profiling",
        {
            "kind": "phase2_profiling",
            "phase": "phase2",
            "requested_stage": "all",
            "configs": {
                "profiling_split": relative_to_project(project_root, profiling_split_config),
                "profiling_features": relative_to_project(project_root, profiling_feature_config),
                "profiling_materialization": relative_to_project(
                    project_root, profiling_materialization_config
                ),
                "profiling_model": relative_to_project(project_root, profiling_model_config),
                "attribution_split": relative_to_project(project_root, attribution_split_config),
                "attribution_features": relative_to_project(
                    project_root, attribution_feature_config
                ),
                "attribution_materialization": relative_to_project(
                    project_root, attribution_materialization_config
                ),
                "signal_extraction": relative_to_project(project_root, extraction_config),
            },
            "stages": stages,
            "artifacts": artifacts,
        },
    )


def run_phase2_track(
    *,
    project_root: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
) -> JsonDict:
    """Run the complete Phase 2 profiling workflow as an independent phase."""
    project_root = project_root or find_project_root()
    if smoke:
        return run_profiling_smoke(project_root=project_root, rebuild=rebuild)
    return _run_phase2_track_full(project_root, rebuild=rebuild)


def run_phase1a_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    preset: str = "authorwise",
    config_path: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
) -> JsonDict:
    """Run Phase 1A baseline attribution as its own root-runnable phase."""
    project_root = project_root or find_project_root()
    phase_definitions = {
        ("authorwise", False): "phase1a",
        ("temporal", False): "phase1a_temporal",
        ("authorwise", True): "phase1a_smoke",
        ("temporal", True): "phase1a_temporal_smoke",
    }
    phase_definition_key = phase_definitions.get((preset, smoke))
    if phase_definition_key is None:
        raise ValueError(f"Unknown Phase 1A preset: {preset}")
    return _run_registered_attribution_phase(
        phase_definition_key,
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
    )


def run_phase1b_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
) -> JsonDict:
    """Run Phase 1B stacked attribution as its own root-runnable phase."""
    project_root = project_root or find_project_root()
    return _run_registered_attribution_phase(
        "phase1b_smoke" if smoke else "phase1b",
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
    )


def _run_predicted_profiling_track(
    phase_key: str,
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
    profiling_representation: str = "probability",
    profiling_scope: str = "all",
) -> JsonDict:
    """Run a predicted-profiling Phase 3 attribution track."""
    project_root = project_root or find_project_root()
    representation = _profiling_representation_name(profiling_representation)
    scope = _profiling_scope_name(profiling_scope)
    if smoke and representation != "probability":
        raise ValueError(f"{phase_key} smoke is only defined for probability profiling.")
    if smoke and scope != "all":
        raise ValueError(f"{phase_key} smoke is only defined for all-signal profiling.")
    phase_definition_key = f"{phase_key}_{representation}"
    if scope == "single_signal":
        phase_definition_key = f"{phase_definition_key}_single_signal"
    if smoke:
        phase_definition_key = f"{phase_definition_key}_smoke"
    return _run_registered_attribution_phase(
        phase_definition_key,
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
    )


def run_phase3a_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
    profiling_representation: str = "probability",
    profiling_scope: str = "all",
) -> JsonDict:
    """Run Phase 3A baseline attribution with profiling signals as its own phase."""
    return _run_predicted_profiling_track(
        "phase3a",
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        smoke=smoke,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
        profiling_representation=profiling_representation,
        profiling_scope=profiling_scope,
    )


def run_phase3b_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    smoke: bool = False,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
    profiling_representation: str = "probability",
    profiling_scope: str = "all",
) -> JsonDict:
    """Run Phase 3B stacked attribution with profiling signals as its own phase."""
    return _run_predicted_profiling_track(
        "phase3b",
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        smoke=smoke,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
        profiling_representation=profiling_representation,
        profiling_scope=profiling_scope,
    )


def run_phase3a_oracle_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
    profiling_scope: str = "all",
) -> JsonDict:
    """Run Phase 3A oracle attribution with ground-truth profiling labels.

    Skips Phase 2 classifier training. Oracle injection (one-hot labels) is
    handled automatically inside run_attribution_model after materialization.
    """
    project_root = project_root or find_project_root()
    scope = _profiling_scope_name(profiling_scope)
    return _run_registered_attribution_phase(
        "phase3a_oracle" if scope == "all" else "phase3a_oracle_single_signal",
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
    )


def run_phase3b_oracle_track(
    *,
    project_root: Path | None = None,
    stage: str = "all",
    config_path: Path | None = None,
    rebuild: bool = False,
    skip_diagnostics: bool = False,
    top_confusions: int = 50,
    selected_candidates_path_override: Path | None = None,
    profiling_scope: str = "all",
) -> JsonDict:
    """Run Phase 3B oracle stacked attribution with ground-truth profiling labels.

    Skips Phase 2 classifier training. Oracle injection (one-hot labels) is
    handled automatically inside run_attribution_model after materialization.
    """
    project_root = project_root or find_project_root()
    scope = _profiling_scope_name(profiling_scope)
    return _run_registered_attribution_phase(
        "phase3b_oracle" if scope == "all" else "phase3b_oracle_single_signal",
        project_root=project_root,
        stage=stage,
        config_path=config_path,
        rebuild=rebuild,
        skip_diagnostics=skip_diagnostics,
        top_confusions=top_confusions,
        selected_candidates_path_override=selected_candidates_path_override,
    )
