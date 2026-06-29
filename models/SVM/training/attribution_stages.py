from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline.materialization import (
    resolve_materialization_stage,
    run_materialization,
)
from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
    write_toml,
)
from models.SVM.signals.profiling_signal_extractor import run_profiling_signal_extraction
from models.SVM.signals.ground_truth_signal_injector import run_ground_truth_signal_injection
from models.SVM.linear_svm_common import _json_safe
from models.SVM.training.train_stacked_attribution import (
    load_selected_stacked_candidates,
    run_final_stacked_evaluation_from_config as run_final_stacked_evaluation,
    run_stacked_experiment_from_config as run_stacked_experiment,
    stacked_search_profiling_blocks,
)
from models.SVM.training.train_svm_attribution import (
    load_selected_direct_candidates,
    run_attribution_experiment_from_config as run_attribution_experiment,
    run_final_attribution_evaluation_from_config as run_final_attribution_evaluation,
)


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class LoadedFinalSelection:
    """Bundle the selected candidates consumed by one staged final attribution run."""

    resolved_stage: JsonDict
    candidates: list[Any]
    payload: JsonDict
    provenance: JsonDict
    predicted_profiling_targets: list[str]
    oracle_targets: list[str]


_STAGES = {"dev", "final", "all"}
_ORACLE_BLOCK_PREFIX = "profiling_oracle_"
_HARD_PROFILING_BLOCK = "profiling_hard"
_HARD_PROFILING_PREFIX = "profiling_hard_"
_RESOLVED_SPEC_FILENAME = "resolved_attribution_run_spec.json"
_PROFILING_PREFIX = "profiling_"


def _load_toml(config_path: Path) -> JsonDict:
    """Read a staged attribution TOML config."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _write_resolved_attribution_stage_spec(
    *,
    project_root: Path,
    results_dir: Path,
    source_config_path: Path,
    resolved_stage: JsonDict,
) -> str:
    """Write the resolved trainer input used by one attribution stage."""
    path = results_dir / _RESOLVED_SPEC_FILENAME
    payload = {
        "schema_version": 1,
        "source_config_path": relative_to_project(project_root, source_config_path),
        "kind": str(resolved_stage["kind"]),
        "stage": str(resolved_stage["stage"]),
        "config": _json_safe(resolved_stage["config"]),
        "materialization": _json_safe(resolved_stage["materialization"]),
    }
    write_json(path, payload)
    return relative_to_project(project_root, path)


def _model_kind(config: JsonDict) -> str:
    """Return the staged attribution model family declared by the config."""
    experiment_cfg = config.get("experiment", {})
    kind = str(experiment_cfg.get("kind", "")).strip().lower()
    if kind not in {"baseline", "stacked"}:
        raise ValueError("experiment.kind must be 'baseline' or 'stacked'.")
    return kind


def _validate_stage(stage: str) -> str:
    """Normalize and validate the requested staged attribution phase."""
    stage_name = str(stage).strip().lower()
    if stage_name not in _STAGES:
        raise ValueError("Model stage must be one of: all, dev, final.")
    return stage_name


def _profiling_block_kind(block: str) -> str | None:
    """Classify profiling feature blocks into the run-level representation axis."""
    block_name = str(block).strip()
    if block_name == "profiling" or (
        block_name.startswith(_PROFILING_PREFIX)
        and block_name != "profiling_oracle"
        and block_name != _HARD_PROFILING_BLOCK
        and not block_name.startswith(_HARD_PROFILING_PREFIX)
        and not block_name.startswith(_ORACLE_BLOCK_PREFIX)
    ):
        return "probability"
    if block_name == _HARD_PROFILING_BLOCK or block_name.startswith(_HARD_PROFILING_PREFIX):
        return "hard"
    if block_name == "profiling_oracle" or block_name.startswith(_ORACLE_BLOCK_PREFIX):
        return "oracle"
    return None


def _profiling_blocks_from_config(config: JsonDict) -> list[str]:
    """Return the ordered profiling block names referenced by an attribution config."""
    blocks: list[str] = []
    kind = _model_kind(config)
    if kind == "baseline":
        for condition in config.get("conditions", []):
            blocks.extend(str(block) for block in condition.get("blocks", []))
    else:
        blocks.extend(stacked_search_profiling_blocks(config))

    seen: set[str] = set()
    unique: list[str] = []
    for block in blocks:
        if _profiling_block_kind(block) is None or block in seen:
            continue
        seen.add(block)
        unique.append(block)
    return unique


def _oracle_targets_for_blocks(project_root: Path, config: JsonDict, blocks: list[str]) -> list[str]:
    """Return oracle targets needed by blocks in source-config target order."""
    requested_targets: set[str] = set()
    include_all_targets = False
    for block in blocks:
        block_name = str(block).strip()
        if block_name == "profiling_oracle":
            include_all_targets = True
        elif block_name.startswith(_ORACLE_BLOCK_PREFIX):
            requested_targets.add(block_name.removeprefix(_ORACLE_BLOCK_PREFIX))

    if not include_all_targets and not requested_targets:
        return []

    raw_path = config.get("oracle_source", {}).get("config_path")
    if not raw_path:
        raise ValueError(
            "Attribution configs that use oracle profiling blocks must define "
            "oracle_source.config_path."
        )
    source_config = _load_toml(resolve_project_path(project_root, raw_path))
    raw_targets = source_config.get("source", {}).get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Oracle source config must define source.targets as a non-empty list.")

    available_targets = [str(target) for target in raw_targets]
    if include_all_targets:
        return available_targets
    return [target for target in available_targets if target in requested_targets]


def _infer_profiling_metadata(project_root: Path, config: JsonDict) -> JsonDict | None:
    """Infer profiling provenance from the concrete blocks used by this run."""
    blocks = _profiling_blocks_from_config(config)
    kinds = {kind for block in blocks if (kind := _profiling_block_kind(block)) is not None}
    if not kinds:
        return None
    if len(kinds) > 1:
        raise ValueError(
            "Canonical attribution configs must not mix profiling representations "
            f"inside one run. Found: {sorted(kinds)}"
        )

    representation = next(iter(kinds))
    if representation == "oracle":
        return {
            "source": "oracle",
            "targets": _oracle_targets_for_blocks(project_root, config, blocks),
            "resolved_blocks": blocks,
        }

    metadata: JsonDict = {
        "source": "predicted",
        "representation": representation,
        "targets": [str(target) for target in config.get("profiling_source", {}).get("targets", [])],
        "resolved_blocks": blocks,
    }
    return metadata


def _materialization_config_path(project_root: Path, config: JsonDict) -> Path:
    """Resolve the materialization config referenced by a staged model config."""
    materialization_cfg = config.get("materialization", {})
    raw_path = materialization_cfg.get("config_path")
    if not raw_path:
        raise ValueError("Model config must define materialization.config_path.")
    return resolve_project_path(project_root, raw_path)


def _resolve_materialization(
    project_root: Path,
    config: JsonDict,
    *,
    model_stage: str,
) -> JsonDict:
    """Resolve materialized data metadata for a model stage."""
    materialization_path = _materialization_config_path(project_root, config)
    return resolve_materialization_stage(
        project_root,
        materialization_path,
        stage=model_stage,
    )


def _common_resolved_sections(
    config: JsonDict,
    materialization: JsonDict,
    *,
    stage: str,
) -> JsonDict:
    """Build trainer config sections shared by direct and stacked attribution."""
    source_cfg = config.get("source", {})
    stage_cfg = config.get(stage, {})
    experiment_cfg = config.get("experiment", {})

    experiment_name = stage_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError(f"Model config must define {stage}.experiment_name.")

    resolved_source = {
        "split_name": str(materialization["split_name"]),
        "materialization_name": str(materialization["materialization_name"]),
        "target": str(source_cfg.get("target", "author")),
        "units": source_cfg.get("units", "all"),
    }
    return {
        "experiment": {
            "name": str(experiment_name),
            "seed": int(experiment_cfg.get("seed", 42)),
            "save_prediction_top_k": int(stage_cfg.get("save_prediction_top_k", 5)),
        },
        "data": dict(config.get("data", {})),
        "source": resolved_source,
    }


def _resolve_baseline_stage_config(
    config: JsonDict,
    materialization: JsonDict,
    *,
    stage: str,
) -> JsonDict:
    """Translate a staged direct-SVM config into trainer-ready config."""
    resolved = _common_resolved_sections(config, materialization, stage=stage)
    fit_cfg = dict(config.get("fit", {}))

    if stage == "dev":
        stage_cfg = config.get("dev", {})
        search_cfg = dict(config.get("search", {}))
        resolved["experiment"]["selection_metric"] = str(stage_cfg.get("selection_metric", "macro_f1"))
        resolved["experiment"]["n_jobs"] = int(stage_cfg.get("n_jobs", 1))
        resolved["model"] = {
            "family": "linear_svm",
            "C_values": search_cfg.get("C_values", [1.0]),
            "class_weights": search_cfg.get("class_weights", ["none"]),
            "max_iter": int(fit_cfg.get("max_iter", 20_000)),
            "tol": float(fit_cfg.get("tol", 1e-4)),
            "dual": fit_cfg.get("dual", "auto"),
            "top_k": fit_cfg.get("top_k", [3, 5]),
        }
        resolved["conditions"] = list(config.get("conditions", []))
        return resolved

    stage_cfg = config.get("final", {})
    resolved["experiment"]["n_jobs"] = int(stage_cfg.get("n_jobs", 1))
    resolved["model"] = {
        "family": "linear_svm",
        "max_iter": int(fit_cfg.get("max_iter", 20_000)),
        "tol": float(fit_cfg.get("tol", 1e-4)),
        "dual": fit_cfg.get("dual", "auto"),
        "top_k": fit_cfg.get("top_k", [3, 5]),
    }
    if stage_cfg.get("selected_candidates_path"):
        resolved["final_eval"] = {"selected_candidates_path": str(stage_cfg["selected_candidates_path"])}
    else:
        resolved["final_eval"] = {}
    return resolved


def _resolve_stacked_stage_config(
    config: JsonDict,
    materialization: JsonDict,
    *,
    stage: str,
) -> JsonDict:
    """Translate a staged stacked-attribution config into trainer-ready config."""
    resolved = _common_resolved_sections(config, materialization, stage=stage)
    fit_cfg = dict(config.get("fit", {}))

    if stage == "dev":
        stage_cfg = config.get("dev", {})
        search_cfg = dict(config.get("search", {}))
        resolved["experiment"]["selection_metric"] = str(stage_cfg.get("selection_metric", "macro_f1"))
        resolved["experiment"]["n_jobs"] = int(stage_cfg.get("n_jobs", 1))
        resolved["model"] = {
            "family": "stacked",
            "inner_cv": int(fit_cfg.get("inner_cv", 3)),
            "base_c_values": search_cfg.get("base_c_values", [1.0]),
            "class_weights": search_cfg.get("class_weights", ["balanced"]),
            "max_iter": int(fit_cfg.get("max_iter", 20_000)),
            "tol": float(fit_cfg.get("tol", 1e-4)),
            "dual": fit_cfg.get("dual", "auto"),
            "top_c_values": search_cfg.get("top_c_values", [1.0]),
            "top_max_iter": int(fit_cfg.get("top_max_iter", 1000)),
            "top_k": fit_cfg.get("top_k", [3, 5]),
        }
        resolved["families"] = list(config.get("families", []))
        resolved["conditions"] = list(config.get("conditions", []))
        return resolved

    stage_cfg = config.get("final", {})
    resolved["experiment"]["n_jobs"] = int(stage_cfg.get("n_jobs", 1))
    resolved["model"] = {
        "family": "stacked",
        "inner_cv": int(fit_cfg.get("inner_cv", 3)),
        "max_iter": int(fit_cfg.get("max_iter", 20_000)),
        "tol": float(fit_cfg.get("tol", 1e-4)),
        "dual": fit_cfg.get("dual", "auto"),
        "top_max_iter": int(fit_cfg.get("top_max_iter", 1000)),
        "top_k": fit_cfg.get("top_k", [3, 5]),
    }
    if stage_cfg.get("selected_candidates_path"):
        resolved["final_eval"] = {"selected_candidates_path": str(stage_cfg["selected_candidates_path"])}
    else:
        resolved["final_eval"] = {}
    return resolved


def _resolve_loaded_model_stage_config(
    project_root: Path,
    config: JsonDict,
    *,
    stage: str,
) -> JsonDict:
    """Resolve an already-loaded staged attribution config into trainer input."""
    stage_name = _validate_stage(stage)
    if stage_name == "all":
        raise ValueError("Resolve one model stage at a time: dev or final.")

    kind = _model_kind(config)
    materialization = _resolve_materialization(project_root, config, model_stage=stage_name)

    if kind == "baseline":
        resolved_config = _resolve_baseline_stage_config(config, materialization, stage=stage_name)
    else:
        resolved_config = _resolve_stacked_stage_config(config, materialization, stage=stage_name)

    return {
        "kind": kind,
        "stage": stage_name,
        "config": resolved_config,
        "materialization": materialization,
        "profiling": _infer_profiling_metadata(project_root, config),
    }


def resolve_model_stage_config(
    project_root: Path,
    config_path: Path,
    *,
    stage: str,
) -> JsonDict:
    """Resolve one staged attribution config file into the exact trainer input."""
    return _resolve_loaded_model_stage_config(
        project_root,
        _load_toml(config_path),
        stage=stage,
    )


def _profile_targets_from_blocks(blocks: list[str], available_targets: list[str]) -> list[str]:
    """Return profiler targets needed by probability or hard predicted blocks."""
    targets: set[str] = set()
    for block in blocks:
        block_name = str(block).strip()
        if block_name == "profiling" or block_name == _HARD_PROFILING_BLOCK:
            targets.update(available_targets)
        elif block_name.startswith(_HARD_PROFILING_PREFIX):
            targets.add(block_name.removeprefix(_HARD_PROFILING_PREFIX))
        elif (
            block_name.startswith(_PROFILING_PREFIX)
            and block_name != "profiling_oracle"
            and not block_name.startswith(_ORACLE_BLOCK_PREFIX)
        ):
            targets.add(block_name.removeprefix(_PROFILING_PREFIX))
    return [target for target in available_targets if target in targets]


def _baseline_dev_profiling_targets(config: JsonDict) -> list[str]:
    """Return profiling targets required by direct-SVM dev search conditions."""
    source = config.get("profiling_source", {})
    available_targets = [str(target) for target in source.get("targets", [])]
    blocks: list[str] = []
    for condition in config.get("conditions", []):
        blocks.extend(str(block) for block in condition.get("blocks", []))
    return _profile_targets_from_blocks(blocks, available_targets)


def _stacked_dev_profiling_targets(config: JsonDict) -> list[str]:
    """Return profiling targets required by stacked dev search conditions."""
    source = config.get("profiling_source", {})
    available_targets = [str(target) for target in source.get("targets", [])]
    return _profile_targets_from_blocks(
        stacked_search_profiling_blocks(config),
        available_targets,
    )


def _baseline_selected_profiling_targets(
    selection_payload: JsonDict, available_targets: list[str]
) -> list[str]:
    """Return profiling targets needed by all selected direct candidates."""
    blocks: list[str] = []
    for candidate in selection_payload.get("selected_candidates", []):
        blocks.extend(str(block) for block in candidate.get("blocks", []))
    return _profile_targets_from_blocks(blocks, available_targets)


def _stacked_selected_profiling_targets(
    selection_payload: JsonDict, available_targets: list[str]
) -> list[str]:
    """Return profiling targets needed by all selected stacked candidates."""
    blocks: list[str] = []
    for candidate in selection_payload.get("selected_candidates", []):
        blocks.extend(str(block) for block in candidate.get("profiling_blocks", []))
    return _profile_targets_from_blocks(blocks, available_targets)


def _run_profiling_extraction_if_needed(
    project_root: Path,
    model_config: JsonDict,
    materialization: JsonDict,
    *,
    stage: str,
    targets: list[str],
    show_progress: bool,
) -> JsonDict | None:
    """Materialize predicted profiling blocks when a stage uses them."""
    if not targets:
        return None

    profiling_source = model_config.get("profiling_source", {})
    required = {
        "profiling_split_name",
        "profiling_materialization_name",
        "profiling_experiment_name",
        "profiling_seed",
    }
    missing = sorted(required - set(profiling_source))
    if missing:
        raise ValueError(
            "Model config requires profiling features but is missing "
            f"[profiling_source] keys: {missing}"
        )

    data_cfg = model_config.get("data", {})
    extraction_config = {
        "data": {
            "splits_dir": data_cfg.get("splits_dir", "data/splits"),
            "artifacts_dir": profiling_source.get("artifacts_dir", "models/artifacts/profiling"),
            "profiling_results_dir": profiling_source.get("profiling_results_dir", "results/models"),
        },
        "source": {
            "attribution_split_name": str(materialization["split_name"]),
            "profiling_split_name": str(profiling_source["profiling_split_name"]),
            "profiling_materialization_name": str(profiling_source["profiling_materialization_name"]),
            "profiling_experiment_name": str(profiling_source["profiling_experiment_name"]),
            "profiling_seed": int(profiling_source["profiling_seed"]),
            "targets": targets,
        },
        "stages": {
            stage: {
                "attribution_materialization_name": str(
                    materialization["materialization_name"]
                ),
            },
        },
    }
    extraction_config_path = (
        Path(materialization["materialized_root"]) / "profiling_extraction_config.toml"
    )
    write_toml(extraction_config_path, extraction_config)
    return run_profiling_signal_extraction(
        extraction_config_path,
        stage=stage,
        show_progress=show_progress,
    )


def _baseline_dev_oracle_targets(project_root: Path, config: JsonDict) -> list[str]:
    """Return oracle targets used by any feature set in a baseline dev config."""
    blocks: list[str] = []
    for condition in config.get("conditions", []):
        blocks.extend(str(b) for b in condition.get("blocks", []))
    return _oracle_targets_for_blocks(project_root, config, blocks)


def _stacked_dev_oracle_targets(project_root: Path, config: JsonDict) -> list[str]:
    """Return oracle targets used by the stacked profiling blocks in a dev config."""
    return _oracle_targets_for_blocks(
        project_root,
        config,
        stacked_search_profiling_blocks(config),
    )


def _baseline_selected_oracle_targets(
    project_root: Path,
    config: JsonDict,
    selection_payload: JsonDict,
) -> list[str]:
    """Return oracle targets needed by all selected direct candidates."""
    blocks: list[str] = []
    for candidate in selection_payload.get("selected_candidates", []):
        blocks.extend(str(block) for block in candidate.get("blocks", []))
    return _oracle_targets_for_blocks(project_root, config, blocks)


def _stacked_selected_oracle_targets(
    project_root: Path,
    config: JsonDict,
    selection_payload: JsonDict,
) -> list[str]:
    """Return oracle targets needed by all selected stacked candidates."""
    blocks: list[str] = []
    for candidate in selection_payload.get("selected_candidates", []):
        blocks.extend(str(block) for block in candidate.get("profiling_blocks", []))
    return _oracle_targets_for_blocks(project_root, config, blocks)


def _run_oracle_injection_if_needed(
    project_root: Path,
    materialization: JsonDict,
    *,
    stage: str,
    targets: list[str],
    show_progress: bool,
) -> JsonDict | None:
    """Write oracle one-hot matrices for the given targets into a materialization stage.

    Builds an inline injection config from the resolved materialization, writes it
    as a TOML file next to the materialized root, and calls the injector. Returns
    None when targets is empty.
    """
    if not targets:
        return None

    injection_config = {
        "data": {"splits_dir": str(Path(materialization["materialized_root"]).parents[2])},
        "source": {
            "attribution_split_name": str(materialization["split_name"]),
            "targets": targets,
        },
        "stages": {
            stage: {
                "attribution_materialization_name": str(materialization["materialization_name"]),
            },
        },
    }
    injection_config_path = (
        Path(materialization["materialized_root"]) / "ground_truth_injection_config.toml"
    )
    write_toml(injection_config_path, injection_config)
    return run_ground_truth_signal_injection(
        injection_config_path,
        stage=stage,
        show_progress=show_progress,
    )


def _patch_stage_manifest(
    project_root: Path,
    manifest: JsonDict,
    *,
    config_path: Path,
    stage: str,
    kind: str,
    materialization: JsonDict,
    resolved_stage: JsonDict,
) -> JsonDict:
    """Attach staged-config provenance to a trainer manifest."""
    results_dir = resolve_project_path(project_root, manifest["results_dir"])
    manifest_path = results_dir / "manifest.json"
    resolved_spec_path = _write_resolved_attribution_stage_spec(
        project_root=project_root,
        results_dir=results_dir,
        source_config_path=config_path,
        resolved_stage=resolved_stage,
    )

    manifest.update(
        {
            "config_path": relative_to_project(project_root, config_path),
            "stage": stage,
            "experiment_kind": kind,
            "materialization_config_path": relative_to_project(project_root, Path(materialization["config_path"])),
            "materialization_stage": str(materialization["stage"]),
            "materialization_name": str(materialization["materialization_name"]),
            "resolved_spec_path": resolved_spec_path,
        }
    )
    if resolved_stage.get("profiling"):
        manifest["profiling"] = resolved_stage["profiling"]

    write_json(manifest_path, manifest)
    return manifest


def _run_resolved_model_stage(
    project_root: Path,
    config_path: Path,
    *,
    stage: str,
    selected_candidates_path_override: Path | None,
    show_progress: bool,
    loaded_final_selection: LoadedFinalSelection | None = None,
) -> JsonDict:
    """Run one resolved direct or stacked attribution trainer stage."""
    resolved = (
        loaded_final_selection.resolved_stage
        if loaded_final_selection is not None
        else resolve_model_stage_config(project_root, config_path, stage=stage)
    )
    kind = str(resolved["kind"])
    resolved_config = resolved["config"]
    materialization = resolved["materialization"]
    preloaded_selection_kwargs = {}
    if loaded_final_selection is not None:
        preloaded_selection_kwargs = {
            "preloaded_candidates": loaded_final_selection.candidates,
            "preloaded_selection_payload": loaded_final_selection.payload,
            "preloaded_selection_source": loaded_final_selection.provenance,
        }

    if kind == "baseline" and stage == "dev":
        manifest = run_attribution_experiment(
            resolved_config,
            project_root=project_root,
            show_progress=show_progress,
        )
    elif kind == "baseline":
        manifest = run_final_attribution_evaluation(
            resolved_config,
            project_root=project_root,
            show_progress=show_progress,
            selected_candidates_path_override=selected_candidates_path_override,
            **preloaded_selection_kwargs,
        )
    elif stage == "dev":
        manifest = run_stacked_experiment(
            resolved_config,
            project_root=project_root,
            show_progress=show_progress,
        )
    else:
        manifest = run_final_stacked_evaluation(
            resolved_config,
            project_root=project_root,
            show_progress=show_progress,
            selected_candidates_path_override=selected_candidates_path_override,
            **preloaded_selection_kwargs,
        )

    return _patch_stage_manifest(
        project_root,
        manifest,
        config_path=config_path,
        stage=stage,
        kind=kind,
        materialization=materialization,
        resolved_stage=resolved,
    )


def _load_final_selection(
    project_root: Path,
    model_config: JsonDict,
    *,
    selected_candidates_path_override: Path | None,
) -> LoadedFinalSelection:
    """Load final selected candidates once for staged extraction and evaluation."""
    resolved = _resolve_loaded_model_stage_config(project_root, model_config, stage="final")
    available_targets = [str(target) for target in model_config.get("profiling_source", {}).get("targets", [])]
    kind = str(resolved["kind"])
    if kind == "baseline":
        candidates, payload, provenance = load_selected_direct_candidates(
            project_root,
            resolved["config"],
            selected_candidates_path_override=selected_candidates_path_override,
        )
        return LoadedFinalSelection(
            resolved_stage=resolved,
            candidates=candidates,
            payload=payload,
            provenance=provenance,
            predicted_profiling_targets=_baseline_selected_profiling_targets(payload, available_targets),
            oracle_targets=_baseline_selected_oracle_targets(project_root, model_config, payload),
        )

    candidates, payload, provenance = load_selected_stacked_candidates(
        project_root,
        resolved["config"],
        selected_candidates_path_override=selected_candidates_path_override,
    )
    return LoadedFinalSelection(
        resolved_stage=resolved,
        candidates=candidates,
        payload=payload,
        provenance=provenance,
        predicted_profiling_targets=_stacked_selected_profiling_targets(payload, available_targets),
        oracle_targets=_stacked_selected_oracle_targets(project_root, model_config, payload),
    )


def run_attribution_model(
    config_path: Path,
    *,
    stage: str = "all",
    rebuild: bool = False,
    show_progress: bool = False,
    selected_candidates_path_override: Path | None = None,
) -> JsonDict:
    """Run staged attribution dev/final workflow from a high-level config."""
    project_root = find_project_root(config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent)
    stage_name = _validate_stage(stage)
    model_config = _load_toml(config_path)
    kind = _model_kind(model_config)
    profiling_metadata = _infer_profiling_metadata(project_root, model_config)
    stages: JsonDict = {}

    if stage_name in {"dev", "all"}:
        dev_materialization = _resolve_materialization(project_root, model_config, model_stage="dev")
        stages["dev_materialization"] = run_materialization(
            _materialization_config_path(project_root, model_config),
            stage=str(dev_materialization["stage"]),
            rebuild=rebuild,
            show_progress=show_progress,
        )
        dev_targets = (
            _baseline_dev_profiling_targets(model_config)
            if kind == "baseline"
            else _stacked_dev_profiling_targets(model_config)
        )
        extraction = _run_profiling_extraction_if_needed(
            project_root,
            model_config,
            dev_materialization,
            stage="dev",
            targets=dev_targets,
            show_progress=show_progress,
        )
        if extraction is not None:
            stages["dev_profiling_extraction"] = extraction
        dev_oracle_targets = (
            _baseline_dev_oracle_targets(project_root, model_config)
            if kind == "baseline"
            else _stacked_dev_oracle_targets(project_root, model_config)
        )
        oracle_injection = _run_oracle_injection_if_needed(
            project_root,
            dev_materialization,
            stage="dev",
            targets=dev_oracle_targets,
            show_progress=show_progress,
        )
        if oracle_injection is not None:
            stages["dev_oracle_injection"] = oracle_injection
        stages["dev"] = _run_resolved_model_stage(
            project_root,
            config_path,
            stage="dev",
            selected_candidates_path_override=None,
            show_progress=show_progress,
        )

    final_override = selected_candidates_path_override
    if stage_name == "all":
        final_override = resolve_project_path(
            project_root,
            stages["dev"]["selected_candidates_path"],
        )

    if stage_name in {"final", "all"}:
        loaded_final_selection = _load_final_selection(
            project_root,
            model_config,
            selected_candidates_path_override=final_override,
        )
        final_materialization = loaded_final_selection.resolved_stage["materialization"]
        stages["final_materialization"] = run_materialization(
            _materialization_config_path(project_root, model_config),
            stage=str(final_materialization["stage"]),
            rebuild=rebuild,
            show_progress=show_progress,
        )
        extraction = _run_profiling_extraction_if_needed(
            project_root,
            model_config,
            final_materialization,
            stage="final",
            targets=loaded_final_selection.predicted_profiling_targets,
            show_progress=show_progress,
        )
        if extraction is not None:
            stages["final_profiling_extraction"] = extraction
        final_oracle_injection = _run_oracle_injection_if_needed(
            project_root,
            final_materialization,
            stage="final",
            targets=loaded_final_selection.oracle_targets,
            show_progress=show_progress,
        )
        if final_oracle_injection is not None:
            stages["final_oracle_injection"] = final_oracle_injection
        stages["final"] = _run_resolved_model_stage(
            project_root,
            config_path,
            stage="final",
            selected_candidates_path_override=final_override,
            show_progress=show_progress,
            loaded_final_selection=loaded_final_selection,
        )

    return {
        "config_path": relative_to_project(project_root, config_path),
        "stage": stage_name,
        "experiment_kind": kind,
        "profiling": profiling_metadata,
        "stages": stages,
    }
