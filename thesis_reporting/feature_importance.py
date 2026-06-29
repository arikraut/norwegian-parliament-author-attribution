"""Feature-importance dispatch and portable report collection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from data_pipeline.utils import relative_to_project, resolve_project_path, write_json
from models.SVM.importance.feature_importance_stacked import (
    run_stacked_condition_importance_analysis,
)
from models.SVM.importance.feature_importance_svm import (
    run_condition_importance_analysis,
)

from .config import ResultSystem, configured_feature_importance_systems, configured_systems
from .provenance import project_relative_outputs


DIRECT_OUTPUT_NAMES = (
    "global_importance",
    "block_importance",
    "stylo_subfamily_importance",
    "per_author_top_features",
    "block_importance_plot",
)

STACKED_OUTPUT_NAMES = (
    "top_model_family_importance",
    "top_model_per_author_family",
    "base_family_global_importance",
    "base_family_block_importance",
    "base_family_stylo_subfamily_importance",
)


def requested_system_keys(raw_systems: str) -> tuple[str, ...]:
    """Normalize comma-separated configured system keys."""

    return tuple(
        system_key.strip()
        for system_key in raw_systems.split(",")
        if system_key.strip()
    )


def selected_systems(system_keys: tuple[str, ...]) -> tuple[ResultSystem, ...]:
    """Return explicitly configured systems selected by stable key."""

    systems_by_key = {system.key: system for system in configured_systems()}
    missing = [key for key in system_keys if key not in systems_by_key]
    if missing:
        raise ValueError(f"Unknown configured system key(s): {missing}")
    return tuple(systems_by_key[key] for key in system_keys)


def condition_manifest_path(system: ResultSystem, results_dir: Path) -> Path:
    """Return the final-run manifest for one configured system."""

    seed_dir = system.condition_dir.parents[1]
    return results_dir / seed_dir / "manifest.json"


def run_system_importance(
    system: ResultSystem,
    *,
    results_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    """Dispatch one system to its public condition-level importance helper."""

    manifest_path = condition_manifest_path(system, results_dir)
    if system.architecture == "stacked":
        return run_stacked_condition_importance_analysis(
            manifest_path,
            condition_id=system.condition_id,
        )
    return run_condition_importance_analysis(
        manifest_path,
        top_n=top_n,
        condition_id=system.condition_id,
    )


def copy_importance_outputs(
    system: ResultSystem,
    summary: dict[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Copy one helper's outputs into the configured result tree."""

    system_dir = output_dir / system.key
    system_dir.mkdir(parents=True, exist_ok=True)
    expected_names = (
        STACKED_OUTPUT_NAMES if system.architecture == "stacked" else DIRECT_OUTPUT_NAMES
    )
    copied: dict[str, str] = {}
    for output_name in expected_names:
        source_path = resolve_project_path(project_root, summary["outputs"][output_name])
        destination = system_dir / source_path.name
        shutil.copyfile(source_path, destination)
        copied[output_name] = str(destination)
    return copied


def portable_helper_summary(
    summary: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Make the known helper-output paths portable without reshaping other metadata."""

    portable = dict(summary)
    if "outputs" in summary:
        portable["outputs"] = {
            key: relative_to_project(
                project_root,
                resolve_project_path(project_root, path),
            )
            for key, path in summary["outputs"].items()
        }
    return portable


def write_feature_importance_manifest(
    *,
    project_root: Path,
    output_dir: Path,
    results_dir: Path,
    systems: tuple[ResultSystem, ...],
    summaries: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, str]],
) -> Path:
    """Write portable provenance for feature-importance copies."""

    manifest_path = output_dir / "manifest.json"
    payload = {
        "run_type": "thesis_feature_importance_additions",
        "results_dir": relative_to_project(project_root, results_dir),
        "systems": [
            {
                "key": system.key,
                "label": system.label,
                "architecture": system.architecture,
                "condition_id": system.condition_id,
                "condition_dir": system.condition_dir.as_posix(),
            }
            for system in systems
        ],
        "helper_summaries": {
            key: portable_helper_summary(summary, project_root=project_root)
            for key, summary in summaries.items()
        },
        "outputs": project_relative_outputs(project_root, outputs),
    }
    write_json(manifest_path, payload)
    return manifest_path


def run_feature_importance_additions(
    systems: tuple[ResultSystem, ...],
    *,
    project_root: Path,
    results_dir: Path,
    output_dir: Path,
    top_n: int,
) -> dict[str, dict[str, str]]:
    """Run configured importance analyses and collect report-ready outputs."""

    summaries: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, str]] = {}
    for system in systems:
        summary = run_system_importance(system, results_dir=results_dir, top_n=top_n)
        summaries[system.key] = summary
        outputs[system.key] = copy_importance_outputs(
            system,
            summary,
            project_root=project_root,
            output_dir=output_dir,
        )
    write_feature_importance_manifest(
        project_root=project_root,
        output_dir=output_dir,
        results_dir=results_dir,
        systems=systems,
        summaries=summaries,
        outputs=outputs,
    )
    return outputs


def write_feature_importance_outputs(
    systems: tuple[ResultSystem, ...],
    *,
    project_root: Path,
    results_dir: Path,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Collect configured feature-importance outputs for the combined report."""

    section_dir = output_dir / "feature_importance"
    section_dir.mkdir(parents=True, exist_ok=True)
    configured_keys = {
        system.key for system in configured_feature_importance_systems()
    }
    feature_systems = tuple(
        system for system in systems if system.key in configured_keys
    )
    nested_outputs = run_feature_importance_additions(
        feature_systems,
        project_root=project_root,
        results_dir=results_dir,
        output_dir=section_dir,
        top_n=top_n,
    )
    outputs = {"manifest": str(section_dir / "manifest.json")}
    for system_key, system_outputs in nested_outputs.items():
        for output_name, output_path in system_outputs.items():
            outputs[f"{system_key}_{output_name}"] = output_path
    return outputs
