"""Portable provenance manifests and Markdown indexes for result reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.utils import relative_to_project, write_json

from .config import ProfileQualityRun, ProfileTarget, ResultSystem, SystemComparison


SECTION_DESCRIPTIONS = {
    "author_performance": (
        "Author Performance",
        "Ranks authors within each configured system by performance and error metrics.",
    ),
    "confusions": (
        "Confusions",
        "Summarizes author- and party-level errors and normalized confusion artifacts.",
    ),
    "profiling_effects": (
        "Profiling Effects",
        "Compares baseline, predicted-profile, oracle-profile, and architecture effects per author.",
    ),
    "topk_confidence": (
        "Top-K And Confidence",
        "Summarizes top-k rescues, score margins, confident errors, and uncertain correct predictions.",
    ),
    "profile_quality": (
        "Profile Quality",
        "Summarizes profiling metrics and their relationship with attribution correctness.",
    ),
    "significance": (
        "Significance",
        "Collects paired bootstrap macro-F1 and McNemar comparisons.",
    ),
    "feature_importance": (
        "Feature Importance",
        "Collects report-ready copies of saved-model feature-importance outputs.",
    ),
}


def project_relative_outputs(
    project_root: Path,
    outputs: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Convert generated output paths to portable project-relative strings."""

    return {
        section: {
            key: relative_to_project(project_root, Path(path))
            for key, path in section_outputs.items()
        }
        for section, section_outputs in outputs.items()
    }


def write_manifest(
    *,
    project_root: Path,
    output_dir: Path,
    results_dir: Path,
    data_dir: Path,
    sections: set[str],
    systems: tuple[ResultSystem, ...],
    comparisons: tuple[SystemComparison, ...],
    profile_run: ProfileQualityRun,
    profile_targets: tuple[ProfileTarget, ...],
    outputs: dict[str, dict[str, str]],
) -> Path:
    """Write portable provenance for all generated result-addition files."""

    manifest_path = output_dir / "manifest.json"
    payload = {
        "run_type": "thesis_results_additions",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": relative_to_project(project_root, results_dir),
        "data_dir": relative_to_project(project_root, data_dir),
        "sections": sorted(sections),
        "systems": [
            {
                "key": system.key,
                "label": system.label,
                "phase": system.phase,
                "split": system.split,
                "architecture": system.architecture,
                "representation": system.representation,
                "scope": system.scope,
                "condition_id": system.condition_id,
                "condition_dir": system.condition_dir.as_posix(),
                "final_predictions_path": system.final_predictions_path.as_posix(),
                "per_author_metrics_path": system.per_author_metrics_path.as_posix(),
                "confusion_pairs_path": system.confusion_pairs_path.as_posix(),
                "normalized_confusion_matrix_path": (
                    system.normalized_confusion_matrix_path.as_posix()
                ),
                "normalized_confusion_heatmap_path": (
                    system.normalized_confusion_heatmap_path.as_posix()
                ),
            }
            for system in systems
        ],
        "comparisons": [
            {
                "key": comparison.key,
                "label": comparison.label,
                "source_system_key": comparison.source_system_key,
                "target_system_key": comparison.target_system_key,
                "purpose": comparison.purpose,
                "comparison_group": comparison.comparison_group,
            }
            for comparison in comparisons
        ],
        "profile_quality_run": {
            "key": profile_run.key,
            "label": profile_run.label,
            "quality_dir": profile_run.quality_dir.as_posix(),
            "attribution_test_metrics_path": (
                profile_run.attribution_test_metrics_path.as_posix()
            ),
            "calibration_summary_path": profile_run.calibration_summary_path.as_posix(),
            "target_summary_path": profile_run.target_summary_path.as_posix(),
        },
        "profile_targets": [
            {
                "key": target.key,
                "label": target.label,
                "prediction_file_key": target.prediction_file_key,
                "test_prediction_path": profile_run.prediction_path(
                    target, "test"
                ).as_posix(),
            }
            for target in profile_targets
        ],
        "outputs": project_relative_outputs(project_root, outputs),
    }
    write_json(manifest_path, payload)
    return manifest_path


def append_output_table(
    lines: list[str],
    *,
    project_root: Path,
    section: str,
    outputs: dict[str, str],
) -> None:
    """Append one consistently formatted section and output table."""

    heading, description = SECTION_DESCRIPTIONS[section]
    lines.extend(
        [
            "",
            f"## {heading}",
            "",
            description,
            "",
            "| Output | Path |",
            "| --- | --- |",
        ]
    )
    for output_name, output_path in outputs.items():
        relative_path = relative_to_project(project_root, Path(output_path))
        lines.append(f"| `{output_name}` | `{relative_path}` |")


def write_summary(
    *,
    project_root: Path,
    output_dir: Path,
    results_dir: Path,
    data_dir: Path,
    systems: tuple[ResultSystem, ...],
    comparisons: tuple[SystemComparison, ...],
    profile_targets: tuple[ProfileTarget, ...],
    outputs: dict[str, dict[str, str]],
) -> Path:
    """Write a concise Markdown index for generated result additions."""

    summary_path = output_dir / "summary.md"
    lines = [
        "# Thesis Results Additions",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Generated from local project result artifacts. Oracle rows are diagnostic only.",
        "",
        "## Source Directories",
        "",
        "| Source | Path |",
        "| --- | --- |",
        f"| Results | `{relative_to_project(project_root, results_dir)}` |",
        f"| Data | `{relative_to_project(project_root, data_dir)}` |",
        "",
        "## Included Systems",
        "",
        "| System key | Label | Split | Representation |",
        "| --- | --- | --- | --- |",
    ]
    for system in systems:
        lines.append(
            f"| `{system.key}` | {system.label} | `{system.split}` | "
            f"`{system.representation}` |"
        )

    if "profiling_effects" in outputs:
        lines.extend(
            [
                "",
                "### Configured Comparisons",
                "",
                "| Comparison key | Purpose |",
                "| --- | --- |",
            ]
        )
        for comparison in comparisons:
            lines.append(f"| `{comparison.key}` | {comparison.purpose} |")

    if "profile_quality" in outputs:
        lines.extend(
            [
                "",
                "### Profile Targets",
                "",
                "| Profile target | Label |",
                "| --- | --- |",
            ]
        )
        for target in profile_targets:
            lines.append(f"| `{target.key}` | {target.label} |")

    for section in SECTION_DESCRIPTIONS:
        if section in outputs:
            append_output_table(
                lines,
                project_root=project_root,
                section=section,
                outputs=outputs[section],
            )

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary_path
