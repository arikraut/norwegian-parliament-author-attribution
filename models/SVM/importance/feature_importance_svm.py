from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_pipeline.row_features.stylometry import stylometry_feature_family
from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
)
from models.SVM.linear_svm_common import _json_safe


CORE_BLOCKS = {"char", "word", "stylo"}
CORE_BLOCK_ALIASES = {"spacy": "stylo"}
PROFILE_NAME_ALIASES = {"left_senter_right": "left_center_right"}
PROFILE_TARGET_ALIASES = {
    "left_center_right": "left_senter_right",
    "left_senter_right": "left_center_right",
}


@dataclass(frozen=True)
class FeatureName:
    """Describe one model-matrix column after block concatenation."""

    name: str
    block: str
    raw_name: str
    subfamily: str


@dataclass(frozen=True)
class FinalUnitPaths:
    """Bundle condition-specific final-model paths for feature-importance analysis."""

    project_root: Path
    manifest_path: Path
    results_dir: Path
    condition_results_dir: Path
    materialized_root: Path
    unit_id: str
    eval_role: str
    condition_id: str
    condition_label: str
    resolved_candidate_path: Path
    model_path: Path | None = None
    model_dir: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON file from a trusted project artifact path."""
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_root(manifest_path: Path) -> Path:
    """Locate the repository root from a model manifest path."""
    return find_project_root(
        manifest_path.resolve().parent,
        Path(__file__).resolve().parent,
    )


def _resolve_project_artifact(project_root: Path, path_value: str | Path) -> Path:
    """Resolve a manifest path field relative to the project root."""
    return resolve_project_path(project_root, str(path_value))


def canonical_feature_name_part(value: str) -> str:
    """Canonicalize historical naming fragments used in feature metadata."""

    canonical = str(value)
    for historical, replacement in PROFILE_NAME_ALIASES.items():
        canonical = canonical.replace(historical, replacement)
    canonical = canonical.replace("left_center_right_senter", "left_center_right_center")
    return canonical


def canonical_feature_block(block: str) -> str:
    """Return the report-facing block name for a raw artifact block name."""

    raw = str(block)
    return canonical_feature_name_part(CORE_BLOCK_ALIASES.get(raw, raw))


def is_core_feature_block(block: str) -> bool:
    """Return whether a raw or canonical block name is a core feature block."""

    return canonical_feature_block(block) in CORE_BLOCKS


def core_column_keys(block: str) -> tuple[str, ...]:
    """Return feature-column JSON keys to try for one core block."""

    raw = str(block)
    canonical = canonical_feature_block(raw)
    keys = [raw, canonical]
    if canonical == "stylo":
        keys.extend(["stylo", "spacy"])
    return tuple(dict.fromkeys(keys))


def profile_target_candidates(target: str) -> tuple[str, ...]:
    """Return raw/canonical target keys accepted in derived feature columns."""

    raw = str(target)
    candidates = [raw, PROFILE_TARGET_ALIASES.get(raw, raw)]
    return tuple(dict.fromkeys(candidates))


def load_final_manifest(
    manifest_path: Path,
    *,
    expected_run_type: str | None = None,
) -> dict[str, Any]:
    """Load a final model manifest and optionally verify its run type."""
    manifest = _read_json(manifest_path)
    if expected_run_type is not None and manifest.get("run_type") != expected_run_type:
        raise ValueError(
            f"{manifest_path} has run_type={manifest.get('run_type')!r}; "
            f"expected {expected_run_type!r}."
        )
    return manifest


def _select_condition_result(
    manifest: dict[str, Any],
    condition_id: str | None,
) -> dict[str, Any]:
    """Return one condition result from a condition-aware final manifest."""
    condition_results = manifest.get("condition_results")
    if not isinstance(condition_results, list) or not condition_results:
        raise ValueError("Final manifest must include non-empty condition_results.")
    if condition_id is None:
        if len(condition_results) != 1:
            available = [str(row.get("condition_id")) for row in condition_results]
            raise ValueError(
                "condition_id is required when a final manifest has multiple "
                f"conditions. Available: {available}"
            )
        return dict(condition_results[0])

    for row in condition_results:
        if str(row.get("condition_id")) == str(condition_id):
            return dict(row)
    available = [str(row.get("condition_id")) for row in condition_results]
    raise ValueError(f"Unknown condition_id {condition_id!r}. Available: {available}")


def _condition_unit_ids(
    manifest: dict[str, Any],
    condition_result: dict[str, Any],
) -> tuple[str, str]:
    """Resolve the final unit id/eval role for a condition result."""
    if condition_result.get("unit_id") and condition_result.get("eval_role"):
        return str(condition_result["unit_id"]), str(condition_result["eval_role"])
    units = manifest.get("source_manifest", {}).get("units", [])
    if isinstance(units, list) and len(units) == 1:
        unit = units[0]
        return str(unit["unit_id"]), str(unit["eval_role"])
    raise ValueError(
        "Final manifest must expose unit_id/eval_role in condition_results "
        "or exactly one source_manifest unit."
    )


def resolve_final_unit_paths(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    condition_id: str | None = None,
) -> FinalUnitPaths:
    """Resolve condition-specific final-model paths from a loaded manifest."""
    project_root = _resolve_project_root(manifest_path)
    condition_result = _select_condition_result(manifest, condition_id)
    resolved_candidate_path = _resolve_project_artifact(
        project_root,
        condition_result["resolved_candidate_path"],
    )
    unit_id, eval_role = _condition_unit_ids(manifest, condition_result)
    model_path = condition_result.get("model_path")
    model_dir = condition_result.get("model_dir")
    return FinalUnitPaths(
        project_root=project_root,
        manifest_path=manifest_path,
        results_dir=_resolve_project_artifact(project_root, manifest["results_dir"]),
        condition_results_dir=resolved_candidate_path.parent,
        materialized_root=_resolve_project_artifact(
            project_root, manifest["materialized_root"]
        ),
        unit_id=unit_id,
        eval_role=eval_role,
        condition_id=str(condition_result["condition_id"]),
        condition_label=str(condition_result.get("condition_label", condition_result["condition_id"])),
        resolved_candidate_path=resolved_candidate_path,
        model_path=(
            _resolve_project_artifact(project_root, model_path)
            if model_path is not None
            else None
        ),
        model_dir=(
            _resolve_project_artifact(project_root, model_dir)
            if model_dir is not None
            else None
        ),
    )


def _resolve_requested_blocks(unit_dir: Path, blocks: list[str]) -> list[str]:
    """Resolve a candidate block list, including the trainer's blocks=['all'] form."""
    if "all" not in blocks:
        return blocks
    if len(blocks) > 1:
        raise ValueError("blocks=['all'] cannot be combined with other blocks.")

    unit_manifest = _read_json(unit_dir / "manifest.json")
    enabled_blocks = [
        str(block)
        for block in unit_manifest.get("enabled_blocks", [])
        if is_core_feature_block(str(block))
    ]
    if not enabled_blocks:
        raise ValueError(f"No core enabled blocks found in {unit_dir / 'manifest.json'}")
    return enabled_blocks


def _feature_subfamily(block: str, raw_name: str) -> str:
    """Map a feature to the subfamily column used in importance CSVs."""
    if canonical_feature_block(block) == "stylo":
        return stylometry_feature_family(raw_name)
    return ""


def _feature_records_for_columns(block: str, columns: list[str]) -> list[FeatureName]:
    """Build FeatureName records for one resolved block and ordered column list."""
    canonical_block = canonical_feature_block(block)
    return [
        FeatureName(
            name=f"{canonical_block}:{canonical_raw_name}",
            block=canonical_block,
            raw_name=canonical_raw_name,
            subfamily=_feature_subfamily(canonical_block, canonical_raw_name),
        )
        for raw_name in columns
        for canonical_raw_name in [canonical_feature_name_part(raw_name)]
    ]


def _target_columns(payload: dict[str, Any], target: str | None) -> list[str]:
    """Return all derived feature columns or the subset belonging to one target."""
    columns = [str(column) for column in payload.get("columns", [])]
    if target is None:
        return columns

    target_columns: list[str] = []
    for candidate in profile_target_candidates(target):
        prefix = f"{candidate}_"
        target_columns = [column for column in columns if column.startswith(prefix)]
        if target_columns:
            break
    if not target_columns:
        raise ValueError(
            f"No derived feature columns found for target {target!r} "
            f"(tried={profile_target_candidates(target)})."
        )
    return target_columns


def _core_feature_columns(
    core_columns: dict[str, Any],
    block: str,
    *,
    unit_dir: Path,
) -> list[str]:
    """Load core feature columns for either canonical or historical block keys."""

    for key in core_column_keys(block):
        columns = [str(column) for column in core_columns.get(key, [])]
        if columns:
            return columns
    raise ValueError(
        f"No feature columns found for block {block!r} in "
        f"{unit_dir / 'feature_columns.json'} "
        f"(tried keys={core_column_keys(block)})."
    )


def _derived_feature_columns(
    materialized_root: Path,
    block: str,
) -> list[str]:
    """Load derived profiling or oracle column names for one requested block."""
    if block == "profiling_oracle":
        return _target_columns(
            _read_json(materialized_root / "oracle_feature_columns.json"),
            None,
        )
    if block.startswith("profiling_oracle_"):
        target = block.removeprefix("profiling_oracle_")
        return _target_columns(
            _read_json(materialized_root / "oracle_feature_columns.json"),
            target,
        )
    if block == "profiling_hard":
        return _target_columns(
            _read_json(materialized_root / "profiling_hard_feature_columns.json"),
            None,
        )
    if block.startswith("profiling_hard_"):
        target = block.removeprefix("profiling_hard_")
        return _target_columns(
            _read_json(materialized_root / "profiling_hard_feature_columns.json"),
            target,
        )
    if block == "profiling":
        return _target_columns(
            _read_json(materialized_root / "profiling_feature_columns.json"),
            None,
        )
    if block.startswith("profiling_"):
        target = block.removeprefix("profiling_")
        return _target_columns(
            _read_json(materialized_root / "profiling_feature_columns.json"),
            target,
        )
    raise ValueError(f"Unsupported derived feature block: {block}")


def load_feature_names_for_blocks(
    materialized_root: Path,
    unit_id: str,
    blocks: list[str],
) -> list[FeatureName]:
    """Load ordered feature names for the same block order used by model training."""
    unit_dir = materialized_root / unit_id
    resolved_blocks = _resolve_requested_blocks(unit_dir, [str(block) for block in blocks])
    core_columns = _read_json(unit_dir / "feature_columns.json")

    feature_names: list[FeatureName] = []
    for block in resolved_blocks:
        if is_core_feature_block(block):
            columns = _core_feature_columns(core_columns, block, unit_dir=unit_dir)
        else:
            columns = _derived_feature_columns(materialized_root, block)
        feature_names.extend(_feature_records_for_columns(block, columns))
    return feature_names


def coef_rows_by_class(
    coef: np.ndarray,
    classes: np.ndarray,
) -> list[tuple[Any, np.ndarray]]:
    """Return one signed coefficient vector per class, including binary models."""
    coef_array = np.asarray(coef)
    class_array = np.asarray(classes)
    if coef_array.shape[0] == len(class_array):
        return [
            (_json_safe(class_label), coef_array[class_idx])
            for class_idx, class_label in enumerate(class_array)
        ]
    if coef_array.shape[0] == 1 and len(class_array) == 2:
        return [
            (_json_safe(class_array[0]), -coef_array[0]),
            (_json_safe(class_array[1]), coef_array[0]),
        ]
    raise ValueError(
        f"Cannot align coef rows {coef_array.shape[0]} with {len(class_array)} classes."
    )


def _feature_metadata_frame(feature_names: list[FeatureName]) -> pd.DataFrame:
    """Convert ordered FeatureName records to a DataFrame."""
    return pd.DataFrame(
        {
            "feature_name": [feature.name for feature in feature_names],
            "block": [feature.block for feature in feature_names],
            "raw_name": [feature.raw_name for feature in feature_names],
            "subfamily": [feature.subfamily for feature in feature_names],
        }
    )


def compute_global_importance(
    coef: np.ndarray,
    feature_names: list[FeatureName],
) -> pd.DataFrame:
    """Compute mean and max absolute coefficient importance per feature."""
    coef_array = np.asarray(coef)
    if coef_array.shape[1] != len(feature_names):
        raise ValueError(
            f"Coefficient width {coef_array.shape[1]} does not match "
            f"{len(feature_names)} feature names."
        )

    importance_mean = np.mean(np.abs(coef_array), axis=0)
    importance_max = np.max(np.abs(coef_array), axis=0)
    total_importance = float(importance_mean.sum())
    metadata = _feature_metadata_frame(feature_names)
    metadata["importance_mean"] = importance_mean
    metadata["importance_max"] = importance_max
    metadata["importance_share"] = (
        importance_mean / total_importance if total_importance else 0.0
    )
    return metadata.sort_values(
        ["importance_mean", "importance_max", "feature_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def compute_block_importance(global_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature importance into block-level totals and shares."""
    grouped = (
        global_df.groupby("block", sort=False)["importance_mean"]
        .agg(total_importance="sum", n_features="size")
        .reset_index()
    )
    total = float(grouped["total_importance"].sum())
    grouped["importance_share"] = (
        grouped["total_importance"] / total if total else 0.0
    )
    grouped["mean_per_feature"] = (
        grouped["total_importance"] / grouped["n_features"].replace(0, np.nan)
    ).fillna(0.0)
    return grouped[
        ["block", "total_importance", "importance_share", "n_features", "mean_per_feature"]
    ].sort_values("total_importance", ascending=False, kind="stable").reset_index(
        drop=True
    )


def compute_stylo_subfamily_importance(global_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stylo feature importance into stylometry subfamilies."""
    stylo_df = global_df[global_df["block"] == "stylo"].copy()
    columns = [
        "subfamily",
        "total_importance",
        "importance_share",
        "n_features",
        "mean_per_feature",
    ]
    if stylo_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        stylo_df.groupby("subfamily", sort=False)["importance_mean"]
        .agg(total_importance="sum", n_features="size")
        .reset_index()
    )
    total = float(grouped["total_importance"].sum())
    grouped["importance_share"] = (
        grouped["total_importance"] / total if total else 0.0
    )
    grouped["mean_per_feature"] = (
        grouped["total_importance"] / grouped["n_features"].replace(0, np.nan)
    ).fillna(0.0)
    return grouped[columns].sort_values(
        "total_importance", ascending=False, kind="stable"
    ).reset_index(drop=True)


def compute_per_author_top_features(
    coef: np.ndarray,
    feature_names: list[FeatureName],
    classes: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the strongest positive and negative features for each class."""
    metadata = _feature_metadata_frame(feature_names)
    rows: list[dict[str, Any]] = []
    top_k = min(int(top_n), len(feature_names))
    for author, class_coef in coef_rows_by_class(coef, classes):
        positive_idx = np.argsort(class_coef)[::-1][:top_k]
        negative_idx = np.argsort(class_coef)[:top_k]
        for direction, indices in [
            ("positive", positive_idx),
            ("negative", negative_idx),
        ]:
            for rank, feature_idx in enumerate(indices, start=1):
                feature_row = metadata.iloc[int(feature_idx)]
                rows.append(
                    {
                        "author": author,
                        "rank": rank,
                        "direction": direction,
                        "feature_name": feature_row["feature_name"],
                        "block": feature_row["block"],
                        "raw_name": feature_row["raw_name"],
                        "subfamily": feature_row["subfamily"],
                        "coef_value": float(class_coef[int(feature_idx)]),
                    }
                )
    return pd.DataFrame(rows)


def plot_block_importance(block_importance_df: pd.DataFrame, output_path: Path) -> None:
    """Write a horizontal bar chart for block-level coefficient importance."""
    plot_df = block_importance_df.sort_values(
        "total_importance", ascending=True, kind="stable"
    )
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(plot_df))))
    ax.barh(plot_df["block"], plot_df["total_importance"], color="#2f6f9f")
    ax.set_xlabel("Total mean |coefficient|")
    ax.set_ylabel("Feature block")
    ax.set_title("Block-level coefficient importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _analysis_outputs(project_root: Path, output_dir: Path) -> dict[str, str]:
    """Return relative paths for the direct SVM output files."""
    return {
        "global_importance": relative_to_project(
            project_root, output_dir / "global_importance.csv"
        ),
        "block_importance": relative_to_project(
            project_root, output_dir / "block_importance.csv"
        ),
        "stylo_subfamily_importance": relative_to_project(
            project_root, output_dir / "stylo_subfamily_importance.csv"
        ),
        "per_author_top_features": relative_to_project(
            project_root, output_dir / "per_author_top_features.csv"
        ),
        "block_importance_plot": relative_to_project(
            project_root, output_dir / "block_importance.png"
        ),
    }


def run_condition_importance_analysis(
    manifest_path: Path,
    top_n: int = 20,
    *,
    condition_id: str,
) -> dict[str, Any]:
    """Run coefficient feature-importance analysis for one final direct condition."""
    manifest_path = manifest_path.resolve()
    manifest = load_final_manifest(
        manifest_path,
        expected_run_type="condition_final_evaluation",
    )
    paths = resolve_final_unit_paths(
        manifest_path,
        manifest,
        condition_id=condition_id,
    )
    if paths.model_path is None:
        raise ValueError(
            f"{manifest_path} condition {paths.condition_id!r} does not record model_path."
        )

    model = joblib.load(paths.model_path)
    resolved_candidate = _read_json(paths.resolved_candidate_path)
    feature_names = load_feature_names_for_blocks(
        paths.materialized_root,
        paths.unit_id,
        [str(block) for block in resolved_candidate["blocks"]],
    )
    global_df = compute_global_importance(model.coef_, feature_names)
    block_df = compute_block_importance(global_df)
    subfamily_df = compute_stylo_subfamily_importance(global_df)
    per_author_df = compute_per_author_top_features(
        model.coef_,
        feature_names,
        np.asarray(model.classes_),
        top_n=top_n,
    )

    output_dir = paths.condition_results_dir / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_df.to_csv(output_dir / "global_importance.csv", index=False)
    block_df.to_csv(output_dir / "block_importance.csv", index=False)
    subfamily_df.to_csv(output_dir / "stylo_subfamily_importance.csv", index=False)
    per_author_df.to_csv(output_dir / "per_author_top_features.csv", index=False)
    plot_block_importance(block_df, output_dir / "block_importance.png")

    return {
        "run_type": "feature_importance",
        "source_manifest": relative_to_project(paths.project_root, manifest_path),
        "condition_id": paths.condition_id,
        "condition_label": paths.condition_label,
        "results_dir": relative_to_project(paths.project_root, output_dir),
        "n_features": len(feature_names),
        "outputs": _analysis_outputs(paths.project_root, output_dir),
    }


def run_importance_analysis(
    manifest_path: Path,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run direct-SVM feature importance for every condition in a final manifest."""
    manifest_path = manifest_path.resolve()
    manifest = load_final_manifest(
        manifest_path,
        expected_run_type="condition_final_evaluation",
    )
    condition_ids = [
        str(row["condition_id"])
        for row in manifest.get("condition_results", [])
    ]
    results = [
        run_condition_importance_analysis(
            manifest_path,
            top_n=top_n,
            condition_id=condition_id,
        )
        for condition_id in condition_ids
    ]
    return {
        "run_type": "feature_importance_all_conditions",
        "source_manifest": relative_to_project(_resolve_project_root(manifest_path), manifest_path),
        "condition_count": len(results),
        "conditions": results,
    }
