from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from data_pipeline.utils import (
    copy_config_outputs,
    relative_to_project,
    resolve_project_path,
)

SUPPORTED_BLOCKS = {
    "char",
    "word",
    "stylo",
    "all",
    "profiling",
    "profiling_party",
    "profiling_female",
    "profiling_age_bin",
    "profiling_left_center_right",
    "profiling_hard",
    "profiling_hard_party",
    "profiling_hard_female",
    "profiling_hard_age_bin",
    "profiling_hard_left_center_right",
    "profiling_oracle",
    "profiling_oracle_party",
    "profiling_oracle_female",
    "profiling_oracle_age_bin",
    "profiling_oracle_left_center_right",
}


@dataclass(frozen=True)
class FeatureLayout:
    """Concrete feature-block layout used by shared direct-SVM matrix assembly."""

    name: str
    blocks: tuple[str, ...]
    normalize_rows: bool
    normalize_each_block: bool
    block_weights: dict[str, float]


def _validate_known_keys(
    section_name: str, payload: dict[str, Any], allowed_keys: set[str]
) -> None:
    """Reject unknown TOML keys at parser boundaries."""
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported keys in [{section_name}]: {unknown_keys}")


def _json_safe(value: Any) -> Any:
    """Convert common artifact values into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _manifest_config_path(project_root: Path, config_path: Path | None) -> str | None:
    """Return a manifest config path when this run came from a concrete file."""
    if config_path is None:
        return None
    return relative_to_project(project_root, config_path)


def _copy_config_outputs_if_available(
    config_path: Path | None,
    results_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Copy human-facing config files for path-based CLI calls only."""
    if config_path is None:
        return
    copy_config_outputs(
        config_path,
        results_dir / "model_config.toml",
        artifacts_dir / "model_config.toml",
    )


def _parse_class_weights(raw_values: list[Any]) -> list[str | None]:
    """Convert config or artifact class-weight values into LinearSVC inputs."""
    class_weights: list[str | None] = []
    for value in raw_values:
        value_str = str(value).strip().lower()
        if value_str in {"none", "null"}:
            class_weights.append(None)
        elif value_str == "balanced":
            class_weights.append("balanced")
        else:
            raise ValueError(f"Unsupported class weight option: {value}")
    if not class_weights:
        raise ValueError("At least one class weight option must be provided.")
    return class_weights


def _load_materialization_summary(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load a materialized-feature manifest and return the selected units."""
    data_cfg = config.get("data", {})
    source_cfg = config.get("source", {})
    splits_root = resolve_project_path(
        project_root, data_cfg.get("splits_dir", "data/splits")
    )

    split_name = str(source_cfg["split_name"])
    materialization_name = str(source_cfg["materialization_name"])
    materialized_root = (
        splits_root / split_name / "materialized_features" / materialization_name
    )
    manifest_path = materialized_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Materialization manifest does not exist: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = manifest.get("units", [])
    if not units:
        raise ValueError(f"No materialization units were found in {manifest_path}")

    selected_units = source_cfg.get("units", "all")
    if isinstance(selected_units, str) and selected_units.lower() == "all":
        return materialized_root, manifest, units
    if not isinstance(selected_units, list):
        raise ValueError("source.units must be 'all' or a list of unit ids.")

    keep_ids = {str(unit_id) for unit_id in selected_units}
    filtered_units = [unit for unit in units if str(unit.get("unit_id")) in keep_ids]
    missing_ids = sorted(
        keep_ids - {str(unit.get("unit_id")) for unit in filtered_units}
    )
    if missing_ids:
        raise ValueError(
            f"Requested materialization units were not found: {missing_ids}"
        )
    return materialized_root, manifest, filtered_units


def _build_provenance_block(
    project_root: Path,
    materialized_root: Path,
    materialization_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build manifest provenance linking model outputs to split and feature inputs."""
    split_name = str(materialization_manifest.get("split_name", ""))
    materialization_name = str(materialization_manifest.get("materialization_name", ""))
    materialization_config_path = materialization_manifest.get("config_path")
    row_feature_name = str(materialization_manifest.get("row_feature_name", ""))

    split_dir = materialized_root.parents[1]
    split_manifest_path = split_dir / "manifest.json"
    split_manifest = (
        json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if split_manifest_path.exists()
        else {}
    )
    split_config_path = split_manifest.get("config_path")

    feature_manifest_path = (
        split_dir / "row_features" / row_feature_name / "manifest.json"
    )
    feature_manifest = (
        json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if feature_manifest_path.exists()
        else {}
    )
    feature_config_path = feature_manifest.get("feature_config_path")

    return {
        "split_name": split_name,
        "split_config_path": split_config_path,
        "feature_set_name": row_feature_name,
        "feature_config_path": feature_config_path,
        "materialization_name": materialization_name,
        "materialization_config_path": materialization_config_path,
        "materialized_root": relative_to_project(project_root, materialized_root),
        "split_manifest_path": (
            relative_to_project(project_root, split_manifest_path)
            if split_manifest_path.exists()
            else None
        ),
        "feature_manifest_path": (
            relative_to_project(project_root, feature_manifest_path)
            if feature_manifest_path.exists()
            else None
        ),
        "materialization_manifest_path": relative_to_project(
            project_root, materialized_root / "manifest.json"
        ),
    }


def _validate_candidate_search_units(units: list[dict[str, Any]]) -> None:
    """Ensure candidate-search materialization units are validation folds."""
    invalid_units: list[str] = []
    for unit in units:
        unit_id = str(unit.get("unit_id", "<missing-unit-id>"))
        eval_role = str(unit.get("eval_role", "<missing-eval-role>"))
        if eval_role != "val":
            invalid_units.append(f"{unit_id} (eval_role={eval_role!r})")

    if invalid_units:
        joined_units = ", ".join(invalid_units)
        raise ValueError(
            "Standard attribution candidate search only supports materialization units with "
            f"eval_role='val'. Found: {joined_units}. Use a dedicated final-evaluation path for test units."
        )


def _validate_final_evaluation_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single test unit required by final attribution evaluation."""
    if len(units) != 1:
        raise ValueError(
            "Final attribution evaluation requires exactly one materialization unit with eval_role='test'. "
            f"Found {len(units)} unit(s)."
        )

    unit = units[0]
    unit_id = str(unit.get("unit_id", "<missing-unit-id>"))
    eval_role = str(unit.get("eval_role", "<missing-eval-role>"))
    if eval_role != "test":
        raise ValueError(
            "Final attribution evaluation requires a materialization unit with eval_role='test'. "
            f"Found {unit_id} (eval_role={eval_role!r})."
        )
    return unit


def _resolve_output_dirs(
    project_root: Path, config: dict[str, Any], config_stem: str
) -> tuple[Path, Path]:
    """Resolve and create model results and artifact directories."""
    data_cfg = config.get("data", {})
    source_cfg = config.get("source", {})
    experiment_cfg = config.get("experiment", {})

    split_name = str(source_cfg["split_name"])
    experiment_name = str(experiment_cfg.get("name", config_stem))
    seed = int(experiment_cfg.get("seed", 42))

    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/models")
    )
    artifacts_root = resolve_project_path(
        project_root, data_cfg.get("artifacts_dir", "models/artifacts/attribution")
    )

    results_dir = results_root / split_name / experiment_name / f"seed_{seed}"
    artifacts_dir = artifacts_root / split_name / experiment_name / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, artifacts_dir


def _load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    """Load a materialized sparse feature matrix as CSR."""
    if not path.exists():
        raise FileNotFoundError(f"Required matrix does not exist: {path}")
    return sparse.load_npz(path).tocsr()


def _load_labels(path: Path) -> np.ndarray:
    """Load a materialized label array."""
    if not path.exists():
        raise FileNotFoundError(f"Required label array does not exist: {path}")
    return np.load(path, allow_pickle=True)


def _matrix_path(unit_dir: Path, role: str, block: str) -> Path:
    """Return the canonical matrix path for a role and feature block."""
    return unit_dir / "matrices" / f"X_{role}_{block}.npz"


def _resolved_variant_blocks(
    unit: dict[str, Any], layout: FeatureLayout
) -> tuple[str, ...]:
    """Resolve ``blocks=['all']`` for a feature layout against a unit manifest."""
    if "all" not in layout.blocks:
        return layout.blocks

    enabled_blocks = unit.get("enabled_blocks")
    if not isinstance(enabled_blocks, list) or not enabled_blocks:
        raise ValueError(
            f"Materialization unit '{unit.get('unit_id')}' does not expose enabled_blocks, "
            "so blocks=['all'] cannot be resolved."
        )
    concrete_blocks = tuple(
        str(block)
        for block in enabled_blocks
        if str(block) in {"char", "word", "stylo"}
    )
    if not concrete_blocks:
        raise ValueError(
            f"Materialization unit '{unit.get('unit_id')}' does not expose any concrete blocks "
            "that can satisfy blocks=['all']."
        )
    return concrete_blocks


def _validate_variant_availability(unit: dict[str, Any], layout: FeatureLayout) -> None:
    """Ensure a materialized unit exposes every block requested by a layout."""
    requested_blocks = _resolved_variant_blocks(unit, layout)
    enabled_blocks = unit.get("enabled_blocks")
    derived_blocks = unit.get("derived_blocks", [])
    if isinstance(enabled_blocks, list):
        enabled_set = {str(block) for block in enabled_blocks} | {
            str(block) for block in derived_blocks
        }
        missing = sorted(set(requested_blocks) - enabled_set)
        if missing:
            raise ValueError(
                f"Materialization unit '{unit.get('unit_id')}' does not provide requested blocks {missing}. "
                f"Enabled blocks: {sorted(enabled_set)}."
            )


def _build_feature_matrix(
    unit: dict[str, Any], unit_dir: Path, role: str, layout: FeatureLayout
) -> sparse.csr_matrix:
    """Assemble and normalize a direct-SVM feature matrix from materialized blocks."""
    requested_blocks = _resolved_variant_blocks(unit, layout)
    matrices: list[sparse.csr_matrix] = []
    for block in requested_blocks:
        matrix = _load_sparse_matrix(_matrix_path(unit_dir, role, block))
        if layout.normalize_each_block:
            matrix = normalize(matrix, norm="l2", axis=1, copy=False)
        weight = float(layout.block_weights.get(block, 1.0))
        if weight != 1.0:
            matrix = matrix.multiply(weight).tocsr()
        matrices.append(matrix)

    if len(matrices) == 1:
        combined = matrices[0]
    else:
        combined = sparse.hstack(matrices, format="csr")

    if layout.normalize_rows:
        combined = normalize(combined, norm="l2", axis=1, copy=False)

    return combined.tocsr()


def _fit_linear_svm(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
    model_cfg: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[LinearSVC, list[str], float]:
    """Fit a LinearSVC from explicit hyperparameters and return fit diagnostics."""
    started_at = time.perf_counter()
    clf = LinearSVC(
        C=float(c_value),
        class_weight=class_weight,
        max_iter=int(model_cfg.get("max_iter", 20_000)),
        tol=float(model_cfg.get("tol", 1e-4)),
        dual=model_cfg.get("dual", "auto"),
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_train, y_train, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - started_at
    convergence_messages = [
        str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    return clf, convergence_messages, fit_seconds


def _top_k_metrics(
    y_true: np.ndarray,
    scores: np.ndarray | None,
    classes: np.ndarray,
    top_k_values: list[int],
) -> dict[str, float]:
    """Compute available top-k accuracy metrics from decision scores."""
    metrics: dict[str, float] = {}
    if scores is None or scores.ndim != 2:
        return metrics

    for k in top_k_values:
        if k <= scores.shape[1]:
            metrics[f"top{k}_accuracy"] = float(
                top_k_accuracy_score(y_true, scores, k=k, labels=classes)
            )
    return metrics


def _evaluate_predictions(
    clf: LinearSVC,
    x: sparse.csr_matrix,
    y: np.ndarray,
    split_name: str,
    top_k_values: list[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, float]:
    """Score a fitted classifier and return metrics, predictions, scores, and time."""
    started_at = time.perf_counter()
    y_pred = clf.predict(x)
    predict_seconds = time.perf_counter() - started_at
    scores = np.asarray(clf.decision_function(x))

    metrics = {
        "split": split_name,
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        "macro_precision": float(
            precision_score(y, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, y_pred, average="macro", zero_division=0)
        ),
    }
    metrics.update(_top_k_metrics(y, scores, clf.classes_, top_k_values))
    return metrics, y_pred, scores, predict_seconds


def _prediction_frame(
    unit_dir: Path,
    eval_role: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    classes: np.ndarray,
    save_top_k: int,
) -> pd.DataFrame:
    """Build the row-level prediction CSV frame for a materialized unit."""
    row_order_path = unit_dir / "row_order" / f"{eval_role}_rows.csv"
    row_order = pd.read_csv(row_order_path)
    if len(row_order) != len(y_true):
        raise ValueError(
            f"Row-order length mismatch for {row_order_path}: expected {len(y_true)}, got {len(row_order)}"
        )

    frame = row_order.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["correct"] = np.asarray(y_true) == np.asarray(y_pred)

    if scores is not None and scores.ndim == 2 and save_top_k > 0:
        top_k = min(save_top_k, scores.shape[1])
        ranked = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        top_labels = classes[ranked]
        frame["pred_score"] = scores.max(axis=1)
        for rank in range(top_k):
            frame[f"top{rank + 1}_label"] = top_labels[:, rank]
            frame[f"top{rank + 1}_score"] = scores[
                np.arange(scores.shape[0]), ranked[:, rank]
            ]

    return frame
