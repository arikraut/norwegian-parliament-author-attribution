"""Evaluate profiling-signal transfer quality on attribution authors.

Compares extracted profiler probabilities with known attribution-author profile
labels and writes the Phase 3 signal decision report.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.SVM.linear_svm_common import _json_safe
from models.SVM.signals.profiling_signal_extractor import (
    resolve_extraction_stage_source,
)

DEFAULT_EXTRACTION_CONFIG = Path(
    "models/configs/profiling/bokmal_profiling_signal_extraction.toml"
)
DEFAULT_MIN_ATTRIBUTION_TRAIN_MACRO_F1 = 0.60
DEFAULT_MIN_PROFILING_CV_MACRO_F1 = 0.60


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a profiling diagnostics TOML config."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    """Read one profiling diagnostics JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def _materialized_root_from_config(
    project_root: Path,
    config: dict[str, Any],
    *,
    stage: str,
) -> Path:
    """Resolve the materialized attribution/profiling root for one extraction stage."""
    data_cfg = config.get("data", {})
    source_cfg = resolve_extraction_stage_source(config, stage=stage)
    splits_dir = resolve_project_path(project_root, data_cfg.get("splits_dir", "data/splits"))
    return (
        splits_dir
        / str(source_cfg["attribution_split_name"])
        / "materialized_features"
        / str(source_cfg["attribution_materialization_name"])
    )


def profiling_quality_output_dir_from_config(
    config_path: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    stage: str = "final",
) -> Path:
    """Return the canonical output directory for a profiling-quality run."""
    project_root = project_root or find_project_root(config_path.resolve().parent, Path.cwd())
    config = _load_toml(config_path)
    source_cfg = resolve_extraction_stage_source(config, stage=stage)
    data_cfg = config.get("data", {})
    quality_root = (
        output_dir
        if output_dir is not None
        else resolve_project_path(project_root, data_cfg.get("profiling_quality_dir", "results/profiling_quality"))
    )
    return (
        quality_root
        / str(source_cfg["attribution_split_name"])
        / str(source_cfg["profiling_experiment_name"])
        / f"seed_{int(source_cfg['profiling_seed'])}"
    )


def profiling_quality_manifest_path(
    config_path: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Return the manifest path for the canonical profiling-quality report."""
    return profiling_quality_output_dir_from_config(
        config_path,
        project_root=project_root,
        output_dir=output_dir,
    ) / "manifest.json"


def _classes_by_target(materialized_root: Path, targets: list[str]) -> dict[str, list[str]]:
    """Read profiling probability class labels from materialization metadata."""
    columns_path = materialized_root / "profiling_feature_columns.json"
    if not columns_path.exists():
        raise FileNotFoundError(
            f"Profiling feature-column metadata not found: {columns_path}. "
            "Run profiling signal extraction before transfer diagnostics."
        )
    payload = _load_json(columns_path)
    columns = [str(col) for col in payload.get("columns", [])]
    if not columns:
        raise ValueError(f"No profiling columns recorded in {columns_path}")

    classes: dict[str, list[str]] = {}
    for target in targets:
        prefix = f"{target}_"
        target_classes = [col[len(prefix) :] for col in columns if col.startswith(prefix)]
        if not target_classes:
            raise ValueError(
                f"No probability columns found for target {target!r} in {columns_path}"
            )
        classes[target] = target_classes
    return classes


def _load_labels(path: Path) -> np.ndarray:
    """Load a required profiling label array as strings."""
    if not path.exists():
        raise FileNotFoundError(f"Required label array not found: {path}")
    return np.asarray(np.load(path, allow_pickle=True)).astype(str)


def _load_probability_matrix(path: Path) -> np.ndarray:
    """Load a required profiling probability matrix as a dense array."""
    if not path.exists():
        raise FileNotFoundError(f"Required profiling probability matrix not found: {path}")
    matrix = sparse.load_npz(path)
    return np.asarray(matrix.toarray(), dtype=float)


def _metric_labels(y_true: np.ndarray, classes: list[str]) -> list[str]:
    """Build the label set used for profiling classification metrics."""
    return sorted(set(y_true.astype(str).tolist()) | set(classes))


def _majority_baseline(y_true: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Compute majority-class baselines for one profiling target."""
    values, counts = np.unique(y_true, return_counts=True)
    majority_label = str(values[np.argmax(counts)])
    y_majority = np.full(shape=y_true.shape, fill_value=majority_label, dtype=object)
    return {
        "majority_label": majority_label,
        "majority_accuracy": float(accuracy_score(y_true, y_majority)),
        "majority_macro_f1": float(
            f1_score(y_true, y_majority, average="macro", labels=labels, zero_division=0)
        ),
    }


def _brier_score(y_true: np.ndarray, y_proba: np.ndarray, classes: list[str]) -> float:
    """Compute multiclass Brier score for profiling probabilities."""
    class_index = {label: idx for idx, label in enumerate(classes)}
    if any(label not in class_index for label in y_true):
        return float("nan")
    one_hot = np.zeros_like(y_proba, dtype=float)
    for row_idx, label in enumerate(y_true):
        one_hot[row_idx, class_index[str(label)]] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute confidence-bin expected calibration error."""
    if len(y_true) == 0:
        return float("nan")
    confidences = np.max(y_proba, axis=1)
    correctness = (y_pred == y_true).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_weight = float(mask.mean())
        ece += bin_weight * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def _evaluate_predictions(
    *,
    target: str,
    unit_id: str,
    role: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one profiling target for one materialized role."""
    if y_proba.ndim != 2:
        raise ValueError(f"{target}/{role} probability matrix must be two-dimensional.")
    if y_proba.shape[0] != len(y_true):
        raise ValueError(
            f"{target}/{role} row mismatch: {y_proba.shape[0]} probabilities vs "
            f"{len(y_true)} labels."
        )
    if y_proba.shape[1] != len(classes):
        raise ValueError(
            f"{target}/{role} class mismatch: {y_proba.shape[1]} probability columns vs "
            f"{len(classes)} recorded classes."
        )

    class_array = np.asarray(classes, dtype=object)
    y_pred = class_array[np.argmax(y_proba, axis=1)].astype(str)
    labels = _metric_labels(y_true, classes)
    missing_from_model = sorted(set(y_true.astype(str).tolist()) - set(classes))
    baseline = _majority_baseline(y_true, labels)

    can_score_probability = not missing_from_model
    target_log_loss = (
        float(log_loss(y_true, y_proba, labels=classes))
        if can_score_probability
        else float("nan")
    )

    max_probability = np.max(y_proba, axis=1)
    correct_mask = y_pred == y_true
    metrics = {
        "target": target,
        "unit_id": unit_id,
        "role": role,
        "n_samples": int(len(y_true)),
        "n_true_classes": int(len(set(y_true.astype(str).tolist()))),
        "n_model_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "log_loss": target_log_loss,
        "brier_score": _brier_score(y_true, y_proba, classes),
        "expected_calibration_error": _expected_calibration_error(y_true, y_pred, y_proba),
        "mean_max_probability": float(max_probability.mean()),
        "mean_correct_probability": (
            float(max_probability[correct_mask].mean()) if correct_mask.any() else float("nan")
        ),
        "mean_incorrect_probability": (
            float(max_probability[~correct_mask].mean()) if (~correct_mask).any() else float("nan")
        ),
        "labels_missing_from_model": ";".join(missing_from_model),
        **baseline,
    }
    metrics["macro_f1_lift_over_majority"] = (
        float(metrics["macro_f1"]) - float(metrics["majority_macro_f1"])
    )
    metrics["accuracy_lift_over_majority"] = (
        float(metrics["accuracy"]) - float(metrics["majority_accuracy"])
    )
    return metrics, y_pred


def _prediction_frame(
    *,
    row_order_path: Path,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    """Build row-level profiling predictions with probability columns."""
    if not row_order_path.exists():
        raise FileNotFoundError(f"Required row-order file not found: {row_order_path}")
    frame = pd.read_csv(row_order_path, dtype={"id_speech": str, "id_person": str})
    if len(frame) != len(y_true):
        raise ValueError(
            f"Row-order length mismatch for {row_order_path}: {len(frame)} vs {len(y_true)} labels."
        )
    frame = frame.copy()
    frame["profile_target"] = target
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["correct"] = y_true == y_pred
    frame["confidence"] = np.max(y_proba, axis=1)
    for col_idx, class_label in enumerate(classes):
        frame[f"prob_{target}_{class_label}"] = y_proba[:, col_idx]
    return frame


def _evaluate_materialized_roles(
    *,
    materialized_root: Path,
    units: list[dict[str, Any]],
    targets: list[str],
    classes: dict[str, list[str]],
    output_dir: Path,
    roles_by_unit: dict[str, list[str]],
    prediction_prefix: str,
    write_predictions: bool,
) -> pd.DataFrame:
    """Evaluate profiling transfer metrics for configured units and roles."""
    rows: list[dict[str, Any]] = []
    predictions_dir = output_dir / "predictions"
    if write_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for unit in units:
        unit_id = str(unit["unit_id"])
        unit_dir = materialized_root / unit_id
        roles = roles_by_unit.get(unit_id, [])
        for role in roles:
            row_order_path = unit_dir / "row_order" / f"{role}_rows.csv"
            for target in targets:
                y_true = _load_labels(unit_dir / "labels" / f"y_{role}_{target}.npy")
                y_proba = _load_probability_matrix(
                    unit_dir / "matrices" / f"X_{role}_profiling_{target}.npz"
                )
                metrics, y_pred = _evaluate_predictions(
                    target=target,
                    unit_id=unit_id,
                    role=role,
                    y_true=y_true,
                    y_proba=y_proba,
                    classes=classes[target],
                )
                rows.append(metrics)
                if write_predictions:
                    pred_frame = _prediction_frame(
                        row_order_path=row_order_path,
                        target=target,
                        y_true=y_true,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        classes=classes[target],
                    )
                    pred_frame.to_csv(
                        predictions_dir / f"{prediction_prefix}_{unit_id}_{role}_{target}_profile_predictions.csv",
                        index=False,
                    )

    return pd.DataFrame(rows)


def _load_profiling_cv_metrics(
    *,
    project_root: Path,
    extraction_config: dict[str, Any],
    targets: list[str],
) -> dict[str, dict[str, Any]]:
    """Load selected profiling-model CV metrics for target-level decisions."""
    data_cfg = extraction_config.get("data", {})
    source_cfg = resolve_extraction_stage_source(extraction_config, stage="final")
    profiling_results_dir = resolve_project_path(
        project_root,
        data_cfg.get("profiling_results_dir", "results/models"),
    )
    base_dir = (
        profiling_results_dir
        / str(source_cfg["profiling_split_name"])
        / str(source_cfg["profiling_experiment_name"])
        / f"seed_{int(source_cfg['profiling_seed'])}"
    )

    metrics: dict[str, dict[str, Any]] = {}
    for target in targets:
        summary_path = base_dir / target / "candidate_summary.csv"
        best_path = base_dir / target / "best_candidate.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Profiling candidate summary not found for target {target!r}: {summary_path}"
            )
        summary = pd.read_csv(summary_path)
        if summary.empty:
            raise ValueError(f"Profiling candidate summary is empty: {summary_path}")
        best_row = summary.iloc[0].to_dict()
        best_payload = _load_json(best_path) if best_path.exists() else {}
        metrics[target] = {
            "profiling_cv_macro_f1": best_row.get("eval_mean_macro_f1"),
            "profiling_cv_accuracy": best_row.get("eval_mean_accuracy"),
            "profiling_cv_macro_precision": best_row.get("eval_mean_macro_precision"),
            "profiling_cv_macro_recall": best_row.get("eval_mean_macro_recall"),
            "profiling_train_macro_f1": best_row.get("train_mean_macro_f1"),
            "profiling_best_candidate_id": best_payload.get(
                "candidate_id",
                best_row.get("candidate_id"),
            ),
            "profiling_best_feature_set": best_payload.get(
                "feature_set",
                best_row.get("feature_set"),
            ),
            "profiling_candidate_summary_path": relative_to_project(project_root, summary_path),
        }
    return metrics


def _single_metric(metrics: pd.DataFrame, target: str, role: str, metric: str) -> float | None:
    """Return one scalar metric for a target/role pair."""
    if metrics.empty:
        return None
    matched = metrics[(metrics["target"] == target) & (metrics["role"] == role)]
    if matched.empty or metric not in matched.columns:
        return None
    return float(matched.iloc[0][metric])


def _build_target_summary(
    *,
    targets: list[str],
    profiling_cv_metrics: dict[str, dict[str, Any]],
    final_metrics: pd.DataFrame,
    min_attribution_train_macro_f1: float,
    min_profiling_cv_macro_f1: float,
) -> pd.DataFrame:
    """Build target inclusion decisions for profiling transfer."""
    rows: list[dict[str, Any]] = []
    for target in targets:
        cv = profiling_cv_metrics.get(target, {})
        profiling_cv_macro_f1 = cv.get("profiling_cv_macro_f1")
        train_macro_f1 = _single_metric(final_metrics, target, "train", "macro_f1")
        train_accuracy = _single_metric(final_metrics, target, "train", "accuracy")
        train_majority_macro_f1 = _single_metric(final_metrics, target, "train", "majority_macro_f1")
        train_lift = _single_metric(final_metrics, target, "train", "macro_f1_lift_over_majority")
        test_macro_f1 = _single_metric(final_metrics, target, "test", "macro_f1")
        test_accuracy = _single_metric(final_metrics, target, "test", "accuracy")

        profiling_ok = (
            profiling_cv_macro_f1 is not None
            and float(profiling_cv_macro_f1) >= min_profiling_cv_macro_f1
        )
        train_ok = (
            train_macro_f1 is not None
            and float(train_macro_f1) >= min_attribution_train_macro_f1
        )
        include = bool(profiling_ok and train_ok)
        if include:
            reason = "profiling_cv_and_attribution_train_thresholds_met"
        elif not profiling_ok and not train_ok:
            reason = "profiling_cv_and_attribution_train_below_threshold"
        elif not profiling_ok:
            reason = "profiling_cv_below_threshold"
        else:
            reason = "attribution_train_below_threshold"

        rows.append(
            {
                "target": target,
                "profiling_cv_macro_f1": profiling_cv_macro_f1,
                "profiling_cv_accuracy": cv.get("profiling_cv_accuracy"),
                "profiling_train_macro_f1": cv.get("profiling_train_macro_f1"),
                "profiling_best_candidate_id": cv.get("profiling_best_candidate_id"),
                "attribution_train_macro_f1": train_macro_f1,
                "attribution_train_accuracy": train_accuracy,
                "attribution_train_majority_macro_f1": train_majority_macro_f1,
                "attribution_train_macro_f1_lift_over_majority": train_lift,
                "attribution_test_macro_f1": test_macro_f1,
                "attribution_test_accuracy": test_accuracy,
                "decision": "include" if include else "exclude",
                "decision_basis": "attribution_train_profile_metrics",
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _calibration_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Project transfer metrics to the calibration-specific summary table."""
    if metrics.empty:
        return pd.DataFrame()
    cols = [
        "target",
        "role",
        "n_samples",
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "mean_max_probability",
        "mean_correct_probability",
        "mean_incorrect_probability",
    ]
    return metrics[[col for col in cols if col in metrics.columns]].copy()


def run_profiling_transfer_diagnostics(
    extraction_config_path: Path,
    *,
    output_dir: Path | None = None,
    min_attribution_train_macro_f1: float = DEFAULT_MIN_ATTRIBUTION_TRAIN_MACRO_F1,
    min_profiling_cv_macro_f1: float = DEFAULT_MIN_PROFILING_CV_MACRO_F1,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Write profiling-quality diagnostics for Phase 3 signal decisions.

    The include/exclude decision uses only the profiling CV score and the
    attribution-train transfer score. Attribution-test transfer scores are
    reported for interpretation, but are explicitly not decision inputs.
    """
    extraction_config_path = extraction_config_path.resolve()
    project_root = find_project_root(
        extraction_config_path.parent,
        Path.cwd(),
        Path(__file__).resolve().parent,
    )
    config = _load_toml(extraction_config_path)
    final_source_cfg = resolve_extraction_stage_source(config, stage="final")
    dev_source_cfg = resolve_extraction_stage_source(config, stage="dev")
    source_cfg = final_source_cfg
    targets = [str(target) for target in source_cfg["targets"]]

    final_root = _materialized_root_from_config(project_root, config, stage="final")
    final_manifest_path = final_root / "manifest.json"
    if not final_manifest_path.exists():
        raise FileNotFoundError(
            f"Final attribution materialization manifest not found: {final_manifest_path}"
        )
    extraction_manifest_path = final_root / "profiling_extraction_manifest.json"
    if not extraction_manifest_path.exists():
        raise FileNotFoundError(
            f"Final profiling extraction manifest not found: {extraction_manifest_path}. "
            "Run final profiling signal extraction before transfer diagnostics."
        )

    quality_dir = profiling_quality_output_dir_from_config(
        extraction_config_path,
        project_root=project_root,
        output_dir=output_dir,
        stage="final",
    )
    quality_dir.mkdir(parents=True, exist_ok=True)

    final_manifest = _load_json(final_manifest_path)
    final_units = final_manifest.get("units", [])
    if not final_units:
        raise ValueError(f"No units found in final materialization manifest: {final_manifest_path}")

    classes = _classes_by_target(final_root, targets)
    roles_by_unit = {
        str(unit["unit_id"]): ["train", str(unit["eval_role"])]
        for unit in final_units
    }

    if show_progress:
        print(
            f"Profiling transfer diagnostics: {len(targets)} target(s), "
            f"{len(final_units)} final attribution unit(s)."
        )

    final_metrics = _evaluate_materialized_roles(
        materialized_root=final_root,
        units=final_units,
        targets=targets,
        classes=classes,
        output_dir=quality_dir,
        roles_by_unit=roles_by_unit,
        prediction_prefix="attribution",
        write_predictions=True,
    )

    train_metrics = final_metrics[final_metrics["role"] == "train"].copy()
    test_metrics = final_metrics[final_metrics["role"] == "test"].copy()
    train_metrics.to_csv(quality_dir / "attribution_train_profile_metrics.csv", index=False)
    test_metrics.to_csv(quality_dir / "attribution_test_profile_metrics.csv", index=False)

    fold_metrics_path: Path | None = None
    dev_root = _materialized_root_from_config(project_root, config, stage="dev")
    dev_manifest_path = dev_root / "manifest.json"
    dev_extraction_manifest_path = dev_root / "profiling_extraction_manifest.json"
    if dev_manifest_path.exists() and dev_extraction_manifest_path.exists():
        dev_manifest = _load_json(dev_manifest_path)
        dev_units = dev_manifest.get("units", [])
        dev_classes = _classes_by_target(dev_root, targets)
        dev_roles_by_unit = {
            str(unit["unit_id"]): [str(unit["eval_role"])]
            for unit in dev_units
        }
        dev_metrics = _evaluate_materialized_roles(
            materialized_root=dev_root,
            units=dev_units,
            targets=targets,
            classes=dev_classes,
            output_dir=quality_dir,
            roles_by_unit=dev_roles_by_unit,
            prediction_prefix="attribution_dev",
            write_predictions=False,
        )
        fold_metrics_path = quality_dir / "attribution_dev_fold_profile_metrics.csv"
        dev_metrics.to_csv(fold_metrics_path, index=False)

    profiling_cv = _load_profiling_cv_metrics(
        project_root=project_root,
        extraction_config=config,
        targets=targets,
    )
    target_summary = _build_target_summary(
        targets=targets,
        profiling_cv_metrics=profiling_cv,
        final_metrics=final_metrics,
        min_attribution_train_macro_f1=min_attribution_train_macro_f1,
        min_profiling_cv_macro_f1=min_profiling_cv_macro_f1,
    )
    target_summary.to_csv(quality_dir / "target_summary.csv", index=False)

    calibration = _calibration_summary(final_metrics)
    calibration.to_csv(quality_dir / "calibration_summary.csv", index=False)

    selected_targets = target_summary.loc[
        target_summary["decision"] == "include", "target"
    ].astype(str).tolist()
    excluded_targets = target_summary.loc[
        target_summary["decision"] == "exclude", "target"
    ].astype(str).tolist()
    decision = {
        "selected_targets": selected_targets,
        "excluded_targets": excluded_targets,
        "decision_basis": "attribution_train_profile_metrics",
        "created_before_phase3_final": True,
        "thresholds": {
            "min_attribution_train_macro_f1": min_attribution_train_macro_f1,
            "min_profiling_cv_macro_f1": min_profiling_cv_macro_f1,
        },
        "note": (
            "Attribution-test profile metrics are written for reporting only and "
            "must not be used to select Phase 3 profiling signals."
        ),
    }
    write_json(quality_dir / "profiling_signal_decision.json", _json_safe(decision))

    manifest = {
        "run_type": "profiling_transfer_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_config_path": relative_to_project(
            project_root,
            extraction_config_path,
        ),
        "attribution_split_name": source_cfg["attribution_split_name"],
        "attribution_dev_materialization_name": dev_source_cfg[
            "attribution_materialization_name"
        ],
        "attribution_final_materialization_name": source_cfg[
            "attribution_materialization_name"
        ],
        "profiling_split_name": source_cfg["profiling_split_name"],
        "profiling_materialization_name": source_cfg["profiling_materialization_name"],
        "profiling_experiment_name": source_cfg["profiling_experiment_name"],
        "profiling_seed": int(source_cfg["profiling_seed"]),
        "targets": targets,
        "quality_dir": relative_to_project(project_root, quality_dir),
        "outputs": {
            "attribution_train_profile_metrics": relative_to_project(
                project_root, quality_dir / "attribution_train_profile_metrics.csv"
            ),
            "attribution_test_profile_metrics": relative_to_project(
                project_root, quality_dir / "attribution_test_profile_metrics.csv"
            ),
            "target_summary": relative_to_project(project_root, quality_dir / "target_summary.csv"),
            "calibration_summary": relative_to_project(
                project_root, quality_dir / "calibration_summary.csv"
            ),
            "profiling_signal_decision": relative_to_project(
                project_root, quality_dir / "profiling_signal_decision.json"
            ),
        },
        "decision": decision,
    }
    if fold_metrics_path is not None:
        manifest["outputs"]["attribution_dev_fold_profile_metrics"] = relative_to_project(
            project_root, fold_metrics_path
        )
    write_json(quality_dir / "manifest.json", _json_safe(manifest))

    if show_progress:
        print(f"Profiling transfer diagnostics complete. Results at: {quality_dir}")

    return _json_safe(manifest)


def _parse_args() -> argparse.Namespace:
    """Parse the profiling transfer diagnostics CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate profiling probability transfer quality on attribution authors.",
    )
    parser.add_argument(
        "--extraction-config",
        default=str(DEFAULT_EXTRACTION_CONFIG),
        help="Profiling signal extraction config with [stages.dev] and [stages.final].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the profiling-quality output directory.",
    )
    parser.add_argument(
        "--min-attribution-train-macro-f1",
        type=float,
        default=DEFAULT_MIN_ATTRIBUTION_TRAIN_MACRO_F1,
        help="Minimum attribution-train transfer macro F1 required for inclusion.",
    )
    parser.add_argument(
        "--min-profiling-cv-macro-f1",
        type=float,
        default=DEFAULT_MIN_PROFILING_CV_MACRO_F1,
        help="Minimum profiling CV macro F1 required for inclusion.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stage logs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for profiling transfer diagnostics."""
    args = _parse_args()
    extraction_config = Path(args.extraction_config).resolve()
    if not extraction_config.exists():
        print(f"Error: extraction config does not exist: {extraction_config}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    manifest = run_profiling_transfer_diagnostics(
        extraction_config,
        output_dir=output_dir,
        min_attribution_train_macro_f1=args.min_attribution_train_macro_f1,
        min_profiling_cv_macro_f1=args.min_profiling_cv_macro_f1,
        show_progress=not args.no_progress,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
