"""Phase 2 profiling classifier trainer.

Trains calibrated ``LinearSVC`` profilers on background authors for the profile
targets used as Phase 3 auxiliary signals. Handles development candidate search
and final refits. See ``models/SVM/README.md`` for workflow and artifact
contracts.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import json
from scipy import sparse
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

from data_pipeline.utils import (
    copy_config_outputs,
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.SVM.linear_svm_common import (
    FeatureLayout,
    _build_feature_matrix,
    _build_provenance_block,
    _evaluate_predictions,
    _fit_linear_svm,
    _load_labels,
    _load_materialization_summary,
    _prediction_frame,
    _resolve_output_dirs,
    _validate_candidate_search_units,
    _validate_known_keys,
    _validate_variant_availability,
)
from models.SVM.training.profiling_selection import (
    ProfilingCandidateSpec,
    best_profiling_candidate,
    profiling_candidate_from_payload,
    profiling_candidate_grid,
    profiling_candidate_metric_row,
    summarize_profiling_candidates,
)

SUPPORTED_MODEL_FAMILY = "linear_svm"
CALIBRATION_CV = 3
CALIBRATION_METHOD = "sigmoid"
_SUPPORTED_WEIGHTING_MODES = {"none", "inverse_author_speech_count"}


# ── Author-weight helpers ─────────────────────────────────────────────────────


def _load_author_ids(unit_dir: Path, role: str) -> np.ndarray:
    """Return the id_person array for all rows in a given unit/role, in row order."""
    row_order = pd.read_csv(
        unit_dir / "row_order" / f"{role}_rows.csv", dtype={"id_person": str}
    )
    return row_order["id_person"].astype(str).to_numpy()


def _inverse_author_weights(
    author_ids: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """Compute per-row inverse-author-frequency weights.

    Each row gets weight 1 / (number of rows belonging to that author). When
    normalize=True the weights are rescaled so their mean is 1.0, keeping the
    effective regularization scale close to the unweighted baseline.
    """
    counts = pd.Series(author_ids).value_counts()
    weights = np.array([1.0 / counts[author_id] for author_id in author_ids], dtype=float)
    if normalize and len(weights):
        weights = weights / weights.mean()
    return weights


def _author_weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, float]:
    """Compute author-weighted versions of the standard classification metrics."""
    return {
        "author_weighted_accuracy": float(
            accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        ),
        "author_weighted_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", sample_weight=sample_weights, zero_division=0)
        ),
        "author_weighted_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weights, zero_division=0)
        ),
        "author_weighted_macro_precision": float(
            precision_score(y_true, y_pred, average="macro", sample_weight=sample_weights, zero_division=0)
        ),
        "author_weighted_macro_recall": float(
            recall_score(y_true, y_pred, average="macro", sample_weight=sample_weights, zero_division=0)
        ),
    }


def _author_weighting_mode(author_balance_cfg: dict[str, Any] | None) -> str:
    """Return the configured author-balance weighting mode."""
    if not author_balance_cfg:
        return "none"
    return str(author_balance_cfg.get("train_sample_weighting", "none")).strip().lower()


def _train_author_weights(
    unit_dir: Path,
    author_balance_cfg: dict[str, Any] | None,
) -> np.ndarray | None:
    """Build train sample weights for profiling units when author balancing is enabled."""
    if _author_weighting_mode(author_balance_cfg) != "inverse_author_speech_count":
        return None
    normalize = bool(author_balance_cfg.get("normalize_train_weights", True))
    return _inverse_author_weights(
        _load_author_ids(unit_dir, "train"),
        normalize=normalize,
    )


def _eval_author_weights(
    unit_dir: Path,
    eval_role: str,
    author_balance_cfg: dict[str, Any] | None,
) -> np.ndarray | None:
    """Build evaluation weights when author-weighted metrics are requested."""
    if not author_balance_cfg or not bool(
        author_balance_cfg.get("report_author_weighted_eval", False)
    ):
        return None
    if _author_weighting_mode(author_balance_cfg) != "inverse_author_speech_count":
        return None
    return _inverse_author_weights(
        _load_author_ids(unit_dir, eval_role),
        normalize=False,
    )


# ── Calibrated model fitting ──────────────────────────────────────────────────


def _fit_calibrated_family_model(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    candidate: ProfilingCandidateSpec,
    model_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[CalibratedClassifierCV, list[str], float, dict[str, Any]]:
    """Fit a CalibratedClassifierCV(LinearSVC) for one profiling fold.

    The optional sample_weight is passed to the calibrator's fit call so that
    inverse-author weighting is applied consistently to both the base SVM and
    the calibration layer. When None, behaviour is identical to the unweighted
    baseline.

    Returns (calibrator, convergence_messages, fit_seconds, calibration_meta).
    """
    method = str(calibration_cfg.get("method", "sigmoid")).strip().lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"Unsupported calibration.method: {method!r}")

    requested_cv = int(calibration_cfg.get("cv", 3))
    if requested_cv < 2:
        raise ValueError("calibration.cv must be >= 2.")

    _, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    effective_cv = min(requested_cv, min_count)
    if effective_cv < 2:
        raise ValueError(
            f"Cannot calibrate with cv={requested_cv} because the minimum class "
            f"support in train is {min_count}."
        )

    started_at = time.perf_counter()
    base = LinearSVC(
        C=float(candidate.c_value),
        class_weight=candidate.class_weight,
        max_iter=int(model_cfg.get("max_iter", 20_000)),
        tol=float(model_cfg.get("tol", 1e-4)),
        dual=model_cfg.get("dual", "auto"),
        random_state=seed,
    )
    calibrator = CalibratedClassifierCV(estimator=base, method=method, cv=effective_cv)

    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        calibrator.fit(x_train, y_train, sample_weight=sample_weight)
        convergence_messages = [
            str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
        ]

    fit_seconds = time.perf_counter() - started_at
    calibration_meta = {
        "requested_cv": requested_cv,
        "effective_cv": effective_cv,
        "min_class_support": min_count,
        "method": method,
    }
    return calibrator, convergence_messages, fit_seconds, calibration_meta


# ── Config parsing and validation ────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse the profiling-classifier trainer CLI."""
    parser = argparse.ArgumentParser(
        description="Train calibrated profiling classifiers from materialized fold features.",
    )
    parser.add_argument(
        "--config",
        default="models/configs/profiling/bokmal_profiling_linear_svm.toml",
        help="Path to a model config TOML file under models/configs/profiling/.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Train final classifiers on all profiling authors (requires dev run first). "
            "Required before Phase 3A or Phase 3B final evaluation."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage logs.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read a profiling-classifier TOML config."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _validate_profiling_config(config: dict[str, Any]) -> None:
    """Validate the profiling model config dict."""
    _validate_known_keys(
        "root",
        config,
        {"experiment", "data", "source", "model", "feature_sets", "author_balance"},
    )
    _validate_known_keys(
        "experiment",
        config.get("experiment", {}),
        {"name", "seed", "selection_metric", "save_prediction_top_k", "n_jobs"},
    )
    _validate_known_keys(
        "data",
        config.get("data", {}),
        {"splits_dir", "results_dir", "artifacts_dir"},
    )
    source_cfg = config.get("source", {})
    _validate_known_keys(
        "source",
        source_cfg,
        {"split_name", "materialization_name", "targets", "units"},
    )
    if "target" in source_cfg:
        raise ValueError(
            "Profiling config uses source.targets (list), not source.target. "
            "Rename the key to 'targets' and provide a list of label names."
        )
    targets = source_cfg.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("source.targets must be a non-empty list of label names.")

    model_cfg = config.get("model", {})
    _validate_known_keys(
        "model",
        model_cfg,
        {"family", "C_values", "class_weights", "max_iter", "tol", "dual"},
    )
    family = str(model_cfg.get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported model.family: {family!r}. Expected {SUPPORTED_MODEL_FAMILY!r}."
        )
    raw_variants = config.get("feature_sets", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Config must define at least one [[feature_sets]] entry.")

    author_balance_cfg = config.get("author_balance", {})
    if author_balance_cfg:
        _validate_known_keys(
            "author_balance",
            author_balance_cfg,
            {"train_sample_weighting", "normalize_train_weights", "report_author_weighted_eval"},
        )
        weighting = str(author_balance_cfg.get("train_sample_weighting", "none")).strip().lower()
        if weighting not in _SUPPORTED_WEIGHTING_MODES:
            raise ValueError(
                f"Unsupported author_balance.train_sample_weighting: {weighting!r}. "
                f"Expected one of {sorted(_SUPPORTED_WEIGHTING_MODES)}."
            )


# ── Calibration diagnostics ───────────────────────────────────────────────────


def _calibration_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Compute calibration quality metrics for one fold.

    Returns log-loss, a multiclass Brier score, and (for binary targets)
    a reliability curve as serialisable lists.
    """
    classes_list = list(classes)
    fold_log_loss = float(log_loss(y_true, y_proba, labels=classes))

    # Multiclass Brier score: mean of sum-of-squared-errors over one-hot rows.
    y_onehot = np.zeros_like(y_proba)
    for i, label in enumerate(y_true):
        y_onehot[i, classes_list.index(label)] = 1.0
    brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))

    result: dict[str, Any] = {"log_loss": fold_log_loss, "brier_score": brier}

    if len(classes) == 2:
        pos_class_idx = 1
        fraction_of_pos, mean_predicted = calibration_curve(
            (y_true == classes[pos_class_idx]).astype(int),
            y_proba[:, pos_class_idx],
            n_bins=10,
        )
        result["reliability_curve"] = {
            "fraction_of_positives": fraction_of_pos.tolist(),
            "mean_predicted_value": mean_predicted.tolist(),
        }

    return result


def _validate_target_unit_label_coverage(
    target_name: str,
    units: list[dict[str, Any]],
    materialized_root: Path,
) -> None:
    """Require every unit train split to contain all labels seen in its eval split."""
    problems: list[str] = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        eval_role = str(unit["eval_role"])
        unit_dir = materialized_root / unit_id
        y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
        y_eval = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")

        train_labels = {str(label) for label in np.unique(y_train)}
        eval_labels = {str(label) for label in np.unique(y_eval)}
        missing = sorted(eval_labels - train_labels)
        if missing:
            problems.append(
                f"{unit_id} ({eval_role}) missing from train: {', '.join(missing)}"
            )

    if problems:
        joined = "; ".join(problems)
        raise ValueError(
            f"Profiling target {target_name!r} has eval labels absent from the "
            f"corresponding train labels: {joined}. Adjust the split config so "
            "each grouped fold can train every label it evaluates, then rebuild."
        )


# ── Parallel worker functions ─────────────────────────────────────────────────


def _search_profiling_unit(
    unit: dict[str, Any],
    layouts_by_name: dict[str, FeatureLayout],
    candidates: list[ProfilingCandidateSpec],
    materialized_root: Path,
    target_name: str,
    model_cfg: dict[str, Any],
    seed: int,
    author_balance_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fit all candidates for one profiling unit and return their metric rows.

    When author_balance_cfg specifies inverse_author_speech_count weighting,
    train sample weights are built from the unit's row-order file and passed
    to the SVM fit. Author-weighted eval metrics are added to each val row when
    report_author_weighted_eval is enabled.
    """
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id

    y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
    y_eval = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")

    train_weights = _train_author_weights(unit_dir, author_balance_cfg)
    eval_weights = _eval_author_weights(unit_dir, eval_role, author_balance_cfg)

    layout_cache: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    for layout_name, layout in layouts_by_name.items():
        _validate_variant_availability(unit, layout)
        layout_cache[layout_name] = (
            _build_feature_matrix(unit, unit_dir, "train", layout),
            _build_feature_matrix(unit, unit_dir, eval_role, layout),
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        x_train, x_eval = layout_cache[candidate.feature_layout.name]
        clf, conv_msgs, fit_sec = _fit_linear_svm(
            x_train,
            y_train,
            c_value=candidate.c_value,
            class_weight=candidate.class_weight,
            model_cfg=model_cfg,
            seed=seed,
            sample_weight=train_weights,
        )
        train_metrics, _, _, t_pred_sec = _evaluate_predictions(
            clf, x_train, y_train, "train", []
        )
        val_metrics, y_pred_eval, _, v_pred_sec = _evaluate_predictions(
            clf, x_eval, y_eval, eval_role, []
        )

        train_row = profiling_candidate_metric_row(
            candidate, unit_id, eval_role, train_metrics, fit_sec, t_pred_sec, conv_msgs
        )
        val_row = profiling_candidate_metric_row(
            candidate, unit_id, eval_role, val_metrics, fit_sec, v_pred_sec, []
        )

        if eval_weights is not None:
            val_row.update(_author_weighted_metrics(y_eval, y_pred_eval, eval_weights))

        rows.append(train_row)
        rows.append(val_row)
    return rows


def _refit_profiling_unit(
    unit: dict[str, Any],
    materialized_root: Path,
    target_name: str,
    best_cand: ProfilingCandidateSpec,
    model_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
    seed: int,
    save_top_k: int,
    models_dir: Path,
    predictions_dir: Path,
    project_root: Path,
    author_balance_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit best candidate with calibration for one unit.

    When author_balance_cfg is provided and specifies inverse weighting, the
    same weighting used during candidate search is also applied to the
    calibrated refit so the two stages optimise the same objective.

    Returns (saved_unit, diagnostics).
    """
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id

    y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
    y_eval = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")
    x_train = _build_feature_matrix(unit, unit_dir, "train", best_cand.feature_layout)
    x_eval = _build_feature_matrix(unit, unit_dir, eval_role, best_cand.feature_layout)

    train_weights = _train_author_weights(unit_dir, author_balance_cfg)

    cal_model, _, _, cal_meta = _fit_calibrated_family_model(
        x_train, y_train, best_cand, model_cfg, calibration_cfg, seed,
        sample_weight=train_weights,
    )

    unit_model_dir = models_dir / unit_id
    unit_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal_model, unit_model_dir / "model.joblib")
    write_json(unit_model_dir / "calibration_meta.json", cal_meta)

    classes = np.asarray(cal_model.classes_)
    scores = cal_model.predict_proba(x_eval)
    y_pred = classes[np.argmax(scores, axis=1)]

    pred_frame = _prediction_frame(
        unit_dir=unit_dir,
        eval_role=eval_role,
        y_true=y_eval,
        y_pred=y_pred,
        scores=scores,
        classes=classes,
        save_top_k=save_top_k,
    )
    pred_frame.to_csv(
        predictions_dir / f"{unit_id}_{eval_role}_predictions.csv", index=False
    )

    diag = _calibration_diagnostics(y_eval, scores, classes)
    diag["unit_id"] = unit_id

    saved_unit = {
        "unit_id": unit_id,
        "eval_role": eval_role,
        "model_path": relative_to_project(
            project_root, unit_model_dir / "model.joblib"
        ),
        "calibration_meta": cal_meta,
    }
    return saved_unit, diag


# ── Per-target training loop ──────────────────────────────────────────────────


def _run_target(
    target_name: str,
    candidates: list[ProfilingCandidateSpec],
    units: list[dict[str, Any]],
    materialized_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    model_cfg: dict[str, Any],
    seed: int,
    selection_metric: str,
    save_top_k: int,
    project_root: Path,
    show_progress: bool,
    n_jobs: int = 1,
    author_balance_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run candidate search and calibrated refit for one profiling target.

    Returns a summary dict that is merged into the top-level manifest under
    ``targets_summary[target_name]``.
    """
    _validate_target_unit_label_coverage(target_name, units, materialized_root)

    target_results_dir = results_dir / target_name
    target_artifacts_dir = artifacts_dir / target_name
    predictions_dir = target_results_dir / "predictions"
    models_dir = target_artifacts_dir / "models"
    for d in (target_results_dir, predictions_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    layouts_by_name = {c.feature_layout.name: c.feature_layout for c in candidates}

    # ── Stage A: candidate search (uncalibrated LinearSVC) ───────────────────

    total_fits = len(units) * len(candidates)
    if show_progress:
        print(
            f"[{target_name}] Starting candidate search "
            f"({len(units)} unit(s) × {len(candidates)} candidate(s) = {total_fits} fits, "
            f"n_jobs={n_jobs})..."
        )

    unit_rows_list: list[list[dict[str, Any]]] = joblib.Parallel(
        n_jobs=n_jobs, backend="loky", verbose=10
    )(
        joblib.delayed(_search_profiling_unit)(
            unit,
            layouts_by_name,
            candidates,
            materialized_root,
            target_name,
            model_cfg,
            seed,
            author_balance_cfg,
        )
        for unit in units
    )
    candidate_rows = [row for unit_rows in unit_rows_list for row in unit_rows]

    metrics_df = pd.DataFrame(candidate_rows)
    summary_df = summarize_profiling_candidates(metrics_df, selection_metric)
    best_cand = best_profiling_candidate(summary_df, candidates)

    best_summary_row = summary_df.iloc[0].to_dict()
    best_layout = best_cand.feature_layout
    best_payload: dict[str, Any] = {
        "candidate_id": best_cand.candidate_id,
        "feature_set": best_layout.name,
        "blocks": list(best_layout.blocks),
        "normalize_rows": best_layout.normalize_rows,
        "normalize_each_block": best_layout.normalize_each_block,
        "block_weights": best_layout.block_weights,
        "c_value": best_cand.c_value,
        "class_weight": best_cand.class_weight_label,
        "selection_metric": selection_metric,
        "summary": {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in best_summary_row.items()
        },
    }

    metrics_df.to_csv(target_results_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(target_results_dir / "candidate_summary.csv", index=False)
    write_json(target_results_dir / "best_candidate.json", best_payload)

    if show_progress:
        print(
            f"[{target_name}] Best candidate: {best_cand.candidate_id}. Refitting with calibration..."
        )

    # ── Stage B: refit best candidate with calibration ────────────────────────

    calibration_cfg = {"method": CALIBRATION_METHOD, "cv": CALIBRATION_CV}

    if show_progress:
        print(f"[{target_name}] Refitting with calibration on {len(units)} unit(s)...")

    refit_results: list[tuple[dict[str, Any], dict[str, Any]]] = joblib.Parallel(
        n_jobs=n_jobs, backend="loky", verbose=10
    )(
        joblib.delayed(_refit_profiling_unit)(
            unit,
            materialized_root,
            target_name,
            best_cand,
            model_cfg,
            calibration_cfg,
            seed,
            save_top_k,
            models_dir,
            predictions_dir,
            project_root,
            author_balance_cfg,
        )
        for unit in units
    )
    saved_units = [r[0] for r in refit_results]
    fold_diagnostics = [r[1] for r in refit_results]
    if show_progress:
        for u in saved_units:
            print(
                f"[{target_name}] [refit] saved model + predictions for {u['unit_id']}"
            )

    write_json(target_results_dir / "calibration_diagnostics.json", fold_diagnostics)

    return {
        "best_candidate": best_payload,
        "saved_units": saved_units,
    }


# ── Public entry point ────────────────────────────────────────────────────────


def run_profiling_experiment(
    config_path: Path, *, show_progress: bool = False
) -> dict[str, Any]:
    """Train profiling classifiers for all targets defined in the config.

    Parameters
    ----------
    config_path:
        Path to a profiling model config TOML file.
    show_progress:
        Whether to print stage logs and progress bars.

    Returns
    -------
    dict
        Top-level manifest written to ``results_dir/manifest.json``.
    """
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    _validate_profiling_config(config)

    experiment_cfg = config.get("experiment", {})
    source_cfg = config.get("source", {})
    model_cfg = config.get("model", {})
    author_balance_cfg = config.get("author_balance") or None

    seed = int(experiment_cfg.get("seed", 42))
    selection_metric = str(experiment_cfg.get("selection_metric", "macro_f1"))
    save_top_k = int(experiment_cfg.get("save_prediction_top_k", 3))
    n_jobs = int(experiment_cfg.get("n_jobs", 1))
    targets: list[str] = list(source_cfg["targets"])

    # Profiling always uses balanced class weights.
    model_cfg.setdefault("class_weights", ["balanced"])

    candidates = profiling_candidate_grid(config)
    materialized_root, materialization_manifest, units = _load_materialization_summary(
        project_root, config
    )
    provenance = _build_provenance_block(
        project_root, materialized_root, materialization_manifest
    )
    _validate_candidate_search_units(units)

    config_name = str(experiment_cfg.get("name", config_path.stem))
    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)
    copy_config_outputs(
        config_path,
        results_dir / "model_config.toml",
        artifacts_dir / "model_config.toml",
    )

    if show_progress:
        print(
            f"Profiling experiment '{config_name}': "
            f"{len(targets)} target(s), {len(candidates)} candidate(s), "
            f"{len(units)} unit(s), seed={seed}, n_jobs={n_jobs}"
        )

    targets_summary: dict[str, Any] = {}
    for target_name in targets:
        if show_progress:
            print(f"\n── Target: {target_name} ──")
        targets_summary[target_name] = _run_target(
            target_name=target_name,
            candidates=candidates,
            units=units,
            materialized_root=materialized_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            model_cfg=model_cfg,
            seed=seed,
            selection_metric=selection_metric,
            save_top_k=save_top_k,
            project_root=project_root,
            show_progress=show_progress,
            n_jobs=n_jobs,
            author_balance_cfg=author_balance_cfg,
        )

    manifest: dict[str, Any] = {
        "run_type": "profiling_candidate_search",
        "experiment_name": config_name,
        "config_path": relative_to_project(project_root, config_path),
        "split_name": source_cfg["split_name"],
        "materialization_name": source_cfg["materialization_name"],
        "targets": targets,
        "selection_metric": selection_metric,
        "seed": seed,
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "candidate_count": len(candidates),
        "unit_count": len(units),
        "provenance": provenance,
        "source_manifest": {
            "path": relative_to_project(
                project_root, materialized_root / "manifest.json"
            ),
            "units": materialization_manifest.get("units", []),
        },
        "targets_summary": targets_summary,
    }
    write_json(results_dir / "manifest.json", manifest)

    if show_progress:
        print(f"\nProfiling experiment complete. Results at: {results_dir}")

    return manifest


# ── Final profiling training ──────────────────────────────────────────────────


def _format_id_sample(values: list[str], *, limit: int = 5) -> str:
    """Format a short sample of ids for validation error messages."""
    sample = values[:limit]
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit} more)"
    return ", ".join(sample) + suffix


def _validate_final_profiling_eval_coverage(
    *,
    split_name: str,
    corpus: pd.DataFrame,
    all_speech_ids: list[str],
    unit_row_counts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that profiling fold eval rows partition corpus/all.csv exactly once."""
    contract = (
        "Final profiling training expects fold eval rows to partition "
        "corpus/all.csv exactly once"
    )
    corpus_ids = corpus["id_speech"].astype(str).tolist()
    corpus_series = pd.Series(corpus_ids)
    duplicate_corpus_ids = corpus_series[corpus_series.duplicated()].tolist()
    if duplicate_corpus_ids:
        raise ValueError(
            f"{contract}; split={split_name}; corpus/all.csv contains duplicate "
            f"id_speech values: {_format_id_sample(duplicate_corpus_ids)}"
        )

    eval_series = pd.Series([str(sid) for sid in all_speech_ids])
    duplicate_eval_ids = eval_series[eval_series.duplicated()].tolist()
    if duplicate_eval_ids:
        raise ValueError(
            f"{contract}; split={split_name}; fold eval rows contain duplicate "
            f"id_speech values: {_format_id_sample(duplicate_eval_ids)}"
        )

    corpus_set = set(corpus_ids)
    eval_set = set(eval_series.tolist())
    missing_from_eval = sorted(corpus_set - eval_set)
    absent_from_corpus = sorted(eval_set - corpus_set)
    if missing_from_eval or absent_from_corpus or len(all_speech_ids) != len(corpus_ids):
        details = [
            f"corpus_rows={len(corpus_ids)}",
            f"collected_eval_rows={len(all_speech_ids)}",
        ]
        if missing_from_eval:
            details.append(
                "corpus ids missing from eval rows: "
                f"{_format_id_sample(missing_from_eval)}"
            )
        if absent_from_corpus:
            details.append(
                "eval ids absent from corpus/all.csv: "
                f"{_format_id_sample(absent_from_corpus)}"
            )
        raise ValueError(f"{contract}; split={split_name}; " + "; ".join(details))

    return {
        "corpus_row_count": len(corpus_ids),
        "collected_eval_row_count": len(all_speech_ids),
        "unit_count": len(unit_row_counts),
        "checked_exactly_once": True,
        "units": unit_row_counts,
    }


def _build_final_stylo_matrix(
    all_speech_ids: list[str],
    stylo_raw_path: Path,
) -> tuple[sparse.csr_matrix, list[str], StandardScaler]:
    """Load, align, filter and scale raw stylometry for the full profiling corpus.

    Selects zero-variance columns on the training corpus. Fits and returns a
    StandardScaler trained on this corpus only. Saved by the caller so the
    signal extractor can apply the same transformation to attribution speeches.
    """
    raw = pd.read_csv(stylo_raw_path, dtype={"id_speech": str})
    raw["id_speech"] = raw["id_speech"].astype(str)

    id_frame = pd.DataFrame({"id_speech": all_speech_ids})
    merged = id_frame.merge(raw, on="id_speech", how="left")

    _STYLO_RAW_METADATA = {"id_speech", "id_person", "outer_role"}
    feature_cols = [c for c in merged.columns if c not in _STYLO_RAW_METADATA]
    x_raw = merged[feature_cols].to_numpy(dtype=float)

    # Drop zero-variance columns (fit on this corpus only).
    col_variance = np.nanvar(x_raw, axis=0)
    keep_mask = col_variance > 0
    kept_cols = [c for c, keep in zip(feature_cols, keep_mask) if keep]
    x_kept = x_raw[:, keep_mask]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_kept)
    return sparse.csr_matrix(x_scaled), kept_cols, scaler


def _assemble_final_blocks(
    candidate: ProfilingCandidateSpec,
    block_matrices: dict[str, sparse.csr_matrix],
) -> sparse.csr_matrix:
    """Assemble a feature matrix from pre-computed block matrices.

    Applies per-block normalization and weighting, then row normalization,
    exactly mirroring the _build_feature_matrix logic from attribution training.
    """
    layout = candidate.feature_layout
    matrices = []
    for block in layout.blocks:
        mat = block_matrices[block]
        if not sparse.issparse(mat):
            mat = sparse.csr_matrix(mat)
        if layout.normalize_each_block:
            mat = normalize(mat, norm="l2", axis=1, copy=False)
        weight = float(layout.block_weights.get(block, 1.0))
        if weight != 1.0:
            mat = mat.multiply(weight).tocsr()
        matrices.append(mat)

    combined = (
        sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]
    )
    if layout.normalize_rows:
        combined = normalize(combined, norm="l2", axis=1, copy=False)
    return combined.tocsr()


def run_final_profiling_training(
    config_path: Path, *, show_progress: bool = False
) -> dict[str, Any]:
    """Train final profiling classifiers on ALL profiling authors.

    Uses the union of each fold's val-split speeches (non-overlapping across
    folds, together covering the full profiling corpus). Reads each target's
    best_candidate.json to reconstruct the exact feature layout selected during
    candidate search, including optional stylo blocks.

    Saves per-block preprocessors only for blocks used by at least one target:
      - ``profiling_artifacts_dir / "final" / "char_vectorizer.joblib"``
      - ``profiling_artifacts_dir / "final" / "word_vectorizer.joblib"``
      - ``profiling_artifacts_dir / "final" / "stylo_scaler.joblib"``
      - ``profiling_artifacts_dir / "final" / "stylo_columns.json"``
      - ``profiling_artifacts_dir / "final" / "feature_build_meta.json"``
      - ``profiling_artifacts_dir / {target} / "models" / "final" / "model.joblib"``

    Returns the manifest dict.
    """
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    _validate_profiling_config(config)

    experiment_cfg = config.get("experiment", {})
    source_cfg = config.get("source", {})
    model_cfg = config.get("model", {})
    author_balance_cfg = config.get("author_balance") or None
    model_cfg.setdefault("class_weights", ["balanced"])

    seed = int(experiment_cfg.get("seed", 42))
    targets: list[str] = list(source_cfg["targets"])
    split_name = str(source_cfg["split_name"])

    materialized_root, materialization_manifest, units = _load_materialization_summary(
        project_root, config
    )
    config_name = str(experiment_cfg.get("name", config_path.stem))
    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)

    data_cfg = config.get("data", {})
    splits_dir = resolve_project_path(
        project_root, data_cfg.get("splits_dir", "data/splits")
    )
    corpus_path = splits_dir / split_name / "corpus" / "all.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Profiling corpus not found: {corpus_path}")
    corpus = pd.read_csv(corpus_path, dtype={"id_speech": str})
    corpus["id_speech"] = corpus["id_speech"].astype(str)
    id_to_text = dict(zip(corpus["id_speech"], corpus["text"].fillna("").astype(str)))

    # Collect all val-split speeches from every fold (non-overlapping → full corpus).
    all_speech_ids: list[str] = []
    all_author_ids: list[str] = []
    all_labels: dict[str, list[Any]] = {target: [] for target in targets}
    unit_row_counts: list[dict[str, Any]] = []

    for unit in units:
        unit_id = str(unit["unit_id"])
        eval_role = str(unit["eval_role"])
        unit_dir = materialized_root / unit_id
        row_order = pd.read_csv(
            unit_dir / "row_order" / f"{eval_role}_rows.csv",
            dtype={"id_speech": str, "id_person": str},
        )
        row_order["id_speech"] = row_order["id_speech"].astype(str)
        row_order["id_person"] = row_order["id_person"].astype(str)
        unit_row_counts.append(
            {
                "unit_id": unit_id,
                "eval_role": eval_role,
                "row_count": int(len(row_order)),
            }
        )
        all_speech_ids.extend(row_order["id_speech"].tolist())
        all_author_ids.extend(row_order["id_person"].tolist())
        for target in targets:
            labels = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target}.npy")
            if len(labels) != len(row_order):
                raise ValueError(
                    "Final profiling training label/row mismatch; "
                    f"split={split_name}; unit={unit_id}; eval_role={eval_role}; "
                    f"target={target}; row_count={len(row_order)}; "
                    f"label_count={len(labels)}"
                )
            all_labels[target].extend(labels.tolist())

    training_corpus_coverage = _validate_final_profiling_eval_coverage(
        split_name=split_name,
        corpus=corpus,
        all_speech_ids=all_speech_ids,
        unit_row_counts=unit_row_counts,
    )
    all_texts = [id_to_text[sid] for sid in all_speech_ids]
    y_all = {target: np.array(all_labels[target]) for target in targets}

    if show_progress:
        print(
            f"Final profiling training: {len(all_texts)} speeches, "
            f"{len(targets)} target(s), seed={seed}"
        )

    # Read best candidates per target to know which blocks are needed.
    best_cands: dict[str, ProfilingCandidateSpec] = {}
    for target in targets:
        best_candidate_path = results_dir / target / "best_candidate.json"
        if not best_candidate_path.exists():
            raise FileNotFoundError(
                f"Dev best_candidate.json not found for target '{target}': {best_candidate_path}. "
                "Run the profiling dev experiment first."
            )
        payload = json.loads(best_candidate_path.read_text(encoding="utf-8"))
        best_cands[target] = profiling_candidate_from_payload(payload)

    needed_blocks: set[str] = set()
    for cand in best_cands.values():
        needed_blocks.update(cand.feature_layout.blocks)

    final_dir = artifacts_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Build per-block preprocessors and matrices only for needed blocks.
    block_matrices: dict[str, sparse.csr_matrix] = {}
    stylo_columns: list[str] = []

    if "char" in needed_blocks or "word" in needed_blocks:
        first_fold = str(units[0]["unit_id"])
        preprocessors_dir = materialized_root / first_fold / "preprocessors"

    if "char" in needed_blocks:
        existing_char_vec = joblib.load(preprocessors_dir / "char_vectorizer.joblib")
        final_char_vec = clone(existing_char_vec)
        if show_progress:
            print("  Fitting final char vectorizer...")
        block_matrices["char"] = final_char_vec.fit_transform(all_texts)
        joblib.dump(final_char_vec, final_dir / "char_vectorizer.joblib")

    if "word" in needed_blocks:
        existing_word_vec = joblib.load(preprocessors_dir / "word_vectorizer.joblib")
        final_word_vec = clone(existing_word_vec)
        if show_progress:
            print("  Fitting final word vectorizer...")
        block_matrices["word"] = final_word_vec.fit_transform(all_texts)
        joblib.dump(final_word_vec, final_dir / "word_vectorizer.joblib")

    if "stylo" in needed_blocks:
        row_feature_name = str(materialization_manifest.get("row_feature_name", ""))
        stylo_raw_path = splits_dir / split_name / "row_features" / row_feature_name / "stylometry_raw.csv.gz"
        if not stylo_raw_path.exists():
            raise FileNotFoundError(
                f"Raw stylometry not found: {stylo_raw_path}. "
                "Run the profiling feature pipeline with stylometry enabled first."
            )
        if show_progress:
            print("  Fitting final stylo scaler...")
        x_stylo, stylo_columns, stylo_scaler = _build_final_stylo_matrix(
            all_speech_ids, stylo_raw_path
        )
        block_matrices["stylo"] = x_stylo
        joblib.dump(stylo_scaler, final_dir / "stylo_scaler.joblib")
        write_json(final_dir / "stylo_columns.json", {"columns": stylo_columns})

    # Compute final sample weights if configured.
    final_train_weights: np.ndarray | None = None
    if _author_weighting_mode(author_balance_cfg) == "inverse_author_speech_count":
        normalize_w = bool(author_balance_cfg.get("normalize_train_weights", True))
        all_author_ids_arr = np.array(all_author_ids, dtype=str)
        final_train_weights = _inverse_author_weights(
            all_author_ids_arr,
            normalize=normalize_w,
        )

    calibration_cfg = {"method": CALIBRATION_METHOD, "cv": CALIBRATION_CV}
    saved_targets: dict[str, Any] = {}
    target_meta: dict[str, Any] = {}

    for target in targets:
        best_cand = best_cands[target]
        best_layout = best_cand.feature_layout
        x_all = _assemble_final_blocks(best_cand, block_matrices)

        cal_model, _, _, cal_meta = _fit_calibrated_family_model(
            x_all, y_all[target], best_cand, model_cfg, calibration_cfg, seed,
            sample_weight=final_train_weights,
        )

        model_final_dir = artifacts_dir / target / "models" / "final"
        model_final_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal_model, model_final_dir / "model.joblib")
        write_json(model_final_dir / "calibration_meta.json", cal_meta)

        saved_targets[target] = {
            "n_train_samples": len(all_texts),
            "model_path": relative_to_project(
                project_root, model_final_dir / "model.joblib"
            ),
            "calibration_meta": cal_meta,
        }
        target_meta[target] = {
            "blocks": list(best_layout.blocks),
            "normalize_each_block": best_layout.normalize_each_block,
            "block_weights": dict(best_layout.block_weights),
            "normalize_rows": best_layout.normalize_rows,
        }

        if show_progress:
            print(
                f"  [{target}] Final model saved → {model_final_dir.relative_to(project_root)}"
            )

    feature_build_meta: dict[str, Any] = {
        "materialization_name": str(source_cfg["materialization_name"]),
        "row_feature_name": str(materialization_manifest.get("row_feature_name", "")),
        "available_blocks": sorted(needed_blocks),
        "stylo_columns": stylo_columns,
        "targets": target_meta,
    }
    write_json(final_dir / "feature_build_meta.json", feature_build_meta)

    manifest = {
        "run_type": "profiling_final_training",
        "experiment_name": config_name,
        "targets": targets,
        "n_train_speeches": len(all_texts),
        "training_corpus_coverage": training_corpus_coverage,
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "saved_targets": saved_targets,
    }
    write_json(results_dir / "final_manifest.json", manifest)
    return manifest


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for profiling-classifier training."""
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)
    show_progress = not args.no_progress
    if args.final:
        run_final_profiling_training(config_path, show_progress=show_progress)
    else:
        run_profiling_experiment(config_path, show_progress=show_progress)


if __name__ == "__main__":
    main()
