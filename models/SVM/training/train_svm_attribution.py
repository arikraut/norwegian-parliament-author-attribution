from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.SVM.training.condition_selection import select_candidates_by_condition
from models.SVM.training.final_condition_outputs import (
    build_final_summary_row,
    final_condition_roots,
    write_final_condition_output,
    write_final_condition_summary,
)
from models.SVM.linear_svm_common import (
    FeatureLayout,
    SUPPORTED_BLOCKS,
    _build_feature_matrix,
    _build_provenance_block,
    _copy_config_outputs_if_available,
    _evaluate_predictions,
    _fit_linear_svm,
    _load_labels,
    _load_materialization_summary,
    _manifest_config_path,
    _parse_class_weights,
    _prediction_frame,
    _resolve_output_dirs,
    _validate_known_keys,
    _validate_candidate_search_units,
    _validate_final_evaluation_units,
    _validate_variant_availability,
)


SUPPORTED_MODEL_FAMILY = "linear_svm"


@dataclass(frozen=True)
class DirectConditionSpec:
    """Declared experiment condition for direct attribution feature selection."""

    condition_id: str
    label: str
    feature_set: str
    blocks: tuple[str, ...]
    normalize_rows: bool
    normalize_each_block: bool
    block_weights: dict[str, float]

    @property
    def feature_layout(self) -> FeatureLayout:
        """Return the shared feature-block layout used for matrix assembly."""
        return FeatureLayout(
            name=self.feature_set,
            blocks=self.blocks,
            normalize_rows=self.normalize_rows,
            normalize_each_block=self.normalize_each_block,
            block_weights=self.block_weights,
        )


@dataclass(frozen=True)
class DirectAttributionCandidateSpec:
    """Direct attribution candidate: one declared condition plus SVM hyperparameters."""

    condition: DirectConditionSpec
    c_value: float
    class_weight: str | None

    @property
    def candidate_id(self) -> str:
        """Return a stable id that remains unique even when conditions share feature sets."""
        class_weight_label = self.class_weight_label
        return f"{self.condition.condition_id}__C={self.c_value:g}__class_weight={class_weight_label}"

    @property
    def class_weight_label(self) -> str:
        """Return the artifact label for the LinearSVC class-weight setting."""
        return "none" if self.class_weight is None else str(self.class_weight)


def _parse_args() -> argparse.Namespace:
    """Parse the direct-SVM attribution trainer CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate author-attribution models from materialized fold features.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a model config TOML file under models/configs/attribution/.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage logs.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read a direct-SVM attribution TOML config."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _validate_condition_keys(raw_conditions: list[dict[str, Any]]) -> None:
    """Validate direct attribution condition-table keys from TOML config."""
    allowed_keys = {
        "id",
        "label",
        "feature_set",
        "blocks",
        "normalize_rows",
        "normalize_each_block",
        "block_weights",
    }
    for idx, raw_condition in enumerate(raw_conditions, start=1):
        _validate_known_keys(f"conditions #{idx}", raw_condition, allowed_keys)


def _validate_model_family(config: dict[str, Any]) -> None:
    """Require the direct attribution trainer to receive a linear-SVM config."""
    model_cfg = config.get("model", {})
    family = str(model_cfg.get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported model.family: {family!r}. Expected {SUPPORTED_MODEL_FAMILY!r}."
        )


def _validate_dev_config(config: dict[str, Any]) -> None:
    """Validate the direct-SVM development search config boundary."""
    _validate_known_keys(
        "root",
        config,
        {"experiment", "data", "source", "model", "conditions"},
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
    _validate_known_keys(
        "source",
        config.get("source", {}),
        {"split_name", "materialization_name", "target", "units"},
    )
    _validate_known_keys(
        "model",
        config.get("model", {}),
        {"family", "C_values", "class_weights", "max_iter", "tol", "dual", "top_k"},
    )
    raw_conditions = config.get("conditions", [])
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions must be a list of tables.")
    _validate_condition_keys(raw_conditions)
    _validate_model_family(config)


def _validate_final_eval_config(config: dict[str, Any]) -> None:
    """Validate the direct-SVM final-evaluation config boundary."""
    _validate_known_keys(
        "root",
        config,
        {"experiment", "data", "source", "model", "final_eval"},
    )
    _validate_known_keys(
        "experiment",
        config.get("experiment", {}),
        {"name", "seed", "save_prediction_top_k", "n_jobs"},
    )
    _validate_known_keys(
        "data",
        config.get("data", {}),
        {"splits_dir", "results_dir", "artifacts_dir"},
    )
    _validate_known_keys(
        "source",
        config.get("source", {}),
        {"split_name", "materialization_name", "target", "units"},
    )
    _validate_known_keys(
        "model",
        config.get("model", {}),
        {"family", "max_iter", "tol", "dual", "top_k"},
    )
    _validate_known_keys(
        "final_eval",
        config.get("final_eval", {}),
        {"selected_candidates_path"},
    )
    _validate_model_family(config)


def _parse_direct_conditions(config: dict[str, Any]) -> list[DirectConditionSpec]:
    """Parse declared direct attribution research conditions from config."""
    raw_conditions = config.get("conditions", [])
    if not raw_conditions:
        raise ValueError("Config must define at least one [[conditions]] entry.")

    conditions: list[DirectConditionSpec] = []
    seen_ids: set[str] = set()
    for raw_condition in raw_conditions:
        condition_id = str(raw_condition["id"]).strip()
        feature_set = str(raw_condition["feature_set"]).strip()
        blocks = tuple(str(block).strip() for block in raw_condition.get("blocks", []))
        if not condition_id:
            raise ValueError("Condition id cannot be empty.")
        if condition_id in seen_ids:
            raise ValueError(f"Duplicate [[conditions]].id value: {condition_id!r}")
        seen_ids.add(condition_id)
        if not feature_set:
            raise ValueError(f"Condition '{condition_id}' must define feature_set.")
        if not blocks:
            raise ValueError(
                f"Condition '{condition_id}' must list at least one feature block."
            )
        unknown_blocks = sorted(set(blocks) - SUPPORTED_BLOCKS)
        if unknown_blocks:
            raise ValueError(
                f"Condition '{condition_id}' uses unsupported blocks: {unknown_blocks}"
            )
        if "all" in blocks and len(blocks) > 1:
            raise ValueError(
                f"Condition '{condition_id}' cannot mix 'all' with other blocks."
            )
        normalize_each_block = bool(raw_condition.get("normalize_each_block", False))
        if normalize_each_block and "all" in blocks:
            raise ValueError(
                f"Condition '{condition_id}' cannot set normalize_each_block=true when blocks=['all']."
            )
        raw_weights = raw_condition.get("block_weights", {})
        block_weights = {block: float(raw_weights.get(block, 1.0)) for block in blocks}
        conditions.append(
            DirectConditionSpec(
                condition_id=condition_id,
                label=str(raw_condition.get("label", condition_id)).strip()
                or condition_id,
                feature_set=feature_set,
                blocks=blocks,
                normalize_rows=bool(raw_condition.get("normalize_rows", True)),
                normalize_each_block=normalize_each_block,
                block_weights=block_weights,
            )
        )
    return conditions


def _candidate_grid(config: dict[str, Any]) -> list[DirectAttributionCandidateSpec]:
    """Build direct attribution candidates as conditions crossed with SVM hyperparameters."""
    model_cfg = config.get("model", {})
    c_values = [float(value) for value in model_cfg.get("C_values", [1.0])]
    if not c_values:
        raise ValueError("model.C_values must contain at least one value.")

    class_weights = _parse_class_weights(list(model_cfg.get("class_weights", ["none"])))
    candidates = [
        DirectAttributionCandidateSpec(
            condition=condition,
            c_value=c_value,
            class_weight=class_weight,
        )
        for condition in _parse_direct_conditions(config)
        for c_value in c_values
        for class_weight in class_weights
    ]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    duplicate_ids = sorted(
        {
            candidate_id
            for candidate_id in candidate_ids
            if candidate_ids.count(candidate_id) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(
            f"Candidate grid contains duplicate candidate identities: {duplicate_ids}"
        )
    return candidates


def _candidate_from_selected_payload(payload: dict[str, Any]) -> DirectAttributionCandidateSpec:
    """Reconstruct one direct attribution candidate from selected_candidates.json."""
    condition_id = str(payload.get("condition_id", "")).strip()
    feature_set = str(payload.get("feature_set", "")).strip()
    if not condition_id:
        raise ValueError("Selected candidate payload must include condition_id.")
    if not feature_set:
        raise ValueError("Selected candidate payload must include feature_set.")

    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("Selected candidate payload must include a non-empty blocks list.")
    blocks = tuple(str(block).strip() for block in raw_blocks)
    unknown_blocks = sorted(set(blocks) - SUPPORTED_BLOCKS)
    if unknown_blocks:
        raise ValueError(
            f"Selected candidate payload uses unsupported blocks: {unknown_blocks}"
        )
    if "all" in blocks and len(blocks) > 1:
        raise ValueError("Selected candidate payload cannot mix 'all' with other blocks.")
    normalize_each_block = bool(payload.get("normalize_each_block", False))
    if normalize_each_block and "all" in blocks:
        raise ValueError(
            "Selected candidate payload cannot set normalize_each_block=true with blocks=['all']."
        )

    raw_block_weights = payload.get("block_weights", {}) or {}
    if not isinstance(raw_block_weights, dict):
        raise ValueError("Selected candidate payload block_weights must be a mapping.")
    block_weights = {
        block: float(raw_block_weights.get(block, 1.0)) for block in blocks
    }
    candidate = DirectAttributionCandidateSpec(
        condition=DirectConditionSpec(
            condition_id=condition_id,
            label=str(payload.get("condition_label", condition_id)).strip()
            or condition_id,
            feature_set=feature_set,
            blocks=blocks,
            normalize_rows=bool(payload.get("normalize_rows", True)),
            normalize_each_block=normalize_each_block,
            block_weights=block_weights,
        ),
        c_value=float(payload["c_value"]),
        class_weight=_parse_class_weights([payload.get("class_weight", "none")])[0],
    )
    payload_candidate_id = payload.get("candidate_id")
    if payload_candidate_id and str(payload_candidate_id) != candidate.candidate_id:
        raise ValueError(
            "Selected candidate payload is inconsistent: "
            f"candidate_id={payload_candidate_id!r} does not match resolved {candidate.candidate_id!r}."
        )
    return candidate


def _candidate_payload(
    candidate: DirectAttributionCandidateSpec,
    *,
    selection_metric: str | None = None,
    dev_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a direct attribution candidate for selection and final artifacts."""
    payload: dict[str, Any] = {
        "condition_id": candidate.condition.condition_id,
        "condition_label": candidate.condition.label,
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.condition.feature_set,
        "blocks": list(candidate.condition.blocks),
        "normalize_rows": candidate.condition.normalize_rows,
        "normalize_each_block": candidate.condition.normalize_each_block,
        "block_weights": candidate.condition.block_weights,
        "c_value": candidate.c_value,
        "class_weight": candidate.class_weight_label,
    }
    if selection_metric is not None:
        payload["selection_metric"] = selection_metric
    if dev_summary is not None:
        payload["dev_summary"] = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in dev_summary.items()
        }
    return payload


def load_selected_direct_candidates(
    project_root: Path,
    config: dict[str, Any],
    *,
    selected_candidates_path_override: Path | None = None,
) -> tuple[list[DirectAttributionCandidateSpec], dict[str, Any], dict[str, Any]]:
    """Load the condition-aware direct attribution selection artifact for final evaluation."""
    final_cfg = config.get("final_eval", {})
    if selected_candidates_path_override is not None:
        selected_candidates_path = selected_candidates_path_override.resolve()
    else:
        raw_path = final_cfg.get("selected_candidates_path")
        if not raw_path:
            raise ValueError(
                "Final evaluation config must define final_eval.selected_candidates_path."
            )
        selected_candidates_path = resolve_project_path(project_root, raw_path)
    if not selected_candidates_path.exists():
        raise FileNotFoundError(
            f"selected_candidates.json does not exist: {selected_candidates_path}"
        )

    payload = json.loads(selected_candidates_path.read_text(encoding="utf-8"))
    raw_candidates = payload.get("selected_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("selected_candidates.json must include selected_candidates.")
    candidates = [_candidate_from_selected_payload(item) for item in raw_candidates]

    selection_results_dir = selected_candidates_path.parent
    selection_manifest_path = selection_results_dir / "manifest.json"
    selection_manifest: dict[str, Any] | None = None
    if selection_manifest_path.exists():
        selection_manifest = json.loads(
            selection_manifest_path.read_text(encoding="utf-8")
        )

    provenance = {
        "selected_candidates_path": relative_to_project(
            project_root, selected_candidates_path
        ),
        "selection_results_dir": relative_to_project(
            project_root, selection_results_dir
        ),
        "selection_manifest_path": (
            relative_to_project(project_root, selection_manifest_path)
            if selection_manifest_path.exists()
            else None
        ),
        "selection_manifest": selection_manifest,
    }
    return candidates, payload, provenance


def _candidate_metric_row(
    candidate: DirectAttributionCandidateSpec,
    unit_id: str,
    eval_role: str,
    split_metrics: dict[str, Any],
    fit_seconds: float,
    predict_seconds: float,
    convergence_messages: list[str],
) -> dict[str, Any]:
    """Create one fold-metrics CSV row for a direct attribution candidate."""
    row = {
        "candidate_id": candidate.candidate_id,
        "condition_id": candidate.condition.condition_id,
        "condition_label": candidate.condition.label,
        "feature_set": candidate.condition.feature_set,
        "blocks": "+".join(candidate.condition.blocks),
        "normalize_rows": candidate.condition.normalize_rows,
        "normalize_each_block": candidate.condition.normalize_each_block,
        "c_value": float(candidate.c_value),
        "class_weight": candidate.class_weight_label,
        "unit_id": unit_id,
        "eval_role": eval_role,
        "split": split_metrics["split"],
        "n_samples": int(split_metrics["n_samples"]),
        "n_classes": int(split_metrics["n_classes"]),
        "accuracy": float(split_metrics["accuracy"]),
        "macro_f1": float(split_metrics["macro_f1"]),
        "weighted_f1": float(split_metrics["weighted_f1"]),
        "macro_precision": float(split_metrics["macro_precision"]),
        "macro_recall": float(split_metrics["macro_recall"]),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "convergence_warning_count": len(convergence_messages),
        "convergence_warning": convergence_messages[0] if convergence_messages else "",
    }
    for metric_name, metric_value in split_metrics.items():
        if metric_name.startswith("top") and metric_name.endswith("_accuracy"):
            row[metric_name] = float(metric_value)
    return row


def _summary_sort_columns(
    summary_df: pd.DataFrame, selection_metric: str
) -> tuple[list[str], list[bool]]:
    """Return the shared candidate ranking columns for direct attribution."""
    primary_sort_col = f"eval_mean_{selection_metric}"
    if primary_sort_col not in summary_df.columns:
        available = sorted(
            col for col in summary_df.columns if col.startswith("eval_mean_")
        )
        raise ValueError(
            f"Selection metric '{selection_metric}' is not available. Available: {available}"
        )

    sort_cols = [primary_sort_col]
    if selection_metric == "author_weighted_macro_f1":
        sort_cols.append("eval_mean_macro_f1")
    sort_cols.extend(["eval_mean_accuracy", "n_eval_units"])
    sort_cols = list(dict.fromkeys(sort_cols))
    sort_cols.extend(["c_value", "condition_id", "class_weight"])
    ascending = [False] * (len(sort_cols) - 3) + [True, True, True]
    return sort_cols, ascending


def _summarize_candidates(
    metrics_df: pd.DataFrame, selection_metric: str
) -> pd.DataFrame:
    """Aggregate dev fold metrics while preserving declared condition identity."""
    if metrics_df.empty:
        raise ValueError("No metrics were collected.")

    eval_df = metrics_df[metrics_df["split"] != "train"].copy()
    if eval_df.empty:
        raise ValueError(
            "Candidate summary requires at least one evaluation split per candidate."
        )

    group_cols = [
        "candidate_id",
        "condition_id",
        "condition_label",
        "feature_set",
        "blocks",
        "normalize_rows",
        "normalize_each_block",
        "c_value",
        "class_weight",
    ]
    top_k_metric_cols = sorted(
        col
        for col in metrics_df.columns
        if col.startswith("top") and col.endswith("_accuracy")
    )
    base_metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "fit_seconds",
        "predict_seconds",
        "convergence_warning_count",
    ]
    metric_cols = base_metric_cols + top_k_metric_cols

    eval_summary = (
        eval_df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    eval_summary.columns = [
        "__".join(str(part) for part in col if part).strip("_")
        for col in eval_summary.columns.to_flat_index()
    ]
    rename_map = {f"{metric}__mean": f"eval_mean_{metric}" for metric in metric_cols}
    rename_map.update(
        {f"{metric}__std": f"eval_std_{metric}" for metric in metric_cols}
    )
    eval_summary = eval_summary.rename(columns=rename_map)
    eval_summary["n_eval_units"] = (
        eval_df.groupby(group_cols, dropna=False).size().values
    )

    train_df = metrics_df[metrics_df["split"] == "train"].copy()
    if not train_df.empty:
        train_metric_cols = [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
        ]
        train_summary = (
            train_df.groupby(group_cols, dropna=False)[train_metric_cols]
            .mean()
            .reset_index()
        )
        train_summary = train_summary.rename(
            columns={col: f"train_mean_{col}" for col in train_metric_cols}
        )
        eval_summary = eval_summary.merge(
            train_summary, on=group_cols, how="left", sort=False
        )

    sort_cols, ascending = _summary_sort_columns(eval_summary, selection_metric)
    return eval_summary.sort_values(
        by=sort_cols,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)


def _select_direct_candidates_by_condition(
    summary_df: pd.DataFrame,
    candidates: list[DirectAttributionCandidateSpec],
    selection_metric: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select the best hyperparameter candidate independently within each condition."""
    sort_cols, ascending = _summary_sort_columns(summary_df, selection_metric)
    return select_candidates_by_condition(
        summary_df,
        candidates,
        selection_metric=selection_metric,
        sort_cols=sort_cols,
        ascending=ascending,
        candidate_id=lambda candidate: candidate.candidate_id,
        condition_id=lambda candidate: candidate.condition.condition_id,
        condition_label=lambda candidate: candidate.condition.label,
        selected_payload=lambda candidate, metric, summary: _candidate_payload(
            candidate,
            selection_metric=metric,
            dev_summary=summary,
        ),
    )


# ── Parallel worker functions ─────────────────────────────────────────────────


def _search_unit(
    unit: dict[str, Any],
    conditions_by_id: dict[str, DirectConditionSpec],
    candidates: list[DirectAttributionCandidateSpec],
    materialized_root: Path,
    target_name: str,
    model_cfg: dict[str, Any],
    seed: int,
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    """Fit all candidates for one unit and return their metric rows."""
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id
    y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
    y_eval = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")

    layout_cache: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    for condition_id, condition in conditions_by_id.items():
        layout = condition.feature_layout
        _validate_variant_availability(unit, layout)
        layout_cache[condition_id] = (
            _build_feature_matrix(unit, unit_dir, "train", layout),
            _build_feature_matrix(unit, unit_dir, eval_role, layout),
        )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        x_train, x_eval = layout_cache[candidate.condition.condition_id]
        clf, convergence_messages, fit_seconds = _fit_linear_svm(
            x_train,
            y_train,
            c_value=candidate.c_value,
            class_weight=candidate.class_weight,
            model_cfg=model_cfg,
            seed=seed,
        )
        train_metrics, _, _, train_predict_seconds = _evaluate_predictions(
            clf, x_train, y_train, "train", top_k_values
        )
        eval_metrics, _, _, eval_predict_seconds = _evaluate_predictions(
            clf, x_eval, y_eval, eval_role, top_k_values
        )
        rows.append(
            _candidate_metric_row(
                candidate,
                unit_id,
                eval_role,
                train_metrics,
                fit_seconds,
                train_predict_seconds,
                convergence_messages,
            )
        )
        rows.append(
            _candidate_metric_row(
                candidate,
                unit_id,
                eval_role,
                eval_metrics,
                fit_seconds,
                eval_predict_seconds,
                [],
            )
        )
    return rows


def run_attribution_experiment_from_config(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run direct attribution development search from an already-resolved config."""
    _validate_dev_config(config)
    candidates = _candidate_grid(config)
    materialized_root, materialization_manifest, units = _load_materialization_summary(
        project_root, config
    )
    provenance = _build_provenance_block(
        project_root, materialized_root, materialization_manifest
    )
    _validate_candidate_search_units(units)
    default_name = config_path.stem if config_path is not None else "attribution_experiment"
    config_name = str(config.get("experiment", {}).get("name", default_name))
    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)
    _copy_config_outputs_if_available(config_path, results_dir, artifacts_dir)

    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    seed = int(experiment_cfg.get("seed", 42))
    selection_metric = str(experiment_cfg.get("selection_metric", "macro_f1"))
    top_k_values = [int(value) for value in model_cfg.get("top_k", [3, 5])]
    n_jobs = int(experiment_cfg.get("n_jobs", 1))

    conditions_by_id = {
        candidate.condition.condition_id: candidate.condition for candidate in candidates
    }
    target_name = str(config["source"].get("target", "author"))

    total_candidate_fits = len(units) * len(candidates)
    if show_progress:
        print(
            f"Loaded {len(units)} unit(s), {len(candidates)} candidate(s), target='{target_name}'."
        )
        print(
            f"Starting candidate search ({total_candidate_fits} fits across {len(units)} unit(s), n_jobs={n_jobs})..."
        )

    unit_rows_list: list[list[dict[str, Any]]] = joblib.Parallel(
        n_jobs=n_jobs, backend="loky", verbose=10
    )(
        joblib.delayed(_search_unit)(
            unit,
            conditions_by_id,
            candidates,
            materialized_root,
            target_name,
            model_cfg,
            seed,
            top_k_values,
        )
        for unit in units
    )
    candidate_rows = [row for unit_rows in unit_rows_list for row in unit_rows]
    if show_progress:
        print("Candidate search completed. Aggregating metrics...")

    metrics_df = pd.DataFrame(candidate_rows)
    summary_df = _summarize_candidates(metrics_df, selection_metric=selection_metric)
    condition_summary_df, selected_candidates = _select_direct_candidates_by_condition(
        summary_df,
        candidates,
        selection_metric,
    )
    if show_progress:
        print(f"Selected {len(selected_candidates)} condition candidate(s).")

    metrics_path = results_dir / "fold_metrics.csv"
    summary_path = results_dir / "candidate_summary.csv"
    condition_summary_path = results_dir / "condition_summary.csv"
    selected_candidates_path = results_dir / "selected_candidates.json"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    condition_summary_df.to_csv(condition_summary_path, index=False)

    selected_payload = {
        "selection_scope": "condition",
        "selection_metric": selection_metric,
        "split_name": config["source"]["split_name"],
        "materialization_name": config["source"]["materialization_name"],
        "target": target_name,
        "selected_candidates": selected_candidates,
    }
    write_json(selected_candidates_path, selected_payload)

    manifest = {
        "run_type": "dev_condition_selection",
        "selection_scope": "condition",
        "experiment_name": config_name,
        "config_path": _manifest_config_path(project_root, config_path),
        "split_name": config["source"]["split_name"],
        "materialization_name": config["source"]["materialization_name"],
        "target": target_name,
        "selection_metric": selection_metric,
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "materialized_root": relative_to_project(project_root, materialized_root),
        "candidate_count": len(candidates),
        "condition_count": len(selected_candidates),
        "unit_count": len(units),
        "fold_metrics_path": relative_to_project(project_root, metrics_path),
        "candidate_summary_path": relative_to_project(project_root, summary_path),
        "condition_summary_path": relative_to_project(
            project_root, condition_summary_path
        ),
        "selected_candidates_path": relative_to_project(
            project_root, selected_candidates_path
        ),
        "provenance": provenance,
        "source_manifest": {
            "path": relative_to_project(
                project_root, materialized_root / "manifest.json"
            ),
            "units": materialization_manifest.get("units", []),
        },
    }
    write_json(results_dir / "manifest.json", manifest)
    return manifest


def run_final_attribution_evaluation(
    config_path: Path,
    *,
    show_progress: bool = False,
    selected_candidates_path_override: Path | None = None,
) -> dict[str, Any]:
    """Run final direct-SVM evaluation from a config file."""
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    return run_final_attribution_evaluation_from_config(
        config,
        project_root=project_root,
        config_path=config_path,
        show_progress=show_progress,
        selected_candidates_path_override=selected_candidates_path_override,
    )


def run_attribution_experiment(
    config_path: Path, *, show_progress: bool = False
) -> dict[str, Any]:
    """Run direct-SVM condition selection from a config file."""
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    return run_attribution_experiment_from_config(
        config,
        project_root=project_root,
        config_path=config_path,
        show_progress=show_progress,
    )


def run_final_attribution_evaluation_from_config(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path | None = None,
    show_progress: bool = False,
    selected_candidates_path_override: Path | None = None,
    preloaded_candidates: list[DirectAttributionCandidateSpec] | None = None,
    preloaded_selection_payload: dict[str, Any] | None = None,
    preloaded_selection_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run direct final attribution evaluation from an already-resolved config."""
    _validate_final_eval_config(config)
    materialized_root, materialization_manifest, units = _load_materialization_summary(
        project_root, config
    )
    provenance = _build_provenance_block(
        project_root, materialized_root, materialization_manifest
    )
    unit = _validate_final_evaluation_units(units)
    if preloaded_candidates is None:
        candidates, selection_payload, selection_source = load_selected_direct_candidates(
            project_root,
            config,
            selected_candidates_path_override=selected_candidates_path_override,
        )
    else:
        candidates = preloaded_candidates
        selection_payload = preloaded_selection_payload
        selection_source = preloaded_selection_source

    default_name = config_path.stem if config_path is not None else "final_attribution_evaluation"
    config_name = str(config.get("experiment", {}).get("name", default_name))
    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)
    _copy_config_outputs_if_available(config_path, results_dir, artifacts_dir)

    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})
    seed = int(experiment_cfg.get("seed", 42))
    top_k_values = [int(value) for value in model_cfg.get("top_k", [3, 5])]
    save_top_k = int(experiment_cfg.get("save_prediction_top_k", 5))
    target_name = str(config["source"].get("target", "author"))
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id

    if show_progress:
        print(
            f"Running final attribution evaluation for {unit_id} across {len(candidates)} condition(s)."
        )

    y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
    y_eval = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")

    final_by_condition_dir, final_artifacts_by_condition_dir = final_condition_roots(
        results_dir,
        artifacts_dir,
    )

    selected_candidates_path = results_dir / "selected_candidates.json"
    write_json(selected_candidates_path, selection_payload)

    condition_results: list[dict[str, Any]] = []
    final_summary_rows: list[dict[str, Any]] = []
    source_payloads = {
        str(payload["candidate_id"]): payload
        for payload in selection_payload.get("selected_candidates", [])
    }

    for candidate in candidates:
        condition_id = candidate.condition.condition_id
        condition_results_dir = final_by_condition_dir / condition_id
        condition_artifacts_dir = final_artifacts_by_condition_dir / condition_id
        condition_artifacts_dir.mkdir(parents=True, exist_ok=True)

        if show_progress:
            print(f"[final] {condition_id}: fitting {candidate.candidate_id}")

        layout = candidate.condition.feature_layout
        _validate_variant_availability(unit, layout)
        x_train = _build_feature_matrix(unit, unit_dir, "train", layout)
        x_eval = _build_feature_matrix(unit, unit_dir, eval_role, layout)

        clf, convergence_messages, fit_seconds = _fit_linear_svm(
            x_train,
            y_train,
            c_value=candidate.c_value,
            class_weight=candidate.class_weight,
            model_cfg=model_cfg,
            seed=seed,
        )
        train_metrics, _, _, train_predict_seconds = _evaluate_predictions(
            clf,
            x_train,
            y_train,
            "train",
            top_k_values,
        )
        eval_metrics, y_pred, scores, predict_seconds = _evaluate_predictions(
            clf,
            x_eval,
            y_eval,
            eval_role,
            top_k_values,
        )

        prediction_frame = _prediction_frame(
            unit_dir=unit_dir,
            eval_role=eval_role,
            y_true=y_eval,
            y_pred=y_pred,
            scores=scores,
            classes=clf.classes_,
            save_top_k=save_top_k,
        )

        model_path = condition_artifacts_dir / "model.joblib"
        joblib.dump(clf, model_path)

        final_metrics_payload = {
            "unit_id": unit_id,
            "eval_role": eval_role,
            "target": target_name,
            "condition_id": condition_id,
            "condition_label": candidate.condition.label,
            "candidate_id": candidate.candidate_id,
            "fit_seconds": float(fit_seconds),
            "train_predict_seconds": float(train_predict_seconds),
            "predict_seconds": float(predict_seconds),
            "convergence_warning_count": len(convergence_messages),
            "convergence_warning": convergence_messages[0] if convergence_messages else "",
            "train_metrics": train_metrics,
            "final_test_metrics": eval_metrics,
        }

        resolved_candidate_payload = _candidate_payload(candidate)
        resolved_candidate_payload["source_payload"] = source_payloads[
            candidate.candidate_id
        ]

        condition_result = {
            "condition_id": condition_id,
            "condition_label": candidate.condition.label,
            "candidate_id": candidate.candidate_id,
            "unit_id": unit_id,
            "eval_role": eval_role,
        }
        condition_results.append(
            {
                **write_final_condition_output(
                    project_root=project_root,
                    condition_results_dir=condition_results_dir,
                    prediction_frame=prediction_frame,
                    metrics_payload=final_metrics_payload,
                    resolved_candidate_payload=resolved_candidate_payload,
                    condition_result=condition_result,
                ),
                "model_path": relative_to_project(project_root, model_path),
            }
        )

        dev_summary = source_payloads[candidate.candidate_id].get("dev_summary", {})
        final_summary_rows.append(
            build_final_summary_row(
                condition_id=condition_id,
                condition_label=candidate.condition.label,
                candidate_id=candidate.candidate_id,
                dev_summary=dev_summary,
                final_metrics=eval_metrics,
                extra_fields={
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                },
            )
        )

    final_condition_summary_path = write_final_condition_summary(
        results_dir,
        final_summary_rows,
    )

    manifest = {
        "run_type": "condition_final_evaluation",
        "selection_scope": "condition",
        "experiment_name": config_name,
        "config_path": _manifest_config_path(project_root, config_path),
        "split_name": config["source"]["split_name"],
        "materialization_name": config["source"]["materialization_name"],
        "target": target_name,
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "materialized_root": relative_to_project(project_root, materialized_root),
        "provenance": provenance,
        "selected_candidates_path": relative_to_project(
            project_root, selected_candidates_path
        ),
        "condition_count": len(candidates),
        "final_condition_summary_path": relative_to_project(
            project_root, final_condition_summary_path
        ),
        "condition_results": condition_results,
        "selection_source": {
            "selected_candidates_path": selection_source["selected_candidates_path"],
            "selection_results_dir": selection_source["selection_results_dir"],
            "selection_manifest_path": selection_source["selection_manifest_path"],
        },
        "source_manifest": {
            "path": relative_to_project(
                project_root, materialized_root / "manifest.json"
            ),
            "units": materialization_manifest.get("units", []),
        },
    }
    write_json(results_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    """CLI entry point for direct-SVM attribution training."""
    args = _parse_args()
    config_path = Path(args.config).resolve()
    manifest = run_attribution_experiment(
        config_path, show_progress=not args.no_progress
    )
    print(f"Finished attribution experiment: {manifest['experiment_name']}")
    print(f"Selected conditions: {manifest['condition_count']}")
    print(f"Results: {manifest['results_dir']}")
    print(f"Artifacts: {manifest['artifacts_dir']}")


if __name__ == "__main__":
    main()
