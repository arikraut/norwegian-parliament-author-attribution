"""Profiling candidate-selection contracts for direct-SVM classifiers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd

from models.SVM.linear_svm_common import (
    FeatureLayout,
    SUPPORTED_BLOCKS,
    _parse_class_weights,
)


@dataclass(frozen=True)
class ProfilingCandidateSpec:
    """One Phase 2 profiling candidate: feature layout plus SVM hyperparameters."""

    feature_layout: FeatureLayout
    c_value: float
    class_weight: str | None

    @property
    def candidate_id(self) -> str:
        """Return the stable profiling candidate artifact identifier."""
        return (
            f"{self.feature_layout.name}__C={self.c_value:g}"
            f"__class_weight={self.class_weight_label}"
        )

    @property
    def class_weight_label(self) -> str:
        """Return the JSON/CSV representation of the class-weight setting."""
        return "none" if self.class_weight is None else str(self.class_weight)


def parse_profiling_feature_layouts(config: dict[str, Any]) -> list[FeatureLayout]:
    """Parse profiling ``[[feature_sets]]`` tables into shared feature layouts."""
    raw_layouts = config.get("feature_sets", [])
    if not raw_layouts:
        raise ValueError("Config must define at least one [[feature_sets]] entry.")

    layouts: list[FeatureLayout] = []
    seen_names: set[str] = set()
    for raw_layout in raw_layouts:
        name = str(raw_layout["name"]).strip()
        blocks = tuple(str(block).strip() for block in raw_layout.get("blocks", []))
        if not name:
            raise ValueError("Feature layout name cannot be empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate [[feature_sets]].name value: {name!r}")
        seen_names.add(name)
        if not blocks:
            raise ValueError(
                f"Feature layout '{name}' must list at least one feature block."
            )
        unknown_blocks = sorted(set(blocks) - SUPPORTED_BLOCKS)
        if unknown_blocks:
            raise ValueError(
                f"Feature layout '{name}' uses unsupported blocks: {unknown_blocks}"
            )
        if "all" in blocks and len(blocks) > 1:
            raise ValueError(
                f"Feature layout '{name}' cannot mix 'all' with other blocks."
            )
        normalize_each_block = bool(raw_layout.get("normalize_each_block", False))
        if normalize_each_block and "all" in blocks:
            raise ValueError(
                f"Feature layout '{name}' cannot set normalize_each_block=true when blocks=['all']."
            )

        raw_weights = raw_layout.get("block_weights", {})
        block_weights = {block: float(raw_weights.get(block, 1.0)) for block in blocks}
        layouts.append(
            FeatureLayout(
                name=name,
                blocks=blocks,
                normalize_rows=bool(raw_layout.get("normalize_rows", True)),
                normalize_each_block=normalize_each_block,
                block_weights=block_weights,
            )
        )
    return layouts


def profiling_candidate_grid(config: dict[str, Any]) -> list[ProfilingCandidateSpec]:
    """Build the Phase 2 profiling candidate grid from layouts and hyperparameters."""
    model_cfg = config.get("model", {})
    c_values = [float(value) for value in model_cfg.get("C_values", [1.0])]
    if not c_values:
        raise ValueError("model.C_values must contain at least one value.")

    class_weights = _parse_class_weights(list(model_cfg.get("class_weights", ["none"])))
    layouts = parse_profiling_feature_layouts(config)
    candidates = [
        ProfilingCandidateSpec(
            feature_layout=layout,
            c_value=c_value,
            class_weight=class_weight,
        )
        for layout, c_value, class_weight in itertools.product(
            layouts, c_values, class_weights
        )
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


def profiling_candidate_from_payload(
    payload: dict[str, Any],
) -> ProfilingCandidateSpec:
    """Reconstruct a profiling candidate from a ``best_candidate.json`` payload."""
    feature_set = str(payload.get("feature_set", "")).strip()
    if not feature_set:
        raise ValueError(
            "Frozen best candidate payload must include a non-empty feature_set."
        )

    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(
            "Frozen best candidate payload must include a non-empty blocks list."
        )
    blocks = tuple(str(block).strip() for block in raw_blocks)
    unknown_blocks = sorted(set(blocks) - SUPPORTED_BLOCKS)
    if unknown_blocks:
        raise ValueError(
            f"Frozen best candidate payload uses unsupported blocks: {unknown_blocks}"
        )
    if "all" in blocks and len(blocks) > 1:
        raise ValueError(
            "Frozen best candidate payload cannot mix 'all' with other blocks."
        )
    normalize_each_block = bool(payload.get("normalize_each_block", False))
    if normalize_each_block and "all" in blocks:
        raise ValueError(
            "Frozen best candidate payload cannot set normalize_each_block=true with blocks=['all']."
        )

    raw_block_weights = payload.get("block_weights", {})
    if raw_block_weights is None:
        raw_block_weights = {}
    if not isinstance(raw_block_weights, dict):
        raise ValueError(
            "Frozen best candidate payload block_weights must be a mapping if provided."
        )
    block_weights = {
        block: float(raw_block_weights.get(block, 1.0)) for block in blocks
    }

    candidate = ProfilingCandidateSpec(
        feature_layout=FeatureLayout(
            name=feature_set,
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
            "Frozen best candidate payload is inconsistent: "
            f"candidate_id={payload_candidate_id!r} does not match resolved {candidate.candidate_id!r}."
        )
    return candidate


def profiling_candidate_metric_row(
    candidate: ProfilingCandidateSpec,
    unit_id: str,
    eval_role: str,
    split_metrics: dict[str, Any],
    fit_seconds: float,
    predict_seconds: float,
    convergence_messages: list[str],
) -> dict[str, Any]:
    """Create one profiling fold-metrics CSV row for a candidate and split."""
    layout = candidate.feature_layout
    row = {
        "candidate_id": candidate.candidate_id,
        "feature_set": layout.name,
        "blocks": "+".join(layout.blocks),
        "normalize_rows": layout.normalize_rows,
        "normalize_each_block": layout.normalize_each_block,
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


def summarize_profiling_candidates(
    metrics_df: pd.DataFrame, selection_metric: str
) -> pd.DataFrame:
    """Aggregate per-unit profiling candidate metrics and rank configurations."""
    if metrics_df.empty:
        raise ValueError("No metrics were collected.")

    eval_df = metrics_df[metrics_df["split"] != "train"].copy()
    if eval_df.empty:
        raise ValueError(
            "Candidate summary requires at least one evaluation split per candidate."
        )

    group_cols = [
        "candidate_id",
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
    author_weighted_metric_cols = sorted(
        col for col in metrics_df.columns if col.startswith("author_weighted_")
    )
    metric_cols = base_metric_cols + top_k_metric_cols + author_weighted_metric_cols

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

    primary_sort_col = f"eval_mean_{selection_metric}"
    if primary_sort_col not in eval_summary.columns:
        available = sorted(
            col for col in eval_summary.columns if col.startswith("eval_mean_")
        )
        raise ValueError(
            f"Selection metric '{selection_metric}' is not available. Available: {available}"
        )

    sort_cols = [primary_sort_col]
    if selection_metric == "author_weighted_macro_f1":
        sort_cols.append("eval_mean_macro_f1")
    sort_cols.extend(["eval_mean_accuracy", "n_eval_units"])
    sort_cols = list(dict.fromkeys(sort_cols))
    ascending = [False] * len(sort_cols)

    return eval_summary.sort_values(
        by=sort_cols + ["c_value", "feature_set", "class_weight"],
        ascending=ascending + [True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def best_profiling_candidate(
    summary_df: pd.DataFrame,
    candidates: list[ProfilingCandidateSpec],
) -> ProfilingCandidateSpec:
    """Return the profiling candidate represented by the first summary row."""
    if summary_df.empty:
        raise ValueError("Candidate summary is empty.")
    best_row = summary_df.iloc[0]
    best_id = str(best_row["candidate_id"])
    for candidate in candidates:
        if candidate.candidate_id == best_id:
            return candidate
    raise KeyError(f"Best candidate '{best_id}' was not present in the candidate grid.")
