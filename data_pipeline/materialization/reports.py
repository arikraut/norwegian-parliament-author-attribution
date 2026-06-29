"""Materialization diagnostics and summary reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.materialization.constants import _PREFERRED_TARGETS


def _build_stylometry_column_report(
    config: dict,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_role: str,
    stylo_feature_cols: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """Evaluate each stylometry column against configured drop criteria."""
    stylo_cfg = config.get("stylometry", {})
    variance_threshold = float(stylo_cfg.get("variance_threshold", 0.0) or 0.0)
    drop_zero_variance_columns = bool(
        stylo_cfg.get("drop_zero_variance_columns", False)
    )
    drop_low_variance_columns = bool(stylo_cfg.get("drop_low_variance_columns", False))
    _validate_finite_stylometry_values(train_df, eval_df, eval_role, stylo_feature_cols)

    report_rows: list[dict[str, object]] = []
    kept_columns: list[str] = []
    for col in stylo_feature_cols:
        train_values = pd.to_numeric(train_df[col], errors="coerce").to_numpy(
            dtype=float
        )
        eval_values = pd.to_numeric(eval_df[col], errors="coerce").to_numpy(
            dtype=float
        )
        variance = float(np.var(train_values, ddof=0)) if train_values.size else 0.0
        is_all_zero = bool(np.allclose(train_values, 0.0)) if train_values.size else True
        is_zero_variance = bool(np.isclose(variance, 0.0))
        is_low_variance = bool(variance <= variance_threshold)
        train_mean = float(train_values.mean()) if train_values.size else 0.0
        eval_mean = float(eval_values.mean()) if eval_values.size else 0.0
        train_std = float(train_values.std(ddof=0)) if train_values.size else 0.0
        eval_std = float(eval_values.std(ddof=0)) if eval_values.size else 0.0
        mean_gap = float(eval_mean - train_mean)
        pooled_scale = float(np.sqrt(((train_std**2) + (eval_std**2)) / 2.0))
        standardized_mean_gap = (
            float(abs(mean_gap) / pooled_scale) if pooled_scale > 0.0 else 0.0
        )

        drop_reason = ""
        if drop_zero_variance_columns and is_zero_variance:
            drop_reason = "zero_variance"
        elif drop_low_variance_columns and is_low_variance:
            drop_reason = "low_variance"

        report_rows.append(
            {
                "feature": col,
                "train_variance": variance,
                "is_all_zero": is_all_zero,
                "is_zero_variance": is_zero_variance,
                "has_nonfinite": False,
                "is_low_variance": is_low_variance,
                "train_mean": train_mean,
                "eval_mean": eval_mean,
                "train_std": train_std,
                "eval_std": eval_std,
                "mean_gap": mean_gap,
                "abs_mean_gap": abs(mean_gap),
                "standardized_mean_gap": standardized_mean_gap,
                "drop_reason": drop_reason,
                "kept": drop_reason == "",
            }
        )
        if drop_reason == "":
            kept_columns.append(col)

    return kept_columns, pd.DataFrame(report_rows)


def _validate_finite_stylometry_values(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_role: str,
    stylo_feature_cols: list[str],
) -> None:
    """Fail if selected raw stylometry contains NaN, Inf, or non-numeric values."""
    issues: list[str] = []
    for role, frame in (("train", train_df), (eval_role, eval_df)):
        for col in stylo_feature_cols:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            nonfinite_count = int((~np.isfinite(values)).sum())
            if nonfinite_count:
                issues.append(f"{role}:{col}={nonfinite_count}")

    if issues:
        raise ValueError(
            "Non-finite stylometry values found; materialization requires finite "
            "stylometry before scaling. Counts by role/column: "
            + ", ".join(issues)
        )


def _summarize_stylometry_drift(
    unit_id: str,
    eval_role: str,
    stylo_column_report: pd.DataFrame,
) -> dict[str, object]:
    """Collapse a per-column stylometry report into unit-level drift statistics."""
    kept_df = stylo_column_report[stylo_column_report["kept"]].copy()
    if kept_df.empty:
        return {
            "unit_id": unit_id,
            "eval_role": eval_role,
            "kept_feature_count": 0,
            "mean_standardized_mean_gap": 0.0,
            "median_standardized_mean_gap": 0.0,
            "max_standardized_mean_gap": 0.0,
            "features_over_gap_0_5": 0,
            "features_over_gap_1_0": 0,
        }

    standardized_gap = kept_df["standardized_mean_gap"].astype(float)
    return {
        "unit_id": unit_id,
        "eval_role": eval_role,
        "kept_feature_count": int(len(kept_df)),
        "mean_standardized_mean_gap": float(standardized_gap.mean()),
        "median_standardized_mean_gap": float(standardized_gap.median()),
        "max_standardized_mean_gap": float(standardized_gap.max()),
        "features_over_gap_0_5": int((standardized_gap > 0.5).sum()),
        "features_over_gap_1_0": int((standardized_gap > 1.0).sum()),
    }


def _build_target_summary(row_targets: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Summarize target coverage for the materialization manifest."""
    summary: dict[str, dict[str, int]] = {}
    for target in _PREFERRED_TARGETS:
        if target not in row_targets.columns:
            continue
        raw = row_targets[target].dropna()
        summary[target] = {
            "classes": int(raw.astype(str).nunique()),
            "non_null_rows": int(len(raw)),
        }
    return summary
