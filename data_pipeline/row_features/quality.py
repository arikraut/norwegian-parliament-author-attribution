"""Stylometry quality report builders."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from data_pipeline.row_features.stylometry import (
    STYLOMETRY_FEATURE_FAMILIES,
    SUBSTITUTION_COUNTER_KEYS,
    stylometry_feature_family,
)


def build_stylometry_quality_reports(
    stylometry_frames: dict[str, pd.DataFrame],
    split_quality: dict[str, dict[str, int | float | str]],
    *,
    low_variance_threshold: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-split quality and low-variance reports from raw stylometry frames."""
    quality_rows: list[dict[str, int | float | str]] = []
    low_variance_rows: list[dict[str, int | float | str | bool]] = []

    def _append_reports(
        split_name: str,
        stylo_df: pd.DataFrame,
        base_quality: dict[str, int | float | str],
    ) -> None:
        """Record quality and low-variance rows for one stylometry split."""
        feature_cols = [
            col
            for col in stylo_df.columns
            if col not in {"id_speech", "id_person", "outer_role"}
        ]
        if feature_cols:
            numeric = (
                stylo_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            )
            variances = numeric.var(axis=0, ddof=0)
            means = numeric.mean(axis=0)
            stds = numeric.std(axis=0, ddof=0)
            all_zero_mask = numeric.abs().sum(axis=0) == 0.0
            zero_variance_mask = variances == 0.0
            low_variance_mask = variances <= float(low_variance_threshold)
            feature_families = {
                col: stylometry_feature_family(col) for col in feature_cols
            }
            flagged_cols = (
                feature_cols
                if split_name == "all_splits"
                else [
                    col
                    for col in feature_cols
                    if bool(all_zero_mask[col]) or bool(low_variance_mask[col])
                ]
            )
            for col in flagged_cols:
                low_variance_rows.append(
                    {
                        "split": split_name,
                        "feature": col,
                        "feature_family": feature_families[col],
                        "mean": float(means[col]),
                        "std": float(stds[col]),
                        "variance": float(variances[col]),
                        "is_all_zero": bool(all_zero_mask[col]),
                        "is_zero_variance": bool(zero_variance_mask[col]),
                        "is_low_variance": bool(low_variance_mask[col]),
                    }
                )
            family_all_zero_counts = {
                family: int(
                    sum(
                        1
                        for col in feature_cols
                        if feature_families[col] == family and bool(all_zero_mask[col])
                    )
                )
                for family in STYLOMETRY_FEATURE_FAMILIES
            }
            family_low_variance_counts = {
                family: int(
                    sum(
                        1
                        for col in feature_cols
                        if feature_families[col] == family
                        and bool(low_variance_mask[col])
                    )
                )
                for family in STYLOMETRY_FEATURE_FAMILIES
            }
            quality_rows.append(
                {
                    **base_quality,
                    "split": split_name,
                    "n_features": int(len(feature_cols)),
                    "all_zero_feature_count": int(all_zero_mask.sum()),
                    "zero_variance_feature_count": int(zero_variance_mask.sum()),
                    "low_variance_feature_count": int(low_variance_mask.sum()),
                    "low_variance_threshold": float(low_variance_threshold),
                    **{
                        f"all_zero_feature_count__{family}": count
                        for family, count in family_all_zero_counts.items()
                    },
                    **{
                        f"low_variance_feature_count__{family}": count
                        for family, count in family_low_variance_counts.items()
                    },
                }
            )
        else:
            quality_rows.append(
                {
                    **base_quality,
                    "split": split_name,
                    "n_features": 0,
                    "all_zero_feature_count": 0,
                    "zero_variance_feature_count": 0,
                    "low_variance_feature_count": 0,
                    "low_variance_threshold": float(low_variance_threshold),
                    **{
                        f"all_zero_feature_count__{family}": 0
                        for family in STYLOMETRY_FEATURE_FAMILIES
                    },
                    **{
                        f"low_variance_feature_count__{family}": 0
                        for family in STYLOMETRY_FEATURE_FAMILIES
                    },
                }
            )

    for split_name, stylo_df in stylometry_frames.items():
        _append_reports(split_name, stylo_df, dict(split_quality.get(split_name, {})))

    if stylometry_frames:
        overall_df = pd.concat(list(stylometry_frames.values()), ignore_index=True)
        overall_quality = Counter()
        for quality in split_quality.values():
            for key in (
                *SUBSTITUTION_COUNTER_KEYS,
                "total_substitutions",
                "nonfinite_output_cells",
            ):
                overall_quality[key] += int(quality.get(key, 0))
            for family in STYLOMETRY_FEATURE_FAMILIES:
                family_key = f"total_substitutions__{family}"
                overall_quality[family_key] += int(quality.get(family_key, 0))
        _append_reports(
            "all_splits",
            overall_df,
            {
                "n_rows": int(len(overall_df)),
                "missing_value_substitutions": int(
                    overall_quality.get("missing_value_substitutions", 0)
                ),
                "nan_substitutions": int(overall_quality.get("nan_substitutions", 0)),
                "inf_substitutions": int(overall_quality.get("inf_substitutions", 0)),
                "non_numeric_substitutions": int(
                    overall_quality.get("non_numeric_substitutions", 0)
                ),
                "total_substitutions": int(
                    overall_quality.get("total_substitutions", 0)
                ),
                "nonfinite_output_cells": int(
                    overall_quality.get("nonfinite_output_cells", 0)
                ),
                **{
                    f"total_substitutions__{family}": int(
                        overall_quality.get(f"total_substitutions__{family}", 0)
                    )
                    for family in STYLOMETRY_FEATURE_FAMILIES
                },
            },
        )

    quality_report = pd.DataFrame(quality_rows)
    low_variance_report = pd.DataFrame(low_variance_rows)
    if not quality_report.empty:
        quality_report = quality_report.sort_values("split").reset_index(drop=True)
    if not low_variance_report.empty:
        low_variance_report = low_variance_report.sort_values(
            ["split", "feature"]
        ).reset_index(drop=True)
    return quality_report, low_variance_report
