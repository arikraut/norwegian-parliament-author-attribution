"""Class-imbalance diagnostics for split bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _imbalance_row(vals: np.ndarray, **extra) -> dict:
    """Compute imbalance metrics for an array of per-author char counts."""
    vals = np.sort(vals.astype(float))
    n = len(vals)
    total = float(vals.sum())
    gini = (
        float((2 * np.arange(1, n + 1) - n - 1).dot(vals) / (n * total))
        if total > 0
        else 0.0
    )
    min_val = float(vals.min())
    row = {
        "n_authors": int(n),
        "min_chars": int(vals.min()),
        "max_chars": int(vals.max()),
        "mean_chars": float(vals.mean()),
        "median_chars": float(np.median(vals)),
        "std_chars": float(vals.std()),
        "imbalance_ratio": float(vals.max() / min_val) if min_val > 0 else float("inf"),
        "gini": round(gini, 4),
    }
    row.update(extra)
    return row


def _compute_class_imbalance_stats(author_fold_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold Gini and imbalance-ratio statistics for class balance reporting.

    Grouped-author profiling folds encode held-out authors as zero support in the opposite
    role, so zero-char rows are excluded to report imbalance only over authors that actually
    participate in each train/val partition.
    """
    rows = []
    for fold_id, group in author_fold_stats_df.groupby("fold_id", sort=True):
        for role in ("train", "val"):
            col = f"{role}_chars"
            if col not in group.columns:
                continue
            vals = (
                pd.to_numeric(group[col], errors="coerce")
                .fillna(0)
                .to_numpy(dtype=float)
            )
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            rows.append(_imbalance_row(vals, fold_id=fold_id, role=role))
    return pd.DataFrame(rows).sort_values(["fold_id", "role"]).reset_index(drop=True)


def _compute_outer_split_imbalance_stats(authors_df: pd.DataFrame) -> pd.DataFrame:
    """Class imbalance statistics for the available outer splits."""
    rows = []
    for role in ("train", "test"):
        col = f"{role}_chars"
        if col not in authors_df.columns:
            continue
        vals = authors_df[col].fillna(0).values
        if float(vals.sum()) == 0:
            continue
        rows.append(_imbalance_row(vals, role=role))
    return pd.DataFrame(rows).reset_index(drop=True)


def _compute_imbalance_stats(
    outer_authors: pd.DataFrame,
    author_fold_stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute outer and fold-level class-imbalance diagnostic tables."""
    outer_stats = _compute_outer_split_imbalance_stats(outer_authors)
    fold_stats = pd.DataFrame()
    if author_fold_stats is not None and not author_fold_stats.empty:
        fold_stats = _compute_class_imbalance_stats(author_fold_stats)
    return outer_stats, fold_stats
