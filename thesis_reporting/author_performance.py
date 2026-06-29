"""Per-author performance analysis for research result reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ResultArtifacts
from .config import ResultSystem


AUTHOR_PERFORMANCE_COLUMNS = [
    "system_key",
    "system_label",
    "phase",
    "split",
    "architecture",
    "representation",
    "scope",
    "condition_id",
    "author_label",
    "author_display",
    "author_name",
    "author_party",
    "support",
    "pred_count",
    "correct_count",
    "error_count",
    "precision",
    "recall",
    "f1",
    "support_share",
    "accuracy_within_true_class",
    "rank_f1_desc",
    "rank_recall_desc",
    "rank_correct_count_desc",
    "rank_error_count_desc",
]


def read_per_author_metrics(
    system: ResultSystem,
    artifacts: ResultArtifacts,
) -> pd.DataFrame:
    """Read and annotate one system's per-author diagnostics table."""

    frame = artifacts.read_csv(system.per_author_metrics_path)
    frame["system_key"] = system.key
    frame["system_label"] = system.label
    frame["phase"] = system.phase
    frame["split"] = system.split
    frame["architecture"] = system.architecture
    frame["representation"] = system.representation
    frame["scope"] = system.scope
    frame["condition_id"] = system.condition_id
    frame["error_count"] = frame["support"] - frame["correct_count"]
    return frame


def build_per_author_rankings(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Combine per-author metrics and add within-system ranking columns."""

    frames = [
        read_per_author_metrics(system, results_dir)
        for system in systems
    ]
    rankings = pd.concat(frames, ignore_index=True)
    group = rankings.groupby("system_key", sort=False)
    rankings["rank_f1_desc"] = group["f1"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    rankings["rank_recall_desc"] = group["recall"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    rankings["rank_correct_count_desc"] = group["correct_count"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    rankings["rank_error_count_desc"] = group["error_count"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    ordered = rankings[AUTHOR_PERFORMANCE_COLUMNS]
    return ordered.sort_values(
        ["system_key", "rank_f1_desc", "rank_recall_desc", "author_display"],
        kind="stable",
    ).reset_index(drop=True)


def ranked_author_slice(
    per_author_rankings: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    top_n: int,
    best: bool,
) -> pd.DataFrame:
    """Select best or worst author rows per system for selected metrics."""

    rows: list[pd.DataFrame] = []
    for _, system_frame in per_author_rankings.groupby("system_key", sort=False):
        for metric in metrics:
            sort_columns, ascending = ranking_sort_order(metric, best=best)
            sorted_frame = system_frame.sort_values(
                sort_columns,
                ascending=ascending,
                kind="stable",
            ).head(top_n)
            selected = sorted_frame.copy()
            selected.insert(8, "ranking_metric", metric)
            selected.insert(9, "ranking_direction", "best" if best else "worst")
            selected.insert(10, "ranking_position", range(1, len(selected) + 1))
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def ranking_sort_order(metric: str, *, best: bool) -> tuple[list[str], list[bool]]:
    """Return explicit sort columns for best/worst author selections."""

    if best:
        return (
            [metric, "f1", "recall", "precision", "correct_count", "author_display"],
            [False, False, False, False, False, True],
        )
    if metric == "error_count":
        return (
            ["error_count", "f1", "recall", "precision", "author_display"],
            [False, True, True, True, True],
        )
    return (
        [metric, "error_count", "f1", "recall", "precision", "author_display"],
        [True, False, True, True, True, True],
    )


def build_best_authors(per_author_rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build top-author rows by F1, recall, and correct-count rankings."""

    return ranked_author_slice(
        per_author_rankings,
        metrics=("f1", "recall", "correct_count"),
        top_n=top_n,
        best=True,
    )


def build_worst_authors(per_author_rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build lowest-performing author rows by F1, recall, and error count."""

    return ranked_author_slice(
        per_author_rankings,
        metrics=("f1", "recall", "error_count"),
        top_n=top_n,
        best=False,
    )


def build_error_concentration(per_author_rankings: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much error is concentrated among the worst authors."""

    cutoffs = (1, 3, 5, 10, 20)
    rows: list[dict[str, Any]] = []
    for system_key, system_frame in per_author_rankings.groupby("system_key", sort=False):
        ordered = system_frame.sort_values(
            ["error_count", "f1", "recall", "author_display"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        total_errors = int(ordered["error_count"].sum())
        system_label = str(ordered["system_label"].iloc[0])
        for cutoff in cutoffs:
            selected = ordered.head(min(cutoff, len(ordered)))
            selected_errors = int(selected["error_count"].sum())
            rows.append(
                {
                    "system_key": system_key,
                    "system_label": system_label,
                    "n_authors": int(len(selected)),
                    "total_errors": total_errors,
                    "selected_author_errors": selected_errors,
                    "share_of_total_errors": (
                        selected_errors / total_errors if total_errors else 0.0
                    ),
                    "selected_authors": "; ".join(
                        selected["author_display"].astype(str).tolist()
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_author_performance_outputs(
    systems: tuple[ResultSystem, ...],
    *,
    results_dir: ResultArtifacts,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Write all files for the author-performance result addition."""

    section_dir = output_dir / "author_performance"
    section_dir.mkdir(parents=True, exist_ok=True)
    artifacts = results_dir
    per_author = build_per_author_rankings(systems, artifacts)
    best_authors = build_best_authors(per_author, top_n)
    worst_authors = build_worst_authors(per_author, top_n)
    error_concentration = build_error_concentration(per_author)

    paths = {
        "per_author_rankings": section_dir / "per_author_rankings.csv",
        "best_authors": section_dir / "best_authors.csv",
        "worst_authors": section_dir / "worst_authors.csv",
        "error_concentration": section_dir / "error_concentration.csv",
    }
    per_author.to_csv(paths["per_author_rankings"], index=False)
    best_authors.to_csv(paths["best_authors"], index=False)
    worst_authors.to_csv(paths["worst_authors"], index=False)
    error_concentration.to_csv(paths["error_concentration"], index=False)
    return {key: str(path) for key, path in paths.items()}
