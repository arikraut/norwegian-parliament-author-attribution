"""Per-author profiling-effect comparisons for research result reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import (
    ResultArtifacts,
    canonical_label,
    id_sample,
    validate_same_key_set,
    validate_unique_key,
)
from .author_performance import read_per_author_metrics
from .config import ResultSystem, SystemComparison


PROFILE_DELTA_METRICS = (
    "pred_count",
    "correct_count",
    "error_count",
    "precision",
    "recall",
    "f1",
    "accuracy_within_true_class",
)


def comparison_system_lookup(
    systems: tuple[ResultSystem, ...],
) -> dict[str, ResultSystem]:
    """Index configured systems by stable system key."""

    return {system.key: system for system in systems}


def delta_direction(delta: float) -> str:
    """Return a readable direction label for higher-is-better deltas."""

    if delta > 0:
        return "improved"
    if delta < 0:
        return "worse"
    return "unchanged"


def error_delta_direction(delta: float) -> str:
    """Return a readable direction label for error-count deltas."""

    if delta < 0:
        return "fewer_errors"
    if delta > 0:
        return "more_errors"
    return "unchanged"


def build_profile_deltas(
    systems: tuple[ResultSystem, ...],
    comparisons: tuple[SystemComparison, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Build per-author metric deltas for configured system comparisons."""

    systems_by_key = comparison_system_lookup(systems)
    metrics_by_system = {
        system.key: read_per_author_metrics(system, results_dir)
        for system in systems
    }
    frames: list[pd.DataFrame] = []
    for comparison in comparisons:
        source_system = systems_by_key[comparison.source_system_key]
        target_system = systems_by_key[comparison.target_system_key]
        source = metrics_by_system[source_system.key].copy()
        target = metrics_by_system[target_system.key].copy()
        source_context = (
            f"{comparison.key} source per-author metrics "
            f"{source_system.key}"
        )
        target_context = (
            f"{comparison.key} target per-author metrics "
            f"{target_system.key}"
        )
        validate_unique_key(source, "author_label", context=source_context)
        validate_unique_key(target, "author_label", context=target_context)
        source["_author_key"] = source["author_label"].map(canonical_label)
        target["_author_key"] = target["author_label"].map(canonical_label)
        validate_unique_key(source, "_author_key", context=source_context)
        validate_unique_key(target, "_author_key", context=target_context)
        validate_same_key_set(
            source,
            target,
            "_author_key",
            left_context=source_context,
            right_context=target_context,
        )
        support = source[["_author_key", "support"]].merge(
            target[["_author_key", "support"]],
            on="_author_key",
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        support_mismatch = support[
            support["support_source"] != support["support_target"]
        ]
        if not support_mismatch.empty:
            raise ValueError(
                f"{comparison.key}: compared systems must use identical per-author "
                "support. "
                f"Found {len(support_mismatch)} mismatched authors "
                f"(sample={id_sample(support_mismatch['_author_key'])})."
            )
        merged = source.merge(
            target,
            on="_author_key",
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        rows = pd.DataFrame(
            {
                "comparison_key": comparison.key,
                "comparison_label": comparison.label,
                "comparison_group": comparison.comparison_group,
                "comparison_purpose": comparison.purpose,
                "source_system_key": source_system.key,
                "source_system_label": source_system.label,
                "target_system_key": target_system.key,
                "target_system_label": target_system.label,
                "author_label": merged["author_label_source"],
                "author_display": merged["author_display_source"],
                "author_name": merged["author_name_source"],
                "author_party": merged["author_party_source"],
                "source_support": merged["support_source"],
                "target_support": merged["support_target"],
                "support_delta": merged["support_target"] - merged["support_source"],
            }
        )
        for metric in PROFILE_DELTA_METRICS:
            rows[f"source_{metric}"] = merged[f"{metric}_source"]
            rows[f"target_{metric}"] = merged[f"{metric}_target"]
            rows[f"{metric}_delta"] = rows[f"target_{metric}"] - rows[f"source_{metric}"]
        rows["f1_delta_direction"] = rows["f1_delta"].map(delta_direction)
        rows["recall_delta_direction"] = rows["recall_delta"].map(delta_direction)
        rows["error_count_delta_direction"] = rows["error_count_delta"].map(
            error_delta_direction
        )
        frames.append(rows)

    deltas = pd.concat(frames, ignore_index=True)
    group = deltas.groupby("comparison_key", sort=False)
    deltas["rank_f1_gain"] = group["f1_delta"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    deltas["rank_f1_loss"] = group["f1_delta"].rank(
        ascending=True,
        method="min",
    ).astype(int)
    return deltas.sort_values(
        ["comparison_key", "rank_f1_gain", "author_display"],
        kind="stable",
    ).reset_index(drop=True)


def weighted_delta(frame: pd.DataFrame, metric: str) -> float:
    """Compute support-weighted mean delta for one comparison frame."""

    total_support = frame["source_support"].sum()
    if total_support == 0:
        return 0.0
    return float((frame[f"{metric}_delta"] * frame["source_support"]).sum() / total_support)


def build_profile_delta_summary(profile_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize author-level profile deltas for each comparison."""

    rows: list[dict[str, Any]] = []
    for comparison_key, frame in profile_deltas.groupby("comparison_key", sort=False):
        by_gain = frame.sort_values(
            ["f1_delta", "recall_delta", "author_display"],
            ascending=[False, False, True],
            kind="stable",
        )
        by_loss = frame.sort_values(
            ["f1_delta", "recall_delta", "author_display"],
            ascending=[True, True, True],
            kind="stable",
        )
        rows.append(
            {
                "comparison_key": comparison_key,
                "comparison_label": frame["comparison_label"].iloc[0],
                "comparison_group": frame["comparison_group"].iloc[0],
                "comparison_purpose": frame["comparison_purpose"].iloc[0],
                "source_system_key": frame["source_system_key"].iloc[0],
                "target_system_key": frame["target_system_key"].iloc[0],
                "author_count": int(len(frame)),
                "authors_f1_improved": int((frame["f1_delta"] > 0).sum()),
                "authors_f1_worse": int((frame["f1_delta"] < 0).sum()),
                "authors_f1_unchanged": int((frame["f1_delta"] == 0).sum()),
                "authors_recall_improved": int((frame["recall_delta"] > 0).sum()),
                "authors_recall_worse": int((frame["recall_delta"] < 0).sum()),
                "mean_f1_delta": float(frame["f1_delta"].mean()),
                "median_f1_delta": float(frame["f1_delta"].median()),
                "support_weighted_f1_delta": weighted_delta(frame, "f1"),
                "mean_recall_delta": float(frame["recall_delta"].mean()),
                "support_weighted_recall_delta": weighted_delta(frame, "recall"),
                "source_correct_total": int(frame["source_correct_count"].sum()),
                "target_correct_total": int(frame["target_correct_count"].sum()),
                "correct_count_delta_total": int(frame["correct_count_delta"].sum()),
                "source_error_total": int(frame["source_error_count"].sum()),
                "target_error_total": int(frame["target_error_count"].sum()),
                "error_count_delta_total": int(frame["error_count_delta"].sum()),
                "largest_f1_gain": float(by_gain["f1_delta"].iloc[0]),
                "largest_f1_gain_author": by_gain["author_display"].iloc[0],
                "largest_f1_loss": float(by_loss["f1_delta"].iloc[0]),
                "largest_f1_loss_author": by_loss["author_display"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_profile_delta_slice(
    profile_deltas: pd.DataFrame,
    *,
    top_n: int,
    gains: bool,
) -> pd.DataFrame:
    """Select the largest per-author gains or losses per comparison."""

    rows: list[pd.DataFrame] = []
    for _, frame in profile_deltas.groupby("comparison_key", sort=False):
        sorted_frame = frame.sort_values(
            ["f1_delta", "recall_delta", "author_display"],
            ascending=[not gains, not gains, True],
            kind="stable",
        ).head(top_n)
        selected = sorted_frame.copy()
        selected.insert(4, "delta_rank", range(1, len(selected) + 1))
        selected.insert(5, "delta_direction", "gain" if gains else "loss")
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def write_profiling_effect_outputs(
    systems: tuple[ResultSystem, ...],
    comparisons: tuple[SystemComparison, ...],
    *,
    results_dir: ResultArtifacts,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Write all files for the profiling-effects result addition."""

    section_dir = output_dir / "profiling_effects"
    section_dir.mkdir(parents=True, exist_ok=True)
    artifacts = results_dir

    profile_deltas = build_profile_deltas(systems, comparisons, artifacts)
    profile_summary = build_profile_delta_summary(profile_deltas)
    top_gains = build_profile_delta_slice(profile_deltas, top_n=top_n, gains=True)
    top_losses = build_profile_delta_slice(profile_deltas, top_n=top_n, gains=False)
    oracle_gap = profile_deltas[
        profile_deltas["comparison_group"] == "oracle_predicted_gap"
    ].copy()

    paths = {
        "per_author_profile_deltas": section_dir / "per_author_profile_deltas.csv",
        "profile_delta_summary": section_dir / "profile_delta_summary.csv",
        "top_profile_gains": section_dir / "top_profile_gains.csv",
        "top_profile_losses": section_dir / "top_profile_losses.csv",
        "oracle_predicted_gap_by_author": (
            section_dir / "oracle_predicted_gap_by_author.csv"
        ),
    }
    profile_deltas.to_csv(paths["per_author_profile_deltas"], index=False)
    profile_summary.to_csv(paths["profile_delta_summary"], index=False)
    top_gains.to_csv(paths["top_profile_gains"], index=False)
    top_losses.to_csv(paths["top_profile_losses"], index=False)
    oracle_gap.to_csv(paths["oracle_predicted_gap_by_author"], index=False)
    return {key: str(path) for key, path in paths.items()}
