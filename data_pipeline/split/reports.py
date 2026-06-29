"""Human-readable split summaries and report tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_split_stats(df_split: pd.DataFrame, split_label: str) -> dict:
    """Compute row/author/char/word summary stats for one split role."""
    if df_split.empty:
        return {
            "split": split_label,
            "n_rows": 0,
            "n_speeches": 0,
            "n_authors": 0,
            "total_words": 0,
            "total_chars": 0,
            "mean_words": 0.0,
            "median_words": 0.0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "female_author_pct": 0.0,
            "min_election": None,
            "max_election": None,
            "min_date": "",
            "max_date": "",
        }

    author_female = df_split.groupby("id_person")["female"].first()
    if "date" in df_split.columns:
        parsed_dates = pd.to_datetime(df_split["date"], errors="coerce")
        min_date = parsed_dates.min()
        max_date = parsed_dates.max()
    else:
        min_date = pd.NaT
        max_date = pd.NaT
    return {
        "split": split_label,
        "n_rows": len(df_split),
        "n_speeches": int(df_split["id_speech"].nunique()),
        "n_authors": int(df_split["id_person"].nunique()),
        "total_words": float(df_split["word_count"].sum()),
        "total_chars": float(df_split["char_count"].sum()),
        "mean_words": float(df_split["word_count"].mean()),
        "median_words": float(df_split["word_count"].median()),
        "mean_chars": float(df_split["char_count"].mean()),
        "median_chars": float(df_split["char_count"].median()),
        "female_author_pct": float(author_female.mean() * 100),
        "min_election": int(df_split["election"].min()),
        "max_election": int(df_split["election"].max()),
        "min_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "",
        "max_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "",
    }


def _compute_distribution(
    df_split: pd.DataFrame, column: str, split_label: str
) -> pd.DataFrame:
    """Return a per-value count and percentage table for one split role."""
    if df_split.empty:
        return pd.DataFrame(columns=["split", column, "count", "pct"])
    counts = df_split[column].value_counts(dropna=False)
    total = counts.sum()
    return pd.DataFrame(
        {
            "split": split_label,
            column: counts.index,
            "count": counts.values,
            "pct": counts.values / total * 100.0,
        }
    )


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine report frames while preserving the empty-frame contract."""
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def _markdown_cell(value: object) -> str:
    """Format one report value so it is safe inside a Markdown table cell."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_markdown_cell(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        rounded = round(float(value), 2)
        if float(rounded).is_integer():
            return str(int(rounded))
        return f"{rounded:.2f}".rstrip("0").rstrip(".")
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def _frame_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows._"

    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _single_row_markdown(title_col: str, values: dict[str, object]) -> pd.DataFrame:
    """Build a two-column table from scalar summary values."""
    return pd.DataFrame(
        [{title_col: key, "value": _markdown_cell(val)} for key, val in values.items()]
    )


def _build_author_summary_frame(
    authors_meta: pd.DataFrame, *, author_col: str
) -> pd.DataFrame:
    """Summarize selected-author demographics for the split report."""
    author_summary = {
        "n_authors": int(authors_meta[author_col].nunique()),
        "female_pct": (
            float(authors_meta["female"].mean() * 100)
            if "female" in authors_meta.columns
            else 0.0
        ),
    }
    ages = authors_meta.get("mean_age")
    if ages is not None:
        ages = ages.dropna()
        if not ages.empty:
            author_summary.update(
                {
                    "mean_age_mean": float(ages.mean()),
                    "mean_age_median": float(ages.median()),
                    "mean_age_min": float(ages.min()),
                    "mean_age_max": float(ages.max()),
                }
            )
    return _single_row_markdown(
        "metric",
        author_summary,
    )


def _build_author_party_distribution(
    authors_meta: pd.DataFrame, *, party_col: str
) -> pd.DataFrame:
    """Count selected authors by party for split diagnostics."""
    author_party_counts = authors_meta[party_col].value_counts(dropna=False)
    return pd.DataFrame(
        {party_col: author_party_counts.index, "count": author_party_counts.values}
    ).assign(pct=lambda x: x["count"] / x["count"].sum() * 100.0)


def _build_author_language_distribution(authors_meta: pd.DataFrame) -> pd.DataFrame:
    """Count selected authors by majority language for split diagnostics."""
    if "language_main" not in authors_meta.columns:
        return pd.DataFrame(columns=["language_main", "count", "pct"])
    author_lang_counts = authors_meta["language_main"].value_counts(dropna=False)
    return pd.DataFrame(
        {"language_main": author_lang_counts.index, "count": author_lang_counts.values}
    ).assign(pct=lambda x: x["count"] / x["count"].sum() * 100.0)


def _write_split_summary_markdown(
    summary_path: Path,
    *,
    split_name: str,
    experiment_name: str,
    split_strategy: str | None,
    manifest: dict,
    corpus_stats: pd.DataFrame,
    party_distribution: pd.DataFrame,
    language_distribution: pd.DataFrame,
    author_summary: pd.DataFrame,
    author_party_distribution: pd.DataFrame,
    author_language_distribution: pd.DataFrame,
    fold_author_summary: pd.DataFrame,
    outer_imbalance_stats: pd.DataFrame,
    fold_imbalance_stats: pd.DataFrame,
) -> None:
    """Write the human-readable split summary beside tabular diagnostics."""
    overview = pd.DataFrame(
        [
            {
                "field": "split_name",
                "value": split_name,
            },
            {
                "field": "experiment_name",
                "value": experiment_name,
            },
            {
                "field": "split_strategy",
                "value": split_strategy or "",
            },
            {
                "field": "selected_authors",
                "value": int(manifest.get("counts", {}).get("selected_authors", 0)),
            },
            {
                "field": "train_rows",
                "value": int(manifest.get("counts", {}).get("train_rows", 0)),
            },
            {
                "field": "test_rows",
                "value": int(manifest.get("counts", {}).get("test_rows", 0)),
            },
            {
                "field": "fold_count",
                "value": int(manifest.get("fold_count", 0)),
            },
            {
                "field": "created_at_utc",
                "value": manifest.get("created_at_utc", ""),
            },
        ]
    )

    support_policy = manifest.get("support_policy", {})
    support_thresholds = _single_row_markdown(
        "metric", support_policy.get("thresholds", {})
    )
    support_totals = _single_row_markdown("metric", support_policy.get("totals", {}))
    support_reasons = pd.DataFrame(
        [
            {"reason": reason, **values}
            for reason, values in sorted(
                support_policy.get("excluded_by_reason", {}).items()
            )
        ]
    )
    dropped_folds = pd.DataFrame(manifest.get("dropped_folds", []))
    kept_folds = pd.DataFrame(manifest.get("folds", []))

    sections = [
        f"# Split Summary: {split_name}",
        "",
        "## Overview",
        _frame_to_markdown(overview),
        "",
        "## Corpus Stats",
        _frame_to_markdown(corpus_stats),
        "",
        "## Author Summary",
        _frame_to_markdown(author_summary),
        "",
        "## Author Party Distribution",
        _frame_to_markdown(author_party_distribution),
    ]

    if not author_language_distribution.empty:
        sections.extend(
            [
                "",
                "## Author Language Distribution",
                _frame_to_markdown(author_language_distribution),
            ]
        )

    sections.extend(
        [
            "",
            "## Speech-Level Party Distribution",
            _frame_to_markdown(party_distribution),
            "",
            "## Speech-Level Language Distribution",
            _frame_to_markdown(language_distribution),
            "",
            "## Kept Folds",
            _frame_to_markdown(kept_folds),
        ]
    )

    if not dropped_folds.empty:
        sections.extend(
            [
                "",
                "## Dropped Folds",
                _frame_to_markdown(dropped_folds),
            ]
        )

    sections.extend(
        [
            "",
            "## Reliability Thresholds",
            _frame_to_markdown(support_thresholds),
            "",
            "## Reliability Totals",
            _frame_to_markdown(support_totals),
        ]
    )

    if not support_reasons.empty:
        sections.extend(
            [
                "",
                "## Reliability Exclusions By Reason",
                _frame_to_markdown(support_reasons),
            ]
        )

    if not fold_author_summary.empty:
        sections.extend(
            [
                "",
                "## Fold Author Summary",
                _frame_to_markdown(fold_author_summary),
            ]
        )

    sections.extend(
        [
            "",
            "## Outer Split Imbalance",
            _frame_to_markdown(outer_imbalance_stats),
        ]
    )

    if not fold_imbalance_stats.empty:
        sections.extend(
            [
                "",
                "## Fold Imbalance",
                _frame_to_markdown(fold_imbalance_stats),
            ]
        )

    summary_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
