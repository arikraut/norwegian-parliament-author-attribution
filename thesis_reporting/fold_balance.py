"""Analysis and rendering for profiling fold-balance reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_pipeline.utils import relative_to_project


DEFAULT_TARGETS = ("party", "female", "age_bin", "left_center_right")


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    """Validate required CSV columns at the report input boundary."""
    missing = set(columns) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path}: missing required columns: {missing_list}")


def _normalize_label(value: object) -> str:
    """Normalize missing and scalar target labels for reporting."""
    if pd.isna(value):
        return "missing"
    return str(value)


def _label_sort_key(label: str) -> tuple[int, str]:
    """Sort missing labels last and all other labels alphabetically."""
    return (label == "missing", label)


def _format_pct(value: float) -> str:
    """Format a percentage for Markdown output."""
    return f"{value:.1f}%"


def _format_float(value: object) -> str:
    """Format floats compactly while preserving non-float values."""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    """Render rows as a Markdown table for the fold-balance report."""
    if not rows:
        return "_No rows._"

    def cell(value: object) -> str:
        """Format one Markdown cell without breaking table syntax."""
        text = _format_float(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines)


def _value_counts(df: pd.DataFrame, target: str) -> pd.Series:
    """Return raw label counts for one target column."""
    labels = df[target].map(_normalize_label)
    return labels.value_counts(dropna=False)


def _with_inverse_author_weights(
    df: pd.DataFrame, *, group_cols: list[str]
) -> pd.DataFrame:
    """Return rows with per-speech weights that sum to 1.0 per author group."""
    weighted = df.copy()
    count_cols = [*group_cols, "id_person"]
    author_counts = weighted.groupby(count_cols, dropna=False)["id_speech"].transform(
        "count"
    )
    weighted["author_weight"] = 1.0 / author_counts
    return weighted


def _weighted_value_sums(df: pd.DataFrame, target: str, weight_col: str) -> pd.Series:
    """Sum inverse-author weights by target label."""
    labels = df[target].map(_normalize_label)
    return df[weight_col].astype(float).groupby(labels, dropna=False).sum()


def _distribution(
    df: pd.DataFrame,
    target: str,
    *,
    weight_col: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return label counts and percentages for one report aggregation mode."""
    counts = (
        _weighted_value_sums(df, target, weight_col)
        if weight_col is not None
        else _value_counts(df, target)
    )
    total = float(counts.sum())
    percentages = counts if total == 0.0 else counts.div(total).mul(100.0)
    return counts, percentages


def _distribution_rows(
    df: pd.DataFrame, targets: list[str], *, scope: str
) -> list[dict[str, object]]:
    """Build unweighted label-distribution rows for report tables."""
    rows: list[dict[str, object]] = []
    for target in targets:
        counts = _value_counts(df, target)
        total = int(counts.sum())
        for label in sorted(counts.index, key=_label_sort_key):
            count = int(counts.loc[label])
            rows.append(
                {
                    "scope": scope,
                    "target": target,
                    "label": label,
                    "count": count,
                    "pct": round(count / total * 100.0, 2) if total else 0.0,
                }
            )
    return rows


def _weighted_distribution_rows(
    df: pd.DataFrame,
    targets: list[str],
    *,
    scope: str,
    weight_col: str,
) -> list[dict[str, object]]:
    """Build inverse-author-weighted label-distribution rows."""
    rows: list[dict[str, object]] = []
    for target in targets:
        weighted_counts = _weighted_value_sums(df, target, weight_col)
        total = float(weighted_counts.sum())
        for label in sorted(weighted_counts.index, key=_label_sort_key):
            effective_count = float(weighted_counts.loc[label])
            rows.append(
                {
                    "scope": scope,
                    "target": target,
                    "label": label,
                    "effective_author_count": round(effective_count, 2),
                    "pct": round(effective_count / total * 100.0, 2) if total else 0.0,
                }
            )
    return rows


def _mode_per_author(targets: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Collapse speech-level target labels to one modal label per author."""
    labels = targets[["id_person", target]].copy()
    labels[target] = labels[target].map(_normalize_label)

    label_counts = (
        labels.groupby(["id_person", target], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["id_person", "count", target], ascending=[True, False, True])
    )
    author_modes = label_counts.drop_duplicates("id_person")[
        ["id_person", target]
    ].copy()

    distinct_counts = labels.groupby("id_person", dropna=False)[target].nunique(
        dropna=False
    )
    mixed_author_count = int((distinct_counts > 1).sum())
    return author_modes, mixed_author_count


def _balance_summary(
    folded: pd.DataFrame,
    overall: pd.DataFrame,
    targets: list[str],
    *,
    unit_col: str,
    scope: str,
    roles: list[str],
    weight_col: str | None = None,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], list[dict[str, object]]]]:
    """Compare fold distributions with their unweighted or weighted target pool."""
    summary_rows: list[dict[str, object]] = []
    distribution_tables: dict[tuple[str, str], list[dict[str, object]]] = {}

    for target in targets:
        _, overall_percentages = _distribution(
            overall,
            target,
            weight_col=weight_col,
        )
        overall_dist = overall_percentages.to_dict()
        for role in roles:
            role_df = folded[folded["fold_role"] == role]
            if role_df.empty:
                continue

            labels = sorted(
                set(overall_dist)
                | set(role_df[target].map(_normalize_label).drop_duplicates().tolist()),
                key=_label_sort_key,
            )
            pivot_rows: list[dict[str, object]] = []

            for fold_id, fold_df in role_df.groupby("fold_id", sort=True):
                fold_counts, fold_percentages = _distribution(
                    fold_df,
                    target,
                    weight_col=weight_col,
                )
                fold_dist = fold_percentages.to_dict()
                signed_deviations = {
                    label: float(fold_dist.get(label, 0.0))
                    - float(overall_dist.get(label, 0.0))
                    for label in labels
                }
                majority_label = (
                    str(fold_counts.idxmax()) if not fold_counts.empty else ""
                )
                majority_pct = (
                    float(fold_dist.get(majority_label, 0.0)) if majority_label else 0.0
                )
                over_label = max(signed_deviations, key=signed_deviations.get)
                under_label = min(signed_deviations, key=signed_deviations.get)

                summary_row: dict[str, object] = {
                    "scope": scope,
                    "role": role,
                    "target": target,
                    "fold_id": fold_id,
                    "majority_label": majority_label,
                    "majority_pct": round(majority_pct, 2),
                    "max_abs_pp_deviation": round(
                        max(abs(value) for value in signed_deviations.values()), 2
                    ),
                    "most_overrepresented_label": over_label,
                    "most_overrepresented_pp": round(
                        signed_deviations[over_label], 2
                    ),
                    "most_underrepresented_label": under_label,
                    "most_underrepresented_pp": round(
                        signed_deviations[under_label], 2
                    ),
                }
                if weight_col is None:
                    summary_row[f"n_{scope}_units"] = int(
                        fold_df[unit_col].nunique()
                    )
                else:
                    summary_row["effective_author_count"] = round(
                        float(fold_df[weight_col].sum()), 2
                    )
                    summary_row["n_authors"] = int(fold_df[unit_col].nunique())
                summary_rows.append(summary_row)

                pivot_row: dict[str, object] = {"fold_id": fold_id}
                for label in labels:
                    pivot_row[label] = _format_pct(float(fold_dist.get(label, 0.0)))
                pivot_rows.append(pivot_row)

            distribution_tables[(scope, f"{target}:{role}")] = pivot_rows

    return summary_rows, distribution_tables


def _merge_complete_targets(
    memberships: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    on: list[str],
    context: str,
) -> pd.DataFrame:
    """Join external fold inputs while rejecting memberships without target rows."""
    merged = memberships.merge(
        targets,
        on=on,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] == "left_only", on]
    if not unmatched.empty:
        sample = unmatched.head(10).to_dict(orient="records")
        raise ValueError(
            f"{context}: {len(unmatched)} fold membership rows have no target row "
            f"(sample={sample})."
        )
    return merged.drop(columns="_merge")


@dataclass(frozen=True)
class FoldInputTables:
    """Hold validated external inputs and their resolved source paths."""

    memberships: pd.DataFrame
    targets: pd.DataFrame
    folds_path: Path
    targets_path: Path


@dataclass(frozen=True)
class BalanceView:
    """Hold one aggregation view used by the fold-balance renderer."""

    overall_rows: list[dict[str, object]]
    summary_rows: list[dict[str, object]]
    distribution_tables: dict[tuple[str, str], list[dict[str, object]]]


@dataclass(frozen=True)
class FoldBalanceAnalysis:
    """Combine focused balance views with report metadata."""

    split: str
    feature: str
    folds_path: Path
    targets_path: Path
    targets: tuple[str, ...]
    roles: tuple[str, ...]
    speech: BalanceView
    author_weighted_speech: BalanceView
    author: BalanceView
    mixed_author_counts: dict[str, int]


def load_fold_inputs(
    project_root: Path,
    *,
    split: str,
    feature: str,
    targets: tuple[str, ...],
) -> FoldInputTables:
    """Load and validate fold memberships and target-feature rows."""

    folds_path = project_root / "data" / "splits" / split / "memberships" / "folds.csv"
    targets_path = (
        project_root
        / "data"
        / "splits"
        / split
        / "row_features"
        / feature
        / "targets.csv"
    )
    folds = pd.read_csv(folds_path)
    row_targets = pd.read_csv(targets_path)
    _require_columns(
        folds,
        folds_path,
        ["fold_id", "id_speech", "id_person", "fold_role"],
    )
    _require_columns(
        row_targets,
        targets_path,
        ["id_speech", "id_person", *targets],
    )
    return FoldInputTables(
        memberships=folds,
        targets=row_targets,
        folds_path=folds_path,
        targets_path=targets_path,
    )


def calculate_speech_level_balance(
    inputs: FoldInputTables,
    *,
    targets: tuple[str, ...],
    roles: tuple[str, ...],
) -> BalanceView:
    """Calculate ordinary speech-level fold balance."""

    speech_targets = inputs.targets[["id_speech", "id_person", *targets]].copy()
    speech_folded = _merge_complete_targets(
        inputs.memberships[["fold_id", "id_speech", "id_person", "fold_role"]],
        speech_targets,
        on=["id_speech", "id_person"],
        context="Speech-level fold balance",
    )
    summary_rows, distribution_tables = _balance_summary(
        speech_folded,
        speech_targets,
        list(targets),
        unit_col="id_speech",
        scope="speech",
        roles=list(roles),
    )
    return BalanceView(
        overall_rows=_distribution_rows(
            speech_targets,
            list(targets),
            scope="speech",
        ),
        summary_rows=summary_rows,
        distribution_tables=distribution_tables,
    )


def calculate_author_weighted_balance(
    inputs: FoldInputTables,
    *,
    targets: tuple[str, ...],
    roles: tuple[str, ...],
) -> BalanceView:
    """Calculate speech-level balance with equal total weight per author."""

    speech_targets = inputs.targets[["id_speech", "id_person", *targets]].copy()
    speech_folded = _merge_complete_targets(
        inputs.memberships[["fold_id", "id_speech", "id_person", "fold_role"]],
        speech_targets,
        on=["id_speech", "id_person"],
        context="Author-weighted speech-level fold balance",
    )
    weighted_targets = _with_inverse_author_weights(speech_targets, group_cols=[])
    weighted_folded = _with_inverse_author_weights(
        speech_folded,
        group_cols=["fold_id", "fold_role"],
    )
    summary_rows, distribution_tables = _balance_summary(
        weighted_folded,
        weighted_targets,
        list(targets),
        unit_col="id_person",
        scope="author_weighted_speech",
        roles=list(roles),
        weight_col="author_weight",
    )
    return BalanceView(
        overall_rows=_weighted_distribution_rows(
            weighted_targets,
            list(targets),
            scope="author_weighted_speech",
            weight_col="author_weight",
        ),
        summary_rows=summary_rows,
        distribution_tables=distribution_tables,
    )


def calculate_author_level_balance(
    inputs: FoldInputTables,
    *,
    targets: tuple[str, ...],
    roles: tuple[str, ...],
) -> tuple[BalanceView, dict[str, int]]:
    """Calculate modal-label author-level fold balance."""

    author_labels = inputs.targets[["id_person"]].drop_duplicates().copy()
    mixed_author_counts: dict[str, int] = {}
    for target in targets:
        target_modes, mixed_count = _mode_per_author(inputs.targets, target)
        mixed_author_counts[target] = mixed_count
        author_labels = author_labels.merge(target_modes, on="id_person", how="left")

    author_folded = _merge_complete_targets(
        inputs.memberships[
            ["fold_id", "id_person", "fold_role"]
        ].drop_duplicates(),
        author_labels,
        on=["id_person"],
        context="Author-level fold balance",
    )
    summary_rows, distribution_tables = _balance_summary(
        author_folded,
        author_labels,
        list(targets),
        unit_col="id_person",
        scope="author",
        roles=list(roles),
    )
    return (
        BalanceView(
            overall_rows=_distribution_rows(
                author_labels,
                list(targets),
                scope="author",
            ),
            summary_rows=summary_rows,
            distribution_tables=distribution_tables,
        ),
        mixed_author_counts,
    )


def analyze_fold_balance(
    inputs: FoldInputTables,
    *,
    split: str,
    feature: str,
    targets: tuple[str, ...],
    roles: tuple[str, ...],
) -> FoldBalanceAnalysis:
    """Build all stable analysis views consumed by the Markdown renderer."""

    author_view, mixed_author_counts = calculate_author_level_balance(
        inputs,
        targets=targets,
        roles=roles,
    )
    return FoldBalanceAnalysis(
        split=split,
        feature=feature,
        folds_path=inputs.folds_path,
        targets_path=inputs.targets_path,
        targets=targets,
        roles=roles,
        speech=calculate_speech_level_balance(
            inputs,
            targets=targets,
            roles=roles,
        ),
        author_weighted_speech=calculate_author_weighted_balance(
            inputs,
            targets=targets,
            roles=roles,
        ),
        author=author_view,
        mixed_author_counts=mixed_author_counts,
    )


def _distribution_sections(
    title: str,
    tables: dict[tuple[str, str], list[dict[str, object]]],
) -> list[str]:
    """Render all target/role distribution tables for one aggregation view."""

    lines = [f"## {title}", ""]
    for (_, key), rows in tables.items():
        if not rows:
            continue
        target, role = key.split(":", 1)
        columns = ["fold_id", *[column for column in rows[0] if column != "fold_id"]]
        lines.extend([f"### {target} ({role})", "", _markdown_table(rows, columns), ""])
    return lines


def render_fold_balance_markdown(
    analysis: FoldBalanceAnalysis,
    *,
    project_root: Path,
) -> str:
    """Render a completed fold-balance analysis as a Markdown document."""

    combined_summaries = [
        *analysis.speech.summary_rows,
        *analysis.author.summary_rows,
    ]
    summary_columns = [
        "scope",
        "role",
        "target",
        "fold_id",
        "n_speech_units",
        "n_author_units",
        "majority_label",
        "majority_pct",
        "max_abs_pp_deviation",
        "most_overrepresented_label",
        "most_overrepresented_pp",
        "most_underrepresented_label",
        "most_underrepresented_pp",
    ]
    summary_columns = [
        column
        for column in summary_columns
        if any(column in row for row in combined_summaries)
    ]
    weighted_summary_columns = [
        "scope",
        "role",
        "target",
        "fold_id",
        "effective_author_count",
        "n_authors",
        "majority_label",
        "majority_pct",
        "max_abs_pp_deviation",
        "most_overrepresented_label",
        "most_overrepresented_pp",
        "most_underrepresented_label",
        "most_underrepresented_pp",
    ]

    lines = [
        f"# Fold Target Balance Report: {analysis.split}",
        "",
        "## Inputs",
        "",
        _markdown_table(
            [
                {"field": "split", "value": analysis.split},
                {"field": "feature", "value": analysis.feature},
                {
                    "field": "folds",
                    "value": relative_to_project(project_root, analysis.folds_path),
                },
                {
                    "field": "targets",
                    "value": relative_to_project(project_root, analysis.targets_path),
                },
                {
                    "field": "reported_targets",
                    "value": ", ".join(analysis.targets),
                },
                {
                    "field": "reported_fold_roles",
                    "value": ", ".join(analysis.roles),
                },
            ],
            ["field", "value"],
        ),
        "",
        "## Overall Speech-Level Distribution",
        "",
        _markdown_table(
            analysis.speech.overall_rows,
            ["scope", "target", "label", "count", "pct"],
        ),
        "",
        "## Overall Author-Weighted Speech-Level Distribution",
        "",
        (
            "Each speech is weighted by inverse author speech count, matching the "
            "profiling evaluator's author-weighted validation metrics."
        ),
        "",
        _markdown_table(
            analysis.author_weighted_speech.overall_rows,
            ["scope", "target", "label", "effective_author_count", "pct"],
        ),
        "",
        "## Overall Author-Level Distribution",
        "",
        "Author-level labels collapse each author to the modal row label for each target.",
        "",
        _markdown_table(
            analysis.author.overall_rows,
            ["scope", "target", "label", "count", "pct"],
        ),
        "",
        "## Authors With Multiple Row Labels",
        "",
        _markdown_table(
            [
                {"target": target, "authors_with_multiple_row_labels": count}
                for target, count in analysis.mixed_author_counts.items()
            ],
            ["target", "authors_with_multiple_row_labels"],
        ),
        "",
        "## Fold Balance Summary",
        "",
        "Deviations are percentage points versus the corresponding overall distribution.",
        "",
        _markdown_table(combined_summaries, summary_columns),
        "",
        "## Author-Weighted Speech-Level Fold Balance Summary",
        "",
        _markdown_table(
            analysis.author_weighted_speech.summary_rows,
            weighted_summary_columns,
        ),
        "",
    ]
    lines.extend(
        _distribution_sections(
            "Speech-Level Fold Distributions",
            analysis.speech.distribution_tables,
        )
    )
    lines.extend(
        _distribution_sections(
            "Author-Weighted Speech-Level Fold Distributions",
            analysis.author_weighted_speech.distribution_tables,
        )
    )
    lines.extend(
        _distribution_sections(
            "Author-Level Fold Distributions",
            analysis.author.distribution_tables,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def write_fold_balance_report(
    output_path: Path,
    analysis: FoldBalanceAnalysis,
    *,
    project_root: Path,
) -> None:
    """Write a rendered fold-balance report to its resolved output path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_fold_balance_markdown(analysis, project_root=project_root),
        encoding="utf-8",
    )

