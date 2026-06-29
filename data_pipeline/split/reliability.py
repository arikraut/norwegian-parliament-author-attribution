"""Split support and reliability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_reliability_thresholds(folds_cfg: dict | None) -> dict[str, int]:
    """Read reliability thresholds from the fold config with zero defaults."""
    cfg = dict(folds_cfg or {})
    return {
        "min_val_chars_per_author_fold_cell": int(
            cfg.get("min_val_chars_per_author_fold_cell", 0) or 0
        ),
        "min_val_speeches_per_author_fold_cell": int(
            cfg.get("min_val_speeches_per_author_fold_cell", 0) or 0
        ),
    }


def _membership_support_columns(author_col: str) -> list[str]:
    """Return the stable column order for membership support diagnostics."""
    return [
        author_col,
        "name",
        "party",
        "support_scope",
        "role",
        "outer_role",
        "fold_id",
        "speech_count",
        "char_count",
        "min_election",
        "max_election",
        "min_date",
        "max_date",
        "below_reliability_threshold",
        "author_level_reliability_status",
        "reliability_exclusion_reason",
    ]


def _build_support_summary(
    df_sub: pd.DataFrame,
    membership: pd.DataFrame | None,
    *,
    scope: str,
    role_col: str,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Aggregate speech memberships into per-author role support rows."""
    columns = _membership_support_columns(author_col)
    if membership is None or membership.empty:
        return pd.DataFrame(columns=columns)

    merge_cols = [
        col
        for col in [
            "id_speech",
            author_col,
            "name",
            "party",
            "char_count",
            "election",
            "date",
        ]
        if col in df_sub.columns
    ]
    membership_cols = [
        col
        for col in ["id_speech", author_col, role_col, "fold_id"]
        if col in membership.columns
    ]
    membership_with_meta = membership[membership_cols].merge(
        df_sub[merge_cols],
        on=["id_speech", author_col],
        how="left",
        sort=False,
    )
    if "name" not in membership_with_meta.columns:
        membership_with_meta["name"] = ""
    if "party" not in membership_with_meta.columns:
        membership_with_meta["party"] = ""
    if "char_count" not in membership_with_meta.columns:
        membership_with_meta["char_count"] = 0
    if "election" not in membership_with_meta.columns:
        membership_with_meta["election"] = pd.NA
    if "date" not in membership_with_meta.columns:
        membership_with_meta["date"] = pd.NA
    membership_with_meta["_date_parsed"] = pd.to_datetime(
        membership_with_meta["date"], errors="coerce"
    )

    group_cols = [author_col, role_col]
    if scope == "fold":
        group_cols.insert(1, "fold_id")

    summary = membership_with_meta.groupby(group_cols, as_index=False).agg(
        name=("name", "first"),
        party=("party", "first"),
        speech_count=("id_speech", "nunique"),
        char_count=("char_count", "sum"),
        min_election=("election", "min"),
        max_election=("election", "max"),
        min_date=("_date_parsed", "min"),
        max_date=("_date_parsed", "max"),
    )
    summary["support_scope"] = scope
    summary["role"] = summary[role_col].astype(str)
    if scope == "outer":
        summary["outer_role"] = summary[role_col].astype(str)
        summary["fold_id"] = ""
    else:
        summary["outer_role"] = ""
    summary["min_election"] = summary["min_election"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    summary["max_election"] = summary["max_election"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    summary["min_date"] = summary["min_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
    )
    summary["max_date"] = summary["max_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
    )
    summary["below_reliability_threshold"] = False
    summary["author_level_reliability_status"] = "not_applicable"
    summary["reliability_exclusion_reason"] = ""
    return summary[columns].copy()


def _reliability_reason(*, below_chars: bool, below_speeches: bool) -> str:
    """Encode which validation-support threshold a fold cell missed."""
    if below_chars and below_speeches:
        return "val_chars_and_val_speeches_below_threshold"
    if below_chars:
        return "val_chars_below_threshold"
    if below_speeches:
        return "val_speeches_below_threshold"
    return ""


def _apply_reliability_policy(
    support_summary: pd.DataFrame,
    thresholds: dict[str, int],
    *,
    author_col: str = "id_person",
) -> tuple[pd.DataFrame, dict]:
    """Flag unreliable validation cells without changing split membership."""
    if support_summary.empty:
        return support_summary.copy(), {
            "policy": "keep_membership_flag_below_threshold_cells_exclude_from_author_level_reliability",
            "thresholds": thresholds,
            "totals": {
                "total_val_cells": 0,
                "excluded_val_cells": 0,
                "authors_with_any_excluded_cells": 0,
                "authors_fully_excluded_from_reliability_summary": 0,
                "authors_with_reliable_val_cells": 0,
            },
            "excluded_by_reason": {},
        }

    summary = support_summary.copy()
    val_mask = (summary["support_scope"] == "fold") & (summary["role"] == "val")
    summary.loc[val_mask, "author_level_reliability_status"] = "included"

    min_chars = int(thresholds.get("min_val_chars_per_author_fold_cell", 0) or 0)
    min_speeches = int(thresholds.get("min_val_speeches_per_author_fold_cell", 0) or 0)

    below_chars = (
        val_mask & (summary["char_count"] < min_chars)
        if min_chars > 0
        else pd.Series(False, index=summary.index)
    )
    below_speeches = (
        val_mask & (summary["speech_count"] < min_speeches)
        if min_speeches > 0
        else pd.Series(False, index=summary.index)
    )
    excluded_mask = below_chars | below_speeches

    summary.loc[excluded_mask, "below_reliability_threshold"] = True
    summary.loc[excluded_mask, "author_level_reliability_status"] = "excluded"
    summary.loc[excluded_mask, "reliability_exclusion_reason"] = [
        _reliability_reason(
            below_chars=bool(chars_flag), below_speeches=bool(speech_flag)
        )
        for chars_flag, speech_flag in zip(
            below_chars[excluded_mask], below_speeches[excluded_mask]
        )
    ]

    val_cells = summary[val_mask].copy()
    excluded_cells = summary[excluded_mask].copy()
    reliable_cells = val_cells[
        val_cells["author_level_reliability_status"] == "included"
    ].copy()

    excluded_by_reason: dict[str, dict[str, int]] = {}
    if not excluded_cells.empty:
        for reason, group in excluded_cells.groupby(
            "reliability_exclusion_reason", sort=True
        ):
            excluded_by_reason[str(reason)] = {
                "excluded_cells": int(len(group)),
                "excluded_authors": int(group[author_col].nunique()),
            }

    totals = {
        "total_val_cells": int(len(val_cells)),
        "excluded_val_cells": int(len(excluded_cells)),
        "authors_with_any_excluded_cells": (
            int(excluded_cells[author_col].nunique()) if not excluded_cells.empty else 0
        ),
        "authors_fully_excluded_from_reliability_summary": 0,
        "authors_with_reliable_val_cells": (
            int(reliable_cells[author_col].nunique()) if not reliable_cells.empty else 0
        ),
    }
    if not val_cells.empty:
        excluded_counts = (
            excluded_cells.groupby(author_col).size()
            if not excluded_cells.empty
            else pd.Series(dtype=int)
        )
        reliable_counts = (
            reliable_cells.groupby(author_col).size()
            if not reliable_cells.empty
            else pd.Series(dtype=int)
        )
        fully_excluded = 0
        for author_id in val_cells[author_col].dropna().unique():
            if (
                int(excluded_counts.get(author_id, 0)) > 0
                and int(reliable_counts.get(author_id, 0)) == 0
            ):
                fully_excluded += 1
        totals["authors_fully_excluded_from_reliability_summary"] = int(fully_excluded)

    return summary, {
        "policy": "keep_membership_flag_below_threshold_cells_exclude_from_author_level_reliability",
        "thresholds": thresholds,
        "totals": totals,
        "excluded_by_reason": excluded_by_reason,
    }


def _build_author_level_reliability_summary(
    support_summary: pd.DataFrame,
    *,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Summarize reliable and excluded validation cells per author."""
    base = (
        support_summary[[author_col, "name", "party"]]
        .drop_duplicates(subset=[author_col])
        .sort_values(author_col)
        .reset_index(drop=True)
    )
    if base.empty:
        return pd.DataFrame(
            columns=[
                author_col,
                "name",
                "party",
                "n_total_val_cells",
                "n_reliable_val_cells",
                "n_excluded_val_cells",
                "had_any_excluded_cells",
                "fully_excluded_from_author_level_reliability",
                "min_reliable_val_chars",
                "median_reliable_val_chars",
                "min_reliable_val_speeches",
                "median_reliable_val_speeches",
                "exclusion_reasons",
            ]
        )

    val_cells = support_summary[
        (support_summary["support_scope"] == "fold")
        & (support_summary["role"] == "val")
    ].copy()
    reliable_cells = val_cells[
        val_cells["author_level_reliability_status"] == "included"
    ].copy()
    excluded_cells = val_cells[
        val_cells["author_level_reliability_status"] == "excluded"
    ].copy()

    summary = base.copy()
    if not val_cells.empty:
        total_counts = val_cells.groupby(author_col).size().rename("n_total_val_cells")
        summary = summary.merge(total_counts, on=author_col, how="left")
    else:
        summary["n_total_val_cells"] = 0

    if not reliable_cells.empty:
        reliable_summary = reliable_cells.groupby(author_col, as_index=False).agg(
            n_reliable_val_cells=("fold_id", "nunique"),
            min_reliable_val_chars=("char_count", "min"),
            median_reliable_val_chars=("char_count", "median"),
            min_reliable_val_speeches=("speech_count", "min"),
            median_reliable_val_speeches=("speech_count", "median"),
        )
        summary = summary.merge(reliable_summary, on=author_col, how="left")
    else:
        summary["n_reliable_val_cells"] = 0
        summary["min_reliable_val_chars"] = np.nan
        summary["median_reliable_val_chars"] = np.nan
        summary["min_reliable_val_speeches"] = np.nan
        summary["median_reliable_val_speeches"] = np.nan

    if not excluded_cells.empty:
        excluded_summary = excluded_cells.groupby(author_col, as_index=False).agg(
            n_excluded_val_cells=("fold_id", "nunique"),
            exclusion_reasons=(
                "reliability_exclusion_reason",
                lambda values: "|".join(sorted(set(values))),
            ),
        )
        summary = summary.merge(excluded_summary, on=author_col, how="left")
    else:
        summary["n_excluded_val_cells"] = 0
        summary["exclusion_reasons"] = ""

    for col in ["n_total_val_cells", "n_reliable_val_cells", "n_excluded_val_cells"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["exclusion_reasons"] = summary["exclusion_reasons"].fillna("")

    summary["had_any_excluded_cells"] = summary["n_excluded_val_cells"] > 0
    summary["fully_excluded_from_author_level_reliability"] = (
        (summary["n_total_val_cells"] > 0)
        & (summary["n_reliable_val_cells"] == 0)
        & (summary["n_excluded_val_cells"] > 0)
    )
    return summary.sort_values(author_col).reset_index(drop=True)


def _build_fold_author_summary_from_support(
    support_summary: pd.DataFrame,
    *,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Build per-fold train/validation support summaries from membership diagnostics."""
    fold_rows = support_summary[support_summary["support_scope"] == "fold"].copy()
    if fold_rows.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "n_authors",
                "min_train_chars",
                "median_train_chars",
                "min_val_chars",
                "median_val_chars",
                "min_train_speeches",
                "median_train_speeches",
                "min_val_speeches",
                "median_val_speeches",
                "train_min_election",
                "train_max_election",
                "val_min_election",
                "val_max_election",
                "train_min_date",
                "train_max_date",
                "val_min_date",
                "val_max_date",
                "excluded_val_cells",
                "excluded_val_authors",
            ]
        )

    rows: list[dict] = []
    for fold_id, fold_group in fold_rows.groupby("fold_id", sort=True):
        train_group = fold_group[fold_group["role"] == "train"].copy()
        val_group = fold_group[fold_group["role"] == "val"].copy()
        excluded_val = val_group[
            val_group["author_level_reliability_status"] == "excluded"
        ].copy()

        rows.append(
            {
                "fold_id": fold_id,
                "n_authors": int(fold_group[author_col].nunique()),
                "min_train_chars": (
                    int(train_group["char_count"].min()) if not train_group.empty else 0
                ),
                "median_train_chars": (
                    float(train_group["char_count"].median())
                    if not train_group.empty
                    else 0.0
                ),
                "min_val_chars": (
                    int(val_group["char_count"].min()) if not val_group.empty else 0
                ),
                "median_val_chars": (
                    float(val_group["char_count"].median())
                    if not val_group.empty
                    else 0.0
                ),
                "min_train_speeches": (
                    int(train_group["speech_count"].min())
                    if not train_group.empty
                    else 0
                ),
                "median_train_speeches": (
                    float(train_group["speech_count"].median())
                    if not train_group.empty
                    else 0.0
                ),
                "min_val_speeches": (
                    int(val_group["speech_count"].min()) if not val_group.empty else 0
                ),
                "median_val_speeches": (
                    float(val_group["speech_count"].median())
                    if not val_group.empty
                    else 0.0
                ),
                "train_min_election": (
                    int(train_group["min_election"].min())
                    if not train_group["min_election"].dropna().empty
                    else None
                ),
                "train_max_election": (
                    int(train_group["max_election"].max())
                    if not train_group["max_election"].dropna().empty
                    else None
                ),
                "val_min_election": (
                    int(val_group["min_election"].min())
                    if not val_group["min_election"].dropna().empty
                    else None
                ),
                "val_max_election": (
                    int(val_group["max_election"].max())
                    if not val_group["max_election"].dropna().empty
                    else None
                ),
                "train_min_date": (
                    str(train_group["min_date"].min())
                    if not train_group["min_date"].replace("", np.nan).dropna().empty
                    else ""
                ),
                "train_max_date": (
                    str(train_group["max_date"].max())
                    if not train_group["max_date"].replace("", np.nan).dropna().empty
                    else ""
                ),
                "val_min_date": (
                    str(val_group["min_date"].min())
                    if not val_group["min_date"].replace("", np.nan).dropna().empty
                    else ""
                ),
                "val_max_date": (
                    str(val_group["max_date"].max())
                    if not val_group["max_date"].replace("", np.nan).dropna().empty
                    else ""
                ),
                "excluded_val_cells": int(len(excluded_val)),
                "excluded_val_authors": (
                    int(excluded_val[author_col].nunique())
                    if not excluded_val.empty
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)
