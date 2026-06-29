"""Shared author support statistics for split creation and diagnostics."""

from __future__ import annotations

import pandas as pd

from data_pipeline.split.selection import build_author_base_stats


def build_author_stats_from_membership(
    df: pd.DataFrame,
    outer_membership: pd.DataFrame,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Compute per-author outer-role support from a split membership table."""
    base = build_author_base_stats(df, author_col=author_col)
    if outer_membership.empty:
        role_stats = pd.DataFrame({author_col: df[author_col].unique()})
    else:
        membership_with_chars = outer_membership.merge(
            df[["id_speech", author_col, "char_count"]],
            on=["id_speech", author_col],
            how="left",
            sort=False,
        )
        grouped = membership_with_chars.groupby(
            [author_col, "outer_role"], as_index=False
        ).agg(
            chars=("char_count", "sum"),
            speeches=("id_speech", "nunique"),
        )

        chars_wide = (
            grouped.pivot(index=author_col, columns="outer_role", values="chars")
            .rename(columns=lambda role: f"{role}_chars")
            .reset_index()
        )
        speeches_wide = (
            grouped.pivot(index=author_col, columns="outer_role", values="speeches")
            .rename(columns=lambda role: f"{role}_speeches")
            .reset_index()
        )
        role_stats = chars_wide.merge(speeches_wide, on=author_col, how="outer")

    author_stats = base.merge(role_stats, on=author_col, how="left")

    for col in [
        "train_chars",
        "train_speeches",
        "test_chars",
        "test_speeches",
        "total_chars_all",
        "total_speeches_all",
    ]:
        if col not in author_stats.columns:
            author_stats[col] = 0
        author_stats[col] = author_stats[col].fillna(0).astype(int)

    return author_stats


def build_author_fold_stats(
    df: pd.DataFrame,
    fold_membership: pd.DataFrame,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Compute per-author support for each train/validation fold role."""
    columns = [
        author_col,
        "fold_id",
        "train_chars",
        "train_speeches",
        "val_chars",
        "val_speeches",
    ]
    if fold_membership.empty:
        return pd.DataFrame(columns=columns)

    membership_with_chars = fold_membership.merge(
        df[["id_speech", author_col, "char_count"]],
        on=["id_speech", author_col],
        how="left",
        sort=False,
    )
    grouped = membership_with_chars.groupby(
        [author_col, "fold_id", "fold_role"], as_index=False
    ).agg(
        chars=("char_count", "sum"),
        speeches=("id_speech", "nunique"),
    )

    chars_wide = (
        grouped.pivot(
            index=[author_col, "fold_id"], columns="fold_role", values="chars"
        )
        .rename(columns=lambda role: f"{role}_chars")
        .reset_index()
    )
    speeches_wide = (
        grouped.pivot(
            index=[author_col, "fold_id"],
            columns="fold_role",
            values="speeches",
        )
        .rename(columns=lambda role: f"{role}_speeches")
        .reset_index()
    )
    fold_stats = chars_wide.merge(
        speeches_wide, on=[author_col, "fold_id"], how="outer"
    )

    for col in ["train_chars", "train_speeches", "val_chars", "val_speeches"]:
        if col not in fold_stats.columns:
            fold_stats[col] = 0
        fold_stats[col] = fold_stats[col].fillna(0).astype(int)

    return fold_stats[columns].copy()
