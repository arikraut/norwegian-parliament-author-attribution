"""Author-grouped split helpers for profiling datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _membership_columns(df: pd.DataFrame, author_col: str, party_col: str) -> list[str]:
    """Choose the speech metadata columns retained in profiling memberships."""
    return [
        col
        for col in ["id_speech", author_col, "election", party_col, "language"]
        if col in df.columns
    ]


def _author_table(
    authors: pd.DataFrame,
    *,
    author_col: str,
    stratify_col: str | None,
) -> pd.DataFrame:
    """Return one sorted row per selected profiling author."""
    cols = [author_col]
    if stratify_col and stratify_col in authors.columns:
        cols.append(stratify_col)
    table = authors[cols].drop_duplicates(subset=[author_col]).sort_values(author_col)
    return table.reset_index(drop=True)


def _membership_from_author_roles(
    df: pd.DataFrame,
    roles_by_author: dict,
    *,
    split_name: str,
    role_col: str,
    author_col: str,
    party_col: str,
    fold_id: str | None = None,
) -> pd.DataFrame:
    """Expand author-level role assignments to speech-level membership rows."""
    base_cols = _membership_columns(df, author_col, party_col)
    membership = df[df[author_col].isin(set(roles_by_author))][base_cols].copy()
    membership["split_name"] = split_name
    if fold_id is not None:
        membership["fold_id"] = fold_id
    membership[role_col] = membership[author_col].map(roles_by_author)
    ordered_cols = ["fold_id"] if fold_id is not None else []
    ordered_cols.extend([*base_cols, "split_name", role_col])
    return membership[ordered_cols].copy()


def build_grouped_author_outer_membership(
    df: pd.DataFrame,
    authors: pd.DataFrame,
    split_name: str,
    *,
    author_col: str = "id_person",
    party_col: str = "party",
) -> pd.DataFrame:
    """Assign every selected profiling author to the outer train role."""
    author_table = _author_table(
        authors,
        author_col=author_col,
        stratify_col=None,
    )
    roles = {author_id: "train" for author_id in author_table[author_col].tolist()}
    return _membership_from_author_roles(
        df,
        roles,
        split_name=split_name,
        role_col="outer_role",
        author_col=author_col,
        party_col=party_col,
    )


def _author_buckets(
    authors: pd.DataFrame,
    n_buckets: int,
    *,
    seed: int,
    author_col: str,
    stratify_col: str,
) -> list[list]:
    """Assign profiling authors to balanced buckets within metadata strata."""
    if n_buckets < 2:
        raise ValueError("Profiling grouped folds require at least two folds.")
    if len(authors) < n_buckets:
        raise ValueError(
            f"Cannot create {n_buckets} author-grouped folds from {len(authors)} training authors."
        )

    rng = np.random.default_rng(seed)
    buckets: list[list] = [[] for _ in range(n_buckets)]

    for _, group in authors.groupby(stratify_col, dropna=False, sort=True):
        shuffled = rng.permutation(group[author_col].to_numpy()).tolist()
        for author_id in shuffled:
            target_idx = min(range(n_buckets), key=lambda idx: (len(buckets[idx]), idx))
            buckets[target_idx].append(author_id)
    return buckets


def build_grouped_author_fold_membership(
    df: pd.DataFrame,
    outer_membership: pd.DataFrame,
    folds_cfg: dict,
    split_name: str,
    *,
    seed: int,
    author_col: str = "id_person",
    party_col: str = "party",
) -> tuple[list[dict], pd.DataFrame]:
    """Build author-disjoint profiling folds from outer-train authors only."""
    mode = str(folds_cfg.get("mode", "none")).lower()
    empty = pd.DataFrame(
        columns=[
            "fold_id",
            "id_speech",
            author_col,
            "election",
            party_col,
            "language",
            "split_name",
            "fold_role",
        ]
    )
    if mode == "none":
        return [], empty
    if mode != "stratified_group_kfold":
        raise ValueError(
            "Profiling folds must use mode='stratified_group_kfold', "
            f"got {mode!r}."
        )

    train_author_ids = sorted(
        outer_membership.loc[
            outer_membership["outer_role"] == "train", author_col
        ]
        .dropna()
        .unique()
        .tolist()
    )
    train_authors = (
        df[df[author_col].isin(train_author_ids)]
        .drop_duplicates(subset=[author_col])
        .sort_values(author_col)
        .reset_index(drop=True)
    )
    stratify_col = str(folds_cfg.get("stratify_by", ""))
    if not stratify_col:
        raise ValueError("stratified_group_kfold requires folds.stratify_by.")
    if stratify_col not in train_authors.columns:
        raise KeyError(
            f"stratified_group_kfold requires a {stratify_col!r} column in the corpus."
        )
    n_splits = int(folds_cfg.get("n_splits", 5))
    buckets = _author_buckets(
        _author_table(
            train_authors,
            author_col=author_col,
            stratify_col=stratify_col,
        ),
        n_splits,
        seed=seed,
        author_col=author_col,
        stratify_col=stratify_col,
    )

    train_author_set = set(train_author_ids)
    fold_defs: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    for idx, val_author_ids in enumerate(buckets, start=1):
        fold_id = f"fold_{idx:02d}"
        val_author_set = set(val_author_ids)
        fold_train_ids = train_author_set - val_author_set
        if not fold_train_ids:
            raise ValueError(f"{fold_id} has no training authors after grouped fold assignment.")

        roles = {author_id: "train" for author_id in fold_train_ids}
        roles.update({author_id: "val" for author_id in val_author_set})
        fold_frames.append(
            _membership_from_author_roles(
                df,
                roles,
                split_name=split_name,
                role_col="fold_role",
                author_col=author_col,
                party_col=party_col,
                fold_id=fold_id,
            )
        )
        fold_defs.append(
            {
                "fold_id": fold_id,
                "mode": mode,
                "train_authors": int(len(fold_train_ids)),
                "val_authors": int(len(val_author_set)),
                "group_col": author_col,
                "stratify_by": stratify_col,
            }
        )

    fold_membership = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else empty
    )
    return fold_defs, fold_membership
