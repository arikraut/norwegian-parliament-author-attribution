"""Temporal (election-based) split assignment and fold construction."""

from __future__ import annotations

import pandas as pd


def build_temporal_outer_membership(
    df: pd.DataFrame,
    split_name: str,
    train_elections: list[int],
    test_elections: list[int],
    author_col: str = "id_person",
    party_col: str = "party",
    n_test_speeches: int | None = None,
) -> pd.DataFrame:
    """Assign temporal outer train/test roles, applying the optional latest-N test cap."""
    sort_cols = [col for col in ["date", "time", "id_speech"] if col in df.columns]
    if not sort_cols:
        sort_cols = ["id_speech"]

    df_train = df[df["election"].isin(train_elections)].copy()
    df_test_all = df[df["election"].isin(test_elections)].copy()

    if n_test_speeches is not None:
        capped_test_frames = []
        for _, author_df in df_test_all.groupby(author_col, sort=True):
            ordered = author_df.sort_values(sort_cols)
            n = len(ordered)
            n_test_actual = min(n_test_speeches, n)
            capped_test_frames.append(ordered.iloc[n - n_test_actual :])
        df_test = (
            pd.concat(capped_test_frames, ignore_index=True)
            if capped_test_frames
            else df_test_all.iloc[:0]
        )
    else:
        df_test = df_test_all

    outer_membership = pd.concat(
        [
            df_train.assign(split_name=split_name, outer_role="train"),
            df_test.assign(split_name=split_name, outer_role="test"),
        ],
        ignore_index=True,
    )
    outer_cols = [
        col
        for col in [
            "id_speech",
            author_col,
            "election",
            party_col,
            "language",
            "split_name",
            "outer_role",
        ]
        if col in outer_membership.columns
    ]
    return outer_membership[outer_cols].copy()


def filter_authors_by_temporal_fold_coverage(
    df: pd.DataFrame,
    author_stats: pd.DataFrame,
    fold_defs: list[dict],
    min_chars_per_fold_election: int,
    author_col: str = "id_person",
) -> pd.DataFrame:
    """Filter author_stats to only authors present in every fold election.

    Used when folds.coverage_policy = 'filter_authors'. Ensures that no fold
    will be dropped later for incomplete coverage — at the cost of a smaller
    author universe.

    An author must have at least min_chars_per_fold_election characters in every
    election that appears as a train or val period across all fold definitions.
    Authors who fall below this threshold in any single election are removed.

    When temporal folds are disabled, this function is skipped and the selected
    author set is not checked against every fold election.
    """
    fold_elections: set[int] = set()
    for fold in fold_defs:
        fold_elections.update(int(e) for e in fold["train_elections"])
        fold_elections.update(int(e) for e in fold["val_elections"])

    if not fold_elections:
        return author_stats.copy()

    surviving = set(author_stats[author_col].dropna().unique())
    df_candidates = df[df[author_col].isin(surviving)]

    for election in sorted(fold_elections):
        chars_by_author = (
            df_candidates[df_candidates["election"] == election]
            .groupby(author_col)["char_count"]
            .sum()
        )
        passing = set(
            chars_by_author[chars_by_author >= min_chars_per_fold_election].index
        )
        failing = surviving - passing
        if failing:
            print(
                f"  fold election {election}: removing {len(failing)} author(s) "
                f"with < {min_chars_per_fold_election:,} chars"
            )
        surviving -= failing

    removed = len(author_stats) - len(surviving)
    print(
        f"filter_authors_by_temporal_fold_coverage: {len(surviving)} authors pass "
        f"({removed} removed across {len(fold_elections)} fold elections)"
    )
    return author_stats[author_stats[author_col].isin(surviving)].copy()


def build_temporal_fold_membership(
    df_sub: pd.DataFrame,
    fold_defs: list[dict] | None,
    split_name: str,
    author_col: str = "id_person",
    party_col: str = "party",
) -> pd.DataFrame:
    """Build temporal train/validation fold memberships from election definitions."""
    fold_membership_frames: list[pd.DataFrame] = []
    for fold_def in fold_defs or []:
        fold_train = df_sub[df_sub["election"].isin(fold_def["train_elections"])].copy()
        fold_val = df_sub[df_sub["election"].isin(fold_def["val_elections"])].copy()
        fold_membership_frames.append(
            pd.concat(
                [
                    fold_train.assign(
                        split_name=split_name,
                        fold_id=fold_def["fold_id"],
                        fold_role="train",
                    ),
                    fold_val.assign(
                        split_name=split_name,
                        fold_id=fold_def["fold_id"],
                        fold_role="val",
                    ),
                ],
                ignore_index=True,
            )
        )

    if fold_membership_frames:
        fold_membership = pd.concat(fold_membership_frames, ignore_index=True)
    else:
        fold_membership = pd.DataFrame(
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

    fold_cols = [
        col
        for col in [
            "fold_id",
            "id_speech",
            author_col,
            "election",
            party_col,
            "language",
            "split_name",
            "fold_role",
        ]
        if col in fold_membership.columns
    ]
    return fold_membership[fold_cols].copy()


def build_fold_definitions(
    train_elections: list[int],
    folds_cfg: dict | None,
) -> list[dict]:
    """Build fold train/val election pair definitions from folds_cfg.

    Returns an empty list when ``folds.mode = 'none'``. For ``mode = 'expanding'``,
    each fold adds one more election to the train window so later folds have more
    training data than earlier ones.
    """
    cfg = dict(folds_cfg or {})
    mode = str(cfg.get("mode", "none")).lower()
    if mode == "none":
        return []
    if mode != "expanding":
        raise ValueError(f"Unsupported fold mode: {mode}")

    source = str(cfg.get("source", "train_only")).lower()
    if source == "train_only":
        periods = list(train_elections)
    else:
        raise ValueError(
            f"Unsupported folds.source: {source!r}. Only 'train_only' is supported."
        )

    periods = list(dict.fromkeys(int(year) for year in periods))
    min_train_periods = int(cfg.get("min_train_periods", 2))
    if min_train_periods < 1:
        raise ValueError("folds.min_train_periods must be >= 1")
    if len(periods) <= min_train_periods:
        return []

    fold_defs = []
    for fold_idx in range(min_train_periods, len(periods)):
        fold_number = len(fold_defs) + 1
        fold_defs.append(
            {
                "fold_id": f"fold_{fold_number:02d}_val_{periods[fold_idx]}",
                "train_elections": periods[:fold_idx],
                "val_elections": [periods[fold_idx]],
                "source": source,
            }
        )
    return fold_defs
