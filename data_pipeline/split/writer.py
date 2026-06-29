"""Split bundle writing and diagnostics."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipeline.split.imbalance import _compute_imbalance_stats
from data_pipeline.split.reliability import (
    _apply_reliability_policy,
    _build_author_level_reliability_summary,
    _build_fold_author_summary_from_support,
    _build_support_summary,
    _membership_support_columns,
    _resolve_reliability_thresholds,
)
from data_pipeline.split.reports import (
    _build_author_language_distribution,
    _build_author_party_distribution,
    _build_author_summary_frame,
    _compute_distribution,
    _compute_split_stats,
    _concat_nonempty,
    _write_split_summary_markdown,
)
from data_pipeline.split.stats import build_author_fold_stats
from data_pipeline.utils import relative_to_project, write_json


# ---------------------------------------------------------------------------
# Materialization (writes corpus + memberships to disk)
# ---------------------------------------------------------------------------


def _ensure_split_output_dirs(
    split_dir: Path, corpus_dir: Path, results_dir: Path
) -> Path:
    """Create the standard split output directory layout and return the memberships directory."""
    split_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    memberships_dir = split_dir / "memberships"
    memberships_dir.mkdir(parents=True, exist_ok=True)
    return memberships_dir


def _validate_outer_membership(
    df_sub: pd.DataFrame,
    outer_membership: pd.DataFrame,
    author_col: str,
    require_full_coverage: bool,
) -> None:
    """Raise if outer memberships contain duplicate speeches or (when required) miss selected speeches."""
    dup_mask = outer_membership["id_speech"].duplicated()
    if dup_mask.any():
        raise ValueError(
            f"outer_membership contains duplicated id_speech values ({int(dup_mask.sum())} duplicates)"
        )

    membership_subset = outer_membership[
        outer_membership[author_col].isin(df_sub[author_col].unique())
    ].copy()
    membership_speech_ids = set(membership_subset["id_speech"].unique())
    extra_in_outer = membership_speech_ids - set(df_sub["id_speech"].unique())
    if extra_in_outer:
        raise ValueError(
            "outer_membership contains speeches outside the selected author corpus "
            f"(extra={len(extra_in_outer)})"
        )

    if require_full_coverage:
        missing_from_outer = set(df_sub["id_speech"].unique()) - membership_speech_ids
        if missing_from_outer:
            raise ValueError(
                "outer_membership must cover each selected speech exactly once "
                f"(missing={len(missing_from_outer)}, extra={len(extra_in_outer)})"
            )


def _derive_outer_role_frames(
    df_sub: pd.DataFrame,
    outer_membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the selected corpus into (df_train, df_test) using outer membership role assignments."""
    roles = set(outer_membership["outer_role"].dropna().astype(str).unique())
    unsupported_roles = roles - {"train", "test"}
    if unsupported_roles:
        raise ValueError(
            "outer_membership only supports train/test outer roles. "
            f"Unsupported role(s): {sorted(unsupported_roles)}"
        )
    outer_role_map = outer_membership[["id_speech", "outer_role"]].copy()
    df_with_roles = df_sub.merge(outer_role_map, on="id_speech", how="left", sort=False)

    df_train = (
        df_with_roles[df_with_roles["outer_role"] == "train"]
        .drop(columns=["outer_role"])
        .copy()
    )
    df_test = (
        df_with_roles[df_with_roles["outer_role"] == "test"]
        .drop(columns=["outer_role"])
        .copy()
    )
    return df_train, df_test


def _filter_infeasible_fold_membership(
    fold_membership: pd.DataFrame | None,
    fold_defs: list[dict] | None,
    all_author_ids: set,
    author_col: str = "id_person",
    party_col: str = "party",
    require_author_coverage: bool = True,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Drop fold memberships that lack required train/validation role coverage."""
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
    if fold_membership is None or fold_membership.empty:
        return [], [], empty

    fold_defs_by_id = {
        str(fd["fold_id"]): fd for fd in (fold_defs or []) if "fold_id" in fd
    }
    fold_ids: list[str] = [
        str(fd["fold_id"]) for fd in (fold_defs or []) if "fold_id" in fd
    ]
    extra_fold_ids = sorted(
        set(fold_membership["fold_id"].dropna().astype(str).unique()) - set(fold_ids)
    )
    fold_ids.extend(extra_fold_ids)

    kept_fold_defs: list[dict] = []
    dropped_fold_records: list[dict] = []
    kept_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        fold_frame = fold_membership[
            fold_membership["fold_id"].astype(str) == fold_id
        ].copy()
        if fold_frame.empty:
            continue

        fold_train = fold_frame[fold_frame["fold_role"] == "train"]
        fold_val = fold_frame[fold_frame["fold_role"] == "val"]

        missing_from_train = (
            all_author_ids - set(fold_train[author_col].unique())
            if require_author_coverage
            else set()
        )
        missing_from_val = (
            all_author_ids - set(fold_val[author_col].unique())
            if require_author_coverage
            else set()
        )
        empty_role = fold_train.empty or fold_val.empty

        if empty_role or missing_from_train or missing_from_val:
            dropped_fold_records.append(
                {
                    "fold_id": fold_id,
                    "reason": "incomplete_author_coverage",
                    "n_missing_from_train": len(missing_from_train),
                    "n_missing_from_val": len(missing_from_val),
                    "missing_from_train": sorted(int(a) for a in missing_from_train),
                    "missing_from_val": sorted(int(a) for a in missing_from_val),
                }
            )
            print(
                f"  Dropping {fold_id}: {len(missing_from_train)} author(s) missing from train, "
                f"{len(missing_from_val)} missing from val."
            )
            continue

        kept_frames.append(fold_frame)
        kept_fold_defs.append(fold_defs_by_id.get(fold_id, {"fold_id": fold_id}))

    if kept_frames:
        fold_membership_selected = pd.concat(kept_frames, ignore_index=True)
    else:
        fold_membership_selected = empty

    return kept_fold_defs, dropped_fold_records, fold_membership_selected


def _write_split_bundle(
    df: pd.DataFrame,
    authors_subset: pd.DataFrame,
    author_stats_full: pd.DataFrame,
    split_name: str,
    experiment_name: str,
    project_root: Path,
    split_dir: Path,
    corpus_dir: Path,
    results_dir: Path,
    config_path: Path,
    source_dataset_path: Path,
    selection_seed: int,
    outer_membership: pd.DataFrame,
    fold_membership: pd.DataFrame | None,
    fold_defs: list[dict] | None,
    train_elections: list[int],
    test_elections: list[int],
    party_col: str = "party",
    author_col: str = "id_person",
    selected_corpus: pd.DataFrame | None = None,
    require_outer_membership_coverage: bool = False,
    require_fold_author_coverage: bool = True,
    split_strategy: str | None = None,
    strategy_config: dict | None = None,
    authors_meta_sort_ascending: list[bool] | tuple[bool, ...] | None = None,
) -> dict:
    """Write corpus, memberships, label arrays, diagnostics, and manifest for one complete split.

    Shared implementation used by the public membership writer.
    Caller is responsible for building outer_membership and fold_membership before calling.
    Returns the split manifest dict.
    """
    memberships_dir = _ensure_split_output_dirs(split_dir, corpus_dir, results_dir)
    if authors_subset.empty:
        raise ValueError("No authors were selected for this split.")
    author_ids = set(authors_subset[author_col].unique())
    if selected_corpus is None:
        df_sub = df[df[author_col].isin(author_ids)].copy()
    else:
        df_sub = selected_corpus[selected_corpus[author_col].isin(author_ids)].copy()
    reliability_thresholds = _resolve_reliability_thresholds(
        (strategy_config or {}).get("folds")
    )

    outer_membership = outer_membership[
        outer_membership[author_col].isin(author_ids)
    ].copy()
    _validate_outer_membership(
        df_sub,
        outer_membership,
        author_col=author_col,
        require_full_coverage=require_outer_membership_coverage,
    )
    df_train, df_test = _derive_outer_role_frames(df_sub, outer_membership)

    df_sub.to_csv(corpus_dir / "all.csv", index=False)
    df_train.to_csv(corpus_dir / "train.csv", index=False)
    df_test.to_csv(corpus_dir / "test.csv", index=False)

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
    outer_membership[outer_cols].to_csv(memberships_dir / "outer.csv", index=False)

    if fold_membership is not None:
        fold_membership = fold_membership[
            fold_membership[author_col].isin(author_ids)
        ].copy()
    kept_fold_defs, dropped_fold_records, fold_membership_selected = (
        _filter_infeasible_fold_membership(
            fold_membership,
            fold_defs,
            all_author_ids=author_ids,
            author_col=author_col,
            party_col=party_col,
            require_author_coverage=require_fold_author_coverage,
        )
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
        if col in fold_membership_selected.columns
    ]
    fold_membership_selected[fold_cols].to_csv(
        memberships_dir / "folds.csv", index=False
    )
    outer_support_summary = _build_support_summary(
        df_sub,
        outer_membership,
        scope="outer",
        role_col="outer_role",
        author_col=author_col,
    )
    fold_support_summary = _build_support_summary(
        df_sub,
        fold_membership_selected,
        scope="fold",
        role_col="fold_role",
        author_col=author_col,
    )
    per_author_support_summary = (
        pd.concat(
            [
                frame
                for frame in [outer_support_summary, fold_support_summary]
                if not frame.empty
            ],
            ignore_index=True,
        )
        if (not outer_support_summary.empty or not fold_support_summary.empty)
        else pd.DataFrame(columns=_membership_support_columns(author_col))
    )
    per_author_support_summary, reliability_policy_summary = _apply_reliability_policy(
        per_author_support_summary,
        reliability_thresholds,
        author_col=author_col,
    )
    per_author_support_summary.to_csv(
        results_dir / "per_author_support_summary.csv", index=False
    )
    author_level_reliability_summary = _build_author_level_reliability_summary(
        per_author_support_summary,
        author_col=author_col,
    )
    author_level_reliability_summary.to_csv(
        results_dir / "author_level_reliability_summary.csv", index=False
    )

    author_fold_stats_df: pd.DataFrame | None = None
    fold_author_summary = pd.DataFrame()
    if not fold_membership_selected.empty:
        author_fold_stats = build_author_fold_stats(
            df_sub, fold_membership_selected, author_col=author_col
        )
        fold_train_support = per_author_support_summary[
            (per_author_support_summary["support_scope"] == "fold")
            & (per_author_support_summary["role"] == "train")
        ][
            [
                author_col,
                "fold_id",
                "min_election",
                "max_election",
                "min_date",
                "max_date",
            ]
        ].rename(
            columns={
                "min_election": "train_min_election",
                "max_election": "train_max_election",
                "min_date": "train_min_date",
                "max_date": "train_max_date",
            }
        )
        fold_val_support = per_author_support_summary[
            (per_author_support_summary["support_scope"] == "fold")
            & (per_author_support_summary["role"] == "val")
        ][
            [
                author_col,
                "fold_id",
                "min_election",
                "max_election",
                "min_date",
                "max_date",
                "below_reliability_threshold",
                "author_level_reliability_status",
                "reliability_exclusion_reason",
            ]
        ].rename(
            columns={
                "min_election": "val_min_election",
                "max_election": "val_max_election",
                "min_date": "val_min_date",
                "max_date": "val_max_date",
                "below_reliability_threshold": "val_below_reliability_threshold",
                "author_level_reliability_status": "val_author_level_reliability_status",
                "reliability_exclusion_reason": "val_reliability_exclusion_reason",
            }
        )
        author_fold_stats = author_fold_stats.merge(
            fold_train_support, on=[author_col, "fold_id"], how="left"
        ).merge(fold_val_support, on=[author_col, "fold_id"], how="left")
        if "val_below_reliability_threshold" not in author_fold_stats.columns:
            author_fold_stats["val_below_reliability_threshold"] = False
        author_fold_stats["val_below_reliability_threshold"] = author_fold_stats[
            "val_below_reliability_threshold"
        ].map(lambda value: bool(value) if pd.notna(value) else False)
        if "val_author_level_reliability_status" not in author_fold_stats.columns:
            author_fold_stats["val_author_level_reliability_status"] = "not_applicable"
        author_fold_stats["val_author_level_reliability_status"] = author_fold_stats[
            "val_author_level_reliability_status"
        ].fillna("not_applicable")
        if "val_reliability_exclusion_reason" not in author_fold_stats.columns:
            author_fold_stats["val_reliability_exclusion_reason"] = ""
        author_fold_stats["val_reliability_exclusion_reason"] = author_fold_stats[
            "val_reliability_exclusion_reason"
        ].fillna("")
        author_fold_stats.to_csv(results_dir / "author_fold_stats.csv", index=False)
        author_fold_stats_df = author_fold_stats

        fold_author_summary = _build_fold_author_summary_from_support(
            per_author_support_summary,
            author_col=author_col,
        )

    # authors_subset already carries the selected party label; drop the duplicate before merge
    # so authors.csv does not silently pick up competing party columns.
    author_stats_no_party = author_stats_full.drop(columns=[party_col], errors="ignore")
    sort_ascending = list(authors_meta_sort_ascending or [True, True, True])
    authors_meta = (
        authors_subset.merge(author_stats_no_party, on=author_col, how="left")
        .sort_values(
            [party_col, "rank_in_party", "selection_metric_value"],
            ascending=sort_ascending,
        )
        .reset_index(drop=True)
    )
    authors_meta.to_csv(split_dir / "authors.csv", index=False)
    outer_imbalance_stats, fold_imbalance_stats = _compute_imbalance_stats(
        authors_meta, author_fold_stats_df
    )

    split_defs = [("train", df_train), ("test", df_test), ("all", df_sub)]

    corpus_stats = pd.DataFrame(
        [_compute_split_stats(frame, label) for label, frame in split_defs]
    )
    corpus_stats.to_csv(results_dir / "corpus_stats.csv", index=False)

    party_distribution = _concat_nonempty(
        [_compute_distribution(frame, party_col, label) for label, frame in split_defs]
    )

    language_distribution = _concat_nonempty(
        [_compute_distribution(frame, "language", label) for label, frame in split_defs]
    )

    author_summary = _build_author_summary_frame(authors_meta, author_col=author_col)
    author_party_distribution = _build_author_party_distribution(
        authors_meta, party_col=party_col
    )
    author_language_distribution = _build_author_language_distribution(authors_meta)

    shutil.copy2(config_path, split_dir / "split_config.toml")

    summary_path = results_dir / "split_summary.md"
    manifest = {
        "experiment_name": experiment_name,
        "split_name": split_name,
        "fold_count": int(len(kept_fold_defs)),
        "fold_ids": [fd["fold_id"] for fd in kept_fold_defs],
        "config_path": relative_to_project(project_root, config_path),
        "source_dataset": relative_to_project(project_root, source_dataset_path),
        "selection_seed": selection_seed,
        "train_elections": train_elections,
        "test_elections": test_elections,
        "folds": kept_fold_defs,
        "dropped_folds": dropped_fold_records,
        "counts": {
            "selected_authors": int(authors_meta[author_col].nunique()),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "outer_membership_rows": int(len(outer_membership)),
            "fold_membership_rows": int(len(fold_membership_selected)),
        },
        "paths": {
            "split_dir": relative_to_project(project_root, split_dir),
            "corpus_dir": relative_to_project(project_root, corpus_dir),
            "memberships_dir": relative_to_project(project_root, memberships_dir),
            "results_dir": relative_to_project(project_root, results_dir),
            "summary_markdown": relative_to_project(project_root, summary_path),
            "per_author_support_summary": relative_to_project(
                project_root, results_dir / "per_author_support_summary.csv"
            ),
            "author_level_reliability_summary": relative_to_project(
                project_root, results_dir / "author_level_reliability_summary.csv"
            ),
            "author_fold_stats": (
                relative_to_project(project_root, results_dir / "author_fold_stats.csv")
                if author_fold_stats_df is not None
                else ""
            ),
        },
        "support_policy": {
            "policy": reliability_policy_summary["policy"],
            "thresholds": reliability_policy_summary["thresholds"],
            "totals": reliability_policy_summary["totals"],
            "excluded_by_reason": reliability_policy_summary["excluded_by_reason"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if split_strategy is not None:
        manifest["split_strategy"] = split_strategy
    if strategy_config is not None:
        manifest["strategy_config"] = strategy_config
    write_json(split_dir / "manifest.json", manifest)

    _write_split_summary_markdown(
        summary_path,
        split_name=split_name,
        experiment_name=experiment_name,
        split_strategy=split_strategy,
        manifest=manifest,
        corpus_stats=corpus_stats,
        party_distribution=party_distribution,
        language_distribution=language_distribution,
        author_summary=author_summary,
        author_party_distribution=author_party_distribution,
        author_language_distribution=author_language_distribution,
        fold_author_summary=fold_author_summary,
        outer_imbalance_stats=outer_imbalance_stats,
        fold_imbalance_stats=fold_imbalance_stats,
    )

    print("Saved split data in", split_dir)
    print("Saved split diagnostics in", results_dir)
    print(f"  train: {len(df_train)} rows  |  test: {len(df_test)} rows")
    print(f"  authors: {authors_meta[author_col].nunique()}  (see authors.csv)")
    print(
        f"  memberships: outer.csv ({len(outer_membership)} rows), "
        f"folds.csv ({len(fold_membership_selected)} rows)"
    )
    print(
        f"  folds kept: {len(kept_fold_defs)}  {[fd['fold_id'] for fd in kept_fold_defs]}"
    )
    if dropped_fold_records:
        for dr in dropped_fold_records:
            print(
                f"  fold dropped: {dr['fold_id']}  "
                f"({dr['n_missing_from_train']} author(s) missing from train, "
                f"{dr['n_missing_from_val']} missing from val)"
            )

    return {
        "split_dir": split_dir,
        "corpus_dir": corpus_dir,
        "memberships_dir": memberships_dir,
        "results_dir": results_dir,
        "authors_path": split_dir / "authors.csv",
        "manifest_path": split_dir / "manifest.json",
        "summary_path": summary_path,
    }


def write_membership_split(
    df: pd.DataFrame,
    authors_subset: pd.DataFrame,
    author_stats_full: pd.DataFrame,
    split_name: str,
    experiment_name: str,
    project_root: Path,
    split_dir: Path,
    corpus_dir: Path,
    results_dir: Path,
    config_path: Path,
    source_dataset_path: Path,
    selection_seed: int,
    outer_membership: pd.DataFrame,
    fold_membership: pd.DataFrame | None,
    fold_defs: list[dict] | None,
    split_strategy: str,
    strategy_config: dict | None = None,
    require_fold_author_coverage: bool = True,
    selected_corpus: pd.DataFrame | None = None,
    authors_meta_sort_ascending: list[bool] | tuple[bool, ...] | None = None,
    party_col: str = "party",
    author_col: str = "id_person",
) -> dict:
    """Write a split bundle from explicit per-speech outer/fold memberships."""
    if "election" in outer_membership.columns:
        train_elections = sorted(
            outer_membership.loc[outer_membership["outer_role"] == "train", "election"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        test_elections = sorted(
            outer_membership.loc[outer_membership["outer_role"] == "test", "election"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    else:
        train_elections = []
        test_elections = []

    return _write_split_bundle(
        df=df,
        authors_subset=authors_subset,
        author_stats_full=author_stats_full,
        split_name=split_name,
        experiment_name=experiment_name,
        project_root=project_root,
        split_dir=split_dir,
        corpus_dir=corpus_dir,
        results_dir=results_dir,
        config_path=config_path,
        source_dataset_path=source_dataset_path,
        selection_seed=selection_seed,
        outer_membership=outer_membership,
        fold_membership=fold_membership,
        fold_defs=fold_defs,
        train_elections=train_elections,
        test_elections=test_elections,
        party_col=party_col,
        author_col=author_col,
        selected_corpus=selected_corpus,
        require_outer_membership_coverage=True,
        require_fold_author_coverage=require_fold_author_coverage,
        split_strategy=split_strategy,
        strategy_config=strategy_config or {},
        authors_meta_sort_ascending=authors_meta_sort_ascending or [True, True, False],
    )
