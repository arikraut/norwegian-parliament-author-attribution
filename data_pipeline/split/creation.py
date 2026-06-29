"""Split creation entry points for scripts"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.split.author_disjointness import (
    assert_author_sets_disjoint,
    load_author_ids_from_csv,
)
from data_pipeline.split.authorwise import (
    build_authorwise_fold_membership,
    build_outer_membership_by_author,
    build_outer_membership_fixed_test,
    filter_authors_by_authorwise_fold_coverage,
)
from data_pipeline.split.context import load_split_run_context
from data_pipeline.split.profiling import (
    build_grouped_author_fold_membership,
    build_grouped_author_outer_membership,
)
from data_pipeline.split.selection import (
    apply_author_split_eligibility_filters,
    apply_temporal_test_eligibility_filters,
    apply_exclusion_filter,
    build_author_base_stats,
    select_authors,
)
from data_pipeline.split.stats import build_author_stats_from_membership
from data_pipeline.split.temporal import (
    build_fold_definitions,
    build_temporal_fold_membership,
    build_temporal_outer_membership,
    filter_authors_by_temporal_fold_coverage,
)
from data_pipeline.split.writer import write_membership_split
from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
)

__all__ = [
    "run_split_creation",
    "run_split_creation_from_context",
    "run_author_split_creation_from_context",
    "run_fixed_test_split_creation_from_context",
    "run_temporal_split_creation_from_context",
    "run_profiling_split_creation_from_context",
]


def _build_run_summary(
    context: dict,
    *,
    selected_authors: pd.DataFrame,
    fold_defs: list[dict] | None,
    artifacts: dict,
) -> dict:
    """Build the split-run manifest payload shared by all split strategies."""
    project_root = context["project_root"]
    return {
        "split_name": context["split_name"],
        "split_strategy": context["outer_split_strategy"],
        "n_authors_selected": int(len(selected_authors)),
        "n_folds": len(fold_defs) if fold_defs else 0,
        "split_dir": str(relative_to_project(project_root, context["split_dir"])),
        "results_dir": str(relative_to_project(project_root, context["results_dir"])),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }


def _load_context_for_run(
    config_path: Path,
) -> dict:
    """Load a split config and source corpus for a top-level split runner."""
    config_path = Path(config_path).resolve()
    project_root = find_project_root(config_path)
    return load_split_run_context(
        project_root,
        config_path=config_path,
    )


def _run_authorwise_pipeline(
    context: dict,
    outer_membership_full: pd.DataFrame,
    split_strategy: str,
) -> dict:
    """Run eligibility filtering, fold coverage filtering, author selection, and split writing.

    Shared by run_author_split_creation_from_context and run_fixed_test_split_creation_from_context.
    The two entry points differ only in how outer_membership_full is constructed before calling here.
    """
    df = context["df"]
    folds_cfg = context["folds_cfg"]
    split_name = context["split_name"]
    coverage_policy = str(folds_cfg.get("coverage_policy", "filter_authors")).lower()
    fold_mode = str(folds_cfg.get("mode", "none")).lower()
    eligibility_cfg = context["split_config"]["eligibility"]

    author_stats = build_author_stats_from_membership(df, outer_membership_full)
    author_stats_eligible = apply_author_split_eligibility_filters(
        author_stats, eligibility_cfg
    )
    print(f"Authors after eligibility filter: {len(author_stats_eligible)}")

    if fold_mode != "none" and coverage_policy != "filter_authors":
        raise ValueError(
            f"Unsupported folds.coverage_policy value: {coverage_policy!r}. "
            "Only 'filter_authors' is supported."
        )

    if fold_mode == "kfold":
        author_stats_eligible = filter_authors_by_authorwise_fold_coverage(
            df,
            outer_membership_full,
            author_stats_eligible,
            folds_cfg,
            split_name,
        )
        print(f"Authors after fold coverage filter: {len(author_stats_eligible)}")

    _, selected_authors = select_authors(
        author_stats_eligible,
        context["pool_cfg"],
        context["selection_cfg"],
        context["selection_seed"],
    )
    print(f"Authors selected: {len(selected_authors)}")

    selected_ids = selected_authors["id_person"].unique()
    selected_outer_membership = outer_membership_full[
        outer_membership_full["id_person"].isin(selected_ids)
    ].copy()
    selected_author_stats = author_stats_eligible[
        author_stats_eligible["id_person"].isin(selected_ids)
    ].copy()

    fold_defs, fold_membership = build_authorwise_fold_membership(
        selected_outer_membership,
        folds_cfg,
        split_name,
        df=df[df["id_person"].isin(selected_ids)].copy(),
        enforce_minima=True,
    )

    artifacts = write_membership_split(
        df=df,
        authors_subset=selected_authors,
        author_stats_full=selected_author_stats,
        split_name=split_name,
        experiment_name=context["experiment_name"],
        project_root=context["project_root"],
        split_dir=context["split_dir"],
        corpus_dir=context["corpus_dir"],
        results_dir=context["results_dir"],
        config_path=context["config_path"],
        source_dataset_path=context["source_dataset_path"],
        selection_seed=context["selection_seed"],
        outer_membership=selected_outer_membership,
        fold_membership=fold_membership,
        fold_defs=fold_defs,
        split_strategy=split_strategy,
        strategy_config={"outer_split": context["outer_split_cfg"], "folds": folds_cfg},
    )

    return _build_run_summary(
        context,
        selected_authors=selected_authors,
        fold_defs=fold_defs,
        artifacts=artifacts,
    )


def run_author_split_creation_from_context(context: dict) -> dict:
    """Run the author-wise (percentage-share) split pipeline from a prepared context."""
    df = context["df"]
    outer_split_cfg = context["outer_split_cfg"]
    print(f"Loaded corpus: {len(df):,} rows, {df['id_person'].nunique():,} authors")
    outer_membership_full = build_outer_membership_by_author(
        df, outer_split_cfg, context["split_name"]
    )
    return _run_authorwise_pipeline(context, outer_membership_full, "author_percentage")


def run_temporal_split_creation_from_context(context: dict) -> dict:
    """Run the full temporal split creation pipeline from a prepared context."""
    df = context["df"]
    outer_split_cfg = context["outer_split_cfg"]
    eligibility_cfg = context["split_config"]["eligibility"]
    pool_cfg = context["pool_cfg"]
    selection_cfg = context["selection_cfg"]
    folds_cfg = context["folds_cfg"]

    split_name = context["split_name"]
    experiment_name = context["experiment_name"]
    selection_seed = context["selection_seed"]

    split_dir = context["split_dir"]
    corpus_dir = context["corpus_dir"]
    results_dir = context["results_dir"]
    project_root = context["project_root"]
    config_path = context["config_path"]
    source_dataset_path = context["source_dataset_path"]

    train_elections = [int(year) for year in outer_split_cfg["train"]]
    test_elections = [int(year) for year in outer_split_cfg["test"]]
    n_test_speeches = outer_split_cfg.get("n_test_speeches")
    n_test_speeches = int(n_test_speeches) if n_test_speeches is not None else None

    fold_mode = str(folds_cfg.get("mode", "none")).lower()
    coverage_policy = str(folds_cfg.get("coverage_policy", "filter_authors")).lower()
    min_chars_per_fold_election = int(
        folds_cfg.get("min_chars_per_fold_election", 0) or 0
    )

    print(
        f"Loaded corpus: {len(df):,} rows, elections: {sorted(df['election'].unique())}"
    )

    outer_membership_full = build_temporal_outer_membership(
        df,
        split_name=split_name,
        train_elections=train_elections,
        test_elections=test_elections,
        n_test_speeches=n_test_speeches,
    )
    author_stats = build_author_stats_from_membership(df, outer_membership_full)
    author_stats_eligible = apply_author_split_eligibility_filters(
        author_stats, eligibility_cfg
    )
    print(f"Authors after eligibility filter: {len(author_stats_eligible)}")

    author_stats_eligible = apply_temporal_test_eligibility_filters(
        author_stats_eligible, eligibility_cfg
    )
    if "min_test_speeches_per_author" in eligibility_cfg:
        print(
            "Authors after temporal test support filter: "
            f"{len(author_stats_eligible)}"
        )

    fold_defs = build_fold_definitions(train_elections, folds_cfg)

    if fold_mode != "none":
        if coverage_policy != "filter_authors":
            raise ValueError(
                f"Unsupported folds.coverage_policy value: {coverage_policy!r}. "
                "Only 'filter_authors' is supported."
            )
        author_stats_eligible = filter_authors_by_temporal_fold_coverage(
            df,
            author_stats_eligible,
            fold_defs,
            min_chars_per_fold_election,
        )
        print(f"Authors after fold coverage filter: {len(author_stats_eligible)}")

    _, selected_authors = select_authors(
        author_stats_eligible,
        pool_cfg,
        selection_cfg,
        selection_seed,
    )
    print(f"Authors selected: {len(selected_authors)}")

    selected_ids = selected_authors["id_person"].unique()
    selected_author_stats = author_stats_eligible[
        author_stats_eligible["id_person"].isin(selected_ids)
    ].copy()
    selected_outer_membership = outer_membership_full[
        outer_membership_full["id_person"].isin(selected_ids)
    ].copy()
    covered_speech_ids = set(selected_outer_membership["id_speech"].unique())
    selected_corpus = df[
        df["id_person"].isin(selected_ids) & df["id_speech"].isin(covered_speech_ids)
    ].copy()
    fold_membership = build_temporal_fold_membership(
        selected_corpus,
        fold_defs,
        split_name,
    )

    artifacts = write_membership_split(
        df=df,
        authors_subset=selected_authors,
        author_stats_full=selected_author_stats,
        split_name=split_name,
        experiment_name=experiment_name,
        project_root=project_root,
        split_dir=split_dir,
        corpus_dir=corpus_dir,
        results_dir=results_dir,
        config_path=config_path,
        source_dataset_path=source_dataset_path,
        selection_seed=selection_seed,
        outer_membership=selected_outer_membership,
        fold_membership=fold_membership,
        fold_defs=fold_defs,
        split_strategy="election_based",
        strategy_config={"outer_split": outer_split_cfg, "folds": folds_cfg},
        selected_corpus=selected_corpus,
        authors_meta_sort_ascending=[True, True, True],
    )

    return _build_run_summary(
        context,
        selected_authors=selected_authors,
        fold_defs=fold_defs,
        artifacts=artifacts,
    )


def run_fixed_test_split_creation_from_context(context: dict) -> dict:
    """Run the fixed-test-count split pipeline from a prepared context.

    Each author contributes exactly ``outer_split.n_test_speeches`` speeches to test
    (the chronologically latest ones); all earlier speeches go to train. Authors with
    fewer speeches than n_test_speeches will have an empty train set and are removed
    by the eligibility filter. Folds (if configured) are built over train speeches.
    """
    df = context["df"]
    print(f"Loaded corpus: {len(df):,} rows, {df['id_person'].nunique():,} authors")
    outer_membership_full = build_outer_membership_fixed_test(
        df, context["outer_split_cfg"], context["split_name"]
    )
    return _run_authorwise_pipeline(
        context,
        outer_membership_full,
        split_strategy="fixed_test_speeches",
    )


def run_profiling_split_creation_from_context(context: dict) -> dict:
    """Run profiling split creation with author-grouped train and fold roles.

    Reads the required ``[exclusion]`` section from the split config:

        [exclusion]
        attribution_authors_path = "data/splits/<split_name>/authors.csv"

    Every author listed in that file is removed from the candidate pool before
    selection. Profiling deliberately does not apply author-attribution
    eligibility or fold-coverage filters; whole authors are assigned to grouped
    train and fold roles so validation authors are unseen during training.
    """
    df = context["df"]
    split_name = context["split_name"]
    print(f"Loaded corpus: {len(df):,} rows, {df['id_person'].nunique():,} authors")

    exclusion_cfg = context["split_config"].get("exclusion", {})
    attr_path = exclusion_cfg.get("attribution_authors_path")
    if not attr_path:
        raise ValueError(
            "Grouped-author profiling split configs must define "
            "[exclusion].attribution_authors_path so profiling authors are "
            "kept disjoint from the configured attribution author set."
        )
    authors_path = resolve_project_path(context["project_root"], attr_path)
    exclusion_ids = load_author_ids_from_csv(
        authors_path,
        label="attribution exclusion",
    )
    print(
        f"Excluding {len(exclusion_ids)} configured attribution authors loaded from {authors_path.name}"
    )

    author_stats = build_author_base_stats(df)
    author_stats = apply_exclusion_filter(author_stats, exclusion_ids)
    print(f"Authors after exclusion filter: {len(author_stats)}")

    _, selected_authors = select_authors(
        author_stats,
        context["pool_cfg"],
        context["selection_cfg"],
        context["selection_seed"],
    )
    print(f"Authors selected: {len(selected_authors)}")
    assert_author_sets_disjoint(
        selected_authors["id_person"],
        exclusion_ids,
        left_label=f"profiling split {split_name}",
        right_label=f"attribution authors from {attr_path}",
    )

    outer_membership = build_grouped_author_outer_membership(
        df,
        selected_authors,
        split_name,
    )
    selected_author_stats = build_author_stats_from_membership(
        df,
        outer_membership,
    )
    selected_author_stats = selected_author_stats[
        selected_author_stats["id_person"].isin(selected_authors["id_person"])
    ].copy()

    fold_defs, fold_membership = build_grouped_author_fold_membership(
        df,
        outer_membership,
        context["folds_cfg"],
        split_name,
        seed=context["selection_seed"],
    )

    artifacts = write_membership_split(
        df=df,
        authors_subset=selected_authors,
        author_stats_full=selected_author_stats,
        split_name=split_name,
        experiment_name=context["experiment_name"],
        project_root=context["project_root"],
        split_dir=context["split_dir"],
        corpus_dir=context["corpus_dir"],
        results_dir=context["results_dir"],
        config_path=context["config_path"],
        source_dataset_path=context["source_dataset_path"],
        selection_seed=context["selection_seed"],
        outer_membership=outer_membership,
        fold_membership=fold_membership,
        fold_defs=fold_defs,
        split_strategy="grouped_authors",
        strategy_config={
            "outer_split": {"strategy": "grouped_authors"},
            "folds": context["folds_cfg"],
            "profiling": {
                "role": "profiling_background",
                "policy": "source_authors_minus_configured_attribution_authors",
                "attribution_authors_path": attr_path,
                "excluded_author_count": len(exclusion_ids),
                "post_selection_overlap_count": 0,
            },
        },
        require_fold_author_coverage=False,
    )

    return _build_run_summary(
        context,
        selected_authors=selected_authors,
        fold_defs=fold_defs,
        artifacts=artifacts,
    )


def run_split_creation(config_path: Path) -> dict:
    """Dispatch split creation based on ``outer_split.strategy`` in the config."""
    context = _load_context_for_run(config_path)
    return run_split_creation_from_context(context)


def run_split_creation_from_context(context: dict) -> dict:
    """Dispatch split creation from a prepared context based on ``outer_split.strategy``."""
    strategy = context["outer_split_strategy"]
    if strategy == "author_percentage":
        return run_author_split_creation_from_context(context)
    if strategy == "fixed_test_speeches":
        return run_fixed_test_split_creation_from_context(context)
    if strategy == "grouped_authors":
        return run_profiling_split_creation_from_context(context)
    if strategy == "election_based":
        return run_temporal_split_creation_from_context(context)
    raise ValueError(
        f"Unsupported outer_split.strategy: {strategy!r}. "
        "Expected 'author_percentage', 'fixed_test_speeches', "
        "'grouped_authors', or 'election_based'."
    )


if __name__ == "__main__":
    import argparse
    import json

    _parser = argparse.ArgumentParser(
        description="Create one split bundle from a split config.",
    )
    _parser.add_argument(
        "--split-config",
        default="data_pipeline/configs/splits/bokmal_authorwise.toml",
        help="Split config path. Can be author-wise or temporal.",
    )
    _args = _parser.parse_args()
    _config = resolve_project_path(find_project_root(), Path(_args.split_config))
    print(json.dumps(run_split_creation(_config), ensure_ascii=False, indent=2))
