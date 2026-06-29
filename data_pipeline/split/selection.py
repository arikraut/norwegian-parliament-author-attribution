"""Author pool filtering and selection logic for split creation."""

from __future__ import annotations

import pandas as pd

from data_pipeline.split.author_disjointness import (
    canonical_author_id_series,
    canonical_author_id_set,
)


def build_author_base_stats(
    df: pd.DataFrame, author_col: str = "id_person"
) -> pd.DataFrame:
    """Compute per-author metadata and full-corpus character/speech totals for later filtering and ranking."""
    author_basic = df.groupby(author_col, as_index=False).agg(
        name=("name", "first"),
        female=("female", "first"),
        party=("party", "first"),
        partyname=("partyname", "first"),
    )

    age_stats = (
        df.groupby(author_col)["age"]
        .agg(min_age="min", max_age="max", mean_age="mean")
        .reset_index()
    )

    lang_mode = (
        df.groupby(author_col)["language"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else None)
        .reset_index(name="language_main")
    )

    total_stats = df.groupby(author_col, as_index=False).agg(
        total_chars_all=("char_count", "sum"),
        total_speeches_all=("id_speech", "nunique"),
    )

    return (
        author_basic.merge(total_stats, on=author_col, how="left")
        .merge(age_stats, on=author_col, how="left")
        .merge(lang_mode, on=author_col, how="left")
    )


def apply_exclusion_filter(
    author_stats: pd.DataFrame,
    exclusion_ids: list,
) -> pd.DataFrame:
    """Remove authors whose id_person appears in the exclusion list."""
    excluded = canonical_author_id_set(exclusion_ids)
    author_ids = canonical_author_id_series(author_stats["id_person"])
    return author_stats[~author_ids.isin(excluded)].copy()


def apply_author_split_eligibility_filters(
    author_stats: pd.DataFrame,
    eligibility_cfg: dict,
) -> pd.DataFrame:
    """Remove attribution authors below the configured outer-train character minimum."""
    min_train_chars = int(eligibility_cfg["min_train_chars_per_author"])
    return author_stats[author_stats["train_chars"] >= min_train_chars].copy()


def apply_temporal_test_eligibility_filters(
    author_stats: pd.DataFrame,
    eligibility_cfg: dict,
) -> pd.DataFrame:
    """Remove temporal authors below the configured capped outer-test speech minimum."""
    min_test_speeches = eligibility_cfg.get("min_test_speeches_per_author")
    if min_test_speeches is None:
        return author_stats.copy()

    min_test_speeches = int(min_test_speeches)
    return author_stats[
        author_stats["test_speeches"] >= min_test_speeches
    ].copy()


# ---------------------------------------------------------------------------
# Author selection
# ---------------------------------------------------------------------------


def filter_author_pool(author_stats_df: pd.DataFrame, pool_cfg: dict) -> pd.DataFrame:
    """Apply party and language pool restrictions before author selection.

    Removes authors outside the configured party list or language mode, then drops any
    party with fewer than pool_cfg.min_authors_per_party remaining candidates.
    """
    authors = author_stats_df.copy()

    if pool_cfg.get("party_mode", "all") == "list":
        authors = authors[authors["party"].isin(pool_cfg.get("chosen_parties", []))]

    if pool_cfg.get("language_mode", "both") == "bokmal_only":
        authors = authors[
            authors["language_main"].isin(pool_cfg.get("bokmal_codes", ["Bokmål"]))
        ]

    min_authors_per_party = int(pool_cfg.get("min_authors_per_party", 0) or 0)
    if min_authors_per_party > 0:
        counts = authors["party"].value_counts()
        keep_parties = counts[counts >= min_authors_per_party].index
        authors = authors[authors["party"].isin(keep_parties)]

    return authors


def add_party_ranks(authors: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Attach a sort-ready metric value and within-party rank to each candidate author."""
    authors = authors.copy()
    authors["selection_metric_value"] = authors[metric_col]
    authors["rank_in_party"] = (
        authors.groupby("party")[metric_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return authors


def _validate_char_ranking_metric(pool: pd.DataFrame, ranking_metric: str) -> None:
    """Reject author-selection metrics that are absent from the candidate pool."""
    if ranking_metric not in pool.columns:
        raise KeyError(f"Unknown ranking metric: {ranking_metric!r}. Available: {sorted(pool.columns.tolist())}")


def _select_alternating_groups(
    group: pd.DataFrame,
    k: int,
    group_col: str,
) -> pd.DataFrame:
    """Select k authors by round-robin across groups so no single party or language dominates the early slots."""
    group = group.sort_values(
        ["selection_metric_value", "rank_in_party", "id_person"],
        ascending=[False, True, True],
    )
    grouped_indices: dict[str, list] = {}
    for idx, value in group[group_col].items():
        grouped_indices.setdefault(str(value), []).append(idx)

    labels = sorted(
        grouped_indices.keys(),
        key=lambda label: (
            -group.loc[grouped_indices[label][0], "selection_metric_value"],
            label,
        ),
    )
    positions = {label: 0 for label in labels}
    selected_idx: list[int] = []
    ptr = 0

    while len(selected_idx) < k and any(
        positions[label] < len(grouped_indices[label]) for label in labels
    ):
        label = labels[ptr]
        pos = positions[label]
        if pos < len(grouped_indices[label]):
            selected_idx.append(grouped_indices[label][pos])
            positions[label] += 1
        ptr = (ptr + 1) % len(labels)

    return group.loc[selected_idx]


def _select_within_group(
    df_group: pd.DataFrame,
    k: int,
    strategy: str,
) -> pd.DataFrame:
    """Select up to k authors from one candidate group using the configured picking strategy.

    Supports top_chars and alternate_parties. Returns the full group unchanged
    when k >= len(group).
    """
    group = df_group.copy()
    if k >= len(group):
        return group

    if strategy == "top_chars":
        return group.sort_values("selection_metric_value", ascending=False).head(k)

    if strategy == "alternate_parties":
        if "party" not in group.columns:
            raise KeyError("alternate_parties strategy requires a 'party' column")
        return _select_alternating_groups(group, k, group_col="party")

    raise ValueError(f"Unknown picking strategy: {strategy}")


def _enforce_min_authors_total_mode(
    selected_subset: pd.DataFrame,
    pool: pd.DataFrame,
    k_target: int,
    min_authors_per_party: int,
    strategy: str,
) -> pd.DataFrame:
    """Refill a total-mode selection when the first pass produces parties below the per-party minimum.

    Drops under-represented parties and replaces their slots with candidates from the parties
    that already meet the minimum. Returns the original selection unchanged when no party
    violates the minimum or when no replacement candidates are available.
    """
    if min_authors_per_party <= 0 or selected_subset.empty:
        return selected_subset

    counts_sel = selected_subset["party"].value_counts()
    good_parties = counts_sel[counts_sel >= min_authors_per_party].index
    bad_parties = counts_sel[counts_sel < min_authors_per_party].index

    if len(bad_parties) == 0:
        return selected_subset

    kept = selected_subset[selected_subset["party"].isin(good_parties)].copy()
    dropped = selected_subset[selected_subset["party"].isin(bad_parties)].copy()

    print(
        f"Dropping {len(dropped)} authors from parties with fewer than "
        f"{min_authors_per_party} authors in the current selection: "
        + ", ".join(sorted(set(dropped["party"])))
    )

    slots_needed = k_target - len(kept)
    if slots_needed <= 0:
        return kept

    if len(good_parties) == 0:
        print(
            "Warning: no party reached the per-party minimum in the current selection; "
            "keeping the original selection."
        )
        return selected_subset

    author_ids_kept = set(kept["id_person"])
    pool_remaining = pool[~pool["id_person"].isin(author_ids_kept)]
    pool_remaining_good = pool_remaining[pool_remaining["party"].isin(good_parties)]

    if pool_remaining_good.empty:
        print(
            "Warning: no remaining candidates from parties that already meet the minimum; "
            "selection will have fewer authors than requested."
        )
        return kept

    refill_df = _select_within_group(pool_remaining_good, slots_needed, strategy)
    refill_subset = refill_df[
        ["party", "id_person", "selection_metric_value", "rank_in_party"]
    ].copy()
    combined = pd.concat([kept, refill_subset], ignore_index=True)

    return combined


def select_authors(
    author_stats_df: pd.DataFrame,
    pool_cfg: dict,
    selection_cfg: dict,
    selection_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the pool, rank candidates, and return the final selected author subset.

    Returns (filtered_pool, selected_authors). The pool has party ranks attached;
    selected_authors is a smaller subset after applying the configured mode and strategy.
    """
    pool = filter_author_pool(author_stats_df, pool_cfg)
    if pool.empty:
        raise ValueError("No authors left after applying party/language filters.")

    ranking_metric = selection_cfg.get("ranking_metric", "train_chars")
    _validate_char_ranking_metric(pool, ranking_metric)

    pool = add_party_ranks(pool, ranking_metric)
    selection_mode = selection_cfg.get("mode", "per_party")
    strategy = selection_cfg.get("strategy", "top_chars")
    min_authors_per_party = int(pool_cfg.get("min_authors_per_party", 0) or 0)

    if selection_mode == "all_eligible":
        selected = pool.copy()
    elif selection_mode == "per_party":
        k = int(selection_cfg["n_authors_per_party"])
        selected_list = []
        for _, sub in pool.groupby("party"):
            if len(sub) < k:
                continue
            selected_list.append(_select_within_group(sub, k, strategy))
        selected = (
            pd.concat(selected_list, ignore_index=False)
            if selected_list
            else pd.DataFrame(columns=pool.columns)
        )
    elif selection_mode == "total":
        k = min(int(selection_cfg["n_authors_total"]), len(pool))
        selected = _select_within_group(pool, k, strategy)
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    if selected.empty:
        empty = pd.DataFrame(
            columns=["party", "id_person", "selection_metric_value", "rank_in_party"]
        )
        return pool, empty

    selected_subset = selected[
        ["party", "id_person", "selection_metric_value", "rank_in_party"]
    ].copy()
    if selection_mode == "total":
        selected_subset = _enforce_min_authors_total_mode(
            selected_subset,
            pool,
            min(int(selection_cfg["n_authors_total"]), len(pool)),
            min_authors_per_party,
            strategy,
        )

    return pool, selected_subset.reset_index(drop=True)
