"""Author-wise chronological split assignment and fold construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.split.stats import build_author_fold_stats


def _validate_role_shares(
    cfg: dict,
    roles: tuple[str, ...],
    section_name: str,
    required_roles: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Validate that role share values are non-negative and sum to 1.

    When required_roles is None, all roles must be present. When required_roles is
    provided, only those roles are required; absent roles default to 0.
    """
    check_roles = roles if required_roles is None else required_roles
    missing_roles = [role for role in check_roles if role not in cfg]
    if missing_roles:
        raise KeyError(
            f"{section_name} is missing required role shares: {missing_roles}"
        )

    if required_roles is None:
        shares = {role: float(cfg[role] or 0.0) for role in roles}
    else:
        shares = {role: float(cfg.get(role, 0.0) or 0.0) for role in roles}
    for role, share in shares.items():
        if share < 0:
            raise ValueError(f"{section_name}.{role} must be >= 0")

    total = sum(shares.values())
    if total <= 0:
        raise ValueError(
            f"{section_name} must assign a positive total share across {roles}"
        )
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"{section_name} shares must sum to 1.0, got {total:.6f}")
    return shares


def _char_values(frame: pd.DataFrame) -> np.ndarray:
    """Return a numeric char-count vector, returning zeros safely when the frame is empty or missing the column."""
    if frame.empty or "char_count" not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return frame["char_count"].fillna(0).astype(float).to_numpy()


def _ordered_author_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort speeches into a deterministic chronological order before any author-wise splitting."""
    if frame.empty:
        return frame.copy()

    sort_cols = [col for col in ["date", "time", "id_speech"] if col in frame.columns]
    if not sort_cols:
        sort_cols = ["id_speech"]
    return frame.sort_values(sort_cols).copy()


def _build_char_buckets(
    ordered: pd.DataFrame, n_buckets: int
) -> tuple[list[np.ndarray], bool]:
    """Partition one ordered author slice into n_buckets whole-speech buckets with similar character mass.

    Fills buckets greedily front-to-back (chronological order) so earlier speeches land in
    lower-indexed buckets, which become the train role after _split_ordered_positions_by_char_share.
    """
    n_items = len(ordered)
    if n_items < n_buckets or n_items == 0:
        return [], True

    char_values = _char_values(ordered)
    total_chars = float(char_values.sum())
    target_chars = total_chars / float(n_buckets) if n_buckets > 0 else 0.0
    buckets: list[np.ndarray] = []
    start = 0

    for bucket_idx in range(n_buckets - 1):
        remaining_buckets = n_buckets - bucket_idx - 1
        bucket_positions: list[int] = []
        bucket_chars = 0.0

        while start < n_items - remaining_buckets:
            bucket_positions.append(start)
            bucket_chars += char_values[start]
            start += 1
            if bucket_chars >= target_chars:
                break

        if not bucket_positions:
            return [], True
        buckets.append(np.array(bucket_positions, dtype=int))

    last_bucket = np.arange(start, n_items, dtype=int)
    if len(last_bucket) == 0:
        return [], True
    buckets.append(last_bucket)
    return buckets, False


def resolve_fold_role_minimums(
    folds_cfg: dict | None,
) -> dict[str, int]:
    """Read per-role minimum character thresholds from folds_cfg, defaulting to zero."""
    cfg = dict(folds_cfg or {})
    return {
        "train": int(cfg.get("min_train_chars_per_author", 0) or 0),
        "val": int(cfg.get("min_val_chars_per_author", 0) or 0),
    }


def _split_ordered_positions_by_char_share(
    ordered: pd.DataFrame,
    shares: dict[str, float],
    role_order: list[str],
) -> dict[str, np.ndarray]:
    """Assign speeches to roles by peeling target character-share buckets back-to-front.

    Evaluation roles (test, val) receive the most recent speeches; train receives the earlier
    ones. This chronological assignment is the intended leakage-safe evaluation strategy.
    """
    positions_by_role: dict[str, np.ndarray] = {
        role: np.array([], dtype=int) for role in role_order
    }
    if ordered.empty:
        return positions_by_role

    char_values = _char_values(ordered)
    total_chars = float(char_values.sum())
    active_roles = [role for role in role_order if shares[role] > 0]
    if not active_roles:
        return positions_by_role

    end = len(ordered)
    for role_idx in range(len(active_roles) - 1, 0, -1):
        role = active_roles[role_idx]
        target_chars = shares[role] * total_chars
        n_earlier_roles = role_idx
        role_positions: list[int] = []
        role_chars = 0.0

        while end > n_earlier_roles and (
            not role_positions or role_chars < target_chars
        ):
            end -= 1
            role_positions.append(end)
            role_chars += char_values[end]

        positions_by_role[role] = np.array(sorted(role_positions), dtype=int)

    positions_by_role[active_roles[0]] = np.arange(end, dtype=int)
    return positions_by_role


def _build_outer_role_rows(
    ordered: pd.DataFrame,
    role_positions: dict[str, np.ndarray],
    role_order: list[str],
    base_cols: list[str],
    split_name: str,
    min_role_chars: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    """Convert per-role speech positions into membership rows and role-level diagnostics.

    Returns (membership_df, role_stats) where role_stats maps each role to its character
    and speech counts and any character deficit against min_role_chars.
    """
    membership_frames: list[pd.DataFrame] = []
    role_stats: dict[str, dict[str, int]] = {}
    total_deficit = 0

    for role in role_order:
        positions = role_positions.get(role, np.array([], dtype=int))
        role_slice = ordered.iloc[positions]

        role_frame = role_slice[base_cols].copy()
        role_frame["split_name"] = split_name
        role_frame["outer_role"] = role
        membership_frames.append(role_frame)

        char_sum = (
            int(role_slice["char_count"].fillna(0).sum())
            if "char_count" in role_slice.columns
            else 0
        )
        speech_count = int(len(role_slice))
        char_deficit = max(0, int(min_role_chars.get(role, 0)) - char_sum)
        total_deficit += char_deficit

        role_stats[role] = {
            "chars": char_sum,
            "speeches": speech_count,
            "char_deficit": int(char_deficit),
        }

    assignment = (
        pd.concat(membership_frames, ignore_index=True)
        if membership_frames
        else pd.DataFrame(columns=[*base_cols, "split_name", "outer_role"])
    )
    diagnostics = {
        "meets_requirements": bool(total_deficit == 0),
        "total_deficit": int(total_deficit),
        "role_stats": role_stats,
    }
    return assignment, diagnostics

def _format_fold_failure_summary(failed_authors: list[dict]) -> str:
    """Format a concise error message listing the first few authors who failed fold role assignment."""
    examples = []
    for record in failed_authors[:3]:
        if record["has_empty_bucket"]:
            examples.append(f"{record['author_id']} (empty fold bucket)")
            continue
        fold_bits = []
        for fold_id, role_stats in record["fold_stats"].items():
            train_stats = role_stats.get("train", {})
            val_stats = role_stats.get("val", {})
            fold_bits.append(
                f"{fold_id}=train_chars:{train_stats.get('chars', 0)}/def:{train_stats.get('char_deficit', 0)},"
                f"train_speeches:{train_stats.get('speeches', 0)},"
                f"val_chars:{val_stats.get('chars', 0)}/def:{val_stats.get('char_deficit', 0)},"
                f"val_speeches:{val_stats.get('speeches', 0)}"
            )
        examples.append(f"{record['author_id']} ({'; '.join(fold_bits)})")
    return (
        "Failed to assign valid author-wise fold roles for "
        f"{len(failed_authors)} author(s) under the configured fold minima. "
        f"Examples: {', '.join(examples)}"
    )


def build_outer_membership_by_author(
    df: pd.DataFrame,
    outer_split_cfg: dict,
    split_name: str,
    author_col: str = "id_person",
    party_col: str = "party",
) -> pd.DataFrame:
    """Assign each author's speeches to outer train/test roles chronologically."""
    shares = _validate_role_shares(
        outer_split_cfg,
        ("train", "test"),
        section_name="outer_split",
    )
    role_order = ["train", "test"]
    min_role_chars = {role: 0 for role in role_order}

    membership_frames: list[pd.DataFrame] = []
    base_cols = [
        col
        for col in ["id_speech", author_col, "election", party_col, "language"]
        if col in df.columns
    ]

    for _, author_df in df.groupby(author_col, sort=True):
        ordered = _ordered_author_rows(author_df)
        role_positions = _split_ordered_positions_by_char_share(
            ordered, shares, role_order
        )
        assignment, _ = _build_outer_role_rows(
            ordered,
            role_positions=role_positions,
            role_order=role_order,
            base_cols=base_cols,
            split_name=split_name,
            min_role_chars=min_role_chars,
        )
        membership_frames.append(assignment)

    if not membership_frames:
        return pd.DataFrame(
            columns=[
                "id_speech",
                author_col,
                "election",
                party_col,
                "language",
                "split_name",
                "outer_role",
            ]
        )
    return pd.concat(membership_frames, ignore_index=True)


def _split_by_fixed_test_speech_count(
    ordered: pd.DataFrame,
    n_test: int,
) -> dict[str, np.ndarray]:
    """Assign the n_test chronologically latest speeches to test and all earlier ones to train.

    When the author has fewer than n_test speeches, all go to test and train is empty.
    The downstream eligibility filter removes such authors before selection.
    """
    n = len(ordered)
    n_test_actual = min(n_test, n)
    positions = np.arange(n)
    return {
        "train": positions[: n - n_test_actual],
        "test": positions[n - n_test_actual :],
    }


def build_outer_membership_fixed_test(
    df: pd.DataFrame,
    outer_split_cfg: dict,
    split_name: str,
    author_col: str = "id_person",
    party_col: str = "party",
) -> pd.DataFrame:
    """Assign each author's speeches to train/test by taking the last n_test_speeches as test.

    The split is purely chronological: the most recent N speeches per author become test,
    everything earlier becomes train. Authors with fewer speeches than n_test will have an
    empty train set and will be dropped by the downstream eligibility filter.

    Config keys read from outer_split_cfg:
        n_test_speeches (int, required): how many speeches per author go to test.
    """
    n_test = int(outer_split_cfg["n_test_speeches"])
    if n_test < 1:
        raise ValueError("outer_split.n_test_speeches must be >= 1")

    role_order = ["train", "test"]
    base_cols = [
        col
        for col in ["id_speech", author_col, "election", party_col, "language"]
        if col in df.columns
    ]

    membership_frames: list[pd.DataFrame] = []
    for _, author_df in df.groupby(author_col, sort=True):
        ordered = _ordered_author_rows(author_df)
        role_positions = _split_by_fixed_test_speech_count(ordered, n_test)
        assignment, _ = _build_outer_role_rows(
            ordered,
            role_positions=role_positions,
            role_order=role_order,
            base_cols=base_cols,
            split_name=split_name,
            min_role_chars={},
        )
        membership_frames.append(assignment)

    if not membership_frames:
        return pd.DataFrame(
            columns=[
                "id_speech",
                author_col,
                "election",
                party_col,
                "language",
                "split_name",
                "outer_role",
            ]
        )
    return pd.concat(membership_frames, ignore_index=True)


def _build_author_kfold_partitions(
    ordered: pd.DataFrame,
    n_splits: int,
) -> tuple[list[dict], bool]:
    """Partition one author's chronologically ordered speeches into k-fold train/val buckets.

    Returns (partitions, has_empty_bucket). Each partition is a dict with
    ``train_positions`` and ``val_positions`` index arrays. ``has_empty_bucket`` is
    true when the author has too few speeches to fill all requested folds.
    """
    partitions: list[dict] = []
    n_items = len(ordered)
    positions = np.arange(n_items)

    bucket_indices, has_empty_bucket = _build_char_buckets(ordered, n_splits)
    if has_empty_bucket:
        return [], True

    for fold_idx, val_positions in enumerate(bucket_indices):
        train_positions = np.setdiff1d(positions, val_positions, assume_unique=True)
        partitions.append(
            {
                "fold_id": f"fold_{fold_idx + 1:02d}",
                "train_positions": train_positions,
                "val_positions": val_positions,
                "included_positions": positions,
                "mode": "kfold",
                "fold_index": int(fold_idx + 1),
                "n_splits": int(n_splits),
                "ordering": "chronological",
            }
        )
    return partitions, False


def _build_fold_role_rows(
    ordered: pd.DataFrame,
    fold_partitions: list[dict],
    base_cols: list[str],
    split_name: str,
    min_role_chars: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    """Convert one author's fold partition results into membership rows and fold diagnostics.

    Returns (fold_membership_df, fold_stats) where fold_stats maps each fold_id to per-role
    character counts and any deficit against min_role_chars.
    """
    fold_frames: list[pd.DataFrame] = []
    fold_stats: dict[str, dict[str, dict[str, int]]] = {}
    total_deficit = 0

    for fold_part in fold_partitions:
        train_positions = fold_part["train_positions"]
        val_positions = fold_part["val_positions"]
        included_positions = np.sort(fold_part["included_positions"])

        train_slice = ordered.iloc[train_positions]
        val_slice = ordered.iloc[val_positions]
        fold_frame = ordered.iloc[included_positions][base_cols].copy()
        fold_frame["split_name"] = split_name
        fold_frame["fold_id"] = fold_part["fold_id"]
        fold_frame["fold_role"] = "train"
        fold_frame.loc[ordered.index[val_positions], "fold_role"] = "val"
        fold_frames.append(fold_frame)

        role_stats: dict[str, dict[str, int]] = {}
        for role, role_slice in [("train", train_slice), ("val", val_slice)]:
            char_sum = (
                int(role_slice["char_count"].fillna(0).sum())
                if "char_count" in role_slice.columns
                else 0
            )
            speech_count = (
                int(role_slice["id_speech"].nunique())
                if "id_speech" in role_slice.columns
                else int(len(role_slice))
            )
            char_deficit = max(0, int(min_role_chars.get(role, 0)) - char_sum)
            total_deficit += char_deficit
            role_stats[role] = {
                "chars": char_sum,
                "speeches": speech_count,
                "char_deficit": int(char_deficit),
            }
        fold_stats[str(fold_part["fold_id"])] = role_stats

    assignment = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames
        else pd.DataFrame(columns=[*base_cols, "split_name", "fold_id", "fold_role"])
    )
    diagnostics = {
        "meets_requirements": bool(total_deficit == 0),
        "total_deficit": int(total_deficit),
        "fold_stats": fold_stats,
    }
    return assignment, diagnostics


def filter_authors_by_authorwise_fold_coverage(
    df: pd.DataFrame,
    outer_membership: pd.DataFrame,
    author_stats: pd.DataFrame,
    folds_cfg: dict,
    split_name: str,
    author_col: str = "id_person",
    party_col: str = "party",
) -> pd.DataFrame:
    """Filter authors so author-wise fold generation keeps full author coverage.

    Coverage is checked from a deterministic fold probe so the surviving author
    set is stable across selection seeds. The probe enforces per-fold
    character-volume minima when those are configured.
    """
    cfg = dict(folds_cfg or {})
    mode = str(cfg.get("mode", "none")).lower()
    if mode == "none":
        return author_stats.copy()

    min_role_chars = resolve_fold_role_minimums(cfg)
    fold_defs, probe_fold_membership = build_authorwise_fold_membership(
        outer_membership=outer_membership,
        folds_cfg=cfg,
        split_name=f"{split_name}__fold_probe__",
        df=df,
        author_col=author_col,
        party_col=party_col,
        enforce_minima=False,
    )
    fold_stats = build_author_fold_stats(
        df, probe_fold_membership, author_col=author_col
    )
    expected_fold_ids = {str(fd["fold_id"]) for fd in fold_defs}

    surviving = set(author_stats[author_col].dropna().unique())
    failing_authors: list[int] = []
    for author_id in list(surviving):
        author_fold_stats = fold_stats[fold_stats[author_col] == author_id].copy()
        if (
            author_fold_stats.empty
            or set(author_fold_stats["fold_id"].astype(str)) != expected_fold_ids
            or (author_fold_stats["train_chars"] < int(min_role_chars["train"])).any()
            or (author_fold_stats["val_chars"] < int(min_role_chars["val"])).any()
        ):
            failing_authors.append(int(author_id))

    if failing_authors:
        surviving -= set(failing_authors)
        print(
            f"filter_authors_by_authorwise_fold_coverage: removing {len(failing_authors)} author(s) "
            "that cannot support author-wise k-folds "
            f"({len(expected_fold_ids)} fold(s)) "
            "under the configured fold minima"
        )

    return author_stats[author_stats[author_col].isin(surviving)].copy()


def build_authorwise_fold_membership(
    outer_membership: pd.DataFrame,
    folds_cfg: dict,
    split_name: str,
    df: pd.DataFrame | None = None,
    author_col: str = "id_person",
    party_col: str = "party",
    enforce_minima: bool = True,
) -> tuple[list[dict], pd.DataFrame]:
    """Create chronological author-wise k-fold memberships from outer train speeches."""
    cfg = dict(folds_cfg or {})
    mode = str(cfg.get("mode", "none")).lower()
    if mode == "none":
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
        return [], empty
    if mode != "kfold":
        raise ValueError(f"Unsupported author fold mode: {mode}")

    source = str(cfg.get("source", "train_only")).lower()
    if source == "train_only":
        source_roles = {"train"}
    else:
        raise ValueError(
            f"Unsupported author folds.source: {source!r}. Only 'train_only' is supported."
        )

    source_membership = outer_membership[
        outer_membership["outer_role"].isin(source_roles)
    ].copy()
    if df is not None and "char_count" not in source_membership.columns:
        source_membership = source_membership.merge(
            df[["id_speech", author_col, "char_count"]],
            on=["id_speech", author_col],
            how="left",
            sort=False,
        )
    n_splits = int(cfg.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("author folds.n_splits must be >= 2")

    min_role_chars = resolve_fold_role_minimums(cfg)
    base_cols = [
        col
        for col in ["id_speech", author_col, "election", party_col, "language"]
        if col in source_membership.columns
    ]

    fold_frames: list[pd.DataFrame] = []
    fold_defs: list[dict] = []
    failed_authors: list[dict] = []
    for author_id, author_df in source_membership.groupby(author_col, sort=True):
        ordered = _ordered_author_rows(author_df)
        author_fold_partitions, has_empty_bucket = _build_author_kfold_partitions(
            ordered,
            n_splits=n_splits,
        )
        if has_empty_bucket:
            if enforce_minima:
                failed_authors.append(
                    {
                        "author_id": int(author_id),
                        "fold_stats": {},
                        "has_empty_bucket": True,
                    }
                )
            continue
        if not fold_defs:
            fold_defs = [
                {
                    "fold_id": part["fold_id"],
                    "mode": part["mode"],
                    "source": source,
                    "source_roles": sorted(source_roles),
                    "fold_index": part["fold_index"],
                    "n_splits": part["n_splits"],
                    "ordering": part["ordering"],
                }
                for part in author_fold_partitions
            ]

        assignment, diag = _build_fold_role_rows(
            ordered,
            fold_partitions=author_fold_partitions,
            base_cols=base_cols,
            split_name=split_name,
            min_role_chars=min_role_chars,
        )

        if not assignment.empty:
            fold_frames.append(assignment)
        if not diag["meets_requirements"] and enforce_minima:
            failed_authors.append(
                {
                    "author_id": int(author_id),
                    "fold_stats": diag.get("fold_stats", {}),
                    "has_empty_bucket": False,
                }
            )

    if failed_authors:
        raise ValueError(_format_fold_failure_summary(failed_authors))

    if not fold_frames:
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
        return fold_defs, empty
    return fold_defs, pd.concat(fold_frames, ignore_index=True)
