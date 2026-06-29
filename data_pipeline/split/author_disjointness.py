"""Author-set helpers for enforcing profiling/attribution split separation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


AUTHOR_ID_COLUMN = "id_person"


def canonical_author_id_series(values: pd.Series) -> pd.Series:
    """Normalize author IDs so CSV integer/float inference does not hide overlaps."""
    series = pd.Series(values, index=values.index)
    result = series.astype("string")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna() & (numeric % 1 == 0)
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype("int64").astype("string")
    return result


def canonical_author_id_set(values) -> set[str]:
    """Return normalized, non-null author IDs as a set for disjointness checks."""
    if isinstance(values, (set, frozenset)):
        values = sorted(values, key=str)
    series = pd.Series(values).dropna()
    if series.empty:
        return set()
    return set(canonical_author_id_series(series).dropna().astype(str))


def load_author_ids_from_csv(path: Path, *, label: str) -> set[str]:
    """Load normalized author IDs from an authors.csv file at a file boundary."""
    path = Path(path)
    try:
        author_ids = pd.read_csv(path, usecols=[AUTHOR_ID_COLUMN])[AUTHOR_ID_COLUMN]
    except ValueError as exc:
        raise ValueError(
            f"{label} authors file must contain an {AUTHOR_ID_COLUMN!r} column: {path}"
        ) from exc
    return canonical_author_id_set(author_ids)


def assert_author_sets_disjoint(
    left_author_ids,
    right_author_ids,
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Raise when two author ID collections overlap."""
    left_ids = canonical_author_id_set(left_author_ids)
    right_ids = canonical_author_id_set(right_author_ids)
    overlap = sorted(left_ids & right_ids, key=lambda value: (len(value), value))
    if overlap:
        sample = ", ".join(overlap[:10])
        raise ValueError(
            f"{left_label} and {right_label} must be author-disjoint; "
            f"found {len(overlap)} overlapping id_person value(s): {sample}"
        )
