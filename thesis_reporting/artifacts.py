"""Artifact-loading and table-key boundaries for research reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_file(path: Path, *, context: str) -> None:
    """Require an external artifact before copying or consuming it."""

    if not path.exists():
        raise FileNotFoundError(f"{context}: expected artifact does not exist: {path}")


def canonical_label(value: object) -> str:
    """Return a stable string representation for labels and external table keys."""

    if pd.isna(value):
        return ""
    text = str(value)
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def id_sample(values: Iterable[object], limit: int = 10) -> list[str]:
    """Return a short deterministic sample of key values for error messages."""

    return sorted({canonical_label(value) for value in values}, key=str)[:limit]


def validate_unique_key(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
) -> None:
    """Validate that an external table has one non-empty row per key."""

    if column not in frame.columns:
        raise ValueError(f"{context}: missing required key column {column!r}.")
    missing = frame[column].isna()
    if bool(missing.any()):
        raise ValueError(
            f"{context}: {int(missing.sum())} rows have missing {column!r} values."
        )
    duplicated = frame.loc[frame[column].duplicated(keep=False), column]
    if not duplicated.empty:
        raise ValueError(
            f"{context}: duplicate {column!r} values found "
            f"(n_duplicate_rows={len(duplicated)}, sample={id_sample(duplicated)})."
        )


def validate_same_key_set(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    *,
    left_context: str,
    right_context: str,
) -> None:
    """Validate that two external tables contain exactly the same key set."""

    left_keys = set(left[column].map(canonical_label))
    right_keys = set(right[column].map(canonical_label))
    if left_keys != right_keys:
        missing_from_right = left_keys - right_keys
        missing_from_left = right_keys - left_keys
        raise ValueError(
            f"{left_context} and {right_context} must contain the same {column!r} "
            "set. "
            f"{right_context} is missing {len(missing_from_right)} keys "
            f"(sample={id_sample(missing_from_right)}); "
            f"{left_context} is missing {len(missing_from_left)} keys "
            f"(sample={id_sample(missing_from_left)})."
        )


class ResultArtifacts:
    """Cache raw result CSVs and return copies at transformation boundaries."""

    def __init__(self, root: Path) -> None:
        """Create a reader rooted at one resolved result directory."""

        self.root = root
        self._csv_cache: dict[Path, pd.DataFrame] = {}

    def path(self, relative_path: Path) -> Path:
        """Resolve one configured artifact path below the result root."""

        return self.root / relative_path

    def read_csv(self, relative_path: Path) -> pd.DataFrame:
        """Load an ordinary result CSV once and return an isolated copy."""

        path = self.path(relative_path)
        if path not in self._csv_cache:
            self._csv_cache[path] = pd.read_csv(path)
        return self._csv_cache[path].copy()
