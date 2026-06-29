"""Row metadata and profiling-target helpers."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

ROW_METADATA_COLUMNS = [
    "id_speech",
    "id_person",
    "name",
    "party",
    "partyname",
    "female",
    "age",
    "age_bin",
    "language",
    "left_center_right",
    "election",
    "date",
    "time",
    "word_count",
    "char_count",
]

DERIVED_TARGET_SOURCE_COLUMNS = {
    "age_bin": "age",
    "left_center_right": "party",
}


def validate_feature_corpus_columns(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    profiling_labels: list[str],
    metadata_columns: list[str] | None = None,
) -> None:
    """Validate loaded corpus splits against target and derived-metadata column needs."""
    requested_columns = set(profiling_labels)
    requested_columns.update(
        column
        for column in (metadata_columns or ROW_METADATA_COLUMNS)
        if column in DERIVED_TARGET_SOURCE_COLUMNS
    )

    required_columns = {"id_speech", "id_person"}
    for column in requested_columns:
        source_column = DERIVED_TARGET_SOURCE_COLUMNS.get(column, column)
        required_columns.add(source_column)

    for split_name, frame in split_frames.items():
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(
                f"{split_name} corpus is missing required feature source columns: {missing_text}"
            )


def normalize_party_name(party: str) -> str:
    """Normalise a party string to a lowercase, accent-stripped, whitespace-free lookup key."""
    if not isinstance(party, str):
        return ""
    party = party.strip()
    party = unicodedata.normalize("NFKD", party)
    party = "".join(ch for ch in party if not unicodedata.combining(ch))
    party = party.lower()
    party = party.replace(" ", "").replace("-", "")
    return party


def build_party_axis_map(axis_cfg: dict[str, list[str]]) -> dict[str, str]:
    """Build a normalised-party-name to axis-label lookup from [party_axis]."""
    mapping: dict[str, str] = {}
    for axis_name, parties in axis_cfg.items():
        for party_name in parties:
            mapping[normalize_party_name(party_name)] = axis_name
    return mapping


def add_age_bin_column(
    df: pd.DataFrame,
    bin_edges: list[int],
    bin_labels: list[str],
    age_col: str = "age",
    new_col: str = "age_bin",
) -> pd.DataFrame:
    """Add a discretised age_bin column to df using the configured bin edges and labels."""
    df = df.copy()
    s = pd.cut(
        df[age_col], bins=bin_edges, labels=bin_labels, right=False, include_lowest=True
    )
    df[new_col] = s.astype(str).replace("nan", "unknown")
    return df


def add_left_center_right_column(
    df: pd.DataFrame,
    party_axis_map: dict[str, str],
    party_col: str = "party",
    new_col: str = "left_center_right",
) -> pd.DataFrame:
    """Add a left/center/right political-axis column using a normalised party lookup."""
    df = df.copy()
    df[new_col] = df[party_col].apply(
        lambda p: party_axis_map.get(normalize_party_name(p), "unknown")
    )
    return df


def build_row_meta_frame(
    df: pd.DataFrame,
    split_name: str,
    outer_role: str,
    meta_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select row metadata columns from df and annotate with split_name and outer_role."""
    cols = meta_columns or ROW_METADATA_COLUMNS
    available = [c for c in cols if c in df.columns]
    frame = df[available].copy()
    frame["split_name"] = split_name
    frame["outer_role"] = outer_role
    return frame


def build_targets_frame(
    df: pd.DataFrame,
    split_name: str,
    outer_role: str,
    save_author_labels: bool,
    profiling_labels: list[str],
) -> pd.DataFrame:
    """Build the targets DataFrame with author and configured profiling label columns."""
    base = pd.DataFrame(
        {
            "id_speech": df["id_speech"].values,
            "id_person": df["id_person"].values,
            "split_name": split_name,
            "outer_role": outer_role,
        }
    )
    if save_author_labels:
        base["author"] = df["id_person"].values
    for col in profiling_labels:
        if col not in df.columns:
            raise KeyError(
                f"Configured profiling label {col!r} is missing from the dataframe. "
                "Check that all derived columns (age_bin, left_center_right) were added before calling this function."
            )
        base[col] = df[col].values
    return base


def build_feature_split_summary(df: pd.DataFrame, split_name: str) -> dict:
    """Compute row, author, party, and text-length statistics for one corpus split."""
    summary: dict = {
        "split": split_name,
        "rows": int(len(df)),
        "authors": int(df["id_person"].nunique()) if "id_person" in df.columns else 0,
        "parties": int(df["party"].nunique()) if "party" in df.columns else 0,
        "unknown_axis_rows": (
            int((df["left_center_right"] == "unknown").sum())
            if "left_center_right" in df.columns
            else 0
        ),
    }
    if "char_count" in df.columns:
        summary.update(
            {
                "total_chars": int(df["char_count"].sum()),
                "mean_chars": float(df["char_count"].mean()),
                "median_chars": float(df["char_count"].median()),
            }
        )
    if "word_count" in df.columns:
        summary.update(
            {
                "total_words": int(df["word_count"].sum()),
                "mean_words": float(df["word_count"].mean()),
                "median_words": float(df["word_count"].median()),
            }
        )
    return summary


def save_target_distributions(
    split_frames: dict[str, pd.DataFrame | None],
    target_cols: list[str],
    target_dist_dir: Path,
) -> pd.DataFrame:
    """Save per-target label distributions and return a summary DataFrame."""
    target_dist_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for target_column in target_cols:
        distribution_frames = []
        for split_name, df in split_frames.items():
            if df is None or target_column not in df.columns:
                continue
            values = df[target_column].fillna("missing").astype(str)
            counts = values.value_counts(dropna=False).sort_values(ascending=False)
            total = int(counts.sum())
            if total == 0:
                continue
            distribution_frames.append(
                pd.DataFrame(
                    {
                        "target": target_column,
                        "split": split_name,
                        "label": counts.index.astype(str),
                        "count": counts.values,
                        "pct": counts.values / total * 100.0,
                    }
                )
            )
            summary_rows.append(
                {
                    "target": target_column,
                    "split": split_name,
                    "n_classes": int(values.nunique(dropna=False)),
                    "majority_label": str(counts.index[0]),
                    "majority_pct": float(counts.iloc[0] / total * 100.0),
                    "missing_count": int((values == "missing").sum()),
                }
            )
        if distribution_frames:
            pd.concat(distribution_frames, ignore_index=True).to_csv(
                target_dist_dir / f"{target_column}.csv", index=False
            )

    if not summary_rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(summary_rows)
        .sort_values(["target", "split"])
        .reset_index(drop=True)
    )
