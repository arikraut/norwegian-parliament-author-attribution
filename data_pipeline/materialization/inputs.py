"""Input loading and row-source alignment for materialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.materialization.config import (
    _parse_enabled_blocks,
    _read_required_json,
    _resolve_materialization_stage,
)
from data_pipeline.utils import read_required_csv


def _select_new_columns(
    source_df: pd.DataFrame,
    existing_cols: set[str],
    *,
    candidate_cols: list[str] | None = None,
    excluded_cols: set[str] | None = None,
) -> list[str]:
    """Build the list of new columns to pull from source_df."""
    excluded = excluded_cols or set()
    source_cols = candidate_cols if candidate_cols is not None else list(source_df.columns)
    cols = [
        col
        for col in source_cols
        if col in source_df.columns
        and col not in excluded
        and (col == "id_speech" or col not in existing_cols)
    ]
    if "id_speech" not in cols and "id_speech" in source_df.columns:
        cols.insert(0, "id_speech")
    return cols


def _ensure_unique(df: pd.DataFrame, column: str, name: str) -> None:
    """Raise ValueError if column contains duplicated values."""
    dup = df[column].duplicated()
    if dup.any():
        raise ValueError(
            f"{name} contains duplicated {column} values ({int(dup.sum())} duplicates)."
        )


def _ensure_membership_ids_have_targets(
    membership: pd.DataFrame,
    row_targets: pd.DataFrame,
) -> None:
    """Fail when matrix rows would be materialized without target labels."""
    missing_mask = ~membership["id_speech"].isin(row_targets["id_speech"])
    if missing_mask.any():
        missing_ids = (
            membership.loc[missing_mask, "id_speech"].head(10).astype(str).tolist()
        )
        sample = ", ".join(missing_ids)
        raise ValueError(
            "targets.csv is missing rows for "
            f"{int(missing_mask.sum())} materialization membership id_speech value(s). "
            f"Sample: {sample}"
        )


def _load_materialization_inputs(
    project_root: Path, config_path: Path, *, stage: str
) -> dict:
    """Parse the config and load all split-level inputs it references."""
    resolved_stage = _resolve_materialization_stage(project_root, config_path, stage=stage)
    config = resolved_stage.config

    enabled_blocks = _parse_enabled_blocks(config)
    split_name = resolved_stage.split_name
    row_feature_name = resolved_stage.row_feature_name
    materialization_name = resolved_stage.materialization_name

    split_dir = resolved_stage.materialized_root.parents[1]
    corpus_dir = split_dir / "corpus"
    memberships_dir = split_dir / "memberships"
    row_feature_dir = split_dir / "row_features" / row_feature_name
    materialized_root = resolved_stage.materialized_root
    split_manifest = _read_required_json(split_dir / "manifest.json")
    split_strategy = str(split_manifest.get("split_strategy", "")).strip()
    if not split_strategy:
        raise ValueError(
            f"Split manifest at {split_dir / 'manifest.json'} must include split_strategy."
        )
    row_feature_manifest = _read_required_json(row_feature_dir / "manifest.json")

    corpus_all = read_required_csv(corpus_dir / "all.csv")
    outer_membership = read_required_csv(memberships_dir / "outer.csv")
    folds_path = memberships_dir / "folds.csv"
    fold_membership = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    row_meta = read_required_csv(row_feature_dir / "row_meta.csv")
    row_targets = read_required_csv(row_feature_dir / "targets.csv")
    if (
        "stylo" in enabled_blocks
        and row_feature_manifest.get("stylometry", {}).get("generated") is not True
    ):
        raise ValueError(
            f"Materialization '{materialization_name}' requests stylo, but "
            f"row-feature bundle '{row_feature_name}' did not generate stylometry."
        )
    row_stylo = (
        read_required_csv(row_feature_dir / "stylometry_raw.csv.gz")
        if "stylo" in enabled_blocks
        else None
    )

    _ensure_unique(corpus_all, "id_speech", "corpus/all.csv")
    _ensure_unique(row_meta, "id_speech", "row_meta.csv")
    _ensure_unique(row_targets, "id_speech", "targets.csv")
    if row_stylo is not None:
        _ensure_unique(row_stylo, "id_speech", "stylometry_raw.csv.gz")

    return {
        "config": config,
        "stage": resolved_stage.stage,
        "selector": resolved_stage.selector,
        "enabled_blocks": enabled_blocks,
        "resolved_config_path": resolved_stage.resolved_config_path,
        "split_name": split_name,
        "row_feature_name": row_feature_name,
        "materialization_name": materialization_name,
        "split_strategy": split_strategy,
        "corpus_dir": corpus_dir,
        "row_feature_dir": row_feature_dir,
        "materialized_root": materialized_root,
        "corpus_all": corpus_all,
        "outer_membership": outer_membership,
        "fold_membership": fold_membership,
        "row_meta": row_meta,
        "row_targets": row_targets,
        "row_stylo": row_stylo,
    }


def _merge_row_sources(
    membership: pd.DataFrame,
    corpus_all: pd.DataFrame,
    row_meta: pd.DataFrame,
    row_targets: pd.DataFrame,
    row_stylo: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join corpus text, metadata, targets, and stylometry onto a membership frame."""
    _ensure_membership_ids_have_targets(membership, row_targets)

    corpus_cols = _select_new_columns(
        corpus_all,
        set(membership.columns),
        candidate_cols=[
            "id_speech",
            "id_person",
            "text",
            "election",
            "word_count",
            "char_count",
        ],
    )
    corpus_frame = corpus_all[corpus_cols].copy()
    merged = membership.merge(corpus_frame, on="id_speech", how="left", sort=False)

    meta_cols = _select_new_columns(
        row_meta, set(merged.columns), excluded_cols={"outer_role"}
    )
    meta_frame = row_meta[meta_cols].copy()
    if len(meta_cols) > 1:
        merged = merged.merge(meta_frame, on="id_speech", how="left", sort=False)

    target_cols = _select_new_columns(
        row_targets, set(merged.columns), excluded_cols={"outer_role"}
    )
    target_frame = row_targets[target_cols].copy()
    if len(target_cols) > 1:
        merged = merged.merge(target_frame, on="id_speech", how="left", sort=False)

    if row_stylo is not None:
        stylo_cols = _select_new_columns(
            row_stylo, set(merged.columns), excluded_cols={"outer_role"}
        )
        stylo_frame = row_stylo[stylo_cols].copy()
        if len(stylo_cols) > 1:
            merged = merged.merge(stylo_frame, on="id_speech", how="left", sort=False)

    if merged["text"].isna().any():
        missing = int(merged["text"].isna().sum())
        raise ValueError(
            f"Failed to align text for {missing} rows while materializing features."
        )

    return merged
