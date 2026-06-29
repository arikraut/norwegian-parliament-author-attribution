"""Shared split config loading and normalized corpus context for scripts and notebooks."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from data_pipeline.utils import resolve_project_path

REQUIRED_SPLIT_SECTIONS = ("split", "data", "outer_split", "pool", "selection")


def normalize_split_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the core split-stage numeric columns from a cleaned corpus."""
    df = df.copy()
    df["election"] = pd.to_numeric(df["election"], errors="raise").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")
    df["char_count"] = pd.to_numeric(df["char_count"], errors="coerce")
    return df


def load_split_run_context(
    project_root: Path,
    *,
    config_path: str | Path,
) -> dict:
    """Load one split config plus its normalized source corpus."""
    project_root = Path(project_root).resolve()
    config_path = resolve_project_path(project_root, config_path).resolve()

    with config_path.open("rb") as handle:
        split_config = tomllib.load(handle)

    missing_sections = [section for section in REQUIRED_SPLIT_SECTIONS if section not in split_config]
    if missing_sections:
        raise KeyError(f"Missing required config sections: {missing_sections}")

    split_meta = split_config["split"]
    data_cfg = split_config["data"]
    outer_split_cfg = split_config["outer_split"]
    pool_cfg = split_config["pool"]
    selection_cfg = split_config["selection"]
    folds_cfg = split_config.get("folds", {"mode": "none"})

    strategy = str(outer_split_cfg["strategy"]).lower()
    split_name = str(split_meta["name"])
    experiment_name = str(split_meta["experiment_name"])
    selection_seed = int(split_meta["selection_seed"])

    source_dataset_path = resolve_project_path(project_root, data_cfg["source_dataset"]).resolve()
    splits_root = resolve_project_path(project_root, data_cfg.get("splits_dir", "data/splits")).resolve()
    results_root = resolve_project_path(project_root, data_cfg.get("results_dir", "results/splits")).resolve()
    split_dir = splits_root / split_name
    results_dir = results_root / split_name

    df = normalize_split_corpus(pd.read_csv(source_dataset_path))

    return {
        "project_root": project_root,
        "config_path": config_path,
        "split_config": split_config,
        "split_meta": split_meta,
        "data_cfg": data_cfg,
        "outer_split_cfg": outer_split_cfg,
        "pool_cfg": pool_cfg,
        "selection_cfg": selection_cfg,
        "folds_cfg": folds_cfg,
        "split_name": split_name,
        "experiment_name": experiment_name,
        "selection_seed": selection_seed,
        "source_dataset_path": source_dataset_path,
        "splits_root": splits_root,
        "results_root": results_root,
        "split_dir": split_dir,
        "corpus_dir": split_dir / "corpus",
        "results_dir": results_dir,
        "outer_split_strategy": strategy,
        "df": df,
    }
