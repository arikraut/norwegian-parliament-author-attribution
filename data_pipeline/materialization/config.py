"""Materialization config and stage resolution."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.materialization.constants import _SUPPORTED_MATERIALIZATION_BLOCKS
from data_pipeline.utils import resolve_project_path


@dataclass
class MaterializationUnit:
    unit_id: str
    eval_role: str
    membership: pd.DataFrame


@dataclass(frozen=True)
class MaterializationStage:
    config: dict
    config_path: Path
    stage: str
    selector: str
    split_name: str
    row_feature_name: str
    materialization_name: str
    materialized_root: Path
    resolved_config_path: Path


def _parse_enabled_blocks(config: dict) -> tuple[str, ...]:
    """Parse and validate the required [blocks].enabled list from a materialization config."""
    raw_blocks = config.get("blocks", {}).get("enabled")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("blocks.enabled must be a non-empty list.")

    normalized = [str(block).strip().lower() for block in raw_blocks]
    unknown = sorted(set(normalized) - set(_SUPPORTED_MATERIALIZATION_BLOCKS))
    if unknown:
        raise ValueError(f"Unsupported materialization blocks: {unknown}")
    return tuple(
        block for block in _SUPPORTED_MATERIALIZATION_BLOCKS if block in normalized
    )


def _read_required_json(path: Path) -> dict:
    """Read and parse a required JSON file, raising FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(config_path: Path) -> dict:
    """Read a materialization TOML config from disk."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_materialization_stage(
    project_root: Path,
    config_path: Path,
    *,
    stage: str,
) -> MaterializationStage:
    """Resolve one materialization stage into concrete split and output paths."""
    stage_name = str(stage).strip().lower()
    if stage_name not in {"dev", "final"}:
        raise ValueError("Materialization stage must be 'dev' or 'final'.")

    config = _load_toml(config_path)
    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(
            "Materialization config must define [stages.dev] and/or [stages.final]."
        )
    stage_cfg = stages.get(stage_name)
    if not isinstance(stage_cfg, dict):
        available = sorted(str(name) for name in stages)
        raise ValueError(
            f"Materialization stage {stage_name!r} is not defined. Available stages: {available}"
        )

    materialization_meta = config.get("materialization")
    if not isinstance(materialization_meta, dict):
        raise ValueError("Materialization config must define [materialization].")
    required_materialization = {"split_name", "row_feature_name"}
    missing_materialization = sorted(
        required_materialization - set(materialization_meta)
    )
    if missing_materialization:
        raise ValueError(
            "Materialization config is missing required [materialization] keys: "
            f"{missing_materialization}"
        )

    required_stage = {"name", "selector"}
    missing_stage = sorted(required_stage - set(stage_cfg))
    if missing_stage:
        raise ValueError(
            f"Materialization stage {stage_name!r} is missing required keys: {missing_stage}"
        )

    selector = str(stage_cfg["selector"]).strip().lower()
    if selector not in {"all", "final"}:
        raise ValueError(
            f"Unsupported materialization selector for stage {stage_name!r}: {selector!r}"
        )
    if stage_name == "dev" and selector != "all":
        raise ValueError("Materialization stage 'dev' must use selector = 'all'.")
    if stage_name == "final" and selector != "final":
        raise ValueError("Materialization stage 'final' must use selector = 'final'.")

    materialization_name = str(stage_cfg["name"]).strip()
    if not materialization_name:
        raise ValueError(
            f"Materialization stage {stage_name!r} must define a non-empty name."
        )

    data_cfg = dict(config.get("data", {}))
    splits_root = resolve_project_path(
        project_root, data_cfg.get("splits_dir", "data/splits")
    )
    split_name = str(materialization_meta["split_name"])
    row_feature_name = str(materialization_meta["row_feature_name"])
    materialized_root = (
        splits_root / split_name / "materialized_features" / materialization_name
    )
    resolved_config_path = materialized_root / "materialization_config.toml"

    resolved_config = {
        "materialization": {
            "name": materialization_name,
            "split_name": split_name,
            "row_feature_name": row_feature_name,
        },
        "data": data_cfg,
        "folds": {"selector": selector},
        "blocks": config.get("blocks", {}),
    }
    for section in ("word_tfidf", "char_tfidf", "stylometry"):
        if section in config:
            resolved_config[section] = config[section]
    _parse_enabled_blocks(resolved_config)

    return MaterializationStage(
        config=resolved_config,
        config_path=config_path,
        stage=stage_name,
        selector=selector,
        split_name=split_name,
        row_feature_name=row_feature_name,
        materialization_name=materialization_name,
        materialized_root=materialized_root,
        resolved_config_path=resolved_config_path,
    )


def resolve_materialization_stage(
    project_root: Path,
    config_path: Path,
    *,
    stage: str,
) -> dict[str, object]:
    """Resolve a materialization config stage without materializing data."""
    resolved = _resolve_materialization_stage(project_root, config_path, stage=stage)
    return {
        "config": resolved.config,
        "config_path": resolved.config_path,
        "stage": resolved.stage,
        "selector": resolved.selector,
        "split_name": resolved.split_name,
        "row_feature_name": resolved.row_feature_name,
        "materialization_name": resolved.materialization_name,
        "materialized_root": resolved.materialized_root,
        "resolved_config_path": resolved.resolved_config_path,
    }


def _validate_stage_units(units: list[dict], *, stage: str, selector: str) -> None:
    """Check that selected materialization units match the dev/final contract."""
    if not units:
        raise ValueError("No folds or final unit were selected for materialization.")

    eval_roles = [str(unit.get("eval_role", "")) for unit in units]
    if selector == "all":
        invalid = sorted(
            str(unit.get("unit_id", "<missing-unit-id>"))
            for unit in units
            if str(unit.get("eval_role", "")) != "val"
        )
        if invalid:
            raise ValueError(
                f"Materialization stage {stage!r} must expose only validation units with eval_role='val'. "
                f"Invalid units: {invalid}"
            )
        return

    if selector == "final":
        if len(units) != 1 or eval_roles != ["test"]:
            raise ValueError(
                f"Materialization stage {stage!r} must expose exactly one final test unit with eval_role='test'. "
                f"Found roles: {eval_roles}"
            )
        return

    raise ValueError(f"Unsupported materialization selector: {selector}")
