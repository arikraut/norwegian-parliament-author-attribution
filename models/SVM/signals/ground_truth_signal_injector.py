"""Ground-truth oracle signal injector for Phase 3 oracle runs.

Writes one-hot ``profiling_oracle*`` matrices from true profile labels, matching
the row-order and derived-block contract used by predicted profiling signals.
See ``models/SVM/README.md`` for the oracle output contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and return a TOML config file as a plain dict."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _normalize_stage(stage: str) -> str:
    """Validate and normalize a stage string to 'dev' or 'final'."""
    stage_name = str(stage).strip().lower()
    if stage_name not in {"dev", "final"}:
        raise ValueError("Ground truth injection stage must be one of: dev, final.")
    return stage_name


def resolve_injection_stage_source(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Return the resolved [source] merged with the stage-specific materialization name."""
    stage_name = _normalize_stage(stage)
    source = config.get("source", {})
    missing = sorted({"attribution_split_name", "targets"} - set(source))
    if missing:
        raise ValueError(
            f"Ground truth injection config is missing required [source] keys: {missing}"
        )
    targets = source.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("source.targets must be a non-empty list of label names.")

    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(
            "Ground truth injection config must define [stages.dev] and/or [stages.final]."
        )
    stage_cfg = stages.get(stage_name)
    if not isinstance(stage_cfg, dict):
        available = sorted(str(k) for k in stages)
        raise ValueError(
            f"Ground truth injection config does not define [stages.{stage_name}]. "
            f"Available stages: {available}"
        )
    if "attribution_materialization_name" not in stage_cfg:
        raise ValueError(
            f"Ground truth injection config is missing "
            f"[stages.{stage_name}].attribution_materialization_name."
        )

    resolved = dict(source)
    resolved["attribution_materialization_name"] = str(
        stage_cfg["attribution_materialization_name"]
    )
    return resolved


def _one_hot_encode(
    labels: np.ndarray,
    class_ordering: list[str],
) -> sparse.csr_matrix:
    """Return a one-hot sparse matrix aligned to class_ordering.

    Classes not in class_ordering produce an all-zero row (handled silently
    since eval sets may occasionally have authors with no train-set class match,
    but for profiling targets this should not occur).
    """
    class_to_col = {cls: i for i, cls in enumerate(class_ordering)}
    n_rows = len(labels)
    n_cols = len(class_ordering)

    rows: list[int] = []
    cols: list[int] = []
    for row_idx, label in enumerate(labels):
        label_str = str(label)
        col_idx = class_to_col.get(label_str)
        if col_idx is not None:
            rows.append(row_idx)
            cols.append(col_idx)

    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_rows, n_cols),
    )


def run_ground_truth_signal_injection(
    config_path: Path,
    *,
    stage: str = "dev",
    show_progress: bool = False,
) -> dict[str, Any]:
    """Write one-hot oracle profiling matrices for all attribution fold units.

    Reads ground-truth label npy files from each fold's ``labels/`` directory,
    one-hot encodes them, and writes oracle-suffixed sparse matrices into each
    fold's ``matrices/`` directory. Updates the attribution materialization
    manifest's ``derived_blocks`` for each processed unit.

    Parameters
    ----------
    config_path:
        Path to a ground truth signal injection config TOML file.
    stage:
        Attribution materialization stage to inject into.
    show_progress:
        Whether to print stage logs.

    Returns
    -------
    dict
        Injection manifest written to the attribution materialized root.
    """
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    stage_name = _normalize_stage(stage)

    data_cfg = config.get("data", {})
    source_cfg = resolve_injection_stage_source(config, stage=stage_name)

    splits_dir = resolve_project_path(project_root, data_cfg.get("splits_dir", "data/splits"))
    attribution_split_name = str(source_cfg["attribution_split_name"])
    attribution_mat_name = str(source_cfg["attribution_materialization_name"])
    targets: list[str] = list(source_cfg["targets"])

    attribution_materialized_root = (
        splits_dir / attribution_split_name / "materialized_features" / attribution_mat_name
    )
    manifest_path = attribution_materialized_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Attribution manifest not found: {manifest_path}. "
            "Run the attribution materialization pipeline first."
        )

    attribution_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attribution_units = attribution_manifest.get("units", [])
    if not attribution_units:
        raise ValueError(
            f"No units found in attribution manifest: {attribution_materialized_root}"
        )

    if show_progress:
        print(
            f"Ground truth signal injection ({stage_name}): "
            f"{len(attribution_units)} unit(s), {len(targets)} target(s)."
        )

    unit_summaries: list[dict[str, Any]] = []
    column_names: list[str] = []

    for unit in attribution_units:
        unit_id = str(unit["unit_id"])
        eval_role = str(unit["eval_role"])
        unit_dir = attribution_materialized_root / unit_id
        labels_dir = unit_dir / "labels"

        if show_progress:
            print(f"[inject] {unit_id} ({eval_role})")

        train_per_target: dict[str, sparse.csr_matrix] = {}
        eval_per_target: dict[str, sparse.csr_matrix] = {}
        fold_column_names: list[str] = []

        for target in targets:
            train_path = labels_dir / f"y_train_{target}.npy"
            eval_path = labels_dir / f"y_{eval_role}_{target}.npy"

            if not train_path.exists():
                raise FileNotFoundError(
                    f"Ground truth train labels not found: {train_path}. "
                    "Ensure the attribution materialization has run."
                )
            if not eval_path.exists():
                raise FileNotFoundError(
                    f"Ground truth eval labels not found: {eval_path}. "
                    "Ensure the attribution materialization has run."
                )

            y_train = np.load(train_path, allow_pickle=True)
            y_eval = np.load(eval_path, allow_pickle=True)

            class_ordering = sorted(set(str(v) for v in y_train.tolist()))  # sorted for determinism

            train_per_target[target] = _one_hot_encode(y_train, class_ordering)
            eval_per_target[target] = _one_hot_encode(y_eval, class_ordering)

            fold_column_names.extend(f"{target}_{cls}" for cls in class_ordering)

        train_combined = sparse.hstack(
            [train_per_target[t] for t in targets], format="csr"
        )
        eval_combined = sparse.hstack(
            [eval_per_target[t] for t in targets], format="csr"
        )

        if not column_names:
            column_names = fold_column_names
        elif column_names != fold_column_names:
            raise RuntimeError(
                f"Column names changed between units — expected {column_names}, "
                f"got {fold_column_names} for unit {unit_id}."
            )

        matrices_dir = unit_dir / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)

        sparse.save_npz(matrices_dir / "X_train_profiling_oracle.npz", train_combined)
        sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling_oracle.npz", eval_combined)
        for target in targets:
            sparse.save_npz(
                matrices_dir / f"X_train_profiling_oracle_{target}.npz",
                train_per_target[target],
            )
            sparse.save_npz(
                matrices_dir / f"X_{eval_role}_profiling_oracle_{target}.npz",
                eval_per_target[target],
            )

        unit_meta = {
            "unit_id": unit_id,
            "eval_role": eval_role,
            "train_rows": int(train_combined.shape[0]),
            "eval_rows": int(eval_combined.shape[0]),
            "oracle_dim": int(train_combined.shape[1]),
            "targets": targets,
            "train_matrix_path": relative_to_project(
                project_root, matrices_dir / "X_train_profiling_oracle.npz"
            ),
            "eval_matrix_path": relative_to_project(
                project_root, matrices_dir / f"X_{eval_role}_profiling_oracle.npz"
            ),
        }
        unit_summaries.append(unit_meta)

        if show_progress:
            print(
                f"[inject] {unit_id}: saved "
                f"({train_combined.shape[0]}×{train_combined.shape[1]}) train, "
                f"({eval_combined.shape[0]}×{eval_combined.shape[1]}) {eval_role}"
            )

    oracle_blocks = ["profiling_oracle"] + [f"profiling_oracle_{t}" for t in targets]
    processed_unit_ids = {str(s["unit_id"]) for s in unit_summaries}
    for manifest_unit in attribution_manifest.get("units", []):
        if str(manifest_unit.get("unit_id")) in processed_unit_ids:
            existing = list(manifest_unit.get("derived_blocks", []))
            for block in oracle_blocks:
                if block not in existing:
                    existing.append(block)
            manifest_unit["derived_blocks"] = existing
    write_json(manifest_path, attribution_manifest)

    if show_progress:
        print(
            f"[inject] Registered derived_blocks {oracle_blocks} "
            f"in {manifest_path.relative_to(project_root)}"
        )

    write_json(
        attribution_materialized_root / "oracle_feature_columns.json",
        {"columns": column_names, "targets": targets},
    )

    injection_manifest = {
        "config_path": relative_to_project(project_root, config_path),
        "stage": stage_name,
        "attribution_split_name": attribution_split_name,
        "attribution_materialization_name": attribution_mat_name,
        "targets": targets,
        "oracle_dim": len(column_names),
        "column_names": column_names,
        "units": unit_summaries,
    }
    write_json(
        attribution_materialized_root / "ground_truth_injection_manifest.json",
        injection_manifest,
    )

    if show_progress:
        print(
            f"Ground truth injection complete. "
            f"{len(unit_summaries)} unit(s), {len(column_names)} oracle column(s)."
        )

    return injection_manifest


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ground truth signal injection CLI."""
    parser = argparse.ArgumentParser(
        description="Inject ground-truth one-hot profiling features for attribution fold units.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a ground truth signal injection config TOML file.",
    )
    parser.add_argument(
        "--stage",
        choices=["dev", "final"],
        default="dev",
        help="Attribution materialization stage to inject into.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stage logs.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for direct CLI invocation of the ground truth signal injector."""
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)
    run_ground_truth_signal_injection(
        config_path,
        stage=args.stage,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
