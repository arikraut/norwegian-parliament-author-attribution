"""Phase 3 predicted profiling signal extractor.

Applies calibrated profiling classifiers to attribution speeches and writes
probability plus hard-label profiling matrices into attribution
materializations. Inputs are rebuilt in the profiler feature space using
fold/final profiling preprocessors, not attribution preprocessors. See
``models/SVM/README.md`` for the fold mapping, feature-space, and output
contracts.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from data_pipeline.utils import (
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from data_pipeline.split.author_disjointness import (
    assert_author_sets_disjoint,
    load_author_ids_from_csv,
)
from models.SVM.linear_svm_common import FeatureLayout
from models.SVM.training.profiling_selection import profiling_candidate_from_payload


# ── Config loading and validation ─────────────────────────────────────────────


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read a profiling signal extraction TOML config."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _normalize_stage(stage: str) -> str:
    """Normalize the profiling extraction stage selector."""
    stage_name = str(stage).strip().lower()
    if stage_name not in {"dev", "final"}:
        raise ValueError("Profiling signal extraction stage must be one of: dev, final.")
    return stage_name


def resolve_extraction_stage_source(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Return [source] merged with the attribution materialization for one stage."""
    stage_name = _normalize_stage(stage)
    source = config.get("source", {})
    required_source_keys = {
        "attribution_split_name",
        "profiling_split_name",
        "profiling_materialization_name",
        "profiling_experiment_name",
        "profiling_seed",
        "targets",
    }
    missing = sorted(required_source_keys - set(source))
    if missing:
        raise ValueError(
            f"Profiling extraction config is missing required [source] keys: {missing}"
        )
    targets = source.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("source.targets must be a non-empty list of label names.")

    stages = config.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(
            "Profiling extraction config must define [stages.dev] and/or [stages.final]."
        )
    stage_cfg = stages.get(stage_name)
    if not isinstance(stage_cfg, dict):
        available = sorted(str(name) for name in stages)
        raise ValueError(
            f"Profiling extraction config does not define [stages.{stage_name}]. "
            f"Available stages: {available}"
        )
    if "attribution_materialization_name" not in stage_cfg:
        raise ValueError(
            f"Profiling extraction config is missing "
            f"[stages.{stage_name}].attribution_materialization_name."
        )

    resolved_source = dict(source)
    resolved_source["attribution_materialization_name"] = str(
        stage_cfg["attribution_materialization_name"]
    )
    return resolved_source


# ── Resource loading ──────────────────────────────────────────────────────────


def _load_corpus(split_name: str, splits_dir: Path) -> pd.DataFrame:
    """Load the full corpus text table for an attribution split."""
    corpus_path = splits_dir / split_name / "corpus" / "all.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Attribution corpus not found: {corpus_path}. "
            "Run the attribution dev pipeline first."
        )
    corpus = pd.read_csv(corpus_path, dtype={"id_speech": str})
    corpus["id_speech"] = corpus["id_speech"].astype(str)
    return corpus[["id_speech", "text"]].copy()


def _read_target_feature_layout(
    profiling_results_dir: Path,
    profiling_split_name: str,
    profiling_experiment_name: str,
    profiling_seed: int,
    target: str,
) -> FeatureLayout:
    """Read the selected feature layout from a target's best_candidate.json."""
    best_candidate_path = (
        profiling_results_dir
        / profiling_split_name
        / profiling_experiment_name
        / f"seed_{profiling_seed}"
        / target
        / "best_candidate.json"
    )
    if not best_candidate_path.exists():
        raise FileNotFoundError(
            f"best_candidate.json not found for target '{target}': {best_candidate_path}. "
            "Run: python -m models.SVM.training.train_profiling_classifiers --config <config>"
    )
    payload = json.loads(best_candidate_path.read_text(encoding="utf-8"))
    return profiling_candidate_from_payload(payload).feature_layout


def _load_profiling_resources(
    profiling_materialized_root: Path,
    profiling_artifacts_dir: Path,
    profiling_results_dir: Path,
    profiling_split_name: str,
    profiling_experiment_name: str,
    profiling_seed: int,
    profiling_fold_id: str,
    targets: list[str],
) -> dict[str, Any]:
    """Load preprocessors, calibrated models, and per-target candidate info for one fold.

    Loads only the preprocessors required by the targets in this call.
    For fold units, preprocessors come from the profiling materialization fold.
    For the final unit (profiling_fold_id == "final"), preprocessors come from
    the final training artifacts directory.

    Returns a dict with keys:
        ``char_vectorizer``  — TfidfVectorizer or None
        ``word_vectorizer``  — TfidfVectorizer or None
        ``stylo_scaler``     — StandardScaler or None
        ``stylo_columns``    — list[str] of column names expected by the scaler
        ``models``           — dict target → CalibratedClassifierCV
        ``target_blocks``    — dict target → list[str]
        ``target_layouts``   — dict target → FeatureLayout
    """
    # Read per-target feature layouts to know which blocks are needed.
    target_layouts: dict[str, FeatureLayout] = {}
    for target in targets:
        target_layouts[target] = _read_target_feature_layout(
            profiling_results_dir,
            profiling_split_name,
            profiling_experiment_name,
            profiling_seed,
            target,
        )

    needed_blocks: set[str] = set()
    for layout in target_layouts.values():
        needed_blocks.update(layout.blocks)

    # Resolve the directory where preprocessors live.
    if profiling_fold_id == "final":
        preprocessors_dir = profiling_artifacts_dir / "final"
        if not preprocessors_dir.exists():
            raise FileNotFoundError(
                f"Final profiling artifacts not found in {preprocessors_dir}. "
                "Run: python -m models.SVM.training.train_profiling_classifiers --config <config> --final"
            )
    else:
        preprocessors_dir = profiling_materialized_root / profiling_fold_id / "preprocessors"

    # Load preprocessors for each needed block.
    char_vectorizer = None
    word_vectorizer = None
    stylo_scaler = None
    stylo_columns: list[str] = []

    if "char" in needed_blocks:
        char_path = preprocessors_dir / "char_vectorizer.joblib"
        if not char_path.exists():
            raise FileNotFoundError(f"char_vectorizer not found: {char_path}")
        char_vectorizer = joblib.load(char_path)

    if "word" in needed_blocks:
        word_path = preprocessors_dir / "word_vectorizer.joblib"
        if not word_path.exists():
            raise FileNotFoundError(f"word_vectorizer not found: {word_path}")
        word_vectorizer = joblib.load(word_path)

    if "stylo" in needed_blocks:
        scaler_path = preprocessors_dir / "stylo_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"stylo_scaler not found: {scaler_path}. "
                "The profiling materialization must include the 'stylo' block."
            )
        stylo_scaler = joblib.load(scaler_path)

        # Column names are stored differently for folds vs. final.
        if profiling_fold_id == "final":
            columns_path = preprocessors_dir / "stylo_columns.json"
            if not columns_path.exists():
                raise FileNotFoundError(f"stylo_columns.json not found: {columns_path}")
            stylo_columns = json.loads(columns_path.read_text(encoding="utf-8"))["columns"]
        else:
            # Fold: column names are in feature_columns.json under key "stylo".
            fc_path = profiling_materialized_root / profiling_fold_id / "feature_columns.json"
            if not fc_path.exists():
                raise FileNotFoundError(f"feature_columns.json not found: {fc_path}")
            fc = json.loads(fc_path.read_text(encoding="utf-8"))
            stylo_columns = fc.get("stylo", [])

    # Load calibrated models.
    target_models: dict[str, Any] = {}
    for target in targets:
        if profiling_fold_id == "final":
            model_path = profiling_artifacts_dir / target / "models" / "final" / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Final profiling model not found: {model_path}. "
                    "Run: python -m models.SVM.training.train_profiling_classifiers --config <config> --final"
                )
        else:
            model_path = profiling_artifacts_dir / target / "models" / profiling_fold_id / "model.joblib"
        target_models[target] = joblib.load(model_path)

    return {
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "stylo_scaler": stylo_scaler,
        "stylo_columns": stylo_columns,
        "models": target_models,
        "target_blocks": {
            target: list(layout.blocks) for target, layout in target_layouts.items()
        },
        "target_layouts": target_layouts,
    }


def _resolve_profiling_fold(
    attribution_unit_id: str,
    available_profiling_folds: list[str],
    *,
    stage: str,
    eval_role: str,
) -> str:
    """Return the profiling resource ID for one attribution unit.

    Development extraction must use the matching profiling fold. Final extraction
    uses the final profiling bundle for test units because final attribution
    materializations do not have fold ids matching profiling folds.
    """
    stage_name = _normalize_stage(stage)
    if stage_name == "dev":
        if attribution_unit_id in available_profiling_folds:
            return attribution_unit_id
        available = ", ".join(sorted(available_profiling_folds))
        raise ValueError(
            "Dev profiling signal extraction requires each attribution unit id "
            f"to match a profiling fold. Got {attribution_unit_id!r}; "
            f"available profiling folds: {available}."
        )

    if str(eval_role).strip().lower() != "test":
        raise ValueError(
            "Final profiling signal extraction can use the final profiling bundle "
            f"only for test units. Got {attribution_unit_id!r} with eval_role={eval_role!r}."
        )
    return "final"


# ── Feature matrix construction ───────────────────────────────────────────────


def _texts_for_role(
    attribution_unit_dir: Path,
    role: str,
    corpus: pd.DataFrame,
) -> list[str]:
    """Return speech texts in the exact row order of the attribution matrices.

    Reads ``{role}_rows.csv`` (which records the row ordering used when the
    attribution matrices were built) and joins on ``id_speech`` with the corpus
    to retrieve text. The returned list has one entry per matrix row.
    """
    row_order_path = attribution_unit_dir / "row_order" / f"{role}_rows.csv"
    if not row_order_path.exists():
        raise FileNotFoundError(f"Row-order file not found: {row_order_path}")

    row_order = pd.read_csv(row_order_path, dtype={"id_speech": str})
    row_order["id_speech"] = row_order["id_speech"].astype(str)

    merged = row_order.merge(corpus, on="id_speech", how="left")
    if merged["text"].isna().any():
        missing = int(merged["text"].isna().sum())
        raise ValueError(
            f"Failed to retrieve text for {missing} row(s) in "
            f"{row_order_path}. Check that the corpus CSV is complete."
        )
    return merged["text"].astype(str).tolist()


def _load_row_order(attribution_unit_dir: Path, role: str) -> pd.DataFrame:
    """Load the row-order CSV for a given unit and role."""
    row_order_path = attribution_unit_dir / "row_order" / f"{role}_rows.csv"
    if not row_order_path.exists():
        raise FileNotFoundError(f"Row-order file not found: {row_order_path}")
    df = pd.read_csv(row_order_path, dtype={"id_speech": str})
    df["id_speech"] = df["id_speech"].astype(str)
    return df


def _align_stylometry(
    raw_stylo: pd.DataFrame,
    row_order: pd.DataFrame,
    stylo_columns: list[str],
) -> np.ndarray:
    """Align raw stylometry to a row-order frame and return selected columns as an array.

    Merges row_order (id_speech in correct position order) with raw_stylo,
    selects only the columns in stylo_columns, and returns a float array.
    Rows missing from raw_stylo receive NaN, which the caller's scaler will
    handle according to the scaler's own behaviour.
    """
    merged = row_order[["id_speech"]].merge(raw_stylo, on="id_speech", how="left")
    return merged[stylo_columns].to_numpy(dtype=float)


def _build_target_input_matrix(
    texts: list[str],
    raw_stylo_aligned: np.ndarray | None,
    resources: dict[str, Any],
    layout: FeatureLayout,
) -> sparse.csr_matrix:
    """Build the feature matrix expected by one target's profiling model.

    Applies per-block normalization, optional block weighting, and final row
    normalization in the same order as _build_feature_matrix from attribution
    training.

    For the stylo block, raw_stylo_aligned must be a pre-aligned (n_texts,
    n_stylo_cols) array already column-selected to match the profiling scaler's
    expected inputs (done by _align_stylometry). resources["stylo_scaler"] must
    be populated.
    """
    matrices: list[sparse.csr_matrix] = []
    for block in layout.blocks:
        if block == "char":
            mat: sparse.csr_matrix = resources["char_vectorizer"].transform(texts)
        elif block == "word":
            mat = resources["word_vectorizer"].transform(texts)
        elif block == "stylo":
            x_scaled = resources["stylo_scaler"].transform(raw_stylo_aligned)
            mat = sparse.csr_matrix(x_scaled)
        else:
            raise ValueError(
                f"Unsupported profiling feature block: {block!r}. "
                "Supported blocks are 'char', 'word', and 'stylo'."
            )

        if layout.normalize_each_block:
            mat = normalize(mat, norm="l2", axis=1, copy=False)
        weight = float(layout.block_weights.get(block, 1.0))
        if weight != 1.0:
            mat = mat.multiply(weight).tocsr()
        matrices.append(mat)

    combined = (
        sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]
    )
    if layout.normalize_rows:
        combined = normalize(combined, norm="l2", axis=1, copy=False)
    return combined.tocsr()


def _hard_label_matrix(y_proba: np.ndarray) -> sparse.csr_matrix:
    """Convert profiler probabilities into sparse one-hot predicted labels.

    Phase 3 hard-label ablations must use the exact same profiler outputs as the
    probability representation. Taking ``argmax`` here keeps the representation
    change controlled while preserving the profiler class column order.
    """
    predicted_cols = np.argmax(y_proba, axis=1)
    row_indices = np.arange(y_proba.shape[0])
    data = np.ones(y_proba.shape[0], dtype=np.float32)
    return sparse.csr_matrix(
        (data, (row_indices, predicted_cols)),
        shape=y_proba.shape,
    )


# ── Public entry point ────────────────────────────────────────────────────────


def run_profiling_signal_extraction(
    config_path: Path,
    *,
    stage: str = "dev",
    show_progress: bool = False,
) -> dict[str, Any]:
    """Extract predicted profiling features for all attribution fold units.

    Writes probability matrices and matching one-hot hard-label matrices into
    each attribution fold's ``matrices/`` directory. Also writes
    ``profiling_block_meta.json`` per unit and a top-level
    ``profiling_extraction_manifest.json`` in the attribution materialized root.

    Parameters
    ----------
    config_path:
        Path to a profiling signal extraction config TOML file.
    stage:
        Attribution materialization stage to extract onto.
    show_progress:
        Whether to print stage logs.

    Returns
    -------
    dict
        Top-level extraction manifest.
    """
    project_root = find_project_root(
        config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent
    )
    config = _load_config(config_path)
    stage_name = _normalize_stage(stage)

    data_cfg = config.get("data", {})
    source_cfg = resolve_extraction_stage_source(config, stage=stage_name)

    splits_dir = resolve_project_path(project_root, data_cfg.get("splits_dir", "data/splits"))
    artifacts_root = resolve_project_path(project_root, data_cfg.get("artifacts_dir", "models/artifacts/profiling"))
    profiling_results_dir = resolve_project_path(project_root, data_cfg.get("profiling_results_dir", "results/models"))

    attribution_split_name = str(source_cfg["attribution_split_name"])
    attribution_mat_name = str(source_cfg["attribution_materialization_name"])
    profiling_split_name = str(source_cfg["profiling_split_name"])
    profiling_mat_name = str(source_cfg["profiling_materialization_name"])
    profiling_experiment_name = str(source_cfg["profiling_experiment_name"])
    profiling_seed = int(source_cfg["profiling_seed"])
    targets: list[str] = list(source_cfg["targets"])

    attribution_materialized_root = (
        splits_dir / attribution_split_name / "materialized_features" / attribution_mat_name
    )
    profiling_materialized_root = (
        splits_dir / profiling_split_name / "materialized_features" / profiling_mat_name
    )
    profiling_artifacts_dir = (
        artifacts_root / profiling_split_name / profiling_experiment_name / f"seed_{profiling_seed}"
    )
    attribution_author_ids = load_author_ids_from_csv(
        splits_dir / attribution_split_name / "authors.csv",
        label=f"attribution split {attribution_split_name}",
    )
    profiling_author_ids = load_author_ids_from_csv(
        splits_dir / profiling_split_name / "authors.csv",
        label=f"profiling split {profiling_split_name}",
    )
    assert_author_sets_disjoint(
        attribution_author_ids,
        profiling_author_ids,
        left_label=f"attribution split {attribution_split_name}",
        right_label=f"profiling split {profiling_split_name}",
    )
    author_disjointness = {
        "attribution_author_count": len(attribution_author_ids),
        "profiling_author_count": len(profiling_author_ids),
        "overlap_count": 0,
    }

    for path, label in [
        (attribution_materialized_root / "manifest.json", "attribution manifest"),
        (profiling_materialized_root / "manifest.json", "profiling manifest"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required {label} not found: {path}. "
                "Check that the relevant pipeline stage has completed."
            )

    attribution_manifest = json.loads(
        (attribution_materialized_root / "manifest.json").read_text(encoding="utf-8")
    )
    attribution_units = attribution_manifest.get("units", [])
    if not attribution_units:
        raise ValueError(f"No units found in attribution manifest: {attribution_materialized_root}")

    profiling_manifest = json.loads(
        (profiling_materialized_root / "manifest.json").read_text(encoding="utf-8")
    )
    available_profiling_folds = [
        str(u["unit_id"]) for u in profiling_manifest.get("units", [])
    ]
    if not available_profiling_folds:
        raise ValueError(
            f"No profiling fold units found in: {profiling_materialized_root}"
        )

    corpus = _load_corpus(attribution_split_name, splits_dir)

    # Determine if any target needs stylometry; if so, load raw attribution stylometry.
    # We load it once outside the unit loop to avoid repeated disk reads.
    # Raw stylometry comes from the attribution split's row_features directory.
    # The attribution materialization manifest has the row_feature_name to locate it.
    attribution_row_feature_name = str(attribution_manifest.get("row_feature_name", ""))
    raw_stylo_df: pd.DataFrame | None = None

    if show_progress:
        print(
            f"Profiling signal extraction ({stage_name}): {len(attribution_units)} attribution unit(s), "
            f"{len(targets)} target(s), {len(available_profiling_folds)} profiling fold(s) available."
        )

    unit_summaries: list[dict[str, Any]] = []
    column_names: list[str] = []  # set on first unit; consistent across units guaranteed below

    for unit in attribution_units:
        unit_id = str(unit["unit_id"])
        eval_role = str(unit["eval_role"])
        unit_dir = attribution_materialized_root / unit_id

        profiling_fold_id = _resolve_profiling_fold(
            unit_id,
            available_profiling_folds,
            stage=stage_name,
            eval_role=eval_role,
        )

        if show_progress:
            print(f"[extract] {unit_id} → profiling fold {profiling_fold_id}")

        resources = _load_profiling_resources(
            profiling_materialized_root,
            profiling_artifacts_dir,
            profiling_results_dir,
            profiling_split_name,
            profiling_experiment_name,
            profiling_seed,
            profiling_fold_id,
            targets,
        )

        # Lazy-load raw attribution stylometry if any target in this unit needs it.
        # Only load once; reuse across units since it covers the full attribution corpus.
        if resources["stylo_scaler"] is not None and raw_stylo_df is None:
            stylo_raw_path = (
                splits_dir
                / attribution_split_name
                / "row_features"
                / attribution_row_feature_name
                / "stylometry_raw.csv.gz"
            )
            if not stylo_raw_path.exists():
                raise FileNotFoundError(
                    f"Attribution raw stylometry not found: {stylo_raw_path}. "
                    "Run the attribution feature pipeline with stylometry enabled first."
                )
            raw_stylo_df = pd.read_csv(stylo_raw_path, dtype={"id_speech": str})
            raw_stylo_df["id_speech"] = raw_stylo_df["id_speech"].astype(str)

        train_texts = _texts_for_role(unit_dir, "train", corpus)
        eval_texts = _texts_for_role(unit_dir, eval_role, corpus)

        # Build per-role aligned stylometry if needed.
        train_raw_stylo: np.ndarray | None = None
        eval_raw_stylo: np.ndarray | None = None
        if resources["stylo_scaler"] is not None:
            stylo_cols = resources["stylo_columns"]
            train_row_order = _load_row_order(unit_dir, "train")
            eval_row_order = _load_row_order(unit_dir, eval_role)
            train_raw_stylo = _align_stylometry(raw_stylo_df, train_row_order, stylo_cols)
            eval_raw_stylo = _align_stylometry(raw_stylo_df, eval_row_order, stylo_cols)

        train_per_target: dict[str, sparse.csr_matrix] = {}
        eval_per_target: dict[str, sparse.csr_matrix] = {}
        train_hard_per_target: dict[str, sparse.csr_matrix] = {}
        eval_hard_per_target: dict[str, sparse.csr_matrix] = {}
        fold_column_names: list[str] = []

        for target in targets:
            layout = resources["target_layouts"][target]
            model = resources["models"][target]

            x_train = _build_target_input_matrix(
                train_texts, train_raw_stylo, resources, layout
            )
            x_eval = _build_target_input_matrix(
                eval_texts, eval_raw_stylo, resources, layout
            )

            train_proba = model.predict_proba(x_train)
            eval_proba = model.predict_proba(x_eval)

            train_per_target[target] = sparse.csr_matrix(train_proba)
            eval_per_target[target] = sparse.csr_matrix(eval_proba)
            train_hard_per_target[target] = _hard_label_matrix(train_proba)
            eval_hard_per_target[target] = _hard_label_matrix(eval_proba)

            classes = [str(c) for c in model.classes_]
            fold_column_names.extend(f"{target}_{cls}" for cls in classes)

        # Stack per-target probability columns into one combined matrix.
        train_combined = sparse.hstack(
            [train_per_target[t] for t in targets], format="csr"
        )
        eval_combined = sparse.hstack(
            [eval_per_target[t] for t in targets], format="csr"
        )
        train_hard_combined = sparse.hstack(
            [train_hard_per_target[t] for t in targets], format="csr"
        )
        eval_hard_combined = sparse.hstack(
            [eval_hard_per_target[t] for t in targets], format="csr"
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
        sparse.save_npz(matrices_dir / "X_train_profiling.npz", train_combined)
        sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling.npz", eval_combined)
        sparse.save_npz(
            matrices_dir / "X_train_profiling_hard.npz",
            train_hard_combined,
        )
        sparse.save_npz(
            matrices_dir / f"X_{eval_role}_profiling_hard.npz",
            eval_hard_combined,
        )
        for target in targets:
            sparse.save_npz(
                matrices_dir / f"X_train_profiling_{target}.npz",
                train_per_target[target],
            )
            sparse.save_npz(
                matrices_dir / f"X_{eval_role}_profiling_{target}.npz",
                eval_per_target[target],
            )
            sparse.save_npz(
                matrices_dir / f"X_train_profiling_hard_{target}.npz",
                train_hard_per_target[target],
            )
            sparse.save_npz(
                matrices_dir / f"X_{eval_role}_profiling_hard_{target}.npz",
                eval_hard_per_target[target],
            )

        cal_metas: dict[str, Any] = {}
        for target in targets:
            meta_path = (
                profiling_artifacts_dir / target / "models" / profiling_fold_id / "calibration_meta.json"
            )
            if meta_path.exists():
                cal_metas[target] = json.loads(meta_path.read_text(encoding="utf-8"))

        unit_meta = {
            "unit_id": unit_id,
            "eval_role": eval_role,
            "profiling_fold_id": profiling_fold_id,
            "train_rows": train_combined.shape[0],
            "eval_rows": eval_combined.shape[0],
            "profiling_dim": train_combined.shape[1],
            "hard_profiling_dim": train_hard_combined.shape[1],
            "targets": targets,
            "calibration_meta": cal_metas,
            "train_matrix_path": relative_to_project(
                project_root, matrices_dir / "X_train_profiling.npz"
            ),
            "eval_matrix_path": relative_to_project(
                project_root, matrices_dir / f"X_{eval_role}_profiling.npz"
            ),
            "train_hard_matrix_path": relative_to_project(
                project_root, matrices_dir / "X_train_profiling_hard.npz"
            ),
            "eval_hard_matrix_path": relative_to_project(
                project_root, matrices_dir / f"X_{eval_role}_profiling_hard.npz"
            ),
        }
        write_json(unit_dir / "profiling_block_meta.json", unit_meta)
        unit_summaries.append(unit_meta)

        if show_progress:
            print(
                f"[extract] {unit_id}: saved "
                f"({train_combined.shape[0]}×{train_combined.shape[1]}) train, "
                f"({eval_combined.shape[0]}×{eval_combined.shape[1]}) {eval_role}"
            )

    # Register derived profiling blocks in the attribution materialization manifest
    # so that _validate_variant_availability in the attribution trainers can find them.
    derived_blocks = (
        ["profiling"]
        + [f"profiling_{target}" for target in targets]
        + ["profiling_hard"]
        + [f"profiling_hard_{target}" for target in targets]
    )
    attribution_manifest_path = attribution_materialized_root / "manifest.json"
    attribution_manifest_loaded = json.loads(attribution_manifest_path.read_text(encoding="utf-8"))
    processed_unit_ids = {str(s["unit_id"]) for s in unit_summaries}
    for manifest_unit in attribution_manifest_loaded.get("units", []):
        if str(manifest_unit.get("unit_id")) in processed_unit_ids:
            existing = list(manifest_unit.get("derived_blocks", []))
            for block in derived_blocks:
                if block not in existing:
                    existing.append(block)
            manifest_unit["derived_blocks"] = existing
    write_json(attribution_manifest_path, attribution_manifest_loaded)

    if show_progress:
        print(
            f"[extract] Registered derived_blocks {derived_blocks} "
            f"in {attribution_manifest_path.relative_to(project_root)}"
        )

    # Write column names to the materialized root for downstream inspection.
    write_json(
        attribution_materialized_root / "profiling_feature_columns.json",
        {"columns": column_names, "targets": targets},
    )
    write_json(
        attribution_materialized_root / "profiling_hard_feature_columns.json",
        {"columns": column_names, "targets": targets},
    )

    manifest = {
        "config_path": relative_to_project(project_root, config_path),
        "stage": stage_name,
        "attribution_split_name": attribution_split_name,
        "attribution_materialization_name": attribution_mat_name,
        "profiling_split_name": profiling_split_name,
        "profiling_materialization_name": profiling_mat_name,
        "profiling_experiment_name": profiling_experiment_name,
        "profiling_seed": profiling_seed,
        "representations": ["probability", "hard"],
        "targets": targets,
        "profiling_dim": len(column_names),
        "hard_profiling_dim": len(column_names),
        "column_names": column_names,
        "author_disjointness": author_disjointness,
        "units": unit_summaries,
    }
    write_json(
        attribution_materialized_root / "profiling_extraction_manifest.json",
        manifest,
    )

    if show_progress:
        print(
            f"Profiling signal extraction complete. "
            f"{len(unit_summaries)} unit(s), {len(column_names)} probability/hard column(s)."
        )

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse the profiling signal extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract predicted profiling features for attribution fold units.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a profiling signal extraction config TOML file.",
    )
    parser.add_argument(
        "--stage",
        choices=["dev", "final"],
        default="dev",
        help="Attribution materialization stage to extract onto.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stage logs.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for profiling signal extraction."""
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)
    run_profiling_signal_extraction(
        config_path,
        stage=args.stage,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
