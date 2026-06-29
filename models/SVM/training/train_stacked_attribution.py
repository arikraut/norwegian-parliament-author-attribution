"""Stacked attribution trainer.

Trains calibrated family-level ``LinearSVC`` models and a ``LogisticRegression``
top model. Inner CV creates leakage-safe training meta-features; outer-fold
base models score validation or test rows. Optional profiling/oracle blocks are
appended at the top-model level. See ``models/SVM/README.md`` for phase-level
details.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from data_pipeline.utils import (
    create_progress_bar,
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.SVM.training.condition_selection import select_candidates_by_condition
from models.SVM.training.final_condition_outputs import (
    build_final_summary_row,
    final_condition_roots,
    write_final_condition_output,
    write_final_condition_summary,
)
from models.SVM.linear_svm_common import (
    _build_provenance_block,
    _copy_config_outputs_if_available,
    _load_labels,
    _load_materialization_summary,
    _load_sparse_matrix,
    _manifest_config_path,
    _matrix_path,
    _parse_class_weights,
    _prediction_frame,
    _resolve_output_dirs,
    _validate_known_keys,
    _validate_candidate_search_units,
    _validate_final_evaluation_units,
)


SUPPORTED_MODEL_FAMILY = "stacked"


@dataclass(frozen=True)
class StackedFamilySpec:
    name: str
    blocks: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """Serialize a reusable stacked feature family for selection artifacts."""
        return {"name": self.name, "blocks": list(self.blocks)}


@dataclass(frozen=True)
class StackedConditionSpec:
    """Declared experiment condition for stacked attribution model selection."""

    condition_id: str
    label: str
    families: tuple[StackedFamilySpec, ...]
    profiling_blocks: tuple[str, ...]


@dataclass(frozen=True)
class StackedCandidateSpec:
    """Stacked attribution candidate: one condition plus base/top hyperparameters."""

    condition: StackedConditionSpec
    base_c: float
    class_weight: str | None
    top_c: float

    @property
    def condition_id(self) -> str:
        """Return the declared condition id."""
        return self.condition.condition_id

    @property
    def condition_label(self) -> str:
        """Return the declared condition label."""
        return self.condition.label

    @property
    def family_set_name(self) -> str:
        """Return a descriptive family-set label derived from family names."""
        return "_".join(family.name for family in self.condition.families)

    @property
    def families(self) -> tuple[StackedFamilySpec, ...]:
        """Return the feature families used by this condition."""
        return self.condition.families

    @property
    def profiling_blocks(self) -> tuple[str, ...]:
        """Return profiling blocks declared by this condition."""
        return self.condition.profiling_blocks

    @property
    def class_weight_label(self) -> str:
        """Return the artifact label for the class-weight setting."""
        return "none" if self.class_weight is None else str(self.class_weight)

    @property
    def candidate_id(self) -> str:
        """Return a stable ID for one stacked condition and hyperparameter setting."""
        return (
            f"{self.condition_id}__baseC={self.base_c:g}"
            f"__topC={self.top_c:g}"
            f"__class_weight={self.class_weight_label}"
        )

    def to_payload(
        self,
        *,
        selection_metric: str | None = None,
        dev_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize this stacked candidate for selected/final artifacts."""
        payload: dict[str, Any] = {
            "condition_id": self.condition_id,
            "condition_label": self.condition_label,
            "candidate_id": self.candidate_id,
            "family_set": self.family_set_name,
            "families": [family.to_payload() for family in self.families],
            "base_c": self.base_c,
            "class_weight": self.class_weight_label,
            "top_c": self.top_c,
            "profiling_blocks": list(self.profiling_blocks),
        }
        if selection_metric is not None:
            payload["selection_metric"] = selection_metric
        if dev_summary is not None:
            payload["dev_summary"] = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in dev_summary.items()
            }
        return payload


@dataclass(frozen=True)
class ReusableFamilyKey:
    """Identify one reusable stacked base-family fit within an outer unit."""

    family_name: str
    base_c: float
    class_weight: str | None


@dataclass(frozen=True)
class ReusableFamilyOutput:
    """Hold base-family probability outputs reused by stacked candidate scoring."""

    train_meta: np.ndarray
    val_meta: np.ndarray


@dataclass(frozen=True)
class StackedUnitData:
    """Hold all loaded matrices and labels for one stacked outer unit."""

    unit_id: str
    eval_role: str
    y_train: np.ndarray
    y_val: np.ndarray
    global_classes: np.ndarray
    profiling_train_by_blocks: dict[tuple[str, ...], np.ndarray | None]
    profiling_val_by_blocks: dict[tuple[str, ...], np.ndarray | None]
    family_train_matrices: dict[str, sparse.csr_matrix]
    family_val_matrices: dict[str, sparse.csr_matrix]


# ── Config loading and validation ──────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse the stacked-attribution trainer CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a stacked attribution model.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a stacked model config TOML file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage logs.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Read a stacked-attribution TOML config."""
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def stacked_search_profiling_blocks(config: dict[str, Any]) -> list[str]:
    """Return the first-seen union of profiling blocks declared by stacked conditions."""
    blocks: dict[str, None] = {}
    for raw_condition in config.get("conditions", []):
        raw_blocks = raw_condition.get("profiling_blocks", [])
        if raw_blocks in (None, []):
            continue
        if not isinstance(raw_blocks, list):
            raise ValueError("[[conditions]].profiling_blocks must be a list of strings.")
        for block in raw_blocks:
            if not isinstance(block, str) or not block.strip():
                raise ValueError("[[conditions]].profiling_blocks entries must be non-empty strings.")
            blocks.setdefault(block.strip(), None)
    return list(blocks)


def _validate_stacked_config(config: dict[str, Any]) -> None:
    """Validate the stacked development-search config boundary."""
    _validate_known_keys(
        "root",
        config,
        {"experiment", "data", "source", "model", "families", "conditions"},
    )
    _validate_known_keys(
        "experiment",
        config.get("experiment", {}),
        {"name", "seed", "selection_metric", "save_prediction_top_k", "n_jobs"},
    )
    _validate_known_keys(
        "data",
        config.get("data", {}),
        {"splits_dir", "results_dir", "artifacts_dir"},
    )
    _validate_known_keys(
        "source",
        config.get("source", {}),
        {"split_name", "materialization_name", "target", "units"},
    )
    _validate_known_keys(
        "model",
        config.get("model", {}),
        {
            "family",
            "inner_cv",
            "base_c_values",
            "class_weights",
            "max_iter",
            "tol",
            "dual",
            "top_c_values",
            "top_max_iter",
            "top_k",
        },
    )
    model_cfg = config.get("model", {})
    family = str(model_cfg.get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported model.family: {family!r}. Expected {SUPPORTED_MODEL_FAMILY!r}."
        )

    families = config.get("families", [])
    if not isinstance(families, list) or not families:
        raise ValueError("Stacked config must define at least one [[families]] entry.")
    for idx, fam in enumerate(families, start=1):
        _validate_known_keys(f"families #{idx}", fam, {"name", "blocks"})
        if "name" not in fam:
            raise ValueError("Each [[families]] entry must have a 'name' field.")
        if "blocks" not in fam or not fam["blocks"]:
            raise ValueError(f"Family '{fam.get('name')}' must have a non-empty 'blocks' list.")
    raw_conditions = config.get("conditions", [])
    if not isinstance(raw_conditions, list):
        raise ValueError("Stacked config conditions must be a list of tables.")
    for idx, raw_condition in enumerate(raw_conditions, start=1):
        _validate_known_keys(
            f"conditions #{idx}",
            raw_condition,
            {"id", "label", "families", "profiling_blocks"},
        )

    source_cfg = config.get("source", {})
    if "target" not in source_cfg:
        raise ValueError("source.target must be defined.")

    _stacked_candidate_grid(config)


def _stacked_families_from_config(config: dict[str, Any]) -> dict[str, StackedFamilySpec]:
    """Parse named base feature families from a stacked config."""
    raw_families = config.get("families", [])
    families_by_name: dict[str, StackedFamilySpec] = {}
    for raw_family in raw_families:
        name = str(raw_family.get("name", "")).strip()
        blocks = tuple(str(block).strip() for block in raw_family.get("blocks", []))
        if not name:
            raise ValueError("Each [[families]] entry must have a non-empty 'name'.")
        if not blocks:
            raise ValueError(f"Family '{name}' must have a non-empty 'blocks' list.")
        if name in families_by_name:
            raise ValueError(f"Duplicate [[families]].name value: {name!r}")
        families_by_name[name] = StackedFamilySpec(name=name, blocks=blocks)
    return families_by_name


def _stacked_conditions_from_config(config: dict[str, Any]) -> list[StackedConditionSpec]:
    """Parse stacked attribution research conditions from config."""
    families_by_name = _stacked_families_from_config(config)
    raw_conditions = config.get("conditions")

    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("Stacked config must define at least one [[conditions]] entry.")

    conditions: list[StackedConditionSpec] = []
    seen_ids: set[str] = set()
    for raw_condition in raw_conditions:
        condition_id = str(raw_condition.get("id", "")).strip()
        raw_names = raw_condition.get("families", [])
        raw_profiling_blocks = raw_condition.get("profiling_blocks", [])
        if not condition_id:
            raise ValueError("Each [[conditions]] entry must have a non-empty 'id'.")
        if condition_id in seen_ids:
            raise ValueError(f"Duplicate [[conditions]].id value: {condition_id!r}")
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(f"Condition '{condition_id}' must define a non-empty families list.")
        missing = [str(name) for name in raw_names if str(name) not in families_by_name]
        if missing:
            raise ValueError(f"Condition '{condition_id}' references unknown families: {missing}")
        if not isinstance(raw_profiling_blocks, list):
            raise ValueError(f"Condition '{condition_id}' profiling_blocks must be a list.")
        profiling_blocks = tuple(str(block).strip() for block in raw_profiling_blocks)
        if any(not block for block in profiling_blocks):
            raise ValueError(f"Condition '{condition_id}' profiling_blocks entries must be non-empty strings.")
        conditions.append(
            StackedConditionSpec(
                condition_id=condition_id,
                label=str(raw_condition.get("label", condition_id)).strip()
                or condition_id,
                families=tuple(families_by_name[str(name)] for name in raw_names),
                profiling_blocks=profiling_blocks,
            )
        )
        seen_ids.add(condition_id)
    return conditions


def _stacked_candidate_grid(config: dict[str, Any]) -> list[StackedCandidateSpec]:
    """Build every stacked candidate implied by conditions and hyperparameter grids."""
    model_cfg = config.get("model", {})
    conditions = _stacked_conditions_from_config(config)

    raw_base_c = model_cfg.get("base_c_values")
    if not isinstance(raw_base_c, list) or not raw_base_c:
        raise ValueError("model.base_c_values must be a non-empty list.")
    base_c_values = [float(value) for value in raw_base_c]

    raw_top_c = model_cfg.get("top_c_values")
    if not isinstance(raw_top_c, list) or not raw_top_c:
        raise ValueError("model.top_c_values must be a non-empty list.")
    top_c_values = [float(value) for value in raw_top_c]

    raw_class_weights = model_cfg.get("class_weights")
    if not isinstance(raw_class_weights, list) or not raw_class_weights:
        raise ValueError("model.class_weights must be a non-empty list.")
    class_weights = _parse_class_weights(raw_class_weights)

    candidates = [
        StackedCandidateSpec(
            condition=condition,
            base_c=base_c,
            class_weight=class_weight,
            top_c=top_c,
        )
        for condition, base_c, class_weight, top_c in itertools.product(
            conditions, base_c_values, class_weights, top_c_values
        )
    ]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    duplicate_ids = sorted(
        {candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate stacked candidate IDs: {duplicate_ids}")
    return candidates


def _stacked_candidate_from_payload(payload: dict[str, Any]) -> StackedCandidateSpec:
    """Reconstruct one selected stacked candidate payload."""
    families_payload = payload.get("families")
    if not isinstance(families_payload, list) or not families_payload:
        raise ValueError("Stacked selected candidate payload must contain a non-empty families list.")
    families = tuple(
        StackedFamilySpec(
            name=str(family["name"]),
            blocks=tuple(str(block) for block in family["blocks"]),
        )
        for family in families_payload
    )
    condition_id = str(payload["condition_id"]).strip()
    candidate = StackedCandidateSpec(
        condition=StackedConditionSpec(
            condition_id=condition_id,
            label=str(payload.get("condition_label", condition_id)).strip()
            or condition_id,
            families=families,
            profiling_blocks=tuple(str(block) for block in payload.get("profiling_blocks", [])),
        ),
        base_c=float(payload["base_c"]),
        class_weight=_parse_class_weights([payload.get("class_weight", "balanced")])[0],
        top_c=float(payload["top_c"]),
    )
    payload_candidate_id = payload.get("candidate_id")
    if payload_candidate_id and str(payload_candidate_id) != candidate.candidate_id:
        raise ValueError(
            "Stacked selected candidate payload is inconsistent: "
            f"candidate_id={payload_candidate_id!r} does not match resolved {candidate.candidate_id!r}."
        )
    return candidate


# ── Class alignment ────────────────────────────────────────────────────────────


def _align_proba_to_global(
    proba: np.ndarray,
    model_classes: np.ndarray,
    global_classes: np.ndarray,
) -> np.ndarray:
    """Align predict_proba output to a fixed global class ordering.

    For classes present in ``model_classes``, copies their probability columns
    into the matching position in ``global_classes``. For classes absent from
    ``model_classes`` (possible in small inner folds), inserts zero columns.

    Parameters
    ----------
    proba : (n_samples, n_model_classes)
    model_classes : array of class labels in the model's own ordering
    global_classes : fixed reference class ordering (sorted, from outer train)

    Returns
    -------
    aligned : (n_samples, len(global_classes))
    """
    n_samples = proba.shape[0]
    n_global = len(global_classes)
    aligned = np.zeros((n_samples, n_global), dtype=np.float64)
    global_to_idx = {cls: i for i, cls in enumerate(global_classes)}
    for model_col_idx, cls in enumerate(model_classes):
        if cls in global_to_idx:
            aligned[:, global_to_idx[cls]] = proba[:, model_col_idx]
    return aligned


# ── Feature matrix loading ─────────────────────────────────────────────────────


def _load_family_matrix(
    unit_dir: Path,
    role: str,
    blocks: list[str],
    normalize_rows: bool = True,
) -> sparse.csr_matrix:
    """Load and combine matrices for a single family's blocks."""
    matrices: list[sparse.csr_matrix] = []
    for block in blocks:
        path = _matrix_path(unit_dir, role, block)
        if not path.exists():
            raise FileNotFoundError(
                f"Required family matrix not found: {path}. "
                f"Ensure block '{block}' has been materialized."
            )
        matrices.append(_load_sparse_matrix(path))
    combined = matrices[0] if len(matrices) == 1 else sparse.hstack(matrices, format="csr")
    if normalize_rows:
        combined = normalize(combined, norm="l2", axis=1, copy=False)
    return combined.tocsr()


def _load_profiling_dense(
    unit_dir: Path,
    role: str,
    profiling_blocks: list[str],
) -> np.ndarray | None:
    """Load and horizontally concatenate profiling matrices for the top model.

    Returns ``None`` when ``profiling_blocks`` is empty (Phase 1B baseline).
    """
    if not profiling_blocks:
        return None
    parts: list[np.ndarray] = []
    for block in profiling_blocks:
        path = _matrix_path(unit_dir, role, block)
        if not path.exists():
            raise FileNotFoundError(
                f"Required profiling matrix not found: {path}. "
                "Run profiling signal extraction first."
            )
        mat = _load_sparse_matrix(path)
        parts.append(mat.toarray())
    return np.hstack(parts) if len(parts) > 1 else parts[0]


# ── Inner CV meta-feature generation ──────────────────────────────────────────


def _fit_calibrated_base(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    base_c: float,
    class_weight: str | None,
    max_iter: int,
    tol: float,
    dual: str | bool,
    seed: int,
) -> CalibratedClassifierCV:
    """Fit and return a CalibratedClassifierCV(LinearSVC) suppressing ConvergenceWarning."""
    base = LinearSVC(
        C=base_c,
        class_weight=class_weight,
        max_iter=max_iter,
        tol=tol,
        dual=dual,
        random_state=seed,
    )
    _, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    cal_cv = min(3, min_count)
    if cal_cv < 2:
        raise ValueError(
            f"Cannot calibrate: minimum class support in inner train is {min_count}."
        )
    calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cal_cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        calibrator.fit(x_train, y_train)
    return calibrator


def _base_fit_settings(model_cfg: dict[str, Any]) -> tuple[int, float, str | bool]:
    """Resolve shared base-SVM fit settings from one stacked model config."""
    return (
        int(model_cfg.get("max_iter", 20_000)),
        float(model_cfg.get("tol", 1e-4)),
        model_cfg.get("dual", "auto"),
    )


def _build_oof_family_train_meta(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    global_classes: np.ndarray,
    *,
    base_c: float,
    class_weight: str | None,
    model_cfg: dict[str, Any],
    seed: int,
    progress: Any | None = None,
) -> np.ndarray:
    """Build original-row-order OOF base probabilities for one stacked family."""
    inner_cv = int(model_cfg.get("inner_cv", 3))
    max_iter, tol, dual = _base_fit_settings(model_cfg)
    train_meta = np.zeros((len(y_train), len(global_classes)), dtype=np.float64)

    skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    for inner_train_idx, inner_val_idx in skf.split(np.zeros(len(y_train)), y_train):
        cal_model = _fit_calibrated_base(
            x_train[inner_train_idx],
            y_train[inner_train_idx],
            base_c,
            class_weight,
            max_iter,
            tol,
            dual,
            seed,
        )
        proba = cal_model.predict_proba(x_train[inner_val_idx])
        train_meta[inner_val_idx] = _align_proba_to_global(
            proba,
            np.asarray(cal_model.classes_),
            global_classes,
        )
        if progress is not None:
            progress.update(1)

    return train_meta


def _fit_family_validation_meta(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_eval: sparse.csr_matrix,
    global_classes: np.ndarray,
    *,
    base_c: float,
    class_weight: str | None,
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, CalibratedClassifierCV]:
    """Fit one final stacked family model and return aligned eval probabilities."""
    max_iter, tol, dual = _base_fit_settings(model_cfg)
    family_model = _fit_calibrated_base(
        x_train,
        y_train,
        base_c,
        class_weight,
        max_iter,
        tol,
        dual,
        seed,
    )
    eval_proba = family_model.predict_proba(x_eval)
    eval_meta = _align_proba_to_global(
        eval_proba,
        np.asarray(family_model.classes_),
        global_classes,
    )
    return eval_meta, family_model


def _assemble_stacked_meta_features(
    *,
    family_outputs: tuple[ReusableFamilyOutput, ...],
    profiling_train: np.ndarray | None,
    profiling_eval: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble top-model train/eval matrices from resolved stacked family outputs."""
    train_meta_cols = [family_output.train_meta for family_output in family_outputs]
    eval_meta_cols = [family_output.val_meta for family_output in family_outputs]
    if profiling_train is not None:
        train_meta_cols.append(profiling_train)
    if profiling_eval is not None:
        eval_meta_cols.append(profiling_eval)
    return np.hstack(train_meta_cols), np.hstack(eval_meta_cols)


def _unique_candidate_families(
    candidates: list[StackedCandidateSpec],
) -> tuple[StackedFamilySpec, ...]:
    """Return the first-seen family specs needed by a stacked candidate set."""
    families_by_name: dict[str, StackedFamilySpec] = {}
    for candidate in candidates:
        for family in candidate.families:
            families_by_name.setdefault(family.name, family)
    return tuple(families_by_name.values())


def _reusable_family_keys_for_candidates(
    candidates: list[StackedCandidateSpec],
) -> tuple[ReusableFamilyKey, ...]:
    """Return unique base-family fit keys needed to score stacked candidates."""
    keys: dict[ReusableFamilyKey, None] = {}
    for candidate in candidates:
        for family in candidate.families:
            key = ReusableFamilyKey(
                family_name=family.name,
                base_c=candidate.base_c,
                class_weight=candidate.class_weight,
            )
            keys.setdefault(key, None)
    return tuple(keys)


def _load_stacked_unit_data(
    unit: dict[str, Any],
    materialized_root: Path,
    candidates: list[StackedCandidateSpec],
    source_cfg: dict[str, Any],
) -> StackedUnitData:
    """Load labels, family matrices, and profiling matrices for one outer unit."""
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id
    target_name = str(source_cfg["target"])

    y_train = _load_labels(unit_dir / "labels" / f"y_train_{target_name}.npy")
    y_val = _load_labels(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy")
    global_classes = np.unique(y_train)

    family_train_matrices: dict[str, sparse.csr_matrix] = {}
    family_val_matrices: dict[str, sparse.csr_matrix] = {}
    for family in _unique_candidate_families(candidates):
        family_train_matrices[family.name] = _load_family_matrix(
            unit_dir, "train", list(family.blocks)
        )
        family_val_matrices[family.name] = _load_family_matrix(
            unit_dir, eval_role, list(family.blocks)
        )

    profiling_train_by_blocks: dict[tuple[str, ...], np.ndarray | None] = {}
    profiling_val_by_blocks: dict[tuple[str, ...], np.ndarray | None] = {}
    for profiling_blocks in dict.fromkeys(
        candidate.profiling_blocks for candidate in candidates
    ):
        profiling_train_by_blocks[profiling_blocks] = _load_profiling_dense(
            unit_dir, "train", list(profiling_blocks)
        )
        profiling_val_by_blocks[profiling_blocks] = _load_profiling_dense(
            unit_dir, eval_role, list(profiling_blocks)
        )

    return StackedUnitData(
        unit_id=unit_id,
        eval_role=eval_role,
        y_train=y_train,
        y_val=y_val,
        global_classes=global_classes,
        profiling_train_by_blocks=profiling_train_by_blocks,
        profiling_val_by_blocks=profiling_val_by_blocks,
        family_train_matrices=family_train_matrices,
        family_val_matrices=family_val_matrices,
    )


def _build_reusable_family_output(
    unit_data: StackedUnitData,
    key: ReusableFamilyKey,
    model_cfg: dict[str, Any],
    seed: int,
    progress: Any | None,
) -> ReusableFamilyOutput:
    """Build one reusable base-family output bundle for stacked dev search."""
    x_train = unit_data.family_train_matrices[key.family_name]
    x_val = unit_data.family_val_matrices[key.family_name]

    train_meta = _build_oof_family_train_meta(
        x_train,
        unit_data.y_train,
        unit_data.global_classes,
        base_c=key.base_c,
        class_weight=key.class_weight,
        model_cfg=model_cfg,
        seed=seed,
        progress=progress,
    )
    val_meta, _ = _fit_family_validation_meta(
        x_train,
        unit_data.y_train,
        x_val,
        unit_data.global_classes,
        base_c=key.base_c,
        class_weight=key.class_weight,
        model_cfg=model_cfg,
        seed=seed,
    )
    if progress is not None:
        progress.update(1)

    return ReusableFamilyOutput(train_meta=train_meta, val_meta=val_meta)


def _build_reusable_base_outputs_for_unit(
    unit: dict[str, Any],
    materialized_root: Path,
    candidates: list[StackedCandidateSpec],
    model_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    seed: int,
    show_progress: bool,
) -> tuple[StackedUnitData, dict[ReusableFamilyKey, ReusableFamilyOutput]]:
    """Build all reusable base-family outputs needed for one stacked dev unit."""
    unit_data = _load_stacked_unit_data(unit, materialized_root, candidates, source_cfg)
    reusable_keys = _reusable_family_keys_for_candidates(candidates)
    inner_cv = int(model_cfg.get("inner_cv", 3))
    progress = create_progress_bar(
        total=len(reusable_keys) * (inner_cv + 1),
        desc=f"[{unit_data.unit_id}] Base outputs",
        unit="fit",
        show_progress=show_progress,
    )

    outputs: dict[ReusableFamilyKey, ReusableFamilyOutput] = {}
    try:
        for key in reusable_keys:
            outputs[key] = _build_reusable_family_output(
                unit_data=unit_data,
                key=key,
                model_cfg=model_cfg,
                seed=seed,
                progress=progress,
            )
    finally:
        if progress is not None:
            progress.close()

    return unit_data, outputs


# ── Top model ─────────────────────────────────────────────────────────────────


def _fit_top_model(
    Z_train_meta: np.ndarray,
    y_train: np.ndarray,
    top_c: float,
    top_max_iter: int,
    seed: int,
) -> LogisticRegression:
    """Fit the logistic-regression meta-classifier for stacked attribution."""
    top_model = LogisticRegression(
        C=top_c,
        max_iter=top_max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    top_model.fit(Z_train_meta, y_train)
    return top_model


def _top_k_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    top_k_values: list[int],
) -> dict[str, float]:
    """Compute configured top-k accuracies from stacked probability scores."""
    metrics: dict[str, float] = {}
    for k in top_k_values:
        if k <= scores.shape[1]:
            metrics[f"top{k}_accuracy"] = float(
                top_k_accuracy_score(y_true, scores, k=k, labels=classes)
            )
    return metrics


def _stacked_validation_metrics(
    candidate: StackedCandidateSpec,
    unit_id: str,
    eval_role: str,
    y_val: np.ndarray,
    global_classes: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    top_fit_sec: float,
    predict_sec: float,
    top_k_values: list[int],
) -> dict[str, Any]:
    """Build the fold-level metrics row for one stacked candidate evaluation."""
    val_metrics = {
        "candidate_id": candidate.candidate_id,
        "condition_id": candidate.condition_id,
        "condition_label": candidate.condition_label,
        "family_set": candidate.family_set_name,
        "families": "+".join(family.name for family in candidate.families),
        "base_c": float(candidate.base_c),
        "class_weight": candidate.class_weight_label,
        "top_c": float(candidate.top_c),
        "profiling_blocks": (
            "+".join(candidate.profiling_blocks)
            if candidate.profiling_blocks
            else "none"
        ),
        "split": eval_role,
        "unit_id": unit_id,
        "n_samples": int(len(y_val)),
        "n_classes": int(len(global_classes)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_val, y_pred, average="weighted")),
        "macro_precision": float(
            precision_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "top_fit_seconds": float(top_fit_sec),
        "predict_seconds": float(predict_sec),
    }
    val_metrics.update(_top_k_metrics(y_val, scores, classes, top_k_values))
    return val_metrics


def _score_stacked_candidate_from_reusable_outputs(
    unit_data: StackedUnitData,
    candidate: StackedCandidateSpec,
    reusable_outputs: dict[ReusableFamilyKey, ReusableFamilyOutput],
    model_cfg: dict[str, Any],
    top_k_values: list[int],
    seed: int,
) -> dict[str, Any]:
    """Score one stacked candidate using precomputed base-family probabilities."""
    top_max_iter = int(model_cfg.get("top_max_iter", 1000))

    family_outputs: list[ReusableFamilyOutput] = []
    for family in candidate.families:
        key = ReusableFamilyKey(
            family_name=family.name,
            base_c=candidate.base_c,
            class_weight=candidate.class_weight,
        )
        family_outputs.append(reusable_outputs[key])

    profiling_train = unit_data.profiling_train_by_blocks[candidate.profiling_blocks]
    profiling_val = unit_data.profiling_val_by_blocks[candidate.profiling_blocks]
    Z_train_meta, Z_val_meta = _assemble_stacked_meta_features(
        family_outputs=tuple(family_outputs),
        profiling_train=profiling_train,
        profiling_eval=profiling_val,
    )

    started_at = time.perf_counter()
    top_model = _fit_top_model(
        Z_train_meta,
        unit_data.y_train,
        candidate.top_c,
        top_max_iter,
        seed,
    )
    top_fit_sec = time.perf_counter() - started_at

    started_at = time.perf_counter()
    y_pred = top_model.predict(Z_val_meta)
    predict_sec = time.perf_counter() - started_at
    proba_val = top_model.predict_proba(Z_val_meta)

    return _stacked_validation_metrics(
        candidate=candidate,
        unit_id=unit_data.unit_id,
        eval_role=unit_data.eval_role,
        y_val=unit_data.y_val,
        global_classes=unit_data.global_classes,
        y_pred=y_pred,
        scores=proba_val,
        classes=np.asarray(top_model.classes_),
        top_fit_sec=top_fit_sec,
        predict_sec=predict_sec,
        top_k_values=top_k_values,
    )


def _run_stacked_unit_candidate_search(
    unit: dict[str, Any],
    materialized_root: Path,
    candidates: list[StackedCandidateSpec],
    model_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    top_k_values: list[int],
    seed: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    """Evaluate all stacked dev candidates for one outer unit with reused bases."""
    unit_id = str(unit["unit_id"])
    if show_progress:
        print(f"\n[stacked] {unit_id}: building reusable base outputs...")

    unit_data, reusable_outputs = _build_reusable_base_outputs_for_unit(
        unit=unit,
        materialized_root=materialized_root,
        candidates=candidates,
        model_cfg=model_cfg,
        source_cfg=source_cfg,
        seed=seed,
        show_progress=show_progress,
    )

    candidate_progress = create_progress_bar(
        total=len(candidates),
        desc=f"[{unit_id}] Top models",
        unit="candidate",
        show_progress=show_progress,
    )
    try:
        fold_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            fold_result = _score_stacked_candidate_from_reusable_outputs(
                unit_data=unit_data,
                candidate=candidate,
                reusable_outputs=reusable_outputs,
                model_cfg=model_cfg,
                top_k_values=top_k_values,
                seed=seed,
            )
            fold_rows.append(fold_result)
            if candidate_progress is not None:
                candidate_progress.update(1)
                candidate_progress.set_postfix_str(
                    f"acc={fold_result['accuracy']:.4f}", refresh=False
                )
            if show_progress:
                print(
                    f"[stacked] {unit_id} {candidate.candidate_id}: "
                    f"acc={fold_result['accuracy']:.4f}, "
                    f"macro_f1={fold_result['macro_f1']:.4f}"
                )
    finally:
        if candidate_progress is not None:
            candidate_progress.close()

    return fold_rows


def _save_stacked_models(
    model_output_dir: Path,
    top_model: LogisticRegression,
    family_models: dict[str, CalibratedClassifierCV],
) -> None:
    """Persist the stacked top model and reusable family models."""
    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(top_model, model_output_dir / "top_model.joblib")
    for family_name, model in family_models.items():
        joblib.dump(model, model_output_dir / f"family_{family_name}.joblib")


# ── Per-fold runner ────────────────────────────────────────────────────────────


def _run_stacked_fold(
    unit: dict[str, Any],
    materialized_root: Path,
    candidate: StackedCandidateSpec,
    model_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    model_output_dir: Path | None,
    top_k_values: list[int],
    save_top_k: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the full stacked training + validation procedure for one outer fold."""
    unit_id = str(unit["unit_id"])
    eval_role = str(unit["eval_role"])
    unit_dir = materialized_root / unit_id

    base_c = candidate.base_c
    class_weight = candidate.class_weight
    top_c = candidate.top_c
    top_max_iter = int(model_cfg.get("top_max_iter", 1000))

    if show_progress:
        print(f"\n[stacked] {unit_id}: loading data...")

    unit_data = _load_stacked_unit_data(
        unit,
        materialized_root,
        [candidate],
        source_cfg,
    )

    # ── Step 1: build leakage-safe Z_train_meta via inner CV ─────────────────
    if show_progress:
        print(f"[stacked] {unit_id}: inner CV meta-feature generation...")

    inner_cv = int(model_cfg.get("inner_cv", 3))
    inner_progress = create_progress_bar(
        total=inner_cv * len(candidate.families),
        desc=f"[{unit_id}] Inner CV",
        unit="fit",
        show_progress=show_progress,
    )
    train_meta_by_family: dict[str, np.ndarray] = {}
    try:
        for family in candidate.families:
            train_meta_by_family[family.name] = _build_oof_family_train_meta(
                unit_data.family_train_matrices[family.name],
                unit_data.y_train,
                unit_data.global_classes,
                base_c=base_c,
                class_weight=class_weight,
                model_cfg=model_cfg,
                seed=seed,
                progress=inner_progress,
            )
    finally:
        if inner_progress is not None:
            inner_progress.close()

    # ── Step 2: fit calibrated base models from the outer training split ──────
    if show_progress:
        print(f"[stacked] {unit_id}: fitting final calibrated base models...")

    final_family_models: dict[str, CalibratedClassifierCV] = {}
    family_outputs: list[ReusableFamilyOutput] = []
    refit_progress = create_progress_bar(
        total=len(candidate.families),
        desc=f"[{unit_id}] Final base families",
        unit="family",
        show_progress=show_progress,
    )
    try:
        for family in candidate.families:
            val_meta, family_model = _fit_family_validation_meta(
                unit_data.family_train_matrices[family.name],
                unit_data.y_train,
                unit_data.family_val_matrices[family.name],
                unit_data.global_classes,
                base_c=base_c,
                class_weight=class_weight,
                model_cfg=model_cfg,
                seed=seed,
            )
            final_family_models[family.name] = family_model
            family_outputs.append(
                ReusableFamilyOutput(
                    train_meta=train_meta_by_family[family.name],
                    val_meta=val_meta,
                )
            )
            if refit_progress is not None:
                refit_progress.update(1)
    finally:
        if refit_progress is not None:
            refit_progress.close()

    profiling_train = unit_data.profiling_train_by_blocks[candidate.profiling_blocks]
    profiling_val = unit_data.profiling_val_by_blocks[candidate.profiling_blocks]
    Z_train_meta, Z_val_meta = _assemble_stacked_meta_features(
        family_outputs=tuple(family_outputs),
        profiling_train=profiling_train,
        profiling_eval=profiling_val,
    )

    # ── Step 3: fit top model on meta-features ────────────────────────────────
    if show_progress:
        print(f"[stacked] {unit_id}: fitting top LogisticRegression...")

    t0 = time.perf_counter()
    top_model = _fit_top_model(Z_train_meta, unit_data.y_train, top_c, top_max_iter, seed)
    top_fit_sec = time.perf_counter() - t0

    # ── Step 4: score ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_pred = top_model.predict(Z_val_meta)
    predict_sec = time.perf_counter() - t0
    proba_val = top_model.predict_proba(Z_val_meta)
    top_model_classes = np.asarray(top_model.classes_)

    val_metrics = _stacked_validation_metrics(
        candidate=candidate,
        unit_id=unit_id,
        eval_role=eval_role,
        y_val=unit_data.y_val,
        global_classes=unit_data.global_classes,
        y_pred=y_pred,
        scores=proba_val,
        classes=top_model_classes,
        top_fit_sec=top_fit_sec,
        predict_sec=predict_sec,
        top_k_values=top_k_values,
    )

    if model_output_dir is not None:
        _save_stacked_models(model_output_dir, top_model, final_family_models)

    pred_frame = _prediction_frame(
        unit_dir=unit_dir,
        eval_role=eval_role,
        y_true=unit_data.y_val,
        y_pred=y_pred,
        scores=proba_val,
        classes=top_model_classes,
        save_top_k=save_top_k,
    )

    if show_progress:
        print(
            f"[stacked] {unit_id} {candidate.candidate_id}: acc={val_metrics['accuracy']:.4f}, "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

    return val_metrics, pred_frame


def _summarize_stacked_candidates(
    metrics_df: pd.DataFrame,
    selection_metric: str,
) -> pd.DataFrame:
    """Aggregate stacked candidate metrics while preserving condition identity."""
    if metrics_df.empty:
        raise ValueError("No stacked metrics were collected.")

    group_cols = [
        "candidate_id",
        "condition_id",
        "condition_label",
        "family_set",
        "families",
        "base_c",
        "class_weight",
        "top_c",
        "profiling_blocks",
    ]
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "top_fit_seconds",
        "predict_seconds",
    ]
    top_k_metric_cols = sorted(
        col
        for col in metrics_df.columns
        if col.startswith("top") and col.endswith("_accuracy")
    )
    metric_cols.extend(top_k_metric_cols)

    summary = (
        metrics_df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "__".join(str(part) for part in col if part).strip("_")
        for col in summary.columns.to_flat_index()
    ]
    rename_map = {f"{metric}__mean": f"eval_mean_{metric}" for metric in metric_cols}
    rename_map.update({f"{metric}__std": f"eval_std_{metric}" for metric in metric_cols})
    summary = summary.rename(columns=rename_map)
    counts = metrics_df.groupby(group_cols, dropna=False).size().reset_index(name="n_eval_units")
    summary = summary.merge(counts, on=group_cols, how="left")

    sort_cols, ascending = _stacked_summary_sort_columns(summary, selection_metric)
    summary = summary.sort_values(
        by=sort_cols,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)
    return summary


def _stacked_summary_sort_columns(
    summary_df: pd.DataFrame,
    selection_metric: str,
) -> tuple[list[str], list[bool]]:
    """Return shared sort columns for global and per-condition stacked ranking."""
    selection_col = f"eval_mean_{selection_metric}"
    if selection_col not in summary_df.columns:
        available = sorted(col for col in summary_df.columns if col.startswith("eval_mean_"))
        raise ValueError(
            f"Selection metric '{selection_metric}' is not available. Available: {available}"
        )
    sort_cols = [
        selection_col,
        "eval_mean_accuracy",
        "n_eval_units",
        "base_c",
        "top_c",
        "condition_id",
        "class_weight",
    ]
    ascending = [False, False, False, True, True, True, True]
    return sort_cols, ascending


def _select_stacked_candidates_by_condition(
    summary_df: pd.DataFrame,
    candidates: list[StackedCandidateSpec],
    selection_metric: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select one stacked candidate independently for each declared condition."""
    sort_cols, ascending = _stacked_summary_sort_columns(summary_df, selection_metric)
    return select_candidates_by_condition(
        summary_df,
        candidates,
        selection_metric=selection_metric,
        sort_cols=sort_cols,
        ascending=ascending,
        candidate_id=lambda candidate: candidate.candidate_id,
        condition_id=lambda candidate: candidate.condition_id,
        condition_label=lambda candidate: candidate.condition_label,
        selected_payload=lambda candidate, metric, summary: candidate.to_payload(
            selection_metric=metric,
            dev_summary=summary,
        ),
    )


def _validate_stacked_selection_manifest(
    *,
    project_root: Path,
    selected_candidates_path: Path,
    selected_count: int,
    selection_manifest: dict[str, Any],
) -> None:
    """Validate that a stacked dev manifest points at the loaded selection artifact."""
    run_type = selection_manifest.get("run_type")
    if run_type != "stacked_condition_selection":
        raise ValueError(
            "Stacked selection manifest has unexpected run_type: "
            f"{run_type!r}; expected 'stacked_condition_selection'."
        )

    selection_scope = selection_manifest.get("selection_scope")
    if selection_scope != "condition":
        raise ValueError(
            "Stacked selection manifest has unexpected selection_scope: "
            f"{selection_scope!r}; expected 'condition'."
        )

    raw_manifest_selection_path = selection_manifest.get("selected_candidates_path")
    if not raw_manifest_selection_path:
        raise ValueError("Stacked selection manifest must include selected_candidates_path.")
    manifest_selection_path = resolve_project_path(project_root, raw_manifest_selection_path)
    if manifest_selection_path.resolve() != selected_candidates_path.resolve():
        raise ValueError(
            "Stacked selection manifest selected_candidates_path does not match "
            f"loaded selected_candidates.json: {manifest_selection_path} != {selected_candidates_path}"
        )

    manifest_condition_count = selection_manifest.get("condition_count")
    if manifest_condition_count is None:
        raise ValueError("Stacked selection manifest must include condition_count.")
    if int(manifest_condition_count) != selected_count:
        raise ValueError(
            "Stacked selection manifest condition_count does not match "
            f"selected_candidates.json: {manifest_condition_count} != {selected_count}"
        )


def load_selected_stacked_candidates(
    project_root: Path,
    config: dict[str, Any],
    *,
    selected_candidates_path_override: Path | None = None,
) -> tuple[list[StackedCandidateSpec], dict[str, Any], dict[str, Any]]:
    """Load the condition-aware stacked selected-candidates artifact."""
    final_cfg = config.get("final_eval", {})
    if selected_candidates_path_override is not None:
        selected_candidates_path = resolve_project_path(project_root, selected_candidates_path_override)
    else:
        raw_path = final_cfg.get("selected_candidates_path")
        if not raw_path:
            raise ValueError(
                "Final stacked evaluation requires [final_eval] selected_candidates_path "
                "in the config or a selected_candidates_path_override from the caller."
            )
        selected_candidates_path = resolve_project_path(project_root, raw_path)

    if not selected_candidates_path.exists():
        raise FileNotFoundError(f"Stacked selected_candidates.json does not exist: {selected_candidates_path}")
    payload = json.loads(selected_candidates_path.read_text(encoding="utf-8"))
    raw_candidates = payload.get("selected_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Stacked selected_candidates.json must include selected_candidates.")
    candidates = [_stacked_candidate_from_payload(candidate) for candidate in raw_candidates]
    selection_manifest_path = selected_candidates_path.parent / "manifest.json"
    if selection_manifest_path.exists():
        selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        _validate_stacked_selection_manifest(
            project_root=project_root,
            selected_candidates_path=selected_candidates_path,
            selected_count=len(candidates),
            selection_manifest=selection_manifest,
        )
    return candidates, payload, {
        "selected_candidates_path": relative_to_project(project_root, selected_candidates_path),
        "selection_results_dir": relative_to_project(project_root, selected_candidates_path.parent),
        "selection_manifest_path": (
            relative_to_project(project_root, selection_manifest_path)
            if selection_manifest_path.exists()
            else None
        ),
    }


def _validate_final_stacked_eval_config(
    config: dict[str, Any],
    *,
    require_selected_candidates_path: bool,
) -> None:
    """Validate the stacked final-evaluation config boundary."""
    _validate_known_keys(
        "root",
        config,
        {"experiment", "data", "source", "model", "final_eval"},
    )
    _validate_known_keys(
        "experiment",
        config.get("experiment", {}),
        {"name", "seed", "save_prediction_top_k", "n_jobs"},
    )
    _validate_known_keys(
        "data",
        config.get("data", {}),
        {"splits_dir", "results_dir", "artifacts_dir"},
    )
    _validate_known_keys(
        "source",
        config.get("source", {}),
        {"split_name", "materialization_name", "target", "units"},
    )
    _validate_known_keys(
        "model",
        config.get("model", {}),
        {"family", "inner_cv", "max_iter", "tol", "dual", "top_max_iter", "top_k"},
    )
    _validate_known_keys(
        "final_eval",
        config.get("final_eval", {}),
        {"selected_candidates_path"},
    )

    model_cfg = config.get("model", {})
    family = str(model_cfg.get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported model.family: {family!r}. Expected {SUPPORTED_MODEL_FAMILY!r}."
        )
    source_cfg = config.get("source", {})
    missing_source = sorted({"split_name", "materialization_name", "target", "units"} - set(source_cfg))
    if missing_source:
        raise ValueError(f"Final stacked config is missing required [source] keys: {missing_source}")
    if require_selected_candidates_path and not config.get("final_eval", {}).get("selected_candidates_path"):
        raise ValueError("Final stacked evaluation config must define final_eval.selected_candidates_path.")


# ── Public entry point ────────────────────────────────────────────────────────


def run_stacked_experiment_from_config(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the Paraboni-style stacked model for all outer folds.

    Returns
    -------
    dict
        Top-level manifest written to ``results_dir/manifest.json``.
    """
    _validate_stacked_config(config)

    experiment_cfg = config.get("experiment", {})
    source_cfg = config.get("source", {})
    model_cfg = config.get("model", {})
    candidates = _stacked_candidate_grid(config)
    profiling_blocks = tuple(stacked_search_profiling_blocks(config))

    seed = int(experiment_cfg.get("seed", 42))
    default_name = config_path.stem if config_path is not None else "stacked_experiment"
    config_name = str(experiment_cfg.get("name", default_name))
    selection_metric = str(experiment_cfg.get("selection_metric", "macro_f1"))
    n_jobs = int(experiment_cfg.get("n_jobs", 1))
    top_k_values = [int(value) for value in model_cfg.get("top_k", [3, 5])]

    materialized_root, materialization_manifest, units = _load_materialization_summary(project_root, config)
    provenance = _build_provenance_block(project_root, materialized_root, materialization_manifest)
    _validate_candidate_search_units(units)

    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)
    _copy_config_outputs_if_available(config_path, results_dir, artifacts_dir)

    uses_profiling = bool(profiling_blocks)
    stacking_variant = "with_profiling" if uses_profiling else "baseline"

    if show_progress:
        print(
            f"Stacked experiment '{config_name}' [{stacking_variant}]: "
            f"{len(units)} fold(s), {len(candidates)} candidate(s), "
            f"seed={seed}, n_jobs={n_jobs}"
        )
        conditions = sorted({candidate.condition_id for candidate in candidates})
        print(f"  Conditions: {conditions}")
        if uses_profiling:
            print(f"  Profiling blocks: {list(profiling_blocks)}")
        else:
            print("  Profiling: none (Phase 1B baseline)")

    unit_tasks = list(units)
    worker_show_progress = show_progress and n_jobs == 1

    if show_progress:
        print(
            "Starting stacked candidate search "
            f"({len(candidates)} candidate(s) across {len(unit_tasks)} fold(s), "
            f"n_jobs={n_jobs})..."
        )

    if n_jobs == 1:
        fold_metrics_rows: list[dict[str, Any]] = []
        fold_progress = create_progress_bar(
            total=len(unit_tasks),
            desc="Stacked folds",
            unit="fold",
            show_progress=show_progress,
        )

        for unit in unit_tasks:
            unit_rows = _run_stacked_unit_candidate_search(
                unit=unit,
                materialized_root=materialized_root,
                candidates=candidates,
                model_cfg=model_cfg,
                source_cfg=source_cfg,
                top_k_values=top_k_values,
                seed=seed,
                show_progress=worker_show_progress,
            )
            fold_metrics_rows.extend(unit_rows)
            if fold_progress is not None:
                fold_progress.update(1)

        if fold_progress is not None:
            fold_progress.close()
    else:
        unit_metric_rows = joblib.Parallel(
            n_jobs=n_jobs, backend="loky", verbose=10 if show_progress else 0
        )(
            joblib.delayed(_run_stacked_unit_candidate_search)(
                unit=unit,
                materialized_root=materialized_root,
                candidates=candidates,
                model_cfg=model_cfg,
                source_cfg=source_cfg,
                top_k_values=top_k_values,
                seed=seed,
                show_progress=False,
            )
            for unit in unit_tasks
        )
        fold_metrics_rows = [
            row
            for unit_rows in unit_metric_rows
            for row in unit_rows
        ]

    # ── Aggregate across folds ────────────────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metrics_rows)
    metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)

    summary_df = _summarize_stacked_candidates(metrics_df, selection_metric)
    summary_df.to_csv(results_dir / "candidate_summary.csv", index=False)
    condition_summary_df, selected_candidates = _select_stacked_candidates_by_condition(
        summary_df,
        candidates,
        selection_metric,
    )
    condition_summary_df.to_csv(results_dir / "condition_summary.csv", index=False)
    selected_payload = {
        "selection_scope": "condition",
        "selection_metric": selection_metric,
        "split_name": source_cfg["split_name"],
        "materialization_name": source_cfg["materialization_name"],
        "target": source_cfg["target"],
        "selected_candidates": selected_candidates,
    }
    write_json(results_dir / "selected_candidates.json", selected_payload)

    if show_progress:
        print(
            f"\nStacked experiment complete. "
            f"Selected {len(selected_candidates)} condition candidate(s)."
        )

    manifest: dict[str, Any] = {
        "run_type": "stacked_condition_selection",
        "selection_scope": "condition",
        "experiment_name": config_name,
        "config_path": _manifest_config_path(project_root, config_path),
        "split_name": source_cfg["split_name"],
        "materialization_name": source_cfg["materialization_name"],
        "target": source_cfg["target"],
        "seed": seed,
        "n_jobs": n_jobs,
        "selection_metric": selection_metric,
        "candidate_count": len(candidates),
        "condition_count": len(selected_candidates),
        "profiling_blocks": list(profiling_blocks),
        "uses_profiling": uses_profiling,
        "stacking_variant": stacking_variant,
        "model_cfg": model_cfg,
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "unit_count": len(units),
        "fold_metrics_path": relative_to_project(project_root, results_dir / "fold_metrics.csv"),
        "candidate_summary_path": relative_to_project(project_root, results_dir / "candidate_summary.csv"),
        "condition_summary_path": relative_to_project(project_root, results_dir / "condition_summary.csv"),
        "selected_candidates_path": relative_to_project(project_root, results_dir / "selected_candidates.json"),
        "provenance": provenance,
    }
    write_json(results_dir / "manifest.json", manifest)

    return manifest


def run_stacked_experiment(config_path: Path, *, show_progress: bool = False) -> dict[str, Any]:
    """Run stacked condition selection from a config file."""
    project_root = find_project_root(config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent)
    config = _load_config(config_path)
    return run_stacked_experiment_from_config(
        config,
        project_root=project_root,
        config_path=config_path,
        show_progress=show_progress,
    )


def run_final_stacked_evaluation_from_config(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_path: Path | None = None,
    show_progress: bool = False,
    selected_candidates_path_override: Path | None = None,
    preloaded_candidates: list[StackedCandidateSpec] | None = None,
    preloaded_selection_payload: dict[str, Any] | None = None,
    preloaded_selection_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run final stacked evaluation for every dev-selected condition candidate."""
    _validate_final_stacked_eval_config(
        config,
        require_selected_candidates_path=(
            selected_candidates_path_override is None and preloaded_candidates is None
        ),
    )

    experiment_cfg = config.get("experiment", {})
    source_cfg = config.get("source", {})
    model_cfg = config.get("model", {})
    if preloaded_candidates is None:
        candidates, selection_payload, selection_source = load_selected_stacked_candidates(
            project_root,
            config,
            selected_candidates_path_override=selected_candidates_path_override,
        )
    else:
        candidates = preloaded_candidates
        selection_payload = preloaded_selection_payload
        selection_source = preloaded_selection_source

    seed = int(experiment_cfg.get("seed", 42))
    default_name = config_path.stem if config_path is not None else "final_stacked_evaluation"
    config_name = str(experiment_cfg.get("name", default_name))
    save_top_k = int(experiment_cfg.get("save_prediction_top_k", 5))
    top_k_values = [int(value) for value in model_cfg.get("top_k", [3, 5])]

    uses_profiling = any(candidate.profiling_blocks for candidate in candidates)
    stacking_variant = "with_profiling" if uses_profiling else "baseline"

    materialized_root, materialization_manifest, units = _load_materialization_summary(project_root, config)
    provenance = _build_provenance_block(project_root, materialized_root, materialization_manifest)
    unit = _validate_final_evaluation_units(units)

    results_dir, artifacts_dir = _resolve_output_dirs(project_root, config, config_name)
    _copy_config_outputs_if_available(config_path, results_dir, artifacts_dir)

    if show_progress:
        print(
            f"Final stacked evaluation '{config_name}' [{stacking_variant}]: "
            f"unit={unit['unit_id']}, conditions={len(candidates)}, seed={seed}"
        )

    selected_candidates_path = results_dir / "selected_candidates.json"
    write_json(selected_candidates_path, selection_payload)

    final_by_condition_dir, final_artifacts_by_condition_dir = final_condition_roots(
        results_dir,
        artifacts_dir,
    )

    source_payloads = {
        str(payload["candidate_id"]): payload
        for payload in selection_payload.get("selected_candidates", [])
    }
    condition_results: list[dict[str, Any]] = []
    final_summary_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        condition_id = candidate.condition_id
        condition_results_dir = final_by_condition_dir / condition_id
        condition_artifacts_dir = final_artifacts_by_condition_dir / condition_id / "final_model"

        final_metrics, prediction_frame = _run_stacked_fold(
            unit=unit,
            materialized_root=materialized_root,
            candidate=candidate,
            model_cfg=model_cfg,
            source_cfg=source_cfg,
            model_output_dir=condition_artifacts_dir,
            top_k_values=top_k_values,
            save_top_k=save_top_k,
            seed=seed,
            show_progress=show_progress,
        )

        final_metrics_payload = {
            "unit_id": unit["unit_id"],
            "eval_role": unit["eval_role"],
            "target": source_cfg["target"],
            "condition_id": condition_id,
            "condition_label": candidate.condition_label,
            "candidate_id": candidate.candidate_id,
            "family_set": candidate.family_set_name,
            "families": [family.to_payload() for family in candidate.families],
            "base_c": float(candidate.base_c),
            "class_weight": candidate.class_weight_label,
            "top_c": float(candidate.top_c),
            "profiling_blocks": list(candidate.profiling_blocks),
            "final_test_metrics": final_metrics,
        }

        resolved_candidate_payload = {
            **candidate.to_payload(),
            "source_payload": source_payloads[candidate.candidate_id],
        }

        condition_result = {
            "condition_id": condition_id,
            "condition_label": candidate.condition_label,
            "candidate_id": candidate.candidate_id,
            "unit_id": unit["unit_id"],
            "eval_role": unit["eval_role"],
        }
        condition_results.append(
            {
                **write_final_condition_output(
                    project_root=project_root,
                    condition_results_dir=condition_results_dir,
                    prediction_frame=prediction_frame,
                    metrics_payload=final_metrics_payload,
                    resolved_candidate_payload=resolved_candidate_payload,
                    condition_result=condition_result,
                ),
                "model_dir": relative_to_project(project_root, condition_artifacts_dir),
            }
        )

        dev_summary = source_payloads[candidate.candidate_id].get("dev_summary", {})
        final_summary_rows.append(
            build_final_summary_row(
                condition_id=condition_id,
                condition_label=candidate.condition_label,
                candidate_id=candidate.candidate_id,
                dev_summary=dev_summary,
                final_metrics=final_metrics,
            )
        )

    final_condition_summary_path = write_final_condition_summary(
        results_dir,
        final_summary_rows,
    )
    profiling_blocks_union = list(
        dict.fromkeys(
            block
            for candidate in candidates
            for block in candidate.profiling_blocks
        )
    )

    if show_progress:
        print(f"\nFinal stacked evaluation complete for {len(candidates)} condition(s).")

    manifest: dict[str, Any] = {
        "run_type": "stacked_condition_final_evaluation",
        "selection_scope": "condition",
        "experiment_name": config_name,
        "config_path": _manifest_config_path(project_root, config_path),
        "split_name": source_cfg["split_name"],
        "materialization_name": source_cfg["materialization_name"],
        "target": source_cfg["target"],
        "seed": seed,
        "selected_candidates_path": relative_to_project(
            project_root, selected_candidates_path
        ),
        "condition_count": len(candidates),
        "final_condition_summary_path": relative_to_project(
            project_root, final_condition_summary_path
        ),
        "condition_results": condition_results,
        "selection_source": selection_source,
        "profiling_blocks": profiling_blocks_union,
        "uses_profiling": uses_profiling,
        "stacking_variant": stacking_variant,
        "model_cfg": model_cfg,
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "materialized_root": relative_to_project(project_root, materialized_root),
        "provenance": provenance,
        "source_manifest": {
            "path": relative_to_project(
                project_root, materialized_root / "manifest.json"
            ),
            "units": materialization_manifest.get("units", []),
        },
    }
    write_json(results_dir / "manifest.json", manifest)
    return manifest


def run_final_stacked_evaluation(
    config_path: Path,
    *,
    show_progress: bool = False,
    selected_candidates_path_override: Path | None = None,
) -> dict[str, Any]:
    """Run final stacked evaluation for every dev-selected condition candidate."""
    project_root = find_project_root(config_path.resolve().parent, Path.cwd(), Path(__file__).resolve().parent)
    config = _load_config(config_path)
    return run_final_stacked_evaluation_from_config(
        config,
        project_root=project_root,
        config_path=config_path,
        show_progress=show_progress,
        selected_candidates_path_override=selected_candidates_path_override,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for stacked attribution training."""
    args = _parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)
    run_stacked_experiment(config_path, show_progress=not args.no_progress)


if __name__ == "__main__":
    main()
