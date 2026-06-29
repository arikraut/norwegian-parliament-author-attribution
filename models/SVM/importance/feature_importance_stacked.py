from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_pipeline.utils import relative_to_project
from models.SVM.importance.feature_importance_svm import (
    FeatureName,
    canonical_feature_block,
    coef_rows_by_class,
    compute_block_importance,
    compute_global_importance,
    compute_stylo_subfamily_importance,
    load_feature_names_for_blocks,
    load_final_manifest,
    resolve_final_unit_paths,
)


def load_stacked_models(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the top model and saved base-family models from a stacked artifact dir."""
    top_model = joblib.load(model_dir / "top_model.joblib")
    family_models: dict[str, Any] = {}
    for path in sorted(model_dir.glob("family_*.joblib")):
        raw_name = path.stem.removeprefix("family_")
        model = joblib.load(path)
        family_models[raw_name] = model
        family_models.setdefault(canonical_feature_block(raw_name), model)
    return top_model, family_models


def family_model_candidates(family_name: str) -> tuple[str, ...]:
    """Return raw/canonical model names accepted for one stacked family."""

    canonical = canonical_feature_block(family_name)
    candidates = [family_name, canonical]
    if canonical == "stylo":
        candidates.extend(["stylo", "spacy"])
    return tuple(dict.fromkeys(candidates))


def select_family_model(
    family_models: dict[str, Any],
    family_name: str,
) -> Any:
    """Return a stacked base-family model, accepting raw/canonical aliases."""

    for candidate in family_model_candidates(family_name):
        if candidate in family_models:
            return family_models[candidate]
    raise KeyError(
        f"Could not find model for family {family_name!r}; "
        f"tried={family_model_candidates(family_name)}, "
        f"available={sorted(family_models)}."
    )


def build_top_model_feature_names(
    families: list[dict[str, Any]],
    classes: np.ndarray,
    materialized_root: Path,
    unit_id: str,
    profiling_blocks: list[str],
) -> list[FeatureName]:
    """Reconstruct ordered stacked top-model meta-feature names."""
    class_labels = [str(class_label) for class_label in classes]
    feature_names: list[FeatureName] = []
    for family in families:
        family_name = canonical_feature_block(str(family["name"]))
        feature_names.extend(
            FeatureName(
                name=f"{family_name}:{class_label}",
                block=family_name,
                raw_name=class_label,
                subfamily="",
            )
            for class_label in class_labels
        )

    if profiling_blocks:
        feature_names.extend(
            load_feature_names_for_blocks(materialized_root, unit_id, profiling_blocks)
        )
    return feature_names


def compute_top_model_family_importance(
    top_model: Any,
    feature_names: list[FeatureName],
) -> pd.DataFrame:
    """Aggregate top-model coefficient importance by family/meta-feature block."""
    global_df = compute_global_importance(np.asarray(top_model.coef_), feature_names)
    family_df = compute_block_importance(global_df).rename(columns={"block": "family"})
    return family_df.rename(columns={"n_features": "n_meta_features"})


def compute_top_model_per_author_family_weights(
    top_model: Any,
    feature_names: list[FeatureName],
) -> pd.DataFrame:
    """Summarize signed and absolute top-model weights per author and family."""
    blocks = [feature.block for feature in feature_names]
    ordered_blocks = list(dict.fromkeys(blocks))
    block_indices = {
        block: np.asarray(
            [idx for idx, feature_block in enumerate(blocks) if feature_block == block],
            dtype=int,
        )
        for block in ordered_blocks
    }

    rows: list[dict[str, Any]] = []
    for author, class_coef in coef_rows_by_class(
        np.asarray(top_model.coef_),
        np.asarray(top_model.classes_),
    ):
        for family, indices in block_indices.items():
            weights = class_coef[indices]
            rows.append(
                {
                    "author": author,
                    "family": family,
                    "weight_sum": float(weights.sum()),
                    "weight_mean": float(weights.mean()) if len(weights) else 0.0,
                    "abs_weight_sum": float(np.abs(weights).sum()),
                    "abs_weight_mean": (
                        float(np.abs(weights).mean()) if len(weights) else 0.0
                    ),
                    "n_meta_features": int(len(weights)),
                }
            )
    return pd.DataFrame(rows)


def calibrated_linear_svc_coef(calibrated_clf: Any) -> np.ndarray:
    """Average underlying LinearSVC coefficients from a CalibratedClassifierCV."""
    coefs: list[np.ndarray] = []
    for calibrated_classifier in calibrated_clf.calibrated_classifiers_:
        estimator = calibrated_classifier.estimator
        if not hasattr(estimator, "coef_"):
            raise ValueError("Calibrated base estimator does not expose coef_.")
        coefs.append(np.asarray(estimator.coef_))

    shapes = {coef.shape for coef in coefs}
    if len(shapes) != 1:
        raise ValueError(f"Calibrated estimator coef_ shapes differ: {sorted(shapes)}")
    return np.mean(coefs, axis=0)


def _family_importance_frames(
    family_name: str,
    coef: np.ndarray,
    feature_names: list[FeatureName],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute global, block, and stylo-subfamily summaries for one base family."""
    global_df = compute_global_importance(coef, feature_names)
    block_df = compute_block_importance(global_df)
    subfamily_df = compute_stylo_subfamily_importance(global_df)

    for frame in [global_df, block_df, subfamily_df]:
        frame.insert(0, "family", family_name)
    return global_df, block_df, subfamily_df


def _stacked_outputs(project_root: Path, output_dir: Path) -> dict[str, str]:
    """Return relative paths for stacked feature-importance output files."""
    return {
        "top_model_family_importance": relative_to_project(
            project_root, output_dir / "top_model_family_importance.csv"
        ),
        "top_model_per_author_family": relative_to_project(
            project_root, output_dir / "top_model_per_author_family.csv"
        ),
        "base_family_global_importance": relative_to_project(
            project_root, output_dir / "base_family_global_importance.csv"
        ),
        "base_family_block_importance": relative_to_project(
            project_root, output_dir / "base_family_block_importance.csv"
        ),
        "base_family_stylo_subfamily_importance": relative_to_project(
            project_root, output_dir / "base_family_stylo_subfamily_importance.csv"
        ),
    }


def run_stacked_condition_importance_analysis(
    manifest_path: Path,
    *,
    condition_id: str,
) -> dict[str, Any]:
    """Run coefficient feature-importance analysis for one final stacked condition."""
    manifest_path = manifest_path.resolve()
    manifest = load_final_manifest(
        manifest_path,
        expected_run_type="stacked_condition_final_evaluation",
    )
    paths = resolve_final_unit_paths(
        manifest_path,
        manifest,
        condition_id=condition_id,
    )
    if paths.model_dir is None:
        raise ValueError(
            f"{manifest_path} condition {paths.condition_id!r} does not record model_dir."
        )

    top_model, family_models = load_stacked_models(paths.model_dir)
    resolved_candidate = load_final_manifest(paths.resolved_candidate_path)
    families = list(resolved_candidate["families"])
    profiling_blocks = [str(block) for block in resolved_candidate["profiling_blocks"]]
    top_feature_names = build_top_model_feature_names(
        families,
        np.asarray(top_model.classes_),
        paths.materialized_root,
        paths.unit_id,
        profiling_blocks,
    )

    top_family_df = compute_top_model_family_importance(top_model, top_feature_names)
    top_author_df = compute_top_model_per_author_family_weights(
        top_model,
        top_feature_names,
    )

    base_global_frames: list[pd.DataFrame] = []
    base_block_frames: list[pd.DataFrame] = []
    base_subfamily_frames: list[pd.DataFrame] = []
    for family in families:
        raw_family_name = str(family["name"])
        family_name = canonical_feature_block(raw_family_name)
        calibrated_clf = select_family_model(family_models, raw_family_name)
        family_feature_names = load_feature_names_for_blocks(
            paths.materialized_root,
            paths.unit_id,
            [str(block) for block in family["blocks"]],
        )
        global_df, block_df, subfamily_df = _family_importance_frames(
            family_name,
            calibrated_linear_svc_coef(calibrated_clf),
            family_feature_names,
        )
        base_global_frames.append(global_df)
        base_block_frames.append(block_df)
        base_subfamily_frames.append(subfamily_df)

    output_dir = paths.condition_results_dir / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_family_df.to_csv(output_dir / "top_model_family_importance.csv", index=False)
    top_author_df.to_csv(output_dir / "top_model_per_author_family.csv", index=False)
    pd.concat(base_global_frames, ignore_index=True).to_csv(
        output_dir / "base_family_global_importance.csv",
        index=False,
    )
    pd.concat(base_block_frames, ignore_index=True).to_csv(
        output_dir / "base_family_block_importance.csv",
        index=False,
    )
    pd.concat(base_subfamily_frames, ignore_index=True).to_csv(
        output_dir / "base_family_stylo_subfamily_importance.csv",
        index=False,
    )

    return {
        "run_type": "stacked_feature_importance",
        "source_manifest": relative_to_project(paths.project_root, manifest_path),
        "condition_id": paths.condition_id,
        "condition_label": paths.condition_label,
        "results_dir": relative_to_project(paths.project_root, output_dir),
        "n_top_model_features": len(top_feature_names),
        "families": [
            canonical_feature_block(str(family["name"]))
            for family in families
        ],
        "outputs": _stacked_outputs(paths.project_root, output_dir),
    }


def run_stacked_importance_analysis(
    manifest_path: Path,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run stacked feature importance for every condition in a final manifest."""
    del top_n  # Kept for CLI parity with direct-SVM feature importance.
    manifest_path = manifest_path.resolve()
    manifest = load_final_manifest(
        manifest_path,
        expected_run_type="stacked_condition_final_evaluation",
    )
    condition_ids = [
        str(row["condition_id"])
        for row in manifest.get("condition_results", [])
    ]
    results = [
        run_stacked_condition_importance_analysis(
            manifest_path,
            condition_id=condition_id,
        )
        for condition_id in condition_ids
    ]
    return {
        "run_type": "stacked_feature_importance_all_conditions",
        "source_manifest": relative_to_project(
            resolve_final_unit_paths(
                manifest_path,
                manifest,
                condition_id=condition_ids[0] if condition_ids else None,
            ).project_root,
            manifest_path,
        ) if condition_ids else str(manifest_path.resolve()),
        "condition_count": len(results),
        "conditions": results,
    }
