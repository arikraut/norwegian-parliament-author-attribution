"""Paired significance calculations and report collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score

from data_pipeline.utils import relative_to_project

from .artifacts import ResultArtifacts
from .config import ResultSystem, SystemComparison
from .profiling_effects import comparison_system_lookup


def _bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iterations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lower, ci_upper) for macro F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    point = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float, float]:
    """Return (b, c, chi2_stat, p_value) for McNemar's test with continuity correction.

    b = A correct, B wrong
    c = A wrong,   B correct
    """
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = float((abs(b - c) - 1) ** 2 / (b + c))
    p = float(chi2.sf(stat, df=1))
    return b, c, stat, p


def _mcnemar_uncorrected(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
) -> tuple[int, int, float, float]:
    """Return (b, c, chi2_stat, p_value) for uncorrected asymptotic McNemar."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = float((b - c) ** 2 / (b + c))
    p = float(chi2.sf(stat, df=1))
    return b, c, stat, p


def _prepare_predictions(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Validate and annotate one prediction table at the artifact boundary."""

    df = df.copy()
    required = {"id_speech", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["correct"] = df["y_true"] == df["y_pred"]
    return df


def _id_sample(ids) -> list[object]:
    """Return a short deterministic sample of mismatched IDs for error messages."""
    return sorted(ids, key=str)[:10]


def _canonical_id(value: object) -> str:
    """Return a stable speech-id string for duplicate checks."""
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


def _validate_unique_ids(path: Path, df: pd.DataFrame) -> None:
    """Ensure a prediction file has one row per speech id before merging."""
    canonical_ids = df["id_speech"].map(_canonical_id)
    duplicated = canonical_ids[canonical_ids.duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"{path}: duplicate id_speech values found before comparison "
            f"(n_duplicate_rows={len(duplicated)}, sample={_id_sample(set(duplicated))})."
        )


def _validate_prediction_alignment(
    path_a: Path,
    path_b: Path,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    """Ensure two prediction files describe the same speeches with the same labels."""
    ids_a = set(df_a["id_speech"])
    ids_b = set(df_b["id_speech"])
    if ids_a != ids_b:
        missing_from_b = ids_a - ids_b
        missing_from_a = ids_b - ids_a
        raise ValueError(
            "Prediction files must contain the same id_speech set. "
            f"{path_b} is missing {len(missing_from_b)} IDs from {path_a} "
            f"(sample: {_id_sample(missing_from_b)}); "
            f"{path_a} is missing {len(missing_from_a)} IDs from {path_b} "
            f"(sample: {_id_sample(missing_from_a)})."
        )

    mismatched_labels = merged[merged["y_true_a"] != merged["y_true_b"]]
    if not mismatched_labels.empty:
        sample = mismatched_labels["id_speech"].head(10).tolist()
        raise ValueError(
            "Prediction files must use the same y_true for every id_speech. "
            f"Found {len(mismatched_labels)} mismatched labels between {path_a} and {path_b} "
            f"(sample IDs: {sample})."
        )


def compare_prediction_frames(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    *,
    path_a: Path,
    path_b: Path,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    mcnemar_method: str = "asymptotic_uncorrected",
) -> dict[str, Any]:
    """Return paired significance statistics from two loaded prediction tables."""

    df_a = _prepare_predictions(df_a, path_a)
    df_b = _prepare_predictions(df_b, path_b)
    _validate_unique_ids(path_a, df_a)
    _validate_unique_ids(path_b, df_b)

    merged = df_a.merge(
        df_b[["id_speech", "y_true", "y_pred", "correct"]],
        on="id_speech",
        suffixes=("_a", "_b"),
        how="inner",
    )
    _validate_prediction_alignment(path_a, path_b, df_a, df_b, merged)
    n_matched = len(merged)

    f1_a, lo_a, hi_a = _bootstrap_macro_f1(
        merged["y_true_a"].values, merged["y_pred_a"].values,
        n_iterations=n_bootstrap, seed=seed,
    )
    f1_b, lo_b, hi_b = _bootstrap_macro_f1(
        merged["y_true_b"].values, merged["y_pred_b"].values,
        n_iterations=n_bootstrap, seed=seed,
    )

    if mcnemar_method == "continuity_corrected":
        mcnemar_test = _mcnemar
    elif mcnemar_method == "asymptotic_uncorrected":
        mcnemar_test = _mcnemar_uncorrected
    else:
        raise ValueError(f"Unsupported McNemar method: {mcnemar_method!r}")
    b, c, stat, p = mcnemar_test(
        merged["correct_a"].values,
        merged["correct_b"].values,
    )

    if p < 0.001:
        sig_label = "p < 0.001"
    elif p < 0.01:
        sig_label = "p < 0.01"
    elif p < 0.05:
        sig_label = "p < 0.05"
    else:
        sig_label = f"p = {p:.3f} (not significant at α=0.05)"

    if b + c > 0:
        if b > c:
            direction = f"{label_a} significantly better than {label_b}"
        else:
            direction = f"{label_b} significantly better than {label_a}"
        if p >= 0.05:
            direction = "no significant difference"
    else:
        direction = "perfect agreement — identical predictions"

    result = {
        "label_a": label_a,
        "label_b": label_b,
        "n_speeches": n_matched,
        "bootstrap_n_iterations": n_bootstrap,
        "system_a": {
            "macro_f1": round(f1_a, 4),
            "ci_95_lower": round(lo_a, 4),
            "ci_95_upper": round(hi_a, 4),
        },
        "system_b": {
            "macro_f1": round(f1_b, 4),
            "ci_95_lower": round(lo_b, 4),
            "ci_95_upper": round(hi_b, 4),
        },
        "mcnemar": {
            "method": mcnemar_method,
            "system_a_correct_system_b_wrong": b,
            "system_a_wrong_system_b_correct": c,
            "chi2_stat": round(stat, 4),
            "p_value": round(p, 6),
            "significance": sig_label,
            "conclusion": direction,
        },
    }

    return result


def run_comparison(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    mcnemar_method: str = "asymptotic_uncorrected",
) -> dict[str, Any]:
    """Load and compare two final-prediction artifacts."""

    return compare_prediction_frames(
        pd.read_csv(path_a),
        pd.read_csv(path_b),
        label_a,
        label_b,
        path_a=path_a,
        path_b=path_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
        mcnemar_method=mcnemar_method,
    )


def build_significance_summary_row(
    comparison: SystemComparison,
    source_system: ResultSystem,
    target_system: ResultSystem,
    result: dict[str, Any],
    json_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Convert one significance result dict into a flat summary row."""

    source_stats = result["system_a"]
    target_stats = result["system_b"]
    mcnemar = result["mcnemar"]
    return {
        "comparison_key": comparison.key,
        "comparison_label": comparison.label,
        "comparison_group": comparison.comparison_group,
        "comparison_purpose": comparison.purpose,
        "source_system_key": source_system.key,
        "source_system_label": source_system.label,
        "target_system_key": target_system.key,
        "target_system_label": target_system.label,
        "n_speeches": result["n_speeches"],
        "bootstrap_n_iterations": result["bootstrap_n_iterations"],
        "source_macro_f1": source_stats["macro_f1"],
        "source_ci_95_lower": source_stats["ci_95_lower"],
        "source_ci_95_upper": source_stats["ci_95_upper"],
        "target_macro_f1": target_stats["macro_f1"],
        "target_ci_95_lower": target_stats["ci_95_lower"],
        "target_ci_95_upper": target_stats["ci_95_upper"],
        "mcnemar_method": mcnemar["method"],
        "source_correct_target_wrong": mcnemar[
            "system_a_correct_system_b_wrong"
        ],
        "source_wrong_target_correct": mcnemar[
            "system_a_wrong_system_b_correct"
        ],
        "mcnemar_chi2_stat": mcnemar["chi2_stat"],
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_significance": mcnemar["significance"],
        "mcnemar_conclusion": mcnemar["conclusion"],
        "json_path": relative_to_project(project_root, json_path),
    }


def write_significance_outputs(
    systems: tuple[ResultSystem, ...],
    comparisons: tuple[SystemComparison, ...],
    *,
    artifacts: ResultArtifacts,
    output_dir: Path,
    project_root: Path,
    n_bootstrap: int,
    seed: int = 42,
    mcnemar_method: str = "asymptotic_uncorrected",
) -> dict[str, str]:
    """Write significance summaries and per-comparison JSON results."""

    section_dir = output_dir / "significance"
    json_dir = section_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    systems_by_key = comparison_system_lookup(systems)
    summary_rows: list[dict[str, Any]] = []
    comparison_results: dict[str, Any] = {}
    json_outputs: dict[str, str] = {}
    for comparison in comparisons:
        source_system = systems_by_key[comparison.source_system_key]
        target_system = systems_by_key[comparison.target_system_key]
        source_relative_path = source_system.final_predictions_path
        target_relative_path = target_system.final_predictions_path
        result = compare_prediction_frames(
            artifacts.read_csv(source_relative_path),
            artifacts.read_csv(target_relative_path),
            source_system.label,
            target_system.label,
            path_a=artifacts.path(source_relative_path),
            path_b=artifacts.path(target_relative_path),
            n_bootstrap=n_bootstrap,
            seed=seed,
            mcnemar_method=mcnemar_method,
        )
        comparison_results[comparison.key] = result
        json_path = json_dir / f"{comparison.key}.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_outputs[f"{comparison.key}_json"] = str(json_path)
        summary_rows.append(
            build_significance_summary_row(
                comparison,
                source_system,
                target_system,
                result,
                json_path,
                project_root=project_root,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = section_dir / "comparisons.csv"
    summary.to_csv(summary_path, index=False)
    combined_json_path = section_dir / "comparisons.json"
    combined_json_path.write_text(
        json.dumps(comparison_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "comparisons_csv": str(summary_path),
        "comparisons_json": str(combined_json_path),
    }
    outputs.update(json_outputs)
    return outputs
