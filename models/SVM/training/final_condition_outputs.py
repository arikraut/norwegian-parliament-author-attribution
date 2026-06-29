"""Shared final-condition artifact writing for attribution models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.utils import relative_to_project, write_json


JsonDict = dict[str, Any]


def final_condition_roots(results_dir: Path, artifacts_dir: Path) -> tuple[Path, Path]:
    """Create and return the shared final-by-condition roots."""
    final_by_condition_dir = results_dir / "final_by_condition"
    final_artifacts_by_condition_dir = artifacts_dir / "final_by_condition"
    final_by_condition_dir.mkdir(parents=True, exist_ok=True)
    final_artifacts_by_condition_dir.mkdir(parents=True, exist_ok=True)
    return final_by_condition_dir, final_artifacts_by_condition_dir


def write_final_condition_output(
    *,
    project_root: Path,
    condition_results_dir: Path,
    prediction_frame: pd.DataFrame,
    metrics_payload: JsonDict,
    resolved_candidate_payload: JsonDict,
    condition_result: JsonDict,
) -> JsonDict:
    """Write the common final-condition files and return manifest metadata."""
    condition_results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = condition_results_dir / "final_test_predictions.csv"
    metrics_path = condition_results_dir / "final_test_metrics.json"
    resolved_candidate_path = condition_results_dir / "resolved_candidate.json"

    prediction_frame.to_csv(predictions_path, index=False)
    write_json(metrics_path, metrics_payload)
    write_json(resolved_candidate_path, resolved_candidate_payload)

    return {
        **condition_result,
        "metrics_path": relative_to_project(project_root, metrics_path),
        "predictions_path": relative_to_project(project_root, predictions_path),
        "resolved_candidate_path": relative_to_project(
            project_root,
            resolved_candidate_path,
        ),
    }


def build_final_summary_row(
    *,
    condition_id: str,
    condition_label: str,
    candidate_id: str,
    dev_summary: JsonDict,
    final_metrics: JsonDict,
    extra_fields: JsonDict | None = None,
) -> JsonDict:
    """Build one row for final_condition_summary.csv."""
    row: JsonDict = {
        "condition_id": condition_id,
        "condition_label": condition_label,
        "candidate_id": candidate_id,
    }
    for key, value in dev_summary.items():
        if key.startswith("eval_mean_"):
            row[key.replace("eval_mean_", "dev_mean_", 1)] = value
        elif key.startswith("eval_std_"):
            row[key.replace("eval_std_", "dev_std_", 1)] = value
    for key, value in final_metrics.items():
        row[f"final_{key}"] = value
    if extra_fields:
        row.update(extra_fields)
    return row


def write_final_condition_summary(results_dir: Path, rows: list[JsonDict]) -> Path:
    """Write final_condition_summary.csv and return its path."""
    path = results_dir / "final_condition_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
