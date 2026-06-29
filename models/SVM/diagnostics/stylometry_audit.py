from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.utils import (
    find_project_root,
    read_optional_csv,
    read_required_csv,
    relative_to_project,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    """Parse the stylometry audit CLI."""
    parser = argparse.ArgumentParser(
        description="Summarize stylometry ablations, quality, and drift for one attribution run.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to one attribution results directory under results/models/...",
    )
    return parser.parse_args()


def _blocks_use_stylometry(block_value: Any) -> bool:
    """Return whether a candidate feature block includes stylometry."""
    parts = {part.strip() for part in str(block_value).split("+") if part.strip()}
    return "stylo" in parts or "all" in parts


def _best_feature_rows(summary_df: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    """Keep the best candidate row for each feature-set family."""
    if summary_df.empty:
        return summary_df.copy()
    score_col = f"eval_mean_{selection_metric}"
    if score_col not in summary_df.columns:
        raise KeyError(f"Selection metric column is missing from candidate_summary.csv: {score_col}")

    ranked = summary_df.sort_values(
        by=[score_col, "eval_mean_accuracy", "feature_set", "c_value", "class_weight"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best_by_feature = ranked.groupby("feature_set", sort=False, as_index=False).first()
    best_by_feature["uses_stylometry"] = best_by_feature["blocks"].map(_blocks_use_stylometry)
    return best_by_feature


def _quality_summary(quality_df: pd.DataFrame, low_variance_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize stylometry extraction quality issues."""
    if quality_df.empty:
        return {
            "has_quality_report": False,
            "split_count": 0,
            "total_substitutions": 0,
            "nonfinite_output_cells": 0,
            "max_low_variance_feature_count": 0,
            "max_all_zero_feature_count": 0,
            "low_variance_rows": 0,
        }

    return {
        "has_quality_report": True,
        "split_count": int(len(quality_df)),
        "total_substitutions": int(quality_df.get("total_substitutions", pd.Series(dtype=int)).fillna(0).sum()),
        "nonfinite_output_cells": int(
            quality_df.get("nonfinite_output_cells", pd.Series(dtype=int)).fillna(0).sum()
        ),
        "max_low_variance_feature_count": int(
            quality_df.get("low_variance_feature_count", pd.Series(dtype=int)).fillna(0).max()
        ),
        "max_all_zero_feature_count": int(
            quality_df.get("all_zero_feature_count", pd.Series(dtype=int)).fillna(0).max()
        ),
        "low_variance_rows": int(len(low_variance_df)),
    }


def _drift_summary(drift_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize train/eval stylometry drift diagnostics."""
    if drift_df.empty:
        return {
            "has_drift_report": False,
            "unit_count": 0,
            "mean_standardized_mean_gap": 0.0,
            "max_standardized_mean_gap": 0.0,
            "features_over_gap_0_5": 0,
            "features_over_gap_1_0": 0,
        }

    return {
        "has_drift_report": True,
        "unit_count": int(len(drift_df)),
        "mean_standardized_mean_gap": float(drift_df["mean_standardized_mean_gap"].mean()),
        "max_standardized_mean_gap": float(drift_df["max_standardized_mean_gap"].max()),
        "features_over_gap_0_5": int(drift_df["features_over_gap_0_5"].sum()),
        "features_over_gap_1_0": int(drift_df["features_over_gap_1_0"].sum()),
    }


def _decision_payload(
    best_by_feature: pd.DataFrame,
    selection_metric: str,
    quality_summary: dict[str, Any],
    drift_summary: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether stylometry is worth keeping for the current run."""
    score_col = f"eval_mean_{selection_metric}"
    nonstylo_df = best_by_feature[~best_by_feature["uses_stylometry"]].copy()
    stylo_df = best_by_feature[best_by_feature["uses_stylometry"]].copy()

    best_nonstylo_row = nonstylo_df.sort_values(
        by=[score_col, "eval_mean_accuracy"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0] if not nonstylo_df.empty else None
    best_stylo_row = stylo_df.sort_values(
        by=[score_col, "eval_mean_accuracy"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0] if not stylo_df.empty else None

    best_nonstylo_score = float(best_nonstylo_row[score_col]) if best_nonstylo_row is not None else None
    best_stylo_score = float(best_stylo_row[score_col]) if best_stylo_row is not None else None
    delta = (
        float(best_stylo_score - best_nonstylo_score)
        if best_stylo_score is not None and best_nonstylo_score is not None
        else None
    )

    quality_issue = bool(
        quality_summary["total_substitutions"] > 0
        or quality_summary["nonfinite_output_cells"] > 0
    )
    drift_issue = bool(
        drift_summary["mean_standardized_mean_gap"] > 0.5
        or drift_summary["max_standardized_mean_gap"] > 1.0
    )
    go_decision = bool(
        best_stylo_score is not None
        and best_nonstylo_score is not None
        and best_stylo_score > best_nonstylo_score
        and not quality_issue
        and not drift_issue
    )

    return {
        "selection_metric": selection_metric,
        "best_nonstylometry_feature_set": (
            str(best_nonstylo_row["feature_set"]) if best_nonstylo_row is not None else None
        ),
        "best_nonstylometry_score": best_nonstylo_score,
        "best_stylometry_feature_set": str(best_stylo_row["feature_set"]) if best_stylo_row is not None else None,
        "best_stylometry_score": best_stylo_score,
        "stylometry_delta_vs_nonstylometry": delta,
        "quality_issue_detected": quality_issue,
        "drift_issue_detected": drift_issue,
        "go_decision": "go" if go_decision else "no_go",
    }


def _write_conclusion_markdown(
    path: Path,
    *,
    decision: dict[str, Any],
    quality_summary: dict[str, Any],
    drift_summary: dict[str, Any],
) -> None:
    """Write a report-ready stylometry audit conclusion."""
    lines = [
        "# Stylometry Audit Conclusion",
        "",
        f"- Decision: `{decision['go_decision']}`",
        f"- Selection metric: `{decision['selection_metric']}`",
        f"- Best non-stylometry feature set: `{decision['best_nonstylometry_feature_set']}`",
        f"- Best stylometry feature set: `{decision['best_stylometry_feature_set']}`",
        f"- Stylometry delta vs non-stylometry: `{decision['stylometry_delta_vs_nonstylometry']}`",
        "",
        "## Evidence",
        "",
        f"- Total substitution count from feature extraction: `{quality_summary['total_substitutions']}`",
        f"- Non-finite output cells after extraction: `{quality_summary['nonfinite_output_cells']}`",
        f"- Max low-variance feature count across splits: `{quality_summary['max_low_variance_feature_count']}`",
        f"- Mean stylometry drift gap across units: `{drift_summary['mean_standardized_mean_gap']}`",
        f"- Max stylometry drift gap across units: `{drift_summary['max_standardized_mean_gap']}`",
        "",
        "## Interpretation",
        "",
    ]

    if decision["go_decision"] == "go":
        lines.append(
            "Stylometry currently looks safe enough to keep investigating in Phase 2 because it improved over the non-stylometry baseline without triggering the configured quality or drift warnings."
        )
    else:
        lines.append(
            "Stylometry should stay a documented negative or unresolved result for Phase 1 unless later reruns show a clear improvement over the non-stylometry baseline with clean quality and drift reports."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stylometry_audit(results_dir: Path) -> dict[str, Any]:
    """Create stylometry quality, drift, and go/no-go artifacts for one run."""
    results_dir = results_dir.resolve()
    project_root = find_project_root(results_dir)

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Results manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_metric = str(manifest.get("selection_metric", "macro_f1"))

    summary_df = read_required_csv(results_dir / "candidate_summary.csv")
    best_by_feature = _best_feature_rows(summary_df, selection_metric)
    score_col = f"eval_mean_{selection_metric}"

    materialized_root = project_root / manifest["materialized_root"]
    materialization_manifest = json.loads((materialized_root / "manifest.json").read_text(encoding="utf-8"))
    split_name = str(materialization_manifest["split_name"])
    row_feature_name = str(materialization_manifest["row_feature_name"])
    row_feature_dir = project_root / "data" / "splits" / split_name / "row_features" / row_feature_name

    quality_df = read_optional_csv(row_feature_dir / "stylometry_quality_report.csv")
    low_variance_df = read_optional_csv(row_feature_dir / "stylometry_low_variance_report.csv")
    drift_df = read_optional_csv(materialized_root / "stylometry_drift_summary.csv")

    quality_summary = _quality_summary(quality_df, low_variance_df)
    drift_summary = _drift_summary(drift_df)
    decision = _decision_payload(best_by_feature, selection_metric, quality_summary, drift_summary)

    best_nonstylo_score = decision["best_nonstylometry_score"]
    if best_nonstylo_score is not None:
        best_by_feature["delta_vs_best_nonstylometry"] = best_by_feature[score_col] - best_nonstylo_score
    else:
        best_by_feature["delta_vs_best_nonstylometry"] = pd.NA

    audit_dir = results_dir / "stylometry_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    ablation_summary_path = audit_dir / "ablation_summary.csv"
    drift_summary_path = audit_dir / "drift_summary.csv"
    quality_summary_path = audit_dir / "quality_summary.csv"
    decision_path = audit_dir / "decision.json"
    conclusion_path = audit_dir / "go_no_go.md"

    best_by_feature.to_csv(ablation_summary_path, index=False)
    if not drift_df.empty:
        drift_df.to_csv(drift_summary_path, index=False)
    else:
        pd.DataFrame([drift_summary]).to_csv(drift_summary_path, index=False)
    pd.DataFrame([quality_summary]).to_csv(quality_summary_path, index=False)
    write_json(decision_path, decision)
    _write_conclusion_markdown(
        conclusion_path,
        decision=decision,
        quality_summary=quality_summary,
        drift_summary=drift_summary,
    )

    audit_manifest = {
        "results_dir": relative_to_project(project_root, results_dir),
        "audit_dir": relative_to_project(project_root, audit_dir),
        "selection_metric": selection_metric,
        "source_paths": {
            "model_manifest": relative_to_project(project_root, manifest_path),
            "candidate_summary": relative_to_project(project_root, results_dir / "candidate_summary.csv"),
            "materialization_manifest": relative_to_project(project_root, materialized_root / "manifest.json"),
            "row_feature_dir": relative_to_project(project_root, row_feature_dir),
        },
        "artifacts": {
            "ablation_summary": relative_to_project(project_root, ablation_summary_path),
            "drift_summary": relative_to_project(project_root, drift_summary_path),
            "quality_summary": relative_to_project(project_root, quality_summary_path),
            "decision": relative_to_project(project_root, decision_path),
            "conclusion": relative_to_project(project_root, conclusion_path),
        },
        "decision": decision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(audit_dir / "manifest.json", audit_manifest)
    return audit_manifest


def main() -> None:
    """CLI entry point for stylometry audit generation."""
    args = _parse_args()
    manifest = run_stylometry_audit(Path(args.results_dir))
    print(f"Stylometry audit completed for {manifest['results_dir']}")
    print(f"Decision: {manifest['decision']['go_decision']}")
    print(f"Audit dir: {manifest['audit_dir']}")


if __name__ == "__main__":
    main()
