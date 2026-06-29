#!/usr/bin/env python3
"""Run coefficient feature-importance analysis for final attribution models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.SVM.importance.feature_importance_stacked import run_stacked_importance_analysis
from models.SVM.importance.feature_importance_svm import (
    load_final_manifest,
    run_importance_analysis,
)


def _parse_args() -> argparse.Namespace:
    """Parse feature-importance command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze feature importance for a final attribution manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to a final model results manifest.json.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of positive and negative per-author features to write.",
    )
    return parser.parse_args()


def run_feature_importance(
    manifest_path: Path,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run feature-importance analysis for every condition in a final manifest."""
    manifest = load_final_manifest(manifest_path)
    run_type = manifest.get("run_type")
    if run_type == "condition_final_evaluation":
        return run_importance_analysis(manifest_path, top_n=top_n)
    if run_type == "stacked_condition_final_evaluation":
        return run_stacked_importance_analysis(manifest_path, top_n=top_n)
    raise ValueError(
        f"{manifest_path} has run_type={run_type!r}; expected a final model "
        "manifest with run_type 'condition_final_evaluation' or "
        "'stacked_condition_final_evaluation'."
    )


def main() -> None:
    """Run the CLI entry point and print the analysis summary."""
    args = _parse_args()
    summary = run_feature_importance(
        args.manifest,
        top_n=args.top_n,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
