"""Build feature-importance reports from saved result artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.utils import find_project_root, resolve_project_path
from thesis_reporting.config import configured_feature_importance_systems
from thesis_reporting.feature_importance import (
    requested_system_keys,
    run_feature_importance_additions,
    selected_systems,
)


def parse_args() -> argparse.Namespace:
    """Parse feature-importance collection arguments."""

    parser = argparse.ArgumentParser(
        description="Build feature-importance reports from saved models.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/result_additions/feature_importance"),
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated configured system keys; defaults to all supported systems.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Resolve project paths and collect selected importance outputs."""

    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    results_dir = resolve_project_path(project_root, args.results_dir)
    output_dir = resolve_project_path(project_root, args.output_dir)
    systems = (
        selected_systems(requested_system_keys(args.systems))
        if args.systems
        else configured_feature_importance_systems()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_feature_importance_additions(
        systems,
        project_root=project_root,
        results_dir=results_dir,
        output_dir=output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
