"""Create a profiling validation-fold balance report from existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.utils import find_project_root, resolve_project_path
from thesis_reporting.fold_balance import (
    DEFAULT_TARGETS,
    analyze_fold_balance,
    load_fold_inputs,
    write_fold_balance_report,
)


def parse_args() -> argparse.Namespace:
    """Parse profiling fold-balance report arguments."""

    parser = argparse.ArgumentParser(
        description="Create a Markdown report for target balance across profiling folds.",
    )
    parser.add_argument("--split", default="bokmal_profiling")
    parser.add_argument("--feature", default=None)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=["train", "val"],
        default=["val"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Report path relative to the project root; defaults below "
            "results/reports/profiling_fold_balance/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Resolve paths, analyze existing fold inputs, and write the report."""

    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    feature = args.feature or args.split
    output_value = args.output or Path(
        "results/reports/profiling_fold_balance"
    ) / f"{args.split}.md"
    output_path = resolve_project_path(project_root, output_value)
    targets = tuple(args.targets)
    roles = tuple(args.roles)
    inputs = load_fold_inputs(
        project_root,
        split=args.split,
        feature=feature,
        targets=targets,
    )
    analysis = analyze_fold_balance(
        inputs,
        split=args.split,
        feature=feature,
        targets=targets,
        roles=roles,
    )
    write_fold_balance_report(
        output_path,
        analysis,
        project_root=project_root,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
