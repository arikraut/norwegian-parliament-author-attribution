#!/usr/bin/env python3
"""Run Phase 3B oracle stacked attribution with ground-truth profiling labels from the repository root."""

from __future__ import annotations

import argparse

from pipelines import run_phase3b_oracle_track
from pipelines.runner_cli import add_common_attribution_arguments, print_manifest


def _parse_args() -> argparse.Namespace:
    """Parse Phase 3B oracle runner arguments from the command line."""
    parser = argparse.ArgumentParser(
        description="Run Phase 3B oracle stacked attribution with ground-truth profiling labels.",
    )
    add_common_attribution_arguments(
        parser,
        include_smoke=False,
        include_profiling_scope=True,
        config_help="Override the Phase 3B oracle stacked model config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Phase 3B oracle and print the phase manifest."""
    args = _parse_args()
    manifest = run_phase3b_oracle_track(
        stage=args.stage,
        config_path=args.config,
        rebuild=args.rebuild,
        skip_diagnostics=args.skip_diagnostics,
        top_confusions=args.top_confusions,
        selected_candidates_path_override=args.selected_candidates_path,
        profiling_scope=args.profiling_scope,
    )
    print_manifest("Phase 3B oracle pipeline completed", manifest)


if __name__ == "__main__":
    main()
