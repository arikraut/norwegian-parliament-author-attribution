#!/usr/bin/env python3
"""Run Phase 1B stacked attribution from the repository root."""

from __future__ import annotations

import argparse

from pipelines import run_phase1b_track
from pipelines.runner_cli import (
    add_common_attribution_arguments,
    print_manifest,
    selected_stage,
    validate_smoke_stage,
)


def _parse_args() -> argparse.Namespace:
    """Parse Phase 1B runner arguments from the command line."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1B stacked attribution as a standalone phase.",
    )
    add_common_attribution_arguments(
        parser,
        include_smoke=True,
        config_help="Override the staged Phase 1B model config.",
        smoke_help="Run the small dev-only Phase 1B smoke config.",
    )
    args = parser.parse_args()
    validate_smoke_stage(parser, args)
    return args


def main() -> None:
    """Run Phase 1B and print the phase manifest."""
    args = _parse_args()
    manifest = run_phase1b_track(
        stage=selected_stage(args),
        config_path=args.config,
        smoke=args.smoke,
        rebuild=args.rebuild,
        skip_diagnostics=args.skip_diagnostics,
        top_confusions=args.top_confusions,
        selected_candidates_path_override=args.selected_candidates_path,
    )
    print_manifest("Phase 1B pipeline completed", manifest)


if __name__ == "__main__":
    main()
