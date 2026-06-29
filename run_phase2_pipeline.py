#!/usr/bin/env python3
"""Run Phase 2 profiling classifiers from the repository root."""

from __future__ import annotations

import argparse

from pipelines import run_phase2_track
from pipelines.runner_cli import print_manifest


def _parse_args() -> argparse.Namespace:
    """Parse Phase 2 runner arguments from the command line."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 profiling training as a standalone phase.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small dev-only Phase 2 profiling smoke path.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rerun phase stages even if their manifests already exist.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Run Phase 2 and print the phase manifest."""
    args = _parse_args()
    manifest = run_phase2_track(
        smoke=args.smoke,
        rebuild=args.rebuild,
    )
    print_manifest("Phase 2 pipeline completed", manifest)


if __name__ == "__main__":
    main()
