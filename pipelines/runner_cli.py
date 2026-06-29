"""Small CLI helpers shared by the repository-root phase runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
STAGE_CHOICES = ("all", "dev", "final")


def add_common_attribution_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_smoke: bool,
    include_profiling_representation: bool = False,
    include_profiling_scope: bool = False,
    config_help: str = "Override the staged model config.",
    smoke_help: str = "Run the small dev-only smoke config.",
) -> None:
    """Add the CLI flags shared by attribution phase runners."""
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default=None if include_smoke else "all",
        help=(
            "Run dev search, final evaluation, or both. Defaults to dev for "
            "--smoke and all otherwise."
            if include_smoke
            else "Run dev search, final evaluation, or both (default: all)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=config_help,
    )
    parser.add_argument(
        "--selected-candidates-path",
        type=Path,
        default=None,
        help="Selected candidates JSON to use for final evaluation.",
    )
    if include_profiling_representation:
        parser.add_argument(
            "--profiling-representation",
            choices=["probability", "hard"],
            default="probability",
            help="Predicted profiling representation to use (default: probability).",
        )
    if include_profiling_scope:
        parser.add_argument(
            "--profiling-scope",
            choices=["all", "single_signal"],
            default="all",
            help="Profiling signal scope to use (default: all).",
        )
    if include_smoke:
        parser.add_argument(
            "--smoke",
            action="store_true",
            help=smoke_help,
        )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rerun phase stages even if their manifests already exist.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip attribution diagnostics after model stages.",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=50,
        help="Number of top confusion pairs to keep in diagnostics (default: 50).",
    )


def validate_smoke_stage(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject final/all stage requests for smoke runners before orchestration starts."""
    if getattr(args, "smoke", False) and args.stage not in {None, "dev"}:
        parser.error("--smoke only supports --stage dev.")


def selected_stage(args: argparse.Namespace) -> str:
    """Return the effective phase stage after applying the smoke default."""
    if args.stage is not None:
        return args.stage
    return "dev" if getattr(args, "smoke", False) else "all"


def print_manifest(title: str, manifest: JsonDict) -> None:
    """Print a runner completion banner and its JSON manifest."""
    print(f"\n== {title} ==")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
