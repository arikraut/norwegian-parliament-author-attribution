#!/usr/bin/env python3
"""Data-only pipeline runner.

Run the default author-wise data pipeline:
    python run_data_pipeline.py

Run temporal split/features/materialization without model training:
    python run_data_pipeline.py --preset temporal

Run only one materialization stage:
    python run_data_pipeline.py --preset temporal --stage dev

Override individual configs:
    python run_data_pipeline.py \\
        --split-config data_pipeline/configs/splits/bokmal_temporal.toml \\
        --feature-config data_pipeline/configs/features/bokmal_temporal.toml \\
        --materialization-config data_pipeline/configs/materializations/bokmal_temporal_char_word_stylo.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import DATA_PIPELINE_PRESETS, run_data_pipeline
from pipelines.runner_cli import print_manifest


DEFAULT_PRESET = "authorwise"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run split creation, feature generation, and materialization only. "
            "No model training or diagnostics are run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATA_PIPELINE_PRESETS),
        default=DEFAULT_PRESET,
        help=f"Config preset to run (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Override the preset split config path.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=None,
        help="Override the preset feature-generation config path.",
    )
    parser.add_argument(
        "--materialization-config",
        type=Path,
        default=None,
        help="Override the preset materialization config path.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "dev", "final"],
        default="all",
        help=(
            "Materialization stage to run. 'all' runs every stage defined by "
            "the materialization config (default: all)."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rerun every data stage even if stage manifests already exist.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Pipeline manifest name under results/pipelines/. Defaults to data_<preset> or data_custom.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preset = DATA_PIPELINE_PRESETS[args.preset]
    has_overrides = any(
        value is not None
        for value in (args.split_config, args.feature_config, args.materialization_config)
    )
    pipeline_name = args.name or (
        "data_custom" if has_overrides else f"data_{args.preset.replace('-', '_')}"
    )

    manifest = run_data_pipeline(
        split_config=args.split_config or preset["split_config"],
        feature_config=args.feature_config or preset["feature_config"],
        materialization_config=args.materialization_config or preset["materialization_config"],
        materialization_stage=args.stage,
        rebuild=args.rebuild,
        pipeline_name=pipeline_name,
    )
    print_manifest("Data pipeline completed", manifest)


if __name__ == "__main__":
    main()
