"""Plot annual training-speech support for the configured temporal split."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.utils import resolve_project_path

try:
    from scripts.plot_authorwise_training_timeline import (
        DEFAULT_FIGURE_DIR,
        configured_split_paths,
        generate_split_figures,
        load_test_period_counts,
        load_training_counts,
        project_root,
        resolve_optional_path,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from plot_authorwise_training_timeline import (
        DEFAULT_FIGURE_DIR,
        configured_split_paths,
        generate_split_figures,
        load_test_period_counts,
        load_training_counts,
        project_root,
        resolve_optional_path,
    )


EXPECTED_AUTHORS = 15
EXPECTED_TRAIN_ROWS = 13_366
EXPECTED_TEST_ROWS = 1_500
ELECTIONS = [2005, 2009, 2013, 2017, 2021]
YEARS = list(range(2005, 2025))
TEST_ELECTIONS = [2021]
TEST_YEARS = list(range(2021, 2025))
TEST_START_YEAR = 2021
PARTY_ORDER = ["Ap", "H", "FrP", "SV", "Sp", "KrF"]
DEFAULT_TEMPORAL_SPLIT_CONFIG = Path(
    "data_pipeline/configs/splits/bokmal_temporal.toml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot annual training-speech counts for the temporal split."
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_TEMPORAL_SPLIT_CONFIG,
        help="Split config used to locate the generated corpus directory.",
    )
    parser.add_argument(
        "--train-corpus",
        type=Path,
        default=None,
        help=(
            "Training corpus CSV. Defaults to "
            "<configured splits_dir>/<split name>/corpus/train.csv."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_DIR / "temporal_training_timeline.png",
        help="Output path for the highest-support-author overview.",
    )
    parser.add_argument(
        "--test-corpus",
        type=Path,
        default=None,
        help=(
            "Test corpus CSV. Defaults to "
            "<configured splits_dir>/<split name>/corpus/test.csv."
        ),
    )
    parser.add_argument(
        "--source-corpus",
        type=Path,
        default=None,
        help="Cleaned source corpus. Defaults to the split config source_dataset.",
    )
    parser.add_argument(
        "--party-output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for the individual party figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    root = project_root()
    split_paths = configured_split_paths(root, args.split_config)
    args.train_corpus = resolve_optional_path(
        root,
        args.train_corpus,
        split_paths.corpus_dir / "train.csv",
    )
    args.test_corpus = resolve_optional_path(
        root,
        args.test_corpus,
        split_paths.corpus_dir / "test.csv",
    )
    args.source_corpus = resolve_optional_path(
        root,
        args.source_corpus,
        split_paths.source_dataset,
    )
    args.output = resolve_project_path(root, args.output)
    args.party_output_dir = resolve_project_path(root, args.party_output_dir)
    return args


def main() -> None:
    args = parse_args()
    counts, authors = load_training_counts(
        args.train_corpus,
        expected_authors=EXPECTED_AUTHORS,
        expected_train_rows=EXPECTED_TRAIN_ROWS,
        allowed_elections=ELECTIONS,
        years=YEARS,
    )
    test_counts, test_authors = load_training_counts(
        args.test_corpus,
        expected_authors=EXPECTED_AUTHORS,
        expected_train_rows=EXPECTED_TEST_ROWS,
        allowed_elections=TEST_ELECTIONS,
        years=TEST_YEARS,
    )
    if set(authors["id_person"]) != set(test_authors["id_person"]):
        raise ValueError("Temporal train and test corpora contain different authors.")
    available_test_counts = load_test_period_counts(
        args.source_corpus,
        args.test_corpus,
        selected_author_ids=set(authors["id_person"]),
        test_elections=TEST_ELECTIONS,
        years=TEST_YEARS,
    )

    output_paths = generate_split_figures(
        counts,
        authors,
        overview_output=args.output,
        party_output_dir=args.party_output_dir,
        filename_prefix="temporal_training_timeline",
        party_order=PARTY_ORDER,
        election_years=ELECTIONS,
        years=YEARS,
        dpi=args.dpi,
        test_counts=test_counts,
        available_test_counts=available_test_counts,
        test_start_year=TEST_START_YEAR,
    )
    for output_path in output_paths:
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
