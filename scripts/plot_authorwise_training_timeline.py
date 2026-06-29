"""Plot annual training-speech support by author.

The figure is generated directly from the configured author-wise training
corpus. It does not reconstruct split membership or depend on model-prediction
outputs.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from data_pipeline.utils import find_project_root, resolve_project_path

EXPECTED_AUTHORS = 50
EXPECTED_TRAIN_ROWS = 51_275
ELECTIONS = [2001, 2005, 2009, 2013, 2017, 2021]
YEARS = list(range(2001, 2025))
PARTY_ORDER = ["Ap", "H", "FrP", "SV", "Sp", "KrF", "V"]
DEFAULT_AUTHORWISE_SPLIT_CONFIG = Path(
    "data_pipeline/configs/splits/bokmal_authorwise.toml"
)
DEFAULT_FIGURE_DIR = Path("results/figures/splits")


class SplitPaths(NamedTuple):
    """Resolved paths derived from a split config."""

    corpus_dir: Path
    source_dataset: Path


def project_root() -> Path:
    return find_project_root(Path(__file__).resolve())


def configured_split_paths(project_root: Path, split_config_path: Path) -> SplitPaths:
    """Resolve corpus and source-corpus paths from a split config."""
    resolved_config_path = resolve_project_path(project_root, split_config_path)
    with resolved_config_path.open("rb") as handle:
        split_config = tomllib.load(handle)

    split_name = str(split_config["split"]["name"])
    data_config = split_config["data"]
    splits_dir = resolve_project_path(
        project_root,
        data_config.get("splits_dir", "data/splits"),
    )
    source_dataset = resolve_project_path(project_root, data_config["source_dataset"])
    return SplitPaths(
        corpus_dir=splits_dir / split_name / "corpus",
        source_dataset=source_dataset,
    )


def resolve_optional_path(
    project_root: Path,
    path_value: Path | None,
    default_path: Path,
) -> Path:
    """Resolve a CLI path relative to the project root, or use a resolved default."""
    if path_value is None:
        return default_path
    return resolve_project_path(project_root, path_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot annual training-speech counts for each selected author."
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_AUTHORWISE_SPLIT_CONFIG,
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
        default=DEFAULT_FIGURE_DIR / "authorwise_training_timeline.png",
        help="Output path for the highest-support-author overview.",
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
    args.output = resolve_project_path(root, args.output)
    args.party_output_dir = resolve_project_path(root, args.party_output_dir)
    return args


def normalized_id(values: pd.Series) -> pd.Series:
    """Normalize CSV identifiers that may have been serialized as floats."""
    numeric = pd.to_numeric(values, errors="raise")
    if not np.all(np.isclose(numeric, np.round(numeric))):
        raise ValueError("Expected integer-valued speech and person identifiers.")
    return numeric.round().astype("int64")


def load_training_counts(
    train_corpus_path: Path,
    *,
    expected_authors: int = EXPECTED_AUTHORS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    allowed_elections: list[int] = ELECTIONS,
    years: list[int] = YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_rows = pd.read_csv(
        train_corpus_path,
        usecols=["date", "id_speech", "id_person", "name", "party", "election"],
    )
    training_rows["id_speech"] = normalized_id(training_rows["id_speech"])
    training_rows["id_person"] = normalized_id(training_rows["id_person"])
    training_rows["election"] = pd.to_numeric(
        training_rows["election"], errors="raise"
    ).astype(int)
    training_rows["year"] = pd.to_datetime(
        training_rows["date"], format="%Y-%m-%d", errors="raise"
    ).dt.year

    if len(training_rows) != expected_train_rows:
        raise ValueError(
            f"Training corpus contains {len(training_rows):,} rows; expected "
            f"{expected_train_rows:,}."
        )
    if not training_rows["id_speech"].is_unique:
        raise ValueError("Training corpus contains duplicate speech identifiers.")
    if not set(training_rows["election"].unique()).issubset(allowed_elections):
        raise ValueError("Training corpus contains an unexpected election period.")
    if not set(training_rows["year"].unique()).issubset(years):
        raise ValueError(
            f"Training corpus contains a year outside {years[0]}--{years[-1]}."
        )

    author_consistency = training_rows.groupby("id_person").agg(
        names=("name", "nunique"), parties=("party", "nunique")
    )
    if (author_consistency > 1).any().any():
        raise ValueError("An author has inconsistent name or party metadata.")

    authors = training_rows.groupby("id_person", as_index=False).agg(
        name=("name", "first"),
        party=("party", "first"),
        speech_count=("id_speech", "size"),
    )
    if len(authors) != expected_authors:
        raise ValueError(f"Expected {expected_authors} authors, found {len(authors)}.")

    counts = (
        training_rows.groupby(["id_person", "year"])
        .size()
        .rename("speech_count")
        .reset_index()
    )
    return counts, authors


def load_test_period_counts(
    source_corpus_path: Path,
    test_corpus_path: Path,
    *,
    selected_author_ids: set[int],
    test_elections: list[int],
    years: list[int],
) -> pd.DataFrame:
    source_rows = pd.read_csv(
        source_corpus_path,
        usecols=["date", "id_speech", "id_person", "election"],
    )
    source_rows["id_speech"] = normalized_id(source_rows["id_speech"])
    source_rows["id_person"] = normalized_id(source_rows["id_person"])
    source_rows["election"] = pd.to_numeric(
        source_rows["election"], errors="raise"
    ).astype(int)
    source_rows["year"] = pd.to_datetime(
        source_rows["date"], format="%Y-%m-%d", errors="raise"
    ).dt.year

    candidate_rows = source_rows.loc[
        source_rows["id_person"].isin(selected_author_ids)
        & source_rows["election"].isin(test_elections)
    ].copy()
    if not set(candidate_rows["year"].unique()).issubset(years):
        raise ValueError("Test-period corpus contains an unexpected calendar year.")

    retained_test = pd.read_csv(test_corpus_path, usecols=["id_speech"])
    retained_test_ids = set(normalized_id(retained_test["id_speech"]))
    candidate_ids = set(candidate_rows["id_speech"])
    missing_test_ids = retained_test_ids - candidate_ids
    if missing_test_ids:
        raise ValueError(
            f"Frozen source corpus is missing {len(missing_test_ids)} retained test rows."
        )

    covered_authors = set(candidate_rows["id_person"].unique())
    if covered_authors != selected_author_ids:
        raise ValueError("Test-period corpus does not cover every selected author.")

    return (
        candidate_rows.groupby(["id_person", "year"])
        .size()
        .rename("speech_count")
        .reset_index()
    )


def plot_timeline(
    counts: pd.DataFrame,
    authors: pd.DataFrame,
    output_path: Path,
    dpi: int,
    *,
    election_years: list[int] = ELECTIONS,
    years: list[int] = YEARS,
    title: str | None = None,
    include_party_in_label: bool = False,
    test_counts: pd.DataFrame | None = None,
    available_test_counts: pd.DataFrame | None = None,
    test_start_year: int | None = None,
) -> None:
    if authors.empty:
        raise ValueError("Cannot plot an empty author selection.")

    author_order = authors.sort_values(
        ["speech_count", "name"], ascending=[False, True]
    ).reset_index(drop=True)
    palette = plt.colormaps["tab10"](
        np.linspace(0.0, 0.9, len(author_order))
    )
    figure, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)

    for index, author in author_order.iterrows():
        author_id = int(author["id_person"])
        series = (
            counts.loc[counts["id_person"] == author_id]
            .set_index("year")["speech_count"]
            .reindex(years, fill_value=0)
        )
        active = series.gt(0)
        if not active.any():
            continue

        first = int(np.flatnonzero(active.to_numpy())[0])
        last = int(np.flatnonzero(active.to_numpy())[-1])
        visible = series.iloc[first : last + 1]
        label = str(author["name"])
        if include_party_in_label:
            label = f"{label} ({author['party']})"
        axis.plot(
            visible.index,
            visible.values,
            color=palette[index],
            linewidth=1.65,
            marker="o",
            markersize=4.2,
            alpha=0.88,
            label=label,
            zorder=2,
        )

        if available_test_counts is not None:
            available_series = (
                available_test_counts.loc[
                    available_test_counts["id_person"] == author_id
                ]
                .set_index("year")["speech_count"]
                .reindex(years, fill_value=0)
            )
            available_active = available_series.gt(0)
            if available_active.any():
                available_first = int(
                    np.flatnonzero(available_active.to_numpy())[0]
                )
                available_last = int(
                    np.flatnonzero(available_active.to_numpy())[-1]
                )
                available_visible = available_series.iloc[
                    available_first : available_last + 1
                ]
                axis.plot(
                    available_visible.index,
                    available_visible.values,
                    color=palette[index],
                    linewidth=1.4,
                    linestyle=":",
                    marker="o",
                    markersize=3.8,
                    alpha=0.25,
                    label="_nolegend_",
                    zorder=2,
                )

        if test_counts is not None:
            test_series = (
                test_counts.loc[test_counts["id_person"] == author_id]
                .set_index("year")["speech_count"]
                .reindex(years, fill_value=0)
            )
            test_active = test_series.gt(0)
            if test_active.any():
                test_first = int(np.flatnonzero(test_active.to_numpy())[0])
                test_last = int(np.flatnonzero(test_active.to_numpy())[-1])
                test_visible = test_series.iloc[test_first : test_last + 1]
                axis.plot(
                    test_visible.index,
                    test_visible.values,
                    color=palette[index],
                    linewidth=2.4,
                    linestyle="--",
                    marker="s",
                    markersize=5.0,
                    alpha=1.0,
                    label="_nolegend_",
                    zorder=3,
                )

    for election_year in election_years:
        if election_year == test_start_year:
            continue
        axis.axvline(
            election_year,
            color="#3f3f3f",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

    if test_start_year is not None:
        axis.axvline(
            test_start_year,
            color="#b22222",
            linewidth=1.4,
            linestyle="-.",
            alpha=0.9,
            zorder=1,
        )

    if title:
        axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Year")
    y_label = (
        "Number of speeches"
        if test_counts is not None
        else "Number of training speeches"
    )
    axis.set_ylabel(y_label)
    axis.set_xticks(years)
    axis.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.75)
    axis.grid(axis="x", color="#ededed", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=8.5)

    for tick_label in axis.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_horizontalalignment("right")
        if int(tick_label.get_text()) in election_years:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#202020")

    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#3f3f3f",
            linewidth=1.0,
            linestyle="--",
        )
    )
    labels.append("Parliamentary election")
    if available_test_counts is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#3f3f3f",
                linewidth=1.4,
                linestyle=":",
                marker="o",
                markersize=3.8,
                alpha=0.35,
            )
        )
        labels.append("Available test-period speeches")
    if test_counts is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#3f3f3f",
                linewidth=2.4,
                linestyle="--",
                marker="s",
                markersize=5.0,
            )
        )
        labels.append("Retained test speeches (latest 100)")
    if test_start_year is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#b22222",
                linewidth=1.4,
                linestyle="-.",
            )
        )
        labels.append("Testing begins")
    axis.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_split_figures(
    counts: pd.DataFrame,
    authors: pd.DataFrame,
    *,
    overview_output: Path,
    party_output_dir: Path,
    filename_prefix: str,
    party_order: list[str],
    election_years: list[int],
    years: list[int],
    dpi: int,
    test_counts: pd.DataFrame | None = None,
    available_test_counts: pd.DataFrame | None = None,
    test_start_year: int | None = None,
) -> list[Path]:
    missing_parties = set(party_order) - set(authors["party"].unique())
    if missing_parties:
        raise ValueError(
            f"Requested parties missing from author metadata: {sorted(missing_parties)}"
        )

    top_authors = pd.concat(
        [
            authors.loc[authors["party"] == party]
            .sort_values(["speech_count", "name"], ascending=[False, True])
            .head(1)
            for party in party_order
        ],
        ignore_index=True,
    )
    plot_timeline(
        counts,
        top_authors,
        overview_output,
        dpi,
        election_years=election_years,
        years=years,
        include_party_in_label=True,
        test_counts=test_counts,
        available_test_counts=available_test_counts,
        test_start_year=test_start_year,
    )

    generated_paths = [overview_output]
    for party in party_order:
        party_authors = authors.loc[authors["party"] == party].copy()
        output_path = party_output_dir / f"{filename_prefix}_{party.lower()}.png"
        plot_timeline(
            counts,
            party_authors,
            output_path,
            dpi,
            election_years=election_years,
            years=years,
            title=party,
            test_counts=test_counts,
            available_test_counts=available_test_counts,
            test_start_year=test_start_year,
        )
        generated_paths.append(output_path)
    return generated_paths


def main() -> None:
    args = parse_args()
    counts, authors = load_training_counts(
        args.train_corpus,
        expected_authors=EXPECTED_AUTHORS,
        expected_train_rows=EXPECTED_TRAIN_ROWS,
        allowed_elections=ELECTIONS,
        years=YEARS,
    )
    output_paths = generate_split_figures(
        counts,
        authors,
        overview_output=args.output,
        party_output_dir=args.party_output_dir,
        filename_prefix="authorwise_training_timeline",
        party_order=PARTY_ORDER,
        election_years=ELECTIONS,
        years=YEARS,
        dpi=args.dpi,
    )
    for output_path in output_paths:
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
