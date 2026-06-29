"""Dataset-level statistics for cleaned preprocessing outputs.

Called by ``preprocessing.run_preprocessing`` once the clean variants are in
memory. Writes CSV tables and figures to a caller-provided results directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_pipeline.utils import write_json


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def corpus_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Single-row summary of the full cleaned corpus."""
    author_female = df.groupby("id_person")["female"].first()
    rows = [{
        "n_speeches":    int(df["id_speech"].nunique()),
        "n_authors":     int(df["id_person"].nunique()),
        "n_parties":     int(df["party"].nunique()),
        "n_elections":   int(df["election"].nunique()),
        "election_range": f"{int(df['election'].min())}–{int(df['election'].max())}",
        "total_words":   int(df["word_count"].sum()),
        "total_chars":   int(df["char_count"].sum()),
        "mean_words_per_speech":   round(float(df["word_count"].mean()), 1),
        "median_words_per_speech": round(float(df["word_count"].median()), 1),
        "female_author_pct":       round(float(author_female.mean() * 100), 1),
    }]
    return pd.DataFrame(rows)


def election_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per election period."""
    rows = []
    for election, grp in df.groupby("election"):
        author_female = grp.groupby("id_person")["female"].first()
        lang_counts = grp["language"].value_counts(normalize=True) * 100
        rows.append({
            "election":         int(election),
            "n_speeches":       int(grp["id_speech"].nunique()),
            "n_authors":        int(grp["id_person"].nunique()),
            "n_parties":        int(grp["party"].nunique()),
            "total_words":      int(grp["word_count"].sum()),
            "mean_words":       round(float(grp["word_count"].mean()), 1),
            "median_words":     round(float(grp["word_count"].median()), 1),
            "female_pct":       round(float(author_female.mean() * 100), 1),
            "bokmal_pct":       round(float(lang_counts.get("Bokmål", 0.0)), 1),
            "nynorsk_pct":      round(float(lang_counts.get("Nynorsk", 0.0)), 1),
        })
    return pd.DataFrame(rows).sort_values("election").reset_index(drop=True)


def party_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per party, sorted by speech count descending."""
    rows = []
    for party, grp in df.groupby("party"):
        author_female = grp.groupby("id_person")["female"].first()
        rows.append({
            "party":            str(party),
            "n_speeches":       int(grp["id_speech"].nunique()),
            "n_authors":        int(grp["id_person"].nunique()),
            "total_words":      int(grp["word_count"].sum()),
            "mean_words":       round(float(grp["word_count"].mean()), 1),
            "median_words":     round(float(grp["word_count"].median()), 1),
            "female_pct":       round(float(author_female.mean() * 100), 1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("n_speeches", ascending=False)
        .reset_index(drop=True)
    )


def author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per author with verbosity and demographic columns."""
    rows = []
    for person_id, grp in df.groupby("id_person"):
        lang_counts = grp["language"].value_counts()
        rows.append({
            "id_person":        person_id,
            "name":             grp["name"].iloc[0],
            "party":            grp["party"].iloc[0],
            "female":           int(grp["female"].iloc[0]),
            "main_language":    lang_counts.index[0] if not lang_counts.empty else None,
            "n_speeches":       int(grp["id_speech"].nunique()),
            "n_elections":      int(grp["election"].nunique()),
            "total_words":      int(grp["word_count"].sum()),
            "total_chars":      int(grp["char_count"].sum()),
            "mean_words":       round(float(grp["word_count"].mean()), 1),
            "median_words":     round(float(grp["word_count"].median()), 1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("total_words", ascending=False)
        .reset_index(drop=True)
    )


def speech_length_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile table for speech word counts."""
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = df["word_count"].quantile([p / 100 for p in pcts])
    rows = [{"percentile": p, "word_count": int(values[p / 100])} for p in pcts]
    return pd.DataFrame(rows)


def age_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Speech-level age summary."""
    ages = pd.to_numeric(df["age"], errors="coerce").dropna()
    rows = [
        {"statistic": "mean", "years": round(float(ages.mean()), 1)},
        {"statistic": "median", "years": int(round(float(ages.median())))},
        {"statistic": "p10", "years": int(round(float(ages.quantile(0.10))))},
        {"statistic": "p90", "years": int(round(float(ages.quantile(0.90))))},
    ]
    return pd.DataFrame(rows)


def language_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Language distribution per election (counts and percentages)."""
    rows = []
    for election, grp in df.groupby("election"):
        total = len(grp)
        lang_counts = grp["language"].value_counts()
        for lang, count in lang_counts.items():
            rows.append({
                "election":  int(election),
                "language":  str(lang),
                "n_speeches": int(count),
                "pct":        round(count / total * 100, 1),
            })
    return pd.DataFrame(rows).sort_values(["election", "language"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_figure(fig, path: Path) -> None:
    """Save one dataset figure and close it to release matplotlib state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_dataset_figures(df: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Generate all dataset-level figures. Returns list of saved paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1 — speeches by election
    elec = election_stats(df).sort_values("election")
    fig, ax = plt.subplots()
    ax.bar(elec["election"].astype(str), elec["n_speeches"])
    ax.set_title("Speeches per election period")
    ax.set_xlabel("Election year")
    ax.set_ylabel("Number of speeches")
    ax.tick_params(axis="x", rotation=45)
    path = figures_dir / "01_speeches_by_election.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 2 — average words per election
    fig, ax = plt.subplots()
    ax.plot(elec["election"].astype(str), elec["mean_words"], marker="o")
    ax.set_title("Mean speech length by election period")
    ax.set_xlabel("Election year")
    ax.set_ylabel("Mean word count")
    ax.tick_params(axis="x", rotation=45)
    path = figures_dir / "02_mean_words_by_election.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 3 — speech length histogram (full range)
    fig, ax = plt.subplots()
    ax.hist(df["word_count"], bins=60, edgecolor="none")
    ax.set_title("Speech length distribution (word count)")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Number of speeches")
    path = figures_dir / "03_speech_length_hist.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 4 — top-20 parties by speech count
    p_stats = party_stats(df).head(20)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(p_stats["party"], p_stats["n_speeches"])
    ax.set_title("Top 20 parties by speech count")
    ax.set_xlabel("Party")
    ax.set_ylabel("Number of speeches")
    ax.tick_params(axis="x", rotation=45)
    path = figures_dir / "04_party_speech_counts.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 5 — language share by election (stacked bar)
    lang = language_stats(df)
    pivot = lang.pivot_table(index="election", columns="language", values="pct", fill_value=0.0)
    fig, ax = plt.subplots()
    bottom = None
    for lang_col in pivot.columns:
        vals = pivot[lang_col].values
        ax.bar(pivot.index.astype(str), vals, bottom=bottom, label=lang_col)
        bottom = vals if bottom is None else bottom + vals
    ax.set_title("Language share by election period (%)")
    ax.set_xlabel("Election year")
    ax.set_ylabel("Share (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    path = figures_dir / "05_language_share_by_election.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 6 — female author share by election
    fig, ax = plt.subplots()
    ax.plot(elec["election"].astype(str), elec["female_pct"], marker="o")
    ax.set_title("Female author share by election period (%)")
    ax.set_xlabel("Election year")
    ax.set_ylabel("Female authors (%)")
    ax.tick_params(axis="x", rotation=45)
    path = figures_dir / "06_female_share_by_election.png"
    _save_figure(fig, path)
    saved.append(str(path))

    # 7 — author verbosity (total words per author)
    a_stats = author_stats(df)
    fig, ax = plt.subplots()
    ax.hist(a_stats["total_words"], bins=40, edgecolor="none")
    ax.set_title("Author verbosity (total words in corpus)")
    ax.set_xlabel("Total word count")
    ax.set_ylabel("Number of authors")
    path = figures_dir / "07_author_verbosity.png"
    _save_figure(fig, path)
    saved.append(str(path))

    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_dataset_stats(
    df: pd.DataFrame,
    results_dir: Path,
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    """Compute and save all dataset statistics. Returns the summary tables."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    tables = {
        "corpus_overview":        corpus_overview(df),
        "election_stats":         election_stats(df),
        "party_stats":            party_stats(df),
        "author_stats":           author_stats(df),
        "speech_length_pcts":     speech_length_percentiles(df),
        "age_stats":              age_stats(df),
        "language_stats":         language_stats(df),
    }

    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)

    saved_figures = make_dataset_figures(df, figures_dir)

    write_json(results_dir / "manifest.json", {
        "source": "data_pipeline.preprocessing.run_preprocessing",
        "n_speeches": int(tables["corpus_overview"]["n_speeches"].iloc[0]),
        "n_authors":  int(tables["corpus_overview"]["n_authors"].iloc[0]),
        "tables":     [f"{k}.csv" for k in tables],
        "figures":    [str(Path(p).relative_to(project_root)) for p in saved_figures],
    })

    return tables
