"""Reporting helpers used while documenting preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_speech_length_threshold_figure(
    df: pd.DataFrame,
    output_path: Path,
    *,
    word_count_column: str = "word_count",
    threshold: int = 150,
    quantile: float = 0.99,
    bins: int = 60,
) -> dict[str, Any]:
    """Document the pre-filter speech-length distribution for dataset reports."""
    if word_count_column not in df.columns:
        raise KeyError(f"Expected column {word_count_column!r} in dataframe.")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in the interval (0, 1].")

    word_counts = pd.to_numeric(df[word_count_column], errors="coerce").dropna()
    if word_counts.empty:
        raise ValueError("No valid word-count values available for plotting.")

    clip_value = float(word_counts.quantile(quantile))
    plotted_counts = word_counts[word_counts <= clip_value]
    clipped_count = int((word_counts > clip_value).sum())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(plotted_counts, bins=bins, edgecolor="white", linewidth=0.4)
    ax.axvline(
        threshold,
        color="red",
        linestyle=":",
        linewidth=2.2,
        label=f"{threshold}-word threshold",
    )
    ax.set_title("Speech length distribution before short-speech filtering")
    ax.set_xlabel("Speech length (words)")
    ax.set_ylabel("Number of speeches")
    ax.set_xlim(left=0, right=clip_value)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.95,
        f"x-axis clipped at {quantile:.0%} quantile ({clip_value:.0f} words)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "figure_path": str(output_path),
        "threshold_words": int(threshold),
        "clip_quantile": float(quantile),
        "clip_word_count": int(round(clip_value)),
        "speeches_plotted": int(plotted_counts.shape[0]),
        "speeches_clipped": clipped_count,
    }
