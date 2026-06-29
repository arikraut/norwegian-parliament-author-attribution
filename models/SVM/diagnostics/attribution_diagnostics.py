from __future__ import annotations

import argparse
import json
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_recall_fscore_support


DEV_SELECTION_RUN_TYPES = {"dev_condition_selection", "stacked_condition_selection"}


@dataclass(frozen=True)
class DevSelectionArtifacts:
    """Loaded direct or stacked dev-selection artifacts used by diagnostics."""

    results_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    run_type: str
    selection_metric: str
    artifact_paths: dict[str, Path]
    fold_metrics: pd.DataFrame
    candidate_summary: pd.DataFrame
    condition_summary: pd.DataFrame
    selected_candidates: dict[str, Any]


def _safe_spearmanr(x: "pd.Series", y: "pd.Series") -> tuple[float, float]:
    """Return (statistic, pvalue), or (nan, nan) if either input is constant."""
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _normalize_label(value: Any) -> str:
    """Normalize labels from prediction CSVs into stable string keys."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    text = str(value).strip()
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass
    return text


def _find_existing_path(start: Path, relative: Path) -> Path | None:
    """Find ``relative`` under *start* or any parent directory."""
    for candidate in [start, *start.parents]:
        path = candidate / relative
        if path.exists():
            return path
    return None


def _build_author_meta_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw id_person/name/party CSV into the canonical author-metadata table.

    Returns an empty DataFrame when the required columns are missing.
    """
    if raw.empty or not {"id_person", "name", "party"} <= set(raw.columns):
        return pd.DataFrame()
    meta = raw[["id_person", "name", "party"]].copy()
    meta = meta.dropna(subset=["id_person"])
    meta["author_label"] = meta["id_person"].map(_normalize_label)
    meta = meta.drop_duplicates(subset=["author_label"], keep="first")
    meta = meta.rename(columns={"name": "author_name", "party": "author_party"})
    name = meta["author_name"].fillna("").astype(str)
    party = meta["author_party"].fillna("").astype(str)
    meta["author_display"] = np.where(
        name.str.len() > 0,
        name
        + np.where(party.str.len() > 0, " (" + party + ")", "")
        + " ["
        + meta["author_label"].astype(str)
        + "]",
        meta["author_label"].astype(str),
    )
    return meta[["author_label", "author_name", "author_party", "author_display"]]


def _try_load_author_metadata(results_dir: Path) -> pd.DataFrame:
    """Try to load an author-label → metadata lookup table for nicer diagnostics.

    This is optional (e.g. unit tests and ad-hoc runs won't have these files).
    Returns an empty dataframe when no metadata source is found.
    """
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    split_name = str(manifest.get("split_name", "")).strip()
    if not split_name:
        return pd.DataFrame()

    # Prefer the split-level support summary (tracked, small-ish, and includes names/party).
    support_summary_path = _find_existing_path(
        results_dir,
        Path("results") / "splits" / split_name / "per_author_support_summary.csv",
    )
    if support_summary_path is not None:
        try:
            raw = pd.read_csv(support_summary_path)
        except Exception:
            raw = pd.DataFrame()
        result = _build_author_meta_from_raw(raw)
        if not result.empty:
            return result

    # Fallback: author metadata shipped with the split materialization (often untracked locally).
    authors_path = _find_existing_path(
        results_dir,
        Path("data") / "splits" / split_name / "authors.csv",
    )
    if authors_path is not None:
        try:
            raw = pd.read_csv(authors_path)
        except Exception:
            raw = pd.DataFrame()
        result = _build_author_meta_from_raw(raw)
        if not result.empty:
            return result

    return pd.DataFrame()


def _ensure_author_display(per_author: pd.DataFrame) -> pd.DataFrame:
    """Guarantee per_author has a non-null author_display column."""
    if "author_display" not in per_author.columns:
        per_author["author_display"] = per_author["author_label"].astype(str)
    else:
        per_author["author_display"] = per_author["author_display"].fillna(
            per_author["author_label"].astype(str)
        )
    return per_author


def _sorted_labels(labels: list[str]) -> list[str]:
    """Sort numeric labels numerically and remaining labels lexically."""
    ints: list[int] = []
    rest: list[str] = []
    for label in labels:
        try:
            ints.append(int(label))
        except Exception:
            rest.append(label)
    return [str(value) for value in sorted(set(ints))] + sorted(set(rest))


def _top_k_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Find top-k prediction label columns in a prediction frame."""
    columns: list[tuple[int, str]] = []
    for column in df.columns:
        match = re.fullmatch(r"top(\d+)_label", column)
        if match:
            columns.append((int(match.group(1)), column))
    return sorted(columns, key=lambda item: item[0])


def _annotate_labels(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add normalized labels and correctness flags to prediction rows."""
    df = predictions.copy()
    df["y_true_label"] = df["y_true"].map(_normalize_label)
    df["y_pred_label"] = df["y_pred"].map(_normalize_label)
    df["correct"] = df["y_true_label"] == df["y_pred_label"]

    for _, column in _top_k_columns(df):
        df[column] = df[column].map(_normalize_label)
    return df


def _build_per_author_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute author-level precision, recall, F1, and support diagnostics."""
    y_true = df["y_true_label"].to_numpy()
    y_pred = df["y_pred_label"].to_numpy()
    labels = _sorted_labels(sorted(set(y_true) | set(y_pred)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    pred_count = df["y_pred_label"].value_counts()
    correct_count = df.loc[df["correct"], "y_true_label"].value_counts()
    out = pd.DataFrame(
        {
            "author_label": labels,
            "support": support.astype(int),
            "pred_count": [int(pred_count.get(label, 0)) for label in labels],
            "correct_count": [int(correct_count.get(label, 0)) for label in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    total = float(out["support"].sum())
    out["support_share"] = out["support"] / total if total > 0 else 0.0
    out["accuracy_within_true_class"] = np.where(out["support"] > 0, out["correct_count"] / out["support"], 0.0)
    return out.sort_values(["f1", "support"], ascending=[True, False]).reset_index(drop=True)


def _build_confusions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the most common true-author to predicted-author errors."""
    errors = df.loc[~df["correct"], ["y_true_label", "y_pred_label"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["y_true_label", "y_pred_label", "count", "error_share"])

    counts = (
        errors.groupby(["y_true_label", "y_pred_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    counts["error_share"] = counts["count"] / float(counts["count"].sum())
    return counts


def _enrich_confusions(confusions: pd.DataFrame, *, per_author: pd.DataFrame, author_meta: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable metadata + more interpretable rates to the confusion-pairs table."""
    if confusions.empty:
        return confusions

    enriched = confusions.copy()
    per_author_idx = per_author.set_index("author_label") if not per_author.empty else pd.DataFrame().set_index([])

    # Rates that are easier to interpret than global error_share.
    if not per_author.empty:
        true_support = per_author_idx["support"]
        true_errors = (per_author_idx["support"] - per_author_idx["correct_count"]).clip(lower=0)
        enriched["true_support"] = enriched["y_true_label"].map(true_support).fillna(0).astype(int)
        enriched["true_error_count"] = enriched["y_true_label"].map(true_errors).fillna(0).astype(int)
        enriched["p_pred_given_true"] = np.where(
            enriched["true_support"] > 0,
            enriched["count"] / enriched["true_support"],
            0.0,
        )
        enriched["share_of_true_errors"] = np.where(
            enriched["true_error_count"] > 0,
            enriched["count"] / enriched["true_error_count"],
            0.0,
        )

    # Optional author-name annotations (if available).
    if not author_meta.empty:
        true_meta = author_meta.rename(
            columns={
                "author_label": "y_true_label",
                "author_name": "y_true_name",
                "author_party": "y_true_party",
                "author_display": "y_true_display",
            }
        )
        pred_meta = author_meta.rename(
            columns={
                "author_label": "y_pred_label",
                "author_name": "y_pred_name",
                "author_party": "y_pred_party",
                "author_display": "y_pred_display",
            }
        )
        enriched = enriched.merge(true_meta, on="y_true_label", how="left", sort=False)
        enriched = enriched.merge(pred_meta, on="y_pred_label", how="left", sort=False)

    return enriched


def _build_support_bins(per_author: pd.DataFrame) -> pd.DataFrame:
    """Group authors into support quartiles for performance diagnostics."""
    if per_author.empty:
        return pd.DataFrame(columns=["support_bin", "n_authors", "mean_support", "mean_recall", "mean_f1"])

    binned = per_author.copy()
    try:
        binned["support_bin"] = pd.qcut(binned["support"], q=4, duplicates="drop")
    except Exception:
        binned["support_bin"] = "all"

    grouped = (
        binned.groupby("support_bin", dropna=False, observed=False)
        .agg(
            n_authors=("author_label", "count"),
            mean_support=("support", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
        )
        .reset_index()
    )
    grouped["support_bin"] = grouped["support_bin"].astype(str)
    return grouped


def _build_normalized_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalized confusion matrix (true label × predicted label).

    Values represent recall per (true, pred) pair: how often each true class
    is predicted as each predicted class. Diagonal = per-class recall.
    """
    y_true = df["y_true_label"].to_numpy()
    y_pred = df["y_pred_label"].to_numpy()
    labels = _sorted_labels(sorted(set(y_true) | set(y_pred)))

    label_index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        matrix[label_index[yt], label_index[yp]] += 1.0

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.where(row_sums > 0, matrix / row_sums, 0.0)
    return pd.DataFrame(normalized, index=labels, columns=labels)


def _plot_shared_attribution_figures(
    plot_dir: Path,
    per_author: pd.DataFrame,
    normalized_conf: pd.DataFrame,
    plt: Any,
    sns: Any,
    *,
    is_final: bool,
) -> list[str]:
    """Draw the final per-author diagnostic figures."""
    saved: list[str] = []

    def _save(fig: Any, name: str) -> str:
        """Save one shared diagnostic figure and close it."""
        path = plot_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # Per-author F1 bar chart (sorted ascending)
    if not per_author.empty:
        pa = per_author.sort_values("f1", ascending=True).reset_index(drop=True)
        n_authors = len(pa)
        label_col = "author_display" if "author_display" in pa.columns else "author_label"
        f1_xlabel = "F1 score (test set)" if is_final else "F1 score (aggregated across folds)"
        f1_title = f"Per-author F1 — final test (n={n_authors})" if is_final else f"Per-author F1 (n={n_authors})"
        fig, ax = plt.subplots(figsize=(8, max(4, n_authors * 0.2)))
        colors = ["#d62728" if f < 0.5 else "#ff7f0e" if f < 0.7 else "#2ca02c" for f in pa["f1"]]
        ax.barh(np.arange(n_authors), pa["f1"], color=colors, height=0.7)
        ax.set_yticks(np.arange(n_authors))
        ax.set_yticklabels(pa[label_col].astype(str), fontsize=7)
        ax.set_xlim(0, 1.0)
        ax.axvline(pa["f1"].mean(), color="black", linestyle="--", linewidth=1, label=f"mean={pa['f1'].mean():.3f}")
        ax.set_xlabel(f1_xlabel)
        ax.set_title(f"{f1_title}\nred <0.5 | orange <0.7 | green ≥0.7")
        ax.legend(fontsize=8)
        fig.tight_layout()
        saved.append(_save(fig, "per_author_f1_bar.png"))

    # Support vs recall scatter (log-scale x, Spearman correlation)
    if not per_author.empty:
        support_context = "Test" if is_final else "Validation"
        scatter_suffix = " — final test" if is_final else ""
        corr, pval = _safe_spearmanr(per_author["support"], per_author["recall"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(per_author["support"], per_author["recall"],
                        c=per_author["f1"], cmap="viridis", s=70, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="F1")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"{support_context} support (log scale)")
        ax.set_ylabel("Recall")
        ax.set_title(f"Support vs recall by author{scatter_suffix}\nSpearman r={corr:.3f}, p={pval:.3f}")
        fig.tight_layout()
        saved.append(_save(fig, "support_vs_recall_scatter.png"))

    # Normalized confusion heatmap (top-N most common true labels by support)
    if not per_author.empty and not normalized_conf.empty:
        top_n = min(20, len(normalized_conf))
        top_labels = per_author.sort_values("support", ascending=False)["author_label"].astype(str).head(top_n).tolist()
        available = [label for label in top_labels if label in normalized_conf.index and label in normalized_conf.columns]
        if available:
            heat = normalized_conf.loc[available, available]
            if "author_display" in per_author.columns:
                display_map = per_author.set_index("author_label")["author_display"].to_dict()
                heat = heat.rename(index=display_map, columns=display_map)
            fig, ax = plt.subplots(figsize=(9, 8))
            sns.heatmap(heat, ax=ax, cmap="Blues", vmin=0, vmax=1,
                        linewidths=0.3, linecolor="white",
                        annot=len(available) <= 12, fmt=".2f", annot_kws={"size": 7})
            ax.set_title(f"Normalized confusion matrix (top-{top_n} authors by support)\nRow = true class, diagonal = recall")
            ax.set_xlabel("Predicted author")
            ax.set_ylabel("True author")
            fig.tight_layout()
            saved.append(_save(fig, "normalized_confusion_heatmap.png"))

    # Support histogram (class imbalance)
    if not per_author.empty:
        split_label = "test" if is_final else "val"
        hist_xlabel = (
            "Test support per author"
            if is_final
            else "Validation support per author (total across folds)"
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(per_author["support"], bins=20, edgecolor="white")
        ax.axvline(per_author["support"].mean(), color="red", linestyle="--",
                   label=f"mean={per_author['support'].mean():.0f}")
        ax.axvline(per_author["support"].median(), color="orange", linestyle="--",
                   label=f"median={per_author['support'].median():.0f}")
        ax.set_xlabel(hist_xlabel)
        ax.set_ylabel("Number of authors")
        ax.set_title(f"Class imbalance: {split_label} support distribution")
        ax.legend()
        fig.tight_layout()
        saved.append(_save(fig, f"{split_label}_support_distribution.png"))

    return saved


def _make_final_diagnostics_figures(
    diagnostics_dir: Path,
    per_author: pd.DataFrame,
    normalized_conf: pd.DataFrame,
) -> list[str]:
    """Generate diagnostic figures for a single final test split."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ImportError:
        return []

    plot_dir = diagnostics_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return _plot_shared_attribution_figures(plot_dir, per_author, normalized_conf, plt, sns, is_final=True)


def _resolve_manifest_artifact(start: Path, path_value: str | Path) -> Path:
    """Resolve a manifest artifact path from a results directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    found = _find_existing_path(start, path)
    if found is not None:
        return found
    return (start / path).resolve()


def _load_dev_selection_artifacts(results_dir: Path) -> DevSelectionArtifacts:
    """Load the direct or stacked dev-selection artifact set for diagnostics."""
    results_dir = Path(results_dir).resolve()
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dev selection manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_type = str(manifest.get("run_type", ""))
    if run_type not in DEV_SELECTION_RUN_TYPES:
        expected = ", ".join(sorted(DEV_SELECTION_RUN_TYPES))
        raise ValueError(f"Unsupported dev selection run_type {run_type!r}; expected one of: {expected}.")
    selection_scope = str(manifest.get("selection_scope", ""))
    if selection_scope != "condition":
        raise ValueError(f"Dev selection diagnostics require selection_scope='condition', got {selection_scope!r}.")

    selection_metric = str(manifest.get("selection_metric", "")).strip()
    if not selection_metric:
        raise ValueError("Dev selection manifest must include selection_metric.")

    path_keys = [
        "fold_metrics_path",
        "candidate_summary_path",
        "condition_summary_path",
        "selected_candidates_path",
    ]
    artifact_paths: dict[str, Path] = {"manifest_path": manifest_path}
    for key in path_keys:
        raw_path = manifest.get(key)
        if not raw_path:
            raise KeyError(f"Dev selection manifest is missing {key}.")
        artifact_paths[key] = _resolve_manifest_artifact(results_dir, raw_path)

    return DevSelectionArtifacts(
        results_dir=results_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        run_type=run_type,
        selection_metric=selection_metric,
        artifact_paths=artifact_paths,
        fold_metrics=pd.read_csv(artifact_paths["fold_metrics_path"]),
        candidate_summary=pd.read_csv(artifact_paths["candidate_summary_path"]),
        condition_summary=pd.read_csv(artifact_paths["condition_summary_path"]),
        selected_candidates=json.loads(
            artifact_paths["selected_candidates_path"].read_text(encoding="utf-8")
        ),
    )


def _selection_metric_column(selection_metric: str) -> str:
    """Return the candidate-summary column name for the configured selection metric."""
    return f"eval_mean_{selection_metric}"


def _selection_std_column(selection_metric: str) -> str:
    """Return the candidate-summary std column name for the configured selection metric."""
    return f"eval_std_{selection_metric}"


def _candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return architecture-specific candidate descriptor columns present in a summary."""
    preferred = [
        "feature_set",
        "blocks",
        "families",
        "family_set",
        "profiling_blocks",
        "c_value",
        "base_c",
        "top_c",
        "class_weight",
    ]
    return [column for column in preferred if column in frame.columns]


def _profiling_source_value(row: pd.Series) -> Any:
    """Return the direct or stacked feature-block field used for profiling classification."""
    if "profiling_blocks" in row and not pd.isna(row["profiling_blocks"]):
        return row["profiling_blocks"]
    if "blocks" in row and not pd.isna(row["blocks"]):
        return row["blocks"]
    return ""


def _split_block_tokens(blocks: Any) -> list[str]:
    """Split serialized block lists from direct or stacked selection artifacts."""
    if isinstance(blocks, (list, tuple)):
        raw_tokens = [str(item) for item in blocks]
    else:
        value = "" if pd.isna(blocks) else str(blocks)
        raw_tokens = re.split(r"[+,]", value)
    return [token.strip() for token in raw_tokens if token.strip() and token.strip().lower() != "none"]


def _classify_profiling_representation(blocks: Any) -> str:
    """Classify direct or stacked profiling blocks for dev selection comparison."""
    representations: set[str] = set()
    for token in _split_block_tokens(blocks):
        if token.startswith("profiling_oracle_"):
            representations.add("profiling_oracle")
        elif token.startswith("profiling_hard_"):
            representations.add("profiling_hard")
        elif token.startswith("profiling_"):
            representations.add("profiling_probability")

    if not representations:
        return "base_only"
    if len(representations) == 1:
        return next(iter(representations))
    return "other"


def _rank_dev_candidates(artifacts: DevSelectionArtifacts) -> pd.DataFrame:
    """Rank dev candidates inside each condition by the configured selection metric."""
    summary = artifacts.candidate_summary.copy()
    metric_col = _selection_metric_column(artifacts.selection_metric)
    if metric_col not in summary.columns:
        raise KeyError(f"Candidate summary is missing selection metric column {metric_col!r}.")

    summary["_original_order"] = range(len(summary))
    sort_cols = ["condition_id", metric_col]
    ascending = [True, False]
    for column in ["eval_mean_accuracy", "n_eval_units"]:
        if column in summary.columns:
            sort_cols.append(column)
            ascending.append(False)
    sort_cols.append("_original_order")
    ascending.append(True)

    ranked = summary.sort_values(sort_cols, ascending=ascending, kind="stable").copy()
    ranked["rank_within_condition"] = ranked.groupby("condition_id", sort=False).cumcount() + 1
    ranked["profiling_representation"] = ranked.apply(
        lambda row: _classify_profiling_representation(_profiling_source_value(row)),
        axis=1,
    )
    return ranked.drop(columns=["_original_order"])


def _selected_candidate_ids(artifacts: DevSelectionArtifacts) -> dict[str, str]:
    """Return selected candidate ids keyed by condition id."""
    if "selected_candidate_id" in artifacts.condition_summary.columns:
        return {
            str(row["condition_id"]): str(row["selected_candidate_id"])
            for _, row in artifacts.condition_summary.iterrows()
        }

    raw_candidates = artifacts.selected_candidates.get("selected_candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("selected_candidates.json must include a selected_candidates list.")
    return {
        str(candidate["condition_id"]): str(candidate["candidate_id"])
        for candidate in raw_candidates
    }


def _build_condition_selection_summary(artifacts: DevSelectionArtifacts) -> pd.DataFrame:
    """Summarize selected candidate strength and selection margins by condition."""
    ranked = _rank_dev_candidates(artifacts)
    metric_col = _selection_metric_column(artifacts.selection_metric)
    std_col = _selection_std_column(artifacts.selection_metric)
    selected_by_condition = _selected_candidate_ids(artifacts)

    rows: list[dict[str, Any]] = []
    feature_columns = _candidate_feature_columns(ranked)
    for condition_id, condition_df in ranked.groupby("condition_id", sort=False):
        selected_candidate_id = selected_by_condition[str(condition_id)]
        selected_rows = condition_df[condition_df["candidate_id"] == selected_candidate_id]
        if selected_rows.empty:
            raise ValueError(
                f"Selected candidate {selected_candidate_id!r} was not found in candidate_summary.csv."
            )
        selected = selected_rows.iloc[0]
        runner_up = condition_df[condition_df["candidate_id"] != selected_candidate_id].head(1)
        runner_up_metric = (
            float(runner_up.iloc[0][metric_col])
            if not runner_up.empty and not pd.isna(runner_up.iloc[0][metric_col])
            else float("nan")
        )
        selected_metric = float(selected[metric_col])
        row: dict[str, Any] = {
            "condition_id": str(condition_id),
            "condition_label": str(selected.get("condition_label", condition_id)),
            "selected_candidate_id": selected_candidate_id,
            "selection_metric": artifacts.selection_metric,
            "selected_metric": selected_metric,
            "runner_up_metric": runner_up_metric,
            "selection_margin": selected_metric - runner_up_metric if not np.isnan(runner_up_metric) else float("nan"),
            "n_candidates": int(len(condition_df)),
            "n_eval_units": int(selected["n_eval_units"]) if "n_eval_units" in selected else 0,
            "profiling_representation": str(selected["profiling_representation"]),
        }
        if std_col in selected.index:
            row[std_col] = selected[std_col]
        for column in feature_columns:
            row[column] = selected[column]
        rows.append(row)

    return pd.DataFrame(rows)


def _build_candidate_rankings(
    artifacts: DevSelectionArtifacts,
    *,
    top_candidates: int,
) -> pd.DataFrame:
    """Build a compact candidate-ranking table for each dev condition."""
    ranked = _rank_dev_candidates(artifacts)
    if top_candidates > 0:
        ranked = ranked[ranked["rank_within_condition"] <= int(top_candidates)].copy()

    metric_col = _selection_metric_column(artifacts.selection_metric)
    std_col = _selection_std_column(artifacts.selection_metric)
    base_columns = [
        "condition_id",
        "condition_label",
        "rank_within_condition",
        "candidate_id",
        metric_col,
        std_col,
        "eval_mean_accuracy",
        "eval_mean_macro_f1",
        "n_eval_units",
    ]
    columns = [
        column
        for column in base_columns + _candidate_feature_columns(ranked) + ["profiling_representation"]
        if column in ranked.columns
    ]
    return ranked[columns].reset_index(drop=True)


def _build_fold_stability(artifacts: DevSelectionArtifacts) -> pd.DataFrame:
    """Summarize evaluation-fold variation for selected dev candidates."""
    metric_col = artifacts.selection_metric
    if metric_col not in artifacts.fold_metrics.columns:
        raise KeyError(f"Fold metrics are missing selection metric column {metric_col!r}.")

    selected_by_condition = _selected_candidate_ids(artifacts)
    selected_ids = set(selected_by_condition.values())
    selected_rows = artifacts.fold_metrics[
        artifacts.fold_metrics["candidate_id"].isin(selected_ids)
    ].copy()
    fold_metrics = artifacts.fold_metrics[
        (artifacts.fold_metrics["split"] != "train")
        & (artifacts.fold_metrics["candidate_id"].isin(selected_ids))
    ].copy()

    rows: list[dict[str, Any]] = []
    for selected_candidate_id, group in fold_metrics.groupby("candidate_id", sort=False):
        ordered = group.sort_values(metric_col, ascending=True, kind="stable")
        worst = ordered.iloc[0]
        best = ordered.iloc[-1]
        condition_id = str(worst["condition_id"])
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "selected_candidate_id": str(selected_candidate_id),
            "selection_metric": artifacts.selection_metric,
            "mean_metric": float(group[metric_col].mean()),
            "std_metric": float(group[metric_col].std()),
            "min_metric": float(worst[metric_col]),
            "max_metric": float(best[metric_col]),
            "worst_unit_id": str(worst["unit_id"]),
            "best_unit_id": str(best["unit_id"]),
            "n_eval_units": int(len(group)),
        }
        if "convergence_warning_count" in group.columns:
            warning_rows = selected_rows[
                selected_rows["candidate_id"] == selected_candidate_id
            ]
            row["convergence_warning_count"] = int(warning_rows["convergence_warning_count"].sum())
        rows.append(row)

    return pd.DataFrame(rows)


def _build_profiling_block_comparison(condition_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare selected validation metrics by profiling block representation."""
    rows: list[dict[str, Any]] = []
    for representation, group in condition_summary.groupby("profiling_representation", sort=True):
        ordered = group.sort_values("selected_metric", ascending=False, kind="stable")
        best = ordered.iloc[0]
        rows.append(
            {
                "profiling_representation": representation,
                "n_conditions": int(len(group)),
                "mean_selected_metric": float(group["selected_metric"].mean()),
                "median_selected_metric": float(group["selected_metric"].median()),
                "best_condition_id": str(best["condition_id"]),
                "best_selected_metric": float(best["selected_metric"]),
            }
        )
    return pd.DataFrame(rows)


def run_dev_attribution_selection_diagnostics(
    results_dir: Path,
    *,
    top_candidates: int = 10,
) -> dict[str, Any]:
    """Generate diagnostics from direct or stacked dev-selection artifacts."""
    artifacts = _load_dev_selection_artifacts(results_dir)
    diagnostics_dir = artifacts.results_dir / "diagnostics"
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    condition_summary = _build_condition_selection_summary(artifacts)
    candidate_rankings = _build_candidate_rankings(
        artifacts,
        top_candidates=top_candidates,
    )
    fold_stability = _build_fold_stability(artifacts)
    profiling_comparison = _build_profiling_block_comparison(condition_summary)

    condition_summary_path = diagnostics_dir / "condition_selection_summary.csv"
    candidate_rankings_path = diagnostics_dir / "candidate_rankings.csv"
    fold_stability_path = diagnostics_dir / "fold_stability.csv"
    profiling_comparison_path = diagnostics_dir / "profiling_block_comparison.csv"

    condition_summary.to_csv(condition_summary_path, index=False)
    candidate_rankings.to_csv(candidate_rankings_path, index=False)
    fold_stability.to_csv(fold_stability_path, index=False)
    profiling_comparison.to_csv(profiling_comparison_path, index=False)

    summary = {
        "results_dir": str(artifacts.results_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": artifacts.run_type,
        "selection_metric": artifacts.selection_metric,
        "top_candidates": int(top_candidates),
        "condition_count": int(len(condition_summary)),
        "candidate_rows_written": int(len(candidate_rankings)),
        "inputs": {key: str(path) for key, path in artifacts.artifact_paths.items()},
        "paths": {
            "condition_selection_summary": str(condition_summary_path),
            "candidate_rankings": str(candidate_rankings_path),
            "fold_stability": str(fold_stability_path),
            "profiling_block_comparison": str(profiling_comparison_path),
        },
    }
    (diagnostics_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_single_final_attribution_diagnostics(
    results_dir: Path,
    *,
    top_confusions: int,
    metadata_results_dir: Path | None = None,
) -> dict[str, Any]:
    """Run diagnostics for one final condition directory."""
    results_dir = Path(results_dir).resolve()
    predictions_path = results_dir / "final_test_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Final test predictions not found: {predictions_path}")

    raw = pd.read_csv(predictions_path)
    missing = {"y_true", "y_pred"} - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns {sorted(missing)} in {predictions_path}")
    if "fold_id" not in raw.columns:
        raw["fold_id"] = "test"
    predictions = _annotate_labels(raw)

    diagnostics_dir = results_dir / "diagnostics"
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    per_author = _build_per_author_metrics(predictions)
    author_meta = _try_load_author_metadata(metadata_results_dir or results_dir)
    if not author_meta.empty:
        per_author = per_author.merge(author_meta, on="author_label", how="left", sort=False)
    per_author = _ensure_author_display(per_author)

    confusions = _enrich_confusions(
        _build_confusions(predictions),
        per_author=per_author,
        author_meta=author_meta,
    )
    support_bins = _build_support_bins(per_author)
    normalized_conf = _build_normalized_confusion(predictions)
    n_authors = int(predictions["y_true_label"].nunique())

    author_path = diagnostics_dir / "per_author_metrics.csv"
    confusion_path = diagnostics_dir / "confusion_pairs.csv"
    top_confusion_path = diagnostics_dir / "top_confusions.csv"
    support_path = diagnostics_dir / "support_vs_performance.csv"
    norm_conf_path = diagnostics_dir / "normalized_confusion_matrix.csv"

    per_author.to_csv(author_path, index=False)
    confusions.to_csv(confusion_path, index=False)
    confusions.head(max(int(top_confusions), 1)).to_csv(top_confusion_path, index=False)
    support_bins.to_csv(support_path, index=False)
    normalized_conf.to_csv(norm_conf_path)

    figure_paths = _make_final_diagnostics_figures(diagnostics_dir, per_author, normalized_conf)

    overall_accuracy = float(predictions["correct"].mean())
    y_true = predictions["y_true_label"].to_numpy()
    y_pred = predictions["y_pred_label"].to_numpy()
    labels = _sorted_labels(sorted(set(y_true) | set(y_pred)))
    overall_macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    raw_corr = _safe_spearmanr(per_author["support"], per_author["recall"])[0] if len(per_author) > 1 else float("nan")
    support_recall_corr = None if np.isnan(raw_corr) else raw_corr

    summary = {
        "results_dir": str(results_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(predictions)),
        "n_authors_true": n_authors,
        "overall_accuracy": overall_accuracy,
        "overall_macro_f1": overall_macro_f1,
        "support_recall_correlation": support_recall_corr,
        "paths": {
            "per_author_metrics": str(author_path),
            "confusion_pairs": str(confusion_path),
            "top_confusions": str(top_confusion_path),
            "support_vs_performance": str(support_path),
            "normalized_confusion_matrix": str(norm_conf_path),
        },
        "figures": figure_paths,
    }
    (diagnostics_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_final_attribution_diagnostics(results_dir: Path, *, top_confusions: int = 50) -> dict[str, Any]:
    """Run diagnostics on all conditions in a final attribution evaluation run."""
    results_dir = Path(results_dir).resolve()
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return _run_single_final_attribution_diagnostics(
            results_dir,
            top_confusions=top_confusions,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition_results = manifest.get("condition_results")
    if not isinstance(condition_results, list) or not condition_results:
        return _run_single_final_attribution_diagnostics(
            results_dir,
            top_confusions=top_confusions,
        )

    diagnostics_dir = results_dir / "diagnostics"
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    condition_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for condition_result in condition_results:
        predictions_path = _resolve_manifest_artifact(
            results_dir,
            condition_result["predictions_path"],
        )
        condition_dir = predictions_path.parent
        condition_summary = _run_single_final_attribution_diagnostics(
            condition_dir,
            top_confusions=top_confusions,
            metadata_results_dir=results_dir,
        )
        condition_summary.update(
            {
                "condition_id": str(condition_result["condition_id"]),
                "condition_label": str(
                    condition_result.get("condition_label", condition_result["condition_id"])
                ),
                "candidate_id": str(condition_result.get("candidate_id", "")),
            }
        )
        condition_summaries.append(condition_summary)
        comparison_rows.append(
            {
                "condition_id": condition_summary["condition_id"],
                "condition_label": condition_summary["condition_label"],
                "candidate_id": condition_summary["candidate_id"],
                "n_rows": condition_summary["n_rows"],
                "n_authors_true": condition_summary["n_authors_true"],
                "overall_accuracy": condition_summary["overall_accuracy"],
                "overall_macro_f1": condition_summary["overall_macro_f1"],
                "diagnostics_dir": condition_summary["diagnostics_dir"],
            }
        )

    comparison_path = diagnostics_dir / "final_condition_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    summary = {
        "results_dir": str(results_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_count": len(condition_summaries),
        "condition_summaries": condition_summaries,
        "paths": {
            "final_condition_comparison": str(comparison_path),
        },
    }
    (diagnostics_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse the attribution diagnostics CLI."""
    parser = argparse.ArgumentParser(
        description="Generate attribution diagnostics from dev-selection or final-evaluation outputs.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to one model run directory, e.g. results/models/<split>/<experiment>/seed_<seed>.",
    )
    parser.add_argument(
        "--mode",
        choices=["final", "dev-selection"],
        default="final",
        help="Diagnostic contract to run for the given results directory.",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=50,
        help="Number of top confusion pairs to write to top_confusions.csv.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=10,
        help="Number of ranked dev candidates to keep per condition in dev-selection mode.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for attribution diagnostics."""
    args = _parse_args()
    if args.mode == "dev-selection":
        summary = run_dev_attribution_selection_diagnostics(
            Path(args.results_dir),
            top_candidates=args.top_candidates,
        )
    else:
        summary = run_final_attribution_diagnostics(
            Path(args.results_dir),
            top_confusions=args.top_confusions,
        )
    print("Diagnostics completed.")
    print(f"Results dir: {summary['results_dir']}")
    print(f"Diagnostics dir: {summary['diagnostics_dir']}")
    if "overall_accuracy" in summary:
        print(f"Overall accuracy: {summary['overall_accuracy']:.6f}")
        print(f"Overall macro_f1: {summary['overall_macro_f1']:.6f}")
    else:
        print(f"Condition count: {summary['condition_count']}")


if __name__ == "__main__":
    main()
