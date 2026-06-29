"""Author- and party-level confusion analysis for research result reports."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ResultArtifacts, require_file
from .config import ResultSystem


def read_confusion_pairs(
    system: ResultSystem,
    artifacts: ResultArtifacts,
) -> pd.DataFrame:
    """Read and annotate one system's directed author-confusion table."""

    frame = artifacts.read_csv(system.confusion_pairs_path)
    frame["system_key"] = system.key
    frame["system_label"] = system.label
    frame["phase"] = system.phase
    frame["split"] = system.split
    frame["architecture"] = system.architecture
    frame["representation"] = system.representation
    frame["scope"] = system.scope
    frame["condition_id"] = system.condition_id
    frame["same_party"] = frame["y_true_party"] == frame["y_pred_party"]
    return frame


def build_directed_top_confusions(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Build the most frequent directed author-confusion rows per system."""

    rows: list[pd.DataFrame] = []
    for system in systems:
        frame = read_confusion_pairs(system, results_dir)
        sorted_frame = frame.sort_values(
            ["count", "error_share", "p_pred_given_true", "y_true_display"],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(top_n)
        selected = sorted_frame.copy()
        selected.insert(8, "confusion_rank", range(1, len(selected) + 1))
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _author_pair_record(row: pd.Series) -> dict[str, Any]:
    """Convert one directed confusion row to an unordered-pair record."""

    true_label = str(row["y_true_label"])
    pred_label = str(row["y_pred_label"])
    if true_label <= pred_label:
        author_a = {
            "label": row["y_true_label"],
            "display": row["y_true_display"],
            "party": row["y_true_party"],
        }
        author_b = {
            "label": row["y_pred_label"],
            "display": row["y_pred_display"],
            "party": row["y_pred_party"],
        }
        count_a_to_b = int(row["count"])
        count_b_to_a = 0
    else:
        author_a = {
            "label": row["y_pred_label"],
            "display": row["y_pred_display"],
            "party": row["y_pred_party"],
        }
        author_b = {
            "label": row["y_true_label"],
            "display": row["y_true_display"],
            "party": row["y_true_party"],
        }
        count_a_to_b = 0
        count_b_to_a = int(row["count"])

    return {
        "system_key": row["system_key"],
        "system_label": row["system_label"],
        "phase": row["phase"],
        "split": row["split"],
        "architecture": row["architecture"],
        "representation": row["representation"],
        "scope": row["scope"],
        "condition_id": row["condition_id"],
        "author_a_label": author_a["label"],
        "author_a_display": author_a["display"],
        "author_a_party": author_a["party"],
        "author_b_label": author_b["label"],
        "author_b_display": author_b["display"],
        "author_b_party": author_b["party"],
        "count_a_to_b": count_a_to_b,
        "count_b_to_a": count_b_to_a,
        "same_party": author_a["party"] == author_b["party"],
    }


def build_symmetric_confusion_pairs(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Aggregate directed errors into unordered author-pair confusions."""

    records: list[dict[str, Any]] = []
    for system in systems:
        frame = read_confusion_pairs(system, results_dir)
        records.extend(_author_pair_record(row) for _, row in frame.iterrows())

    pair_rows = pd.DataFrame(records)
    grouped = (
        pair_rows.groupby(
            [
                "system_key",
                "system_label",
                "phase",
                "split",
                "architecture",
                "representation",
                "scope",
                "condition_id",
                "author_a_label",
                "author_a_display",
                "author_a_party",
                "author_b_label",
                "author_b_display",
                "author_b_party",
                "same_party",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            count_a_to_b=("count_a_to_b", "sum"),
            count_b_to_a=("count_b_to_a", "sum"),
        )
    )
    grouped["total_pair_confusions"] = (
        grouped["count_a_to_b"] + grouped["count_b_to_a"]
    )
    totals = grouped.groupby("system_key")["total_pair_confusions"].transform("sum")
    grouped["share_of_system_errors"] = grouped["total_pair_confusions"] / totals
    ranked = grouped.sort_values(
        [
            "system_key",
            "total_pair_confusions",
            "share_of_system_errors",
            "author_a_display",
            "author_b_display",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    return ranked.groupby("system_key", sort=False).head(top_n).reset_index(drop=True)


def build_party_confusion_summary(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Summarize same-party and cross-party author-confusion counts."""

    frames = [
        read_confusion_pairs(system, results_dir)
        for system in systems
    ]
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(
            [
                "system_key",
                "system_label",
                "phase",
                "split",
                "architecture",
                "representation",
                "scope",
                "condition_id",
                "same_party",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            confusion_count=("count", "sum"),
            n_directed_pairs=("count", "size"),
        )
    )
    totals = grouped.groupby("system_key")["confusion_count"].transform("sum")
    grouped["share_of_system_errors"] = grouped["confusion_count"] / totals
    grouped["confusion_group"] = grouped["same_party"].map(
        {True: "same_party", False: "cross_party"}
    )
    return grouped[
        [
            "system_key",
            "system_label",
            "phase",
            "split",
            "architecture",
            "representation",
            "scope",
            "condition_id",
            "confusion_group",
            "same_party",
            "confusion_count",
            "share_of_system_errors",
            "n_directed_pairs",
        ]
    ].sort_values(
        ["system_key", "same_party"],
        ascending=[True, False],
        kind="stable",
    )


def build_party_pair_confusions(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Aggregate directed author errors by true-party and predicted-party."""

    frames = [
        read_confusion_pairs(system, results_dir)
        for system in systems
    ]
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(
            [
                "system_key",
                "system_label",
                "phase",
                "split",
                "architecture",
                "representation",
                "scope",
                "condition_id",
                "y_true_party",
                "y_pred_party",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            confusion_count=("count", "sum"),
            n_directed_author_pairs=("count", "size"),
        )
    )
    totals = grouped.groupby("system_key")["confusion_count"].transform("sum")
    grouped["share_of_system_errors"] = grouped["confusion_count"] / totals
    grouped["same_party"] = grouped["y_true_party"] == grouped["y_pred_party"]
    return grouped.sort_values(
        ["system_key", "confusion_count", "y_true_party", "y_pred_party"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def copy_normalized_confusion_matrices(
    systems: tuple[ResultSystem, ...],
    *,
    results_dir: Path,
    section_dir: Path,
) -> dict[str, str]:
    """Copy normalized confusion matrices into the result-additions tree."""

    matrix_dir = section_dir / "normalized_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for system in systems:
        source_path = results_dir / system.normalized_confusion_matrix_path
        matrix = pd.read_csv(source_path, index_col=0)
        output_path = matrix_dir / f"{system.key}.csv"
        matrix.to_csv(output_path, index_label="author_label")
        outputs[f"normalized_matrix_{system.key}"] = str(output_path)
    return outputs


def copy_normalized_confusion_heatmaps(
    systems: tuple[ResultSystem, ...],
    *,
    results_dir: Path,
    section_dir: Path,
) -> dict[str, str]:
    """Copy normalized confusion heatmaps into the result-additions tree."""

    figure_dir = section_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for system in systems:
        source_path = results_dir / system.normalized_confusion_heatmap_path
        require_file(
            source_path,
            context=(
                "Required normalized confusion heatmap "
                f"for {system.key} ({system.label})"
            ),
        )
        output_path = figure_dir / f"{system.key}_normalized_confusion_heatmap.png"
        shutil.copyfile(source_path, output_path)
        outputs[f"normalized_heatmap_{system.key}"] = str(output_path)
    return outputs


def write_confusion_outputs(
    systems: tuple[ResultSystem, ...],
    *,
    results_dir: ResultArtifacts,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Write all files for the confusion-analysis result addition."""

    section_dir = output_dir / "confusions"
    section_dir.mkdir(parents=True, exist_ok=True)
    artifacts = results_dir

    directed = build_directed_top_confusions(
        systems,
        artifacts,
        top_n=top_n,
    )
    symmetric = build_symmetric_confusion_pairs(
        systems,
        artifacts,
        top_n=top_n,
    )
    party_summary = build_party_confusion_summary(systems, artifacts)
    party_pairs = build_party_pair_confusions(systems, artifacts)

    paths = {
        "directed_top_confusions": section_dir / "directed_top_confusions.csv",
        "symmetric_confusion_pairs": section_dir / "symmetric_confusion_pairs.csv",
        "party_confusion_summary": section_dir / "party_confusion_summary.csv",
        "party_pair_confusions": section_dir / "party_pair_confusions.csv",
    }
    directed.to_csv(paths["directed_top_confusions"], index=False)
    symmetric.to_csv(paths["symmetric_confusion_pairs"], index=False)
    party_summary.to_csv(paths["party_confusion_summary"], index=False)
    party_pairs.to_csv(paths["party_pair_confusions"], index=False)

    outputs = {key: str(path) for key, path in paths.items()}
    outputs.update(
        copy_normalized_confusion_matrices(
            systems,
            results_dir=artifacts.root,
            section_dir=section_dir,
        )
    )
    outputs.update(
        copy_normalized_confusion_heatmaps(
            systems,
            results_dir=artifacts.root,
            section_dir=section_dir,
        )
    )
    return outputs

