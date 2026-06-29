"""Top-k rescue and confidence analysis for research result reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ResultArtifacts, canonical_label
from .author_performance import read_per_author_metrics
from .config import ResultSystem


def available_topk_values(frame: pd.DataFrame) -> tuple[int, ...]:
    """Return available top-k ranks in a prediction table."""

    values: list[int] = []
    rank = 1
    while f"top{rank}_label" in frame.columns:
        values.append(rank)
        rank += 1
    return tuple(values)


def read_final_predictions(
    system: ResultSystem,
    artifacts: ResultArtifacts,
) -> pd.DataFrame:
    """Read and annotate one system's final prediction table."""

    frame = artifacts.read_csv(system.final_predictions_path)
    frame["system_key"] = system.key
    frame["system_label"] = system.label
    frame["phase"] = system.phase
    frame["split"] = system.split
    frame["architecture"] = system.architecture
    frame["representation"] = system.representation
    frame["scope"] = system.scope
    frame["condition_id"] = system.condition_id
    if "id_speech" in frame.columns:
        frame["id_speech_key"] = frame["id_speech"].map(canonical_label)
    frame["y_true_author_label"] = frame["y_true"].map(canonical_label)
    frame["y_pred_author_label"] = frame["y_pred"].map(canonical_label)
    for rank in available_topk_values(frame):
        frame[f"top{rank}_author_label"] = frame[f"top{rank}_label"].map(
            canonical_label
        )
    if "top2_score" in frame.columns:
        frame["top1_margin"] = frame["top1_score"] - frame["top2_score"]
    if "top3_score" in frame.columns:
        frame["top1_top3_margin"] = frame["top1_score"] - frame["top3_score"]
    for rank in available_topk_values(frame):
        top_columns = [f"top{top_rank}_author_label" for top_rank in range(1, rank + 1)]
        frame[f"top{rank}_correct"] = frame[top_columns].eq(
            frame["y_true_author_label"],
            axis=0,
        ).any(axis=1)
    return frame


def author_metadata(
    system: ResultSystem,
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Read author metadata for joining prediction summaries to display names."""

    frame = read_per_author_metrics(system, results_dir)
    metadata = frame[
        ["system_key", "author_label", "author_display", "author_name", "author_party"]
    ].copy()
    metadata["author_label"] = metadata["author_label"].map(canonical_label)
    return metadata


def attach_author_metadata(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    label_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach author display metadata to a prediction-derived table."""

    renamed = metadata.rename(
        columns={
            "author_label": label_column,
            "author_display": f"{prefix}_author_display",
            "author_name": f"{prefix}_author_name",
            "author_party": f"{prefix}_author_party",
        }
    )
    return frame.merge(
        renamed,
        on=["system_key", label_column],
        how="left",
        validate="many_to_one",
    )


def build_overall_topk_rescue(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Build system-level top-k accuracy and rescue counts."""

    rows: list[dict[str, Any]] = []
    for system in systems:
        frame = read_final_predictions(system, results_dir)
        prediction_count = int(len(frame))
        top1_correct = int(frame["top1_correct"].sum())
        top1_errors = prediction_count - top1_correct
        for rank in available_topk_values(frame):
            topk_correct = int(frame[f"top{rank}_correct"].sum())
            rescue_count = int(
                (frame[f"top{rank}_correct"] & ~frame["top1_correct"]).sum()
            )
            rows.append(
                {
                    "system_key": system.key,
                    "system_label": system.label,
                    "phase": system.phase,
                    "split": system.split,
                    "architecture": system.architecture,
                    "representation": system.representation,
                    "scope": system.scope,
                    "condition_id": system.condition_id,
                    "k": rank,
                    "prediction_count": prediction_count,
                    "top1_correct": top1_correct,
                    "top1_accuracy": top1_correct / prediction_count,
                    "topk_correct": topk_correct,
                    "topk_accuracy": topk_correct / prediction_count,
                    "additional_correct_over_top1": topk_correct - top1_correct,
                    "rescue_count": rescue_count,
                    "rescue_rate_among_top1_errors": (
                        rescue_count / top1_errors if top1_errors else 0.0
                    ),
                    "remaining_errors_after_topk": prediction_count - topk_correct,
                }
            )
    return pd.DataFrame(rows)


def build_per_author_topk_rescue(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Build per-author top-k accuracy and rescue counts."""

    rows: list[dict[str, Any]] = []
    for system in systems:
        frame = read_final_predictions(system, results_dir)
        metadata = author_metadata(system, results_dir)
        frame = attach_author_metadata(
            frame,
            metadata,
            label_column="y_true_author_label",
            prefix="true",
        )
        for author_label, author_frame in frame.groupby(
            "y_true_author_label",
            sort=False,
        ):
            support = int(len(author_frame))
            top1_correct = int(author_frame["top1_correct"].sum())
            top1_errors = support - top1_correct
            for rank in available_topk_values(author_frame):
                topk_correct = int(author_frame[f"top{rank}_correct"].sum())
                rescue_count = int(
                    (
                        author_frame[f"top{rank}_correct"]
                        & ~author_frame["top1_correct"]
                    ).sum()
                )
                rows.append(
                    {
                        "system_key": system.key,
                        "system_label": system.label,
                        "phase": system.phase,
                        "split": system.split,
                        "architecture": system.architecture,
                        "representation": system.representation,
                        "scope": system.scope,
                        "condition_id": system.condition_id,
                        "author_label": author_label,
                        "author_display": author_frame[
                            "true_author_display"
                        ].iloc[0],
                        "author_name": author_frame["true_author_name"].iloc[0],
                        "author_party": author_frame["true_author_party"].iloc[0],
                        "k": rank,
                        "support": support,
                        "top1_correct": top1_correct,
                        "top1_accuracy": top1_correct / support,
                        "topk_correct": topk_correct,
                        "topk_accuracy": topk_correct / support,
                        "rescue_count": rescue_count,
                        "rescue_rate_among_top1_errors": (
                            rescue_count / top1_errors if top1_errors else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_score_margin_summary(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Summarize top-1 score and score margins by correctness."""

    frames = [
        read_final_predictions(system, results_dir)
        for system in systems
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["correct_group"] = combined["top1_correct"].map(
        {True: "correct", False: "incorrect"}
    )
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
                "correct_group",
                "top1_correct",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            prediction_count=("top1_margin", "size"),
            mean_top1_score=("top1_score", "mean"),
            median_top1_score=("top1_score", "median"),
            mean_top1_margin=("top1_margin", "mean"),
            median_top1_margin=("top1_margin", "median"),
            p25_top1_margin=("top1_margin", lambda values: values.quantile(0.25)),
            p75_top1_margin=("top1_margin", lambda values: values.quantile(0.75)),
            min_top1_margin=("top1_margin", "min"),
            max_top1_margin=("top1_margin", "max"),
        )
    )
    return grouped.sort_values(
        ["system_key", "top1_correct"],
        ascending=[True, False],
        kind="stable",
    )


def build_per_author_margin_summary(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Summarize score margins per true author."""

    frames: list[pd.DataFrame] = []
    for system in systems:
        frame = read_final_predictions(system, results_dir)
        metadata = author_metadata(system, results_dir)
        frame = attach_author_metadata(
            frame,
            metadata,
            label_column="y_true_author_label",
            prefix="true",
        )
        frames.append(frame)

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
                "y_true_author_label",
                "true_author_display",
                "true_author_name",
                "true_author_party",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            support=("top1_correct", "size"),
            top1_correct_count=("top1_correct", "sum"),
            mean_top1_score=("top1_score", "mean"),
            mean_top1_margin=("top1_margin", "mean"),
            median_top1_margin=("top1_margin", "median"),
            min_top1_margin=("top1_margin", "min"),
            max_top1_margin=("top1_margin", "max"),
        )
    )
    grouped["top1_accuracy"] = grouped["top1_correct_count"] / grouped["support"]
    grouped["top1_error_count"] = grouped["support"] - grouped["top1_correct_count"]
    return grouped.rename(
        columns={
            "y_true_author_label": "author_label",
            "true_author_display": "author_display",
            "true_author_name": "author_name",
            "true_author_party": "author_party",
        }
    ).sort_values(
        ["system_key", "top1_error_count", "mean_top1_margin", "author_display"],
        ascending=[True, False, False, True],
        kind="stable",
    )


def prediction_slice_columns(frame: pd.DataFrame) -> list[str]:
    """Return stable output columns for prediction-level confidence slices."""

    base_columns = [
        "system_key",
        "system_label",
        "phase",
        "split",
        "architecture",
        "representation",
        "scope",
        "condition_id",
        "confidence_rank",
        "confidence_slice",
        "row_idx",
        "id_speech",
        "election",
        "party",
        "y_true_author_label",
        "true_author_display",
        "true_author_party",
        "y_pred_author_label",
        "pred_author_display",
        "pred_author_party",
        "top1_score",
        "top2_score",
        "top1_margin",
        "top1_top3_margin",
    ]
    topk_columns: list[str] = []
    for rank in available_topk_values(frame):
        topk_columns.append(f"top{rank}_author_label")
        score_column = f"top{rank}_score"
        if score_column in frame.columns:
            topk_columns.append(score_column)
    return [column for column in base_columns + topk_columns if column in frame.columns]


def build_prediction_confidence_slice(
    systems: tuple[ResultSystem, ...],
    results_dir: ResultArtifacts,
    *,
    top_n: int,
    confident_errors: bool,
) -> pd.DataFrame:
    """Select prediction-level confident errors or low-margin correct rows."""

    rows: list[pd.DataFrame] = []
    for system in systems:
        frame = read_final_predictions(system, results_dir)
        metadata = author_metadata(system, results_dir)
        frame = attach_author_metadata(
            frame,
            metadata,
            label_column="y_true_author_label",
            prefix="true",
        )
        frame = attach_author_metadata(
            frame,
            metadata,
            label_column="y_pred_author_label",
            prefix="pred",
        )
        if confident_errors:
            selected = frame.loc[~frame["top1_correct"]].sort_values(
                ["top1_margin", "top1_score", "id_speech"],
                ascending=[False, False, True],
                kind="stable",
            )
            slice_name = "confident_error"
        else:
            selected = frame.loc[frame["top1_correct"]].sort_values(
                ["top1_margin", "top1_score", "id_speech"],
                ascending=[True, True, True],
                kind="stable",
            )
            slice_name = "low_margin_correct"
        selected = selected.head(top_n).copy()
        selected.insert(8, "confidence_rank", range(1, len(selected) + 1))
        selected.insert(9, "confidence_slice", slice_name)
        rows.append(selected[prediction_slice_columns(selected)])
    return pd.concat(rows, ignore_index=True)


def write_topk_confidence_outputs(
    systems: tuple[ResultSystem, ...],
    *,
    results_dir: ResultArtifacts,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Write all files for the top-k and confidence result addition."""

    section_dir = output_dir / "topk_confidence"
    section_dir.mkdir(parents=True, exist_ok=True)
    artifacts = results_dir

    overall_topk = build_overall_topk_rescue(systems, artifacts)
    per_author_topk = build_per_author_topk_rescue(systems, artifacts)
    margin_summary = build_score_margin_summary(systems, artifacts)
    per_author_margin = build_per_author_margin_summary(systems, artifacts)
    confident_errors = build_prediction_confidence_slice(
        systems,
        artifacts,
        top_n=top_n,
        confident_errors=True,
    )
    low_margin_correct = build_prediction_confidence_slice(
        systems,
        artifacts,
        top_n=top_n,
        confident_errors=False,
    )

    paths = {
        "topk_rescue_summary": section_dir / "topk_rescue_summary.csv",
        "topk_rescue_by_author": section_dir / "topk_rescue_by_author.csv",
        "confidence_margin_summary": section_dir / "confidence_margin_summary.csv",
        "per_author_margin_summary": section_dir / "per_author_margin_summary.csv",
        "confident_errors": section_dir / "confident_errors.csv",
        "uncertain_correct": section_dir / "uncertain_correct.csv",
    }
    overall_topk.to_csv(paths["topk_rescue_summary"], index=False)
    per_author_topk.to_csv(paths["topk_rescue_by_author"], index=False)
    margin_summary.to_csv(paths["confidence_margin_summary"], index=False)
    per_author_margin.to_csv(paths["per_author_margin_summary"], index=False)
    confident_errors.to_csv(paths["confident_errors"], index=False)
    low_margin_correct.to_csv(paths["uncertain_correct"], index=False)
    return {key: str(path) for key, path in paths.items()}

