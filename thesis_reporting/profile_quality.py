"""Profiling quality and attribution-interaction analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ResultArtifacts, canonical_label, id_sample, validate_same_key_set, validate_unique_key
from .config import ProfileQualityRun, ProfileTarget, ResultSystem
from .topk_confidence import (
    attach_author_metadata,
    author_metadata,
    read_final_predictions,
)


def profile_target_labels(
    targets: tuple[ProfileTarget, ...],
) -> dict[str, str]:
    """Return target labels by canonical profile-target key."""

    return {target.key: target.label for target in targets}


def profile_target_key_aliases(
    targets: tuple[ProfileTarget, ...],
) -> dict[str, str]:
    """Map canonical and artifact profile-target keys to canonical keys."""

    aliases: dict[str, str] = {}
    for target in targets:
        aliases[target.key] = target.key
        aliases[target.prediction_file_key] = target.key
    return aliases


def canonicalize_profile_label_value(target_key: str, value: object) -> str:
    """Canonicalize profile-label values read from external artifacts."""

    text = "" if pd.isna(value) else str(value)
    if target_key == "left_center_right" and text == "senter":
        return "center"
    return text


def canonicalize_profile_label_columns(
    frame: pd.DataFrame,
    *,
    target_column: str = "target",
) -> pd.DataFrame:
    """Canonicalize target-specific label columns after target-name mapping."""

    canonicalized = frame.copy()
    if target_column not in canonicalized.columns:
        return canonicalized
    for column in ("majority_label", "labels_missing_from_model"):
        if column in canonicalized.columns:
            canonicalized[column] = [
                canonicalize_profile_label_value(target_key, value)
                for target_key, value in zip(
                    canonicalized[target_column],
                    canonicalized[column],
                )
            ]
    return canonicalized


def canonicalize_profile_target_metrics(
    frame: pd.DataFrame,
    targets: tuple[ProfileTarget, ...],
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Canonicalize external profile-target names while preserving metric values."""

    if "target" not in frame.columns:
        raise ValueError(f"{source_path}: missing required 'target' column.")
    aliases = profile_target_key_aliases(targets)
    canonical = frame["target"].astype(str).map(aliases)
    unknown = frame.loc[canonical.isna(), "target"]
    if not unknown.empty:
        raise ValueError(
            f"{source_path}: unknown profile target values "
            f"(sample={id_sample(unknown)})."
        )
    canonicalized = frame.copy()
    canonicalized["target"] = canonical
    canonicalized["target_label"] = canonicalized["target"].map(
        profile_target_labels(targets)
    )
    return canonicalize_profile_label_columns(canonicalized)


def read_profile_target_metrics(
    profile_run: ProfileQualityRun,
    targets: tuple[ProfileTarget, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Read and label attribution-test profile target metrics."""

    reader = results_dir
    source_path = reader.path(profile_run.attribution_test_metrics_path)
    frame = reader.read_csv(profile_run.attribution_test_metrics_path)
    frame["profile_quality_key"] = profile_run.key
    frame["profile_quality_label"] = profile_run.label
    return canonicalize_profile_target_metrics(
        frame,
        targets,
        source_path=source_path,
    )


def read_profile_calibration_summary(
    profile_run: ProfileQualityRun,
    targets: tuple[ProfileTarget, ...],
    results_dir: ResultArtifacts,
) -> pd.DataFrame:
    """Read and label profile calibration summary metrics."""

    reader = results_dir
    source_path = reader.path(profile_run.calibration_summary_path)
    frame = reader.read_csv(profile_run.calibration_summary_path)
    frame["profile_quality_key"] = profile_run.key
    frame["profile_quality_label"] = profile_run.label
    return canonicalize_profile_target_metrics(
        frame,
        targets,
        source_path=source_path,
    )


def bool_series(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values to booleans."""

    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def read_profile_predictions(
    profile_run: ProfileQualityRun,
    target: ProfileTarget,
    results_dir: ResultArtifacts,
    *,
    role: str,
) -> pd.DataFrame:
    """Read profile predictions for one target and attribution split role."""

    reader = results_dir
    prediction_path = profile_run.prediction_path(target, role)
    source_path = reader.path(prediction_path)
    frame = reader.read_csv(prediction_path)
    frame["profile_quality_key"] = profile_run.key
    frame["profile_quality_label"] = profile_run.label
    if "profile_target" in frame.columns:
        aliases = profile_target_key_aliases((target,))
        canonical_profile_target = frame["profile_target"].astype(str).map(aliases)
        invalid_targets = frame.loc[
            canonical_profile_target != target.key,
            "profile_target",
        ]
        if not invalid_targets.empty:
            raise ValueError(
                f"{source_path}: expected profile target {target.key!r}; "
                f"found unexpected targets {id_sample(invalid_targets)}."
            )
    frame["profile_target"] = target.key
    frame["profile_target_label"] = target.label
    frame["profile_role"] = role
    if "role" in frame.columns:
        invalid_roles = frame.loc[frame["role"].astype(str) != role, "role"]
        if not invalid_roles.empty:
            raise ValueError(
                f"{source_path}: expected profile prediction role {role!r}; "
                f"found unexpected roles {id_sample(invalid_roles)}."
            )
    frame["id_speech_key"] = frame["id_speech"].map(canonical_label)
    frame["profile_y_true"] = frame["y_true"].map(
        lambda value: canonicalize_profile_label_value(target.key, value)
    )
    frame["profile_y_pred"] = frame["y_pred"].map(
        lambda value: canonicalize_profile_label_value(target.key, value)
    )
    frame["profile_correct"] = bool_series(frame["correct"])
    return frame



def load_profile_prediction_tables(
    profile_run: ProfileQualityRun,
    targets: tuple[ProfileTarget, ...],
    artifacts: ResultArtifacts,
) -> dict[str, pd.DataFrame]:
    """Load and normalize each profile-target prediction table exactly once."""

    return {
        target.key: read_profile_predictions(
            profile_run,
            target,
            artifacts,
            role="test",
        )
        for target in targets
    }


def build_profile_target_confusions(
    profile_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build confusion pairs from prepared profile-prediction tables."""

    combined = pd.concat(profile_predictions.values(), ignore_index=True)
    errors = combined.loc[~combined["profile_correct"]].copy()
    grouped = (
        errors.groupby(
            [
                "profile_quality_key",
                "profile_quality_label",
                "profile_target",
                "profile_target_label",
                "profile_y_true",
                "profile_y_pred",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            confusion_count=("profile_correct", "size"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
        )
    )
    totals = grouped.groupby("profile_target")["confusion_count"].transform("sum")
    grouped["share_of_target_errors"] = grouped["confusion_count"] / totals
    return grouped.sort_values(
        ["profile_target", "confusion_count", "mean_confidence"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)



def build_profile_confusion_summary(
    metrics: pd.DataFrame,
    confusions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine prepared target metrics with each target's top confusion."""

    metric_columns = [
        "profile_quality_key",
        "profile_quality_label",
        "target",
        "target_label",
        "role",
        "n_samples",
        "n_true_classes",
        "n_model_classes",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "majority_label",
        "majority_accuracy",
        "majority_macro_f1",
        "macro_f1_lift_over_majority",
        "accuracy_lift_over_majority",
    ]
    summary = metrics[[column for column in metric_columns if column in metrics.columns]]
    if confusions.empty:
        summary = summary.copy()
        summary["most_common_true_profile"] = pd.NA
        summary["most_common_pred_profile"] = pd.NA
        summary["most_common_confusion_count"] = 0
        summary["most_common_confusion_error_share"] = 0.0
        return summary

    top_confusion = (
        confusions.sort_values(
            ["profile_target", "confusion_count", "mean_confidence"],
            ascending=[True, False, False],
            kind="stable",
        )
        .groupby("profile_target", as_index=False, sort=False)
        .head(1)
        [
            [
                "profile_target",
                "profile_y_true",
                "profile_y_pred",
                "confusion_count",
                "share_of_target_errors",
            ]
        ]
        .rename(
            columns={
                "profile_target": "target",
                "profile_y_true": "most_common_true_profile",
                "profile_y_pred": "most_common_pred_profile",
                "confusion_count": "most_common_confusion_count",
                "share_of_target_errors": "most_common_confusion_error_share",
            }
        )
    )
    return summary.merge(top_confusion, on="target", how="left", validate="one_to_one")



def build_profile_confidence_summary(
    profile_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize confidence from prepared profile-prediction tables."""

    combined = pd.concat(profile_predictions.values(), ignore_index=True)
    grouped = (
        combined.groupby(
            [
                "profile_quality_key",
                "profile_quality_label",
                "profile_target",
                "profile_target_label",
                "profile_correct",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            prediction_count=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            p25_confidence=("confidence", lambda values: values.quantile(0.25)),
            p75_confidence=("confidence", lambda values: values.quantile(0.75)),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        )
    )
    totals = grouped.groupby("profile_target")["prediction_count"].transform("sum")
    grouped["share_of_target_predictions"] = grouped["prediction_count"] / totals
    grouped["profile_correct_group"] = grouped["profile_correct"].map(
        {True: "profile_correct", False: "profile_wrong"}
    )
    return grouped.sort_values(
        ["profile_target", "profile_correct"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)



def build_profile_confident_errors(
    targets: tuple[ProfileTarget, ...],
    profile_predictions: dict[str, pd.DataFrame],
    *,
    top_n: int,
) -> pd.DataFrame:
    """Select highest-confidence errors from prepared prediction tables."""

    rows: list[pd.DataFrame] = []
    for target in targets:
        frame = profile_predictions[target.key]
        selected = frame.loc[~frame["profile_correct"]].sort_values(
            ["confidence", "id_speech"],
            ascending=[False, True],
            kind="stable",
        ).head(top_n)
        selected = selected.copy()
        selected.insert(4, "profile_error_rank", range(1, len(selected) + 1))
        rows.append(
            selected[
                [
                    "profile_quality_key",
                    "profile_quality_label",
                    "profile_target",
                    "profile_target_label",
                    "profile_error_rank",
                    "id_speech",
                    "id_person",
                    "election",
                    "party",
                    "profile_y_true",
                    "profile_y_pred",
                    "confidence",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def attribution_profile_systems(
    systems: tuple[ResultSystem, ...],
) -> tuple[ResultSystem, ...]:
    """Return attribution systems that consume predicted profile signals."""

    return tuple(
        system
        for system in systems
        if system.representation == "probability" and system.scope == "all_signal"
    )


def validate_profile_attribution_ids(
    profile: pd.DataFrame,
    attribution: pd.DataFrame,
    *,
    system: ResultSystem,
    target: ProfileTarget,
) -> None:
    """Validate exact speech-id coverage before joining profile and attribution."""

    profile_context = f"profile predictions target={target.key}"
    attribution_context = f"attribution predictions system={system.key}"
    validate_unique_key(profile, "id_speech_key", context=profile_context)
    validate_unique_key(attribution, "id_speech_key", context=attribution_context)
    validate_same_key_set(
        profile,
        attribution,
        "id_speech_key",
        left_context=profile_context,
        right_context=attribution_context,
    )



def load_attribution_prediction_tables(
    systems: tuple[ResultSystem, ...],
    artifacts: ResultArtifacts,
) -> dict[str, pd.DataFrame]:
    """Load each profile-consuming attribution prediction table once."""

    selected_systems = attribution_profile_systems(systems)
    return {
        system.key: read_final_predictions(system, artifacts)
        for system in selected_systems
    }


def load_attribution_metadata_tables(
    systems: tuple[ResultSystem, ...],
    artifacts: ResultArtifacts,
) -> dict[str, pd.DataFrame]:
    """Load each profile-consuming system's author metadata once."""

    return {
        system.key: author_metadata(system, artifacts)
        for system in attribution_profile_systems(systems)
    }


def build_attribution_vs_profile_correctness(
    systems: tuple[ResultSystem, ...],
    targets: tuple[ProfileTarget, ...],
    attribution_predictions: dict[str, pd.DataFrame],
    profile_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join prepared attribution and profile correctness tables by speech."""

    rows: list[pd.DataFrame] = []
    for system in attribution_profile_systems(systems):
        attribution = attribution_predictions[system.key][
            [
                "id_speech_key",
                "y_true_author_label",
                "y_pred_author_label",
                "top1_correct",
                "top1_score",
                "top1_margin",
            ]
        ].rename(columns={"top1_correct": "attribution_correct"})
        for target in targets:
            profile = profile_predictions[target.key]
            validate_profile_attribution_ids(
                profile,
                attribution,
                system=system,
                target=target,
            )
            merged = profile.merge(
                attribution,
                on="id_speech_key",
                how="inner",
                validate="one_to_one",
            )
            merged["system_key"] = system.key
            merged["system_label"] = system.label
            merged["phase"] = system.phase
            merged["split"] = system.split
            merged["architecture"] = system.architecture
            merged["representation"] = system.representation
            merged["scope"] = system.scope
            merged["condition_id"] = system.condition_id
            rows.append(merged)

    combined = pd.concat(rows, ignore_index=True)
    output_columns = [
        "system_key",
        "system_label",
        "phase",
        "split",
        "architecture",
        "representation",
        "scope",
        "condition_id",
        "profile_target",
        "profile_target_label",
        "id_speech",
        "id_person",
        "election",
        "party",
        "profile_y_true",
        "profile_y_pred",
        "profile_correct",
        "confidence",
        "y_true_author_label",
        "y_pred_author_label",
        "attribution_correct",
        "top1_score",
        "top1_margin",
    ]
    return combined[[column for column in output_columns if column in combined.columns]]



def build_profile_correctness_vs_attribution(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize a prepared attribution/profile correctness join."""

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
                "profile_target",
                "profile_target_label",
                "profile_correct",
                "attribution_correct",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            speech_count=("id_speech", "size"),
            mean_profile_confidence=("confidence", "mean"),
            mean_attribution_margin=("top1_margin", "mean"),
        )
    )
    totals = grouped.groupby(["system_key", "profile_target"])["speech_count"].transform(
        "sum"
    )
    grouped["share_of_system_target_speeches"] = grouped["speech_count"] / totals
    grouped["profile_correct_group"] = grouped["profile_correct"].map(
        {True: "profile_correct", False: "profile_wrong"}
    )
    grouped["attribution_correct_group"] = grouped["attribution_correct"].map(
        {True: "attribution_correct", False: "attribution_wrong"}
    )
    return grouped.sort_values(
        [
            "system_key",
            "profile_target",
            "profile_correct",
            "attribution_correct",
        ],
        ascending=[True, True, False, False],
        kind="stable",
    )



def build_profile_wrong_attribution_wrong_examples(
    systems: tuple[ResultSystem, ...],
    targets: tuple[ProfileTarget, ...],
    attribution_predictions: dict[str, pd.DataFrame],
    attribution_metadata: dict[str, pd.DataFrame],
    profile_predictions: dict[str, pd.DataFrame],
    *,
    top_n: int,
) -> pd.DataFrame:
    """Select joint errors from prepared profile and attribution tables."""

    output_columns = [
        "system_key",
        "system_label",
        "profile_attribution_error_rank",
        "profile_target",
        "profile_target_label",
        "id_speech",
        "id_person",
        "election",
        "party",
        "profile_y_true",
        "profile_y_pred",
        "confidence",
        "y_true_author_label",
        "true_author_display",
        "y_pred_author_label",
        "pred_author_display",
        "top1_score",
        "top1_margin",
    ]
    rows: list[pd.DataFrame] = []
    for system in attribution_profile_systems(systems):
        attribution = attribution_predictions[system.key].copy()
        metadata = attribution_metadata[system.key]
        attribution = attach_author_metadata(
            attribution,
            metadata,
            label_column="y_true_author_label",
            prefix="true",
        )
        attribution = attach_author_metadata(
            attribution,
            metadata,
            label_column="y_pred_author_label",
            prefix="pred",
        )
        attribution = attribution[
            [
                "id_speech_key",
                "y_true_author_label",
                "true_author_display",
                "y_pred_author_label",
                "pred_author_display",
                "top1_correct",
                "top1_score",
                "top1_margin",
            ]
        ].rename(columns={"top1_correct": "attribution_correct"})
        for target in targets:
            profile = profile_predictions[target.key]
            validate_profile_attribution_ids(
                profile,
                attribution,
                system=system,
                target=target,
            )
            merged = profile.merge(
                attribution,
                on="id_speech_key",
                how="inner",
                validate="one_to_one",
            )
            selected = merged.loc[
                (~merged["profile_correct"]) & (~merged["attribution_correct"])
            ].sort_values(
                ["confidence", "top1_margin", "id_speech"],
                ascending=[False, False, True],
                kind="stable",
            )
            if selected.empty:
                continue
            target_slice = selected.head(top_n).copy()
            target_slice.insert(0, "system_key", system.key)
            target_slice.insert(1, "system_label", system.label)
            target_slice.insert(
                2,
                "profile_attribution_error_rank",
                range(1, len(target_slice) + 1),
            )
            rows.append(target_slice)
    if not rows:
        return pd.DataFrame(columns=output_columns)
    examples = pd.concat(rows, ignore_index=True)
    return examples[output_columns]


def write_profile_quality_outputs(
    systems: tuple[ResultSystem, ...],
    profile_run: ProfileQualityRun,
    targets: tuple[ProfileTarget, ...],
    *,
    results_dir: ResultArtifacts,
    output_dir: Path,
    top_n: int,
) -> dict[str, str]:
    """Load source tables once and write all profile-quality outputs."""

    section_dir = output_dir / "profile_quality"
    section_dir.mkdir(parents=True, exist_ok=True)

    target_metrics = read_profile_target_metrics(profile_run, targets, results_dir)
    calibration = read_profile_calibration_summary(
        profile_run,
        targets,
        results_dir,
    )
    profile_predictions = load_profile_prediction_tables(
        profile_run,
        targets,
        results_dir,
    )
    attribution_predictions = load_attribution_prediction_tables(
        systems,
        results_dir,
    )
    attribution_metadata = load_attribution_metadata_tables(
        systems,
        results_dir,
    )
    confusions = build_profile_target_confusions(profile_predictions)
    confusion_summary = build_profile_confusion_summary(target_metrics, confusions)
    confidence_summary = build_profile_confidence_summary(profile_predictions)
    confident_errors = build_profile_confident_errors(
        targets,
        profile_predictions,
        top_n=top_n,
    )
    correctness = build_attribution_vs_profile_correctness(
        systems,
        targets,
        attribution_predictions,
        profile_predictions,
    )
    correctness_summary = build_profile_correctness_vs_attribution(correctness)
    joint_errors = build_profile_wrong_attribution_wrong_examples(
        systems,
        targets,
        attribution_predictions,
        attribution_metadata,
        profile_predictions,
        top_n=top_n,
    )

    paths = {
        "profile_target_metrics": section_dir / "profile_target_metrics.csv",
        "profile_calibration_summary": section_dir / "profile_calibration_summary.csv",
        "profile_confusions_by_target": (
            section_dir / "profile_confusions_by_target.csv"
        ),
        "profile_confusion_summary": section_dir / "profile_confusion_summary.csv",
        "profile_confidence_summary": section_dir / "profile_confidence_summary.csv",
        "profile_confident_errors": section_dir / "profile_confident_errors.csv",
        "attribution_vs_profile_correctness": (
            section_dir / "attribution_vs_profile_correctness.csv"
        ),
        "attribution_vs_profile_correctness_summary": (
            section_dir / "attribution_vs_profile_correctness_summary.csv"
        ),
        "profile_wrong_attribution_wrong_examples": (
            section_dir / "profile_wrong_attribution_wrong_examples.csv"
        ),
    }
    target_metrics.to_csv(paths["profile_target_metrics"], index=False)
    calibration.to_csv(paths["profile_calibration_summary"], index=False)
    confusions.to_csv(paths["profile_confusions_by_target"], index=False)
    confusion_summary.to_csv(paths["profile_confusion_summary"], index=False)
    confidence_summary.to_csv(paths["profile_confidence_summary"], index=False)
    confident_errors.to_csv(paths["profile_confident_errors"], index=False)
    correctness.to_csv(paths["attribution_vs_profile_correctness"], index=False)
    correctness_summary.to_csv(
        paths["attribution_vs_profile_correctness_summary"],
        index=False,
    )
    joint_errors.to_csv(paths["profile_wrong_attribution_wrong_examples"], index=False)
    return {key: str(path) for key, path in paths.items()}
