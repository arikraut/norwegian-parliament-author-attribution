"""Feature extraction runner for row-level metadata, targets, and stylometry."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from data_pipeline.row_features.quality import build_stylometry_quality_reports
from data_pipeline.row_features.stylometry import (
    _load_stylometry_pipeline,
    _safe_float,
    char_distribution_features,
    extract_spacy_stylometry_from_df,
    function_word_features_from_doc,
    load_bokmal_function_words,
    stylometry_feature_family,
    textdescriptives_features_from_doc,
)
from data_pipeline.row_features.targets import (
    add_age_bin_column,
    add_left_center_right_column,
    build_feature_split_summary,
    build_party_axis_map,
    build_row_meta_frame,
    build_targets_frame,
    save_target_distributions,
    validate_feature_corpus_columns,
)
from data_pipeline.utils import (
    copy_config_outputs,
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)


def _stylometry_enabled(stylo_cfg: dict[str, object]) -> bool:
    """Read and validate the required [stylometry].enabled flag from a feature config."""
    enabled = stylo_cfg.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("[stylometry].enabled must be set to true or false.")
    return enabled


def save_feature_manifest(
    split_name: str,
    feature_set_name: str,
    project_root: Path,
    feature_config_path: Path,
    row_feature_dir: Path,
    feature_results_dir: Path,
    row_counts: dict[str, int],
    stylometry_generated: bool,
    stylo_cols: list[str],
    stylo_batch_size: int | None = None,
    quality_low_variance_threshold: float | None = None,
) -> None:
    """Write manifest.json for a row-feature bundle, recording artifact paths and stylometry status."""
    artifacts = {
        "row_meta": "row_meta.csv",
        "targets": "targets.csv",
    }
    stylometry = {
        "generated": stylometry_generated,
        "n_features": len(stylo_cols),
    }
    if stylometry_generated:
        artifacts.update(
            {
                "stylometry_raw": "stylometry_raw.csv.gz",
                "stylometry_quality_report": "stylometry_quality_report.csv",
                "stylometry_low_variance_report": "stylometry_low_variance_report.csv",
            }
        )
        stylometry["batch_size"] = int(stylo_batch_size or 0)
        if quality_low_variance_threshold is not None:
            stylometry["quality_low_variance_threshold"] = float(
                quality_low_variance_threshold
            )

    manifest = {
        "split_name": split_name,
        "feature_set_name": feature_set_name,
        "feature_config_path": relative_to_project(project_root, feature_config_path),
        "split_dir": relative_to_project(project_root, row_feature_dir.parent.parent),
        "row_feature_dir": relative_to_project(project_root, row_feature_dir),
        "feature_results_dir": relative_to_project(project_root, feature_results_dir),
        "row_counts": row_counts,
        "stylometry": stylometry,
        "artifacts": artifacts,
    }
    write_json(row_feature_dir / "manifest.json", manifest)


def _default_party_axis_config() -> dict[str, list[str]]:
    """Return the default Norwegian left/center/right party axis config."""
    return {
        "left": [
            "a",
            "ap",
            "arbeiderpartiet",
            "sv",
            "sosialistiskvenstreparti",
            "r",
            "rodt",
            "rødt",
        ],
        "center": [
            "sp",
            "senterpartiet",
            "v",
            "venstre",
            "krf",
            "kristelig folkeparti",
            "kp",
            "kystpartiet",
            "mdg",
            "miljøpartiet de grønne",
        ],
        "right": ["h", "høyre", "frp", "fremskrittspartiet"],
    }


def run_feature_generation(config_path: Path) -> dict[str, object]:
    """Run row-level feature generation from a feature config."""
    config_path = Path(config_path).resolve()
    project_root = find_project_root(config_path)

    with config_path.open("rb") as fh:
        feature_config = tomllib.load(fh)

    feature_meta = feature_config["feature"]
    data_cfg = feature_config.get("data", {})
    target_cfg = feature_config.get("targets", {})
    age_bin_cfg = feature_config.get("age_bins", {})
    stylo_cfg = feature_config.get("stylometry", {})
    stylo_enabled = _stylometry_enabled(stylo_cfg)
    party_axis_cfg = feature_config.get("party_axis", _default_party_axis_config())
    party_axis_map = build_party_axis_map(party_axis_cfg)

    feature_set_name = str(feature_meta["name"])
    split_name = str(feature_meta["split_name"])
    spacy_model = str(feature_meta.get("spacy_model", "nb_core_news_sm"))
    save_author_labels = bool(target_cfg.get("save_author_labels", True))
    profiling_labels = list(
        target_cfg.get(
            "profiling_labels",
            ["party", "age", "age_bin", "female", "language", "left_center_right"],
        )
    )
    age_bin_edges = [int(e) for e in age_bin_cfg.get("edges", [0, 30, 60, 200])]
    age_bin_labels = list(age_bin_cfg.get("labels", ["<30", "30-60", "60+"]))
    stylo_batch_size = int(stylo_cfg["batch_size"]) if stylo_enabled else None
    stylo_low_var_thr = (
        float(stylo_cfg["low_variance_threshold_for_report"]) if stylo_enabled else None
    )

    if len(age_bin_edges) != len(age_bin_labels) + 1:
        raise ValueError(
            "age_bins.edges must have exactly one more value than age_bins.labels"
        )

    splits_root = resolve_project_path(
        project_root, data_cfg.get("splits_dir", "data/splits")
    )
    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/features")
    )
    split_dir = splits_root / split_name
    corpus_dir = split_dir / "corpus"
    row_feature_dir = split_dir / "row_features" / feature_set_name
    feature_results_dir = results_root / split_name / feature_set_name
    target_dist_dir = feature_results_dir / "target_distributions"

    for directory in (row_feature_dir, feature_results_dir, target_dist_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copy_config_outputs(config_path, row_feature_dir / "feature_config.toml")

    train_path = corpus_dir / "train.csv"
    test_path = corpus_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected corpus files do not exist under {corpus_dir}"
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    validate_feature_corpus_columns(
        {"train": df_train, "test": df_test},
        profiling_labels=profiling_labels,
    )

    df_train = add_age_bin_column(df_train, age_bin_edges, age_bin_labels)
    df_test = add_age_bin_column(df_test, age_bin_edges, age_bin_labels)

    df_train = add_left_center_right_column(df_train, party_axis_map)
    df_test = add_left_center_right_column(df_test, party_axis_map)

    if "female" in df_train.columns and df_train["female"].dtype == bool:
        df_train["female"] = df_train["female"].astype(int)
        df_test["female"] = df_test["female"].astype(int)

    row_meta_df = pd.concat(
        [
            build_row_meta_frame(df_train, split_name, "train"),
            build_row_meta_frame(df_test, split_name, "test"),
        ],
        ignore_index=True,
    )
    row_meta_df.to_csv(row_feature_dir / "row_meta.csv", index=False)

    row_targets_df = pd.concat(
        [
            build_targets_frame(
                df_train, split_name, "train", save_author_labels, profiling_labels
            ),
            build_targets_frame(
                df_test, split_name, "test", save_author_labels, profiling_labels
            ),
        ],
        ignore_index=True,
    )
    row_targets_df.to_csv(row_feature_dir / "targets.csv", index=False)

    stylo_cols: list[str] = []
    function_word_count = 0
    if stylo_enabled:
        nlp = _load_stylometry_pipeline(spacy_model)
        function_words = load_bokmal_function_words()
        function_word_count = len(function_words)

        stylo_parts: dict[str, pd.DataFrame] = {}
        stylo_quality_parts: dict[str, dict[str, int | float | str]] = {}
        for outer_role, frame in [("train", df_train), ("test", df_test)]:
            stylo_df, quality_summary = extract_spacy_stylometry_from_df(
                frame,
                nlp,
                function_words,
                batch_size=int(stylo_batch_size),
                desc=outer_role,
                return_quality=True,
            )
            stylo_parts[outer_role] = stylo_df
            stylo_quality_parts[outer_role] = quality_summary

        stylo_train = stylo_parts["train"]
        stylo_cols = [
            col for col in stylo_train.columns if col not in ("id_speech", "id_person")
        ]

        row_stylo_df = pd.concat(
            [stylo.assign(outer_role=role) for role, stylo in stylo_parts.items()],
            ignore_index=True,
        )
        row_stylo_df.to_csv(
            row_feature_dir / "stylometry_raw.csv.gz", index=False, compression="gzip"
        )

        stylometry_quality_df, stylometry_low_variance_df = (
            build_stylometry_quality_reports(
                stylo_parts,
                stylo_quality_parts,
                low_variance_threshold=float(stylo_low_var_thr),
            )
        )
        stylometry_quality_df.to_csv(
            row_feature_dir / "stylometry_quality_report.csv", index=False
        )
        stylometry_low_variance_df.to_csv(
            row_feature_dir / "stylometry_low_variance_report.csv", index=False
        )

    split_summary_df = pd.DataFrame(
        [
            build_feature_split_summary(df_train, "train"),
            build_feature_split_summary(df_test, "test"),
        ]
    )
    split_summary_df.to_csv(feature_results_dir / "split_summary.csv", index=False)

    target_split_frames = {
        str(role): frame.drop(columns=["outer_role"], errors="ignore")
        for role, frame in row_targets_df.groupby("outer_role", sort=False)
    }
    target_cols = (["author"] if save_author_labels else []) + profiling_labels
    target_summary_df = save_target_distributions(
        target_split_frames, target_cols, target_dist_dir
    )
    if not target_summary_df.empty:
        target_summary_df.to_csv(
            feature_results_dir / "target_summary.csv", index=False
        )

    row_counts = {
        "train": int(len(df_train)),
        "test": int(len(df_test)),
    }
    save_feature_manifest(
        split_name=split_name,
        feature_set_name=feature_set_name,
        project_root=project_root,
        feature_config_path=config_path,
        row_feature_dir=row_feature_dir,
        feature_results_dir=feature_results_dir,
        row_counts=row_counts,
        stylometry_generated=stylo_enabled,
        stylo_cols=stylo_cols,
        stylo_batch_size=stylo_batch_size,
        quality_low_variance_threshold=stylo_low_var_thr,
    )

    summary = {
        "split_name": split_name,
        "feature_set_name": feature_set_name,
        "feature_config_path": relative_to_project(project_root, config_path),
        "row_feature_dir": relative_to_project(project_root, row_feature_dir),
        "feature_results_dir": relative_to_project(project_root, feature_results_dir),
        "target_distribution_dir": relative_to_project(project_root, target_dist_dir),
        "spacy_model": spacy_model,
        "stylometry_generated": stylo_enabled,
        "function_word_count": int(function_word_count),
        "row_counts": row_counts,
        "n_stylometry_features": int(len(stylo_cols)),
    }
    if stylo_enabled:
        summary["quality_report_path"] = relative_to_project(
            project_root,
            row_feature_dir / "stylometry_quality_report.csv",
        )
        summary["low_variance_report_path"] = relative_to_project(
            project_root,
            row_feature_dir / "stylometry_low_variance_report.csv",
        )
    return summary
