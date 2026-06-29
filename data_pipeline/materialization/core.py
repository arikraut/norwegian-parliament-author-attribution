"""Materialization runner."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data_pipeline.materialization.blocks import (
    _fit_stylometry_scaler,
    _make_row_order,
    _save_feature_columns,
    _save_label_arrays,
    _save_sparse_matrix,
)
from data_pipeline.materialization.config import (
    _read_required_json,
    _resolve_materialization_stage,
    _validate_stage_units,
    resolve_materialization_stage,
)
from data_pipeline.materialization.inputs import (
    _load_materialization_inputs,
    _merge_row_sources,
)
from data_pipeline.materialization.reports import (
    _build_stylometry_column_report,
    _build_target_summary,
    _summarize_stylometry_drift,
)
from data_pipeline.materialization.units import _select_units
from data_pipeline.utils import (
    create_progress_bar,
    find_project_root,
    relative_to_project,
    write_json,
    write_toml,
)


def run_materialization(
    config_path: Path,
    *,
    stage: str,
    rebuild: bool = False,
    show_progress: bool = False,
) -> dict:
    """Materialize TF-IDF and stylometry feature matrices for each selected unit."""
    project_root = find_project_root(config_path.parent)
    resolved_stage = _resolve_materialization_stage(project_root, config_path, stage=stage)

    materialized_root = resolved_stage.materialized_root
    if rebuild and materialized_root.exists():
        shutil.rmtree(materialized_root)
    materialized_root.mkdir(parents=True, exist_ok=True)
    write_toml(resolved_stage.resolved_config_path, resolved_stage.config)

    manifest_path = materialized_root / "manifest.json"
    if manifest_path.exists() and not rebuild:
        existing_manifest = _read_required_json(manifest_path)
        _validate_stage_units(
            list(existing_manifest.get("units", [])),
            stage=resolved_stage.stage,
            selector=resolved_stage.selector,
        )
        existing_manifest.update(
            {
                "config_path": relative_to_project(project_root, config_path),
                "stage": resolved_stage.stage,
                "selector": resolved_stage.selector,
                "split_name": resolved_stage.split_name,
                "row_feature_name": resolved_stage.row_feature_name,
                "materialization_name": resolved_stage.materialization_name,
                "resolved_config_path": relative_to_project(
                    project_root, resolved_stage.resolved_config_path
                ),
            }
        )
        write_json(manifest_path, existing_manifest)
        return existing_manifest

    loaded = _load_materialization_inputs(
        project_root, config_path, stage=resolved_stage.stage
    )

    config = loaded["config"]
    enabled_blocks = loaded["enabled_blocks"]
    materialization_name = loaded["materialization_name"]
    word_cfg = config.get("word_tfidf", {})
    char_cfg = config.get("char_tfidf", {})
    units = _select_units(
        config,
        loaded["outer_membership"],
        loaded["fold_membership"],
        split_strategy=loaded["split_strategy"],
    )
    _validate_stage_units(
        [{"unit_id": unit.unit_id, "eval_role": unit.eval_role} for unit in units],
        stage=loaded["stage"],
        selector=loaded["selector"],
    )

    corpus_all = loaded["corpus_all"]
    row_meta = loaded["row_meta"]
    row_targets = loaded["row_targets"]
    row_stylo = loaded["row_stylo"]

    stylo_feature_cols = (
        [
            col
            for col in row_stylo.columns
            if col not in {"id_speech", "id_person", "outer_role"}
        ]
        if row_stylo is not None
        else []
    )
    if "stylo" in enabled_blocks and not stylo_feature_cols:
        raise ValueError("No raw stylometry columns were found in stylometry_raw.csv.gz.")

    run_summary = {
        "split_name": loaded["split_name"],
        "row_feature_name": loaded["row_feature_name"],
        "materialization_name": materialization_name,
        "split_strategy": loaded["split_strategy"],
        "stage": loaded["stage"],
        "selector": loaded["selector"],
        "enabled_blocks": list(enabled_blocks),
        "combined_available": False,
        "config_path": relative_to_project(project_root, config_path),
        "resolved_config_path": relative_to_project(
            project_root, loaded["resolved_config_path"]
        ),
        "materialized_root": relative_to_project(project_root, materialized_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "units": [],
    }
    stylometry_drift_rows: list[dict[str, object]] = []
    progress = create_progress_bar(
        total=len(units),
        desc="Materializing",
        unit="unit",
        show_progress=show_progress,
    )
    if show_progress:
        print(
            f"Loaded {len(units)} materialization unit(s) for "
            f"{loaded['split_name']}/{materialization_name}."
        )

    for unit in units:
        if show_progress and progress is None:
            print(f"[materialize] {unit.unit_id}")
        membership = unit.membership.copy()
        role_column = "fold_role" if "fold_role" in membership.columns else "outer_role"
        train_membership = membership[membership[role_column] == "train"].copy()
        eval_membership = membership[membership[role_column] == unit.eval_role].copy()
        if train_membership.empty or eval_membership.empty:
            raise ValueError(
                f"Unit {unit.unit_id} is missing train or {unit.eval_role} rows."
            )

        train_df = _merge_row_sources(
            train_membership, corpus_all, row_meta, row_targets, row_stylo
        )
        eval_df = _merge_row_sources(
            eval_membership, corpus_all, row_meta, row_targets, row_stylo
        )

        unit_dir = materialized_root / unit.unit_id
        preprocessors_dir = unit_dir / "preprocessors"
        matrices_dir = unit_dir / "matrices"
        labels_dir = unit_dir / "labels"
        row_order_dir = unit_dir / "row_order"
        for directory in [preprocessors_dir, matrices_dir, labels_dir, row_order_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        train_text = train_df["text"].astype(str)
        eval_text = eval_df["text"].astype(str)

        word_vectorizer: TfidfVectorizer | None = None
        char_vectorizer: TfidfVectorizer | None = None
        kept_stylo_cols: list[str] | None = None

        x_train_word = sparse.csr_matrix((len(train_df), 0))
        x_eval_word = sparse.csr_matrix((len(eval_df), 0))
        x_train_char = sparse.csr_matrix((len(train_df), 0))
        x_eval_char = sparse.csr_matrix((len(eval_df), 0))
        x_train_stylo_sparse = sparse.csr_matrix((len(train_df), 0))
        x_eval_stylo_sparse = sparse.csr_matrix((len(eval_df), 0))

        if "word" in enabled_blocks:
            word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(
                    int(word_cfg.get("min_n", 1)),
                    int(word_cfg.get("max_n", 3)),
                ),
                min_df=word_cfg.get("min_df", 5),
                max_df=word_cfg.get("max_df", 1.0),
                max_features=int(word_cfg.get("max_features", 100_000)),
            )
            x_train_word = word_vectorizer.fit_transform(train_text)
            x_eval_word = word_vectorizer.transform(eval_text)
            _save_sparse_matrix(matrices_dir / "X_train_word.npz", x_train_word)
            _save_sparse_matrix(
                matrices_dir / f"X_{unit.eval_role}_word.npz", x_eval_word
            )
            joblib.dump(word_vectorizer, preprocessors_dir / "word_vectorizer.joblib")

        if "char" in enabled_blocks:
            char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(
                    int(char_cfg.get("min_n", 2)),
                    int(char_cfg.get("max_n", 5)),
                ),
                min_df=char_cfg.get("min_df", 5),
                max_df=char_cfg.get("max_df", 1.0),
                max_features=int(char_cfg.get("max_features", 100_000)),
            )
            x_train_char = char_vectorizer.fit_transform(train_text)
            x_eval_char = char_vectorizer.transform(eval_text)
            _save_sparse_matrix(matrices_dir / "X_train_char.npz", x_train_char)
            _save_sparse_matrix(
                matrices_dir / f"X_{unit.eval_role}_char.npz", x_eval_char
            )
            joblib.dump(char_vectorizer, preprocessors_dir / "char_vectorizer.joblib")

        if "stylo" in enabled_blocks:
            kept_stylo_cols, stylo_column_report = _build_stylometry_column_report(
                config,
                train_df,
                eval_df,
                unit.eval_role,
                stylo_feature_cols,
            )
            stylo_column_report.to_csv(
                unit_dir / "stylometry_column_report.csv", index=False
            )
            stylometry_drift_summary = _summarize_stylometry_drift(
                unit.unit_id, unit.eval_role, stylo_column_report
            )
            stylometry_drift_rows.append(stylometry_drift_summary)
            x_train_stylo_raw = (
                train_df[kept_stylo_cols].to_numpy(dtype=float)
                if kept_stylo_cols
                else np.zeros((len(train_df), 0), dtype=float)
            )
            x_eval_stylo_raw = (
                eval_df[kept_stylo_cols].to_numpy(dtype=float)
                if kept_stylo_cols
                else np.zeros((len(eval_df), 0), dtype=float)
            )
            scaler, x_train_stylo, x_eval_stylo = _fit_stylometry_scaler(
                config, x_train_stylo_raw, x_eval_stylo_raw
            )
            x_train_stylo_sparse = sparse.csr_matrix(x_train_stylo)
            x_eval_stylo_sparse = sparse.csr_matrix(x_eval_stylo)
            _save_sparse_matrix(
                matrices_dir / "X_train_stylo.npz", x_train_stylo_sparse
            )
            _save_sparse_matrix(
                matrices_dir / f"X_{unit.eval_role}_stylo.npz", x_eval_stylo_sparse
            )
            joblib.dump(scaler, preprocessors_dir / "stylo_scaler.joblib")
        else:
            stylometry_drift_summary = None

        feature_columns = _save_feature_columns(
            unit_dir,
            char_vectorizer=char_vectorizer,
            word_vectorizer=word_vectorizer,
            stylo_feature_cols=kept_stylo_cols,
        )

        saved_targets = _save_label_arrays(labels_dir, train_df, eval_df, unit.eval_role)

        train_rows = _make_row_order(train_df, unit.unit_id, "train")
        eval_rows = _make_row_order(eval_df, unit.unit_id, unit.eval_role)
        train_rows.to_csv(row_order_dir / "train_rows.csv", index=False)
        eval_rows.to_csv(row_order_dir / f"{unit.eval_role}_rows.csv", index=False)

        manifest = {
            "unit_id": unit.unit_id,
            "split_name": loaded["split_name"],
            "row_feature_name": loaded["row_feature_name"],
            "materialization_name": materialization_name,
            "split_strategy": loaded["split_strategy"],
            "stage": loaded["stage"],
            "selector": loaded["selector"],
            "enabled_blocks": list(enabled_blocks),
            "combined_available": False,
            "config_path": relative_to_project(project_root, config_path),
            "resolved_config_path": relative_to_project(
                project_root, loaded["resolved_config_path"]
            ),
            "source_paths": {
                "corpus_all": relative_to_project(
                    project_root, loaded["corpus_dir"] / "all.csv"
                ),
                "row_meta": relative_to_project(
                    project_root, loaded["row_feature_dir"] / "row_meta.csv"
                ),
                "targets": relative_to_project(
                    project_root, loaded["row_feature_dir"] / "targets.csv"
                ),
            },
            "roles": {
                "train": {
                    "rows": int(len(train_df)),
                    "elections": (
                        sorted(
                            train_df["election"]
                            .dropna()
                            .astype(int)
                            .unique()
                            .tolist()
                        )
                        if "election" in train_df.columns
                        else []
                    ),
                },
                unit.eval_role: {
                    "rows": int(len(eval_df)),
                    "elections": (
                        sorted(
                            eval_df["election"]
                            .dropna()
                            .astype(int)
                            .unique()
                            .tolist()
                        )
                        if "election" in eval_df.columns
                        else []
                    ),
                },
            },
            "dimensions": {
                "word_tfidf": {
                    "generated": "word" in enabled_blocks,
                    "train_rows": int(x_train_word.shape[0]),
                    f"{unit.eval_role}_rows": int(x_eval_word.shape[0]),
                    "cols": int(x_train_word.shape[1]),
                },
                "char_tfidf": {
                    "generated": "char" in enabled_blocks,
                    "train_rows": int(x_train_char.shape[0]),
                    f"{unit.eval_role}_rows": int(x_eval_char.shape[0]),
                    "cols": int(x_train_char.shape[1]),
                },
                "stylo": {
                    "generated": "stylo" in enabled_blocks,
                    "train_rows": int(x_train_stylo_sparse.shape[0]),
                    f"{unit.eval_role}_rows": int(x_eval_stylo_sparse.shape[0]),
                    "cols": int(x_train_stylo_sparse.shape[1]),
                },
                "combined": {
                    "generated": False,
                    "train_rows": int(len(train_df)),
                    f"{unit.eval_role}_rows": int(len(eval_df)),
                    "cols": 0,
                },
            },
            "saved_targets": saved_targets,
            "feature_columns": {
                "path": relative_to_project(
                    project_root, unit_dir / "feature_columns.json"
                ),
                "counts": {
                    block: len(columns) for block, columns in feature_columns.items()
                },
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if "stylo" in enabled_blocks:
            manifest["source_paths"]["stylometry_raw"] = relative_to_project(
                project_root,
                loaded["row_feature_dir"] / "stylometry_raw.csv.gz",
            )
            manifest["stylometry_column_selection"] = {
                "path": relative_to_project(
                    project_root, unit_dir / "stylometry_column_report.csv"
                ),
                "input_columns": int(len(stylo_feature_cols)),
                "kept_columns": int(len(kept_stylo_cols)),
                "dropped_columns": int(
                    len(stylo_feature_cols) - len(kept_stylo_cols)
                ),
                "mean_standardized_mean_gap": float(
                    stylometry_drift_summary["mean_standardized_mean_gap"]
                ),
                "max_standardized_mean_gap": float(
                    stylometry_drift_summary["max_standardized_mean_gap"]
                ),
            }
        write_json(unit_dir / "manifest.json", manifest)

        run_summary["units"].append(
            {
                "unit_id": unit.unit_id,
                "eval_role": unit.eval_role,
                "enabled_blocks": list(enabled_blocks),
                "combined_available": False,
                "path": relative_to_project(project_root, unit_dir),
                "train_rows": int(len(train_df)),
                f"{unit.eval_role}_rows": int(len(eval_df)),
                "word_dim": int(x_train_word.shape[1]),
                "char_dim": int(x_train_char.shape[1]),
                "stylo_dim": int(x_train_stylo_sparse.shape[1]),
                "feature_columns_path": relative_to_project(
                    project_root, unit_dir / "feature_columns.json"
                ),
            }
        )
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"unit={unit.unit_id}", refresh=False)

    run_summary["target_summary"] = _build_target_summary(loaded["row_targets"])
    if stylometry_drift_rows:
        drift_summary = (
            pd.DataFrame(stylometry_drift_rows)
            .sort_values("unit_id")
            .reset_index(drop=True)
        )
        drift_summary_path = materialized_root / "stylometry_drift_summary.csv"
        drift_summary.to_csv(drift_summary_path, index=False)
        run_summary["stylometry_drift_summary"] = {
            "path": relative_to_project(project_root, drift_summary_path),
            "unit_count": int(len(drift_summary)),
        }

    if progress is not None:
        progress.close()
    write_json(materialized_root / "manifest.json", run_summary)
    return run_summary
