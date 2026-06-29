from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from models.SVM.importance.feature_importance_stacked import run_stacked_importance_analysis
from models.SVM.importance.feature_importance_svm import (
    load_feature_names_for_blocks,
    run_importance_analysis,
)
from run_feature_importance import run_feature_importance


def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON artifact for a test project."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_project_root(project_root: Path) -> None:
    """Create the minimal files required by find_project_root."""
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='toy'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    (project_root / "data").mkdir()


class FeatureImportanceTests(unittest.TestCase):
    """Regression tests for final-manifest feature-importance contracts."""

    def test_direct_importance_loads_core_and_profiling_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            unit_dir = materialized_root / "final_test"
            _write_json(
                unit_dir / "feature_columns.json",
                {"char": ["aa", "bb"]},
            )
            _write_json(
                materialized_root / "profiling_feature_columns.json",
                {"columns": ["female_false", "female_true"], "targets": ["female"]},
            )

            artifacts_dir = (
                project_root
                / "models"
                / "artifacts"
                / "attribution"
                / "toy_split"
                / "toy_final"
                / "seed_42"
                / "final_by_condition"
                / "char_profiled"
            )
            artifacts_dir.mkdir(parents=True)
            x_train = np.array(
                [
                    [2.0, 0.0, 0.9, 0.1],
                    [1.8, 0.0, 0.8, 0.2],
                    [0.0, 2.0, 0.2, 0.8],
                    [0.0, 1.8, 0.1, 0.9],
                ]
            )
            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            model = LinearSVC(dual="auto", max_iter=5000).fit(x_train, y_train)
            joblib.dump(model, artifacts_dir / "model.joblib")

            results_dir = (
                project_root / "results" / "models" / "toy_split" / "toy_final" / "seed_42"
            )
            condition_dir = results_dir / "final_by_condition" / "char_profiled"
            _write_json(
                condition_dir / "resolved_candidate.json",
                {
                    "condition_id": "char_profiled",
                    "condition_label": "char_profiled",
                    "candidate_id": "char_profiled__C=1__class_weight=balanced",
                    "blocks": ["char", "profiling_female"],
                },
            )
            manifest_path = results_dir / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "run_type": "condition_final_evaluation",
                    "results_dir": "results/models/toy_split/toy_final/seed_42",
                    "materialized_root": "data/splits/toy_split/materialized_features/toy_final_materialization",
                    "condition_results": [
                        {
                            "condition_id": "char_profiled",
                            "condition_label": "char_profiled",
                            "candidate_id": "char_profiled__C=1__class_weight=balanced",
                            "unit_id": "final_test",
                            "eval_role": "test",
                            "resolved_candidate_path": (
                                "results/models/toy_split/toy_final/seed_42/"
                                "final_by_condition/char_profiled/resolved_candidate.json"
                            ),
                            "model_path": (
                                "models/artifacts/attribution/toy_split/toy_final/seed_42/"
                                "final_by_condition/char_profiled/model.joblib"
                            ),
                        },
                    ],
                },
            )

            summary = run_importance_analysis(manifest_path, top_n=1)

            self.assertEqual(summary["condition_count"], 1)
            self.assertEqual(summary["conditions"][0]["n_features"], 4)
            output_dir = condition_dir / "feature_importance"
            global_df = pd.read_csv(output_dir / "global_importance.csv")
            block_df = pd.read_csv(output_dir / "block_importance.csv")
            self.assertEqual(set(global_df["block"]), {"char", "profiling_female"})
            self.assertEqual(set(block_df["block"]), {"char", "profiling_female"})
            self.assertTrue((output_dir / "block_importance.png").exists())

    def test_direct_importance_loads_hard_profiling_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            _write_json(
                materialized_root / "final_test" / "feature_columns.json",
                {"char": ["aa"]},
            )
            _write_json(
                materialized_root / "profiling_hard_feature_columns.json",
                {"columns": ["female_false", "female_true"], "targets": ["female"]},
            )

            feature_names = load_feature_names_for_blocks(
                materialized_root,
                "final_test",
                ["char", "profiling_hard_female"],
            )

            self.assertEqual(
                [feature.name for feature in feature_names],
                [
                    "char:aa",
                    "profiling_hard_female:female_false",
                    "profiling_hard_female:female_true",
                ],
            )

    def test_feature_names_accept_historical_spacy_core_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            unit_dir = materialized_root / "final_test"
            _write_json(
                unit_dir / "feature_columns.json",
                {
                    "char": ["aa"],
                    "word": ["word_a"],
                    "spacy": ["sent_len", "dep_dist"],
                },
            )
            _write_json(
                unit_dir / "manifest.json",
                {"enabled_blocks": ["char", "word", "spacy"]},
            )

            spacy_names = load_feature_names_for_blocks(
                materialized_root,
                "final_test",
                ["spacy"],
            )
            all_names = load_feature_names_for_blocks(
                materialized_root,
                "final_test",
                ["all"],
            )

            self.assertEqual(
                [feature.name for feature in spacy_names],
                ["stylo:sent_len", "stylo:dep_dist"],
            )
            self.assertEqual({feature.block for feature in spacy_names}, {"stylo"})
            self.assertIn("stylo:sent_len", [feature.name for feature in all_names])

    def test_feature_names_accept_historical_left_senter_right_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            _write_json(materialized_root / "final_test" / "feature_columns.json", {})
            _write_json(
                materialized_root / "oracle_feature_columns.json",
                {
                    "columns": [
                        "left_senter_right_left",
                        "left_senter_right_senter",
                    ],
                    "targets": ["left_senter_right"],
                },
            )

            feature_names = load_feature_names_for_blocks(
                materialized_root,
                "final_test",
                ["profiling_oracle_left_center_right"],
            )

            self.assertEqual(
                [feature.name for feature in feature_names],
                [
                    "profiling_oracle_left_center_right:left_center_right_left",
                    "profiling_oracle_left_center_right:left_center_right_center",
                ],
            )

    def test_stacked_importance_reconstructs_top_and_base_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            _write_json(
                materialized_root / "final_test" / "feature_columns.json",
                {"char": ["aa", "bb"]},
            )

            model_dir = (
                project_root
                / "models"
                / "artifacts"
                / "attribution"
                / "toy_split"
                / "toy_stacked_final"
                / "seed_42"
                / "final_by_condition"
                / "char_only"
                / "final_model"
            )
            model_dir.mkdir(parents=True)
            y_train = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)
            x_family = np.array(
                [
                    [2.0, 0.0],
                    [1.8, 0.0],
                    [1.6, 0.1],
                    [1.4, 0.2],
                    [0.0, 2.0],
                    [0.0, 1.8],
                    [0.1, 1.6],
                    [0.2, 1.4],
                ]
            )
            family_model = CalibratedClassifierCV(
                estimator=LinearSVC(dual="auto", max_iter=5000),
                method="sigmoid",
                cv=2,
            ).fit(x_family, y_train)
            top_model = LogisticRegression(max_iter=500).fit(
                np.array(
                    [
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.2, 0.8],
                        [0.1, 0.9],
                    ]
                ),
                np.array(["A", "A", "B", "B"], dtype=object),
            )
            joblib.dump(top_model, model_dir / "top_model.joblib")
            joblib.dump(family_model, model_dir / "family_char.joblib")

            results_dir = (
                project_root
                / "results"
                / "models"
                / "toy_split"
                / "toy_stacked_final"
                / "seed_42"
            )
            condition_dir = results_dir / "final_by_condition" / "char_only"
            _write_json(
                condition_dir / "resolved_candidate.json",
                {
                    "condition_id": "char_only",
                    "condition_label": "char_only",
                    "candidate_id": "char_only__baseC=1__topC=1__class_weight=balanced",
                    "families": [{"name": "char", "blocks": ["char"]}],
                    "profiling_blocks": [],
                },
            )
            manifest_path = results_dir / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "run_type": "stacked_condition_final_evaluation",
                    "results_dir": "results/models/toy_split/toy_stacked_final/seed_42",
                    "materialized_root": "data/splits/toy_split/materialized_features/toy_final_materialization",
                    "condition_results": [
                        {
                            "condition_id": "char_only",
                            "condition_label": "char_only",
                            "candidate_id": "char_only__baseC=1__topC=1__class_weight=balanced",
                            "unit_id": "final_test",
                            "eval_role": "test",
                            "resolved_candidate_path": (
                                "results/models/toy_split/toy_stacked_final/seed_42/"
                                "final_by_condition/char_only/resolved_candidate.json"
                            ),
                            "model_dir": (
                                "models/artifacts/attribution/toy_split/toy_stacked_final/seed_42/"
                                "final_by_condition/char_only/final_model"
                            ),
                        },
                    ],
                },
            )

            summary = run_stacked_importance_analysis(manifest_path)

            self.assertEqual(summary["condition_count"], 1)
            self.assertEqual(summary["conditions"][0]["families"], ["char"])
            top_df = pd.read_csv(
                condition_dir / "feature_importance" / "top_model_family_importance.csv"
            )
            base_df = pd.read_csv(
                condition_dir / "feature_importance" / "base_family_global_importance.csv"
            )
            self.assertEqual(top_df.loc[0, "family"], "char")
            self.assertEqual(set(base_df["family"]), {"char"})

    def test_stacked_importance_accepts_historical_spacy_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_final_materialization"
            )
            _write_json(
                materialized_root / "final_test" / "feature_columns.json",
                {"spacy": ["sent_len", "dep_dist"]},
            )

            model_dir = (
                project_root
                / "models"
                / "artifacts"
                / "attribution"
                / "toy_split"
                / "toy_stacked_final"
                / "seed_42"
                / "final_by_condition"
                / "char_word_spacy"
                / "final_model"
            )
            model_dir.mkdir(parents=True)
            y_train = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)
            x_family = np.array(
                [
                    [2.0, 0.0],
                    [1.8, 0.0],
                    [1.6, 0.1],
                    [1.4, 0.2],
                    [0.0, 2.0],
                    [0.0, 1.8],
                    [0.1, 1.6],
                    [0.2, 1.4],
                ]
            )
            family_model = CalibratedClassifierCV(
                estimator=LinearSVC(dual="auto", max_iter=5000),
                method="sigmoid",
                cv=2,
            ).fit(x_family, y_train)
            top_model = LogisticRegression(max_iter=500).fit(
                np.array(
                    [
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.2, 0.8],
                        [0.1, 0.9],
                    ]
                ),
                np.array(["A", "A", "B", "B"], dtype=object),
            )
            joblib.dump(top_model, model_dir / "top_model.joblib")
            joblib.dump(family_model, model_dir / "family_spacy.joblib")

            results_dir = (
                project_root
                / "results"
                / "models"
                / "toy_split"
                / "toy_stacked_final"
                / "seed_42"
            )
            condition_dir = results_dir / "final_by_condition" / "char_word_spacy"
            _write_json(
                condition_dir / "resolved_candidate.json",
                {
                    "condition_id": "char_word_spacy",
                    "condition_label": "char_word_spacy",
                    "candidate_id": (
                        "char_word_spacy__baseC=1__topC=1__class_weight=balanced"
                    ),
                    "families": [{"name": "spacy", "blocks": ["spacy"]}],
                    "profiling_blocks": [],
                },
            )
            manifest_path = results_dir / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "run_type": "stacked_condition_final_evaluation",
                    "results_dir": "results/models/toy_split/toy_stacked_final/seed_42",
                    "materialized_root": (
                        "data/splits/toy_split/materialized_features/"
                        "toy_final_materialization"
                    ),
                    "condition_results": [
                        {
                            "condition_id": "char_word_spacy",
                            "condition_label": "char_word_spacy",
                            "candidate_id": (
                                "char_word_spacy__baseC=1__topC=1__"
                                "class_weight=balanced"
                            ),
                            "unit_id": "final_test",
                            "eval_role": "test",
                            "resolved_candidate_path": (
                                "results/models/toy_split/toy_stacked_final/seed_42/"
                                "final_by_condition/char_word_spacy/"
                                "resolved_candidate.json"
                            ),
                            "model_dir": (
                                "models/artifacts/attribution/toy_split/"
                                "toy_stacked_final/seed_42/final_by_condition/"
                                "char_word_spacy/final_model"
                            ),
                        },
                    ],
                },
            )

            summary = run_stacked_importance_analysis(manifest_path)

            self.assertEqual(summary["conditions"][0]["families"], ["stylo"])
            top_df = pd.read_csv(
                condition_dir / "feature_importance" / "top_model_family_importance.csv"
            )
            base_df = pd.read_csv(
                condition_dir / "feature_importance" / "base_family_global_importance.csv"
            )
            self.assertEqual(set(top_df["family"]), {"stylo"})
            self.assertEqual(set(base_df["family"]), {"stylo"})
            self.assertEqual(set(base_df["block"]), {"stylo"})

    def test_runner_rejects_dev_search_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project_root(project_root)
            manifest_path = project_root / "results" / "models" / "toy" / "manifest.json"
            _write_json(manifest_path, {"run_type": "dev_condition_selection"})

            with self.assertRaisesRegex(ValueError, "expected a final model manifest"):
                run_feature_importance(manifest_path)


if __name__ == "__main__":
    unittest.main()
