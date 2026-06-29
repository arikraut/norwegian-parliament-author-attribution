from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from data_pipeline.utils import write_json as _write_json
from models.SVM.training.train_svm_attribution import (
    load_selected_direct_candidates,
    run_attribution_experiment,
    run_final_attribution_evaluation,
    run_final_attribution_evaluation_from_config,
)

def _write_upstream_manifests(
    project_root: Path,
    *,
    split_name: str,
    row_feature_name: str,
    materialization_name: str,
) -> None:
    split_dir = project_root / "data" / "splits" / split_name
    row_feature_dir = split_dir / "row_features" / row_feature_name
    row_feature_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        split_dir / "manifest.json",
        {
            "split_name": split_name,
            "config_path": f"data_pipeline/configs/splits/{split_name}.toml",
        },
    )
    _write_json(
        row_feature_dir / "manifest.json",
        {
            "split_name": split_name,
            "feature_set_name": row_feature_name,
            "feature_config_path": f"data_pipeline/configs/features/{row_feature_name}.toml",
        },
    )


def _write_unit(
    unit_dir: Path,
    unit_id: str,
    eval_role: str,
    x_train_char: sparse.csr_matrix,
    x_eval_char: sparse.csr_matrix,
    x_train_word: sparse.csr_matrix,
    x_eval_word: sparse.csr_matrix,
    x_train_stylo: sparse.csr_matrix | None,
    x_eval_stylo: sparse.csr_matrix | None,
    y_train: np.ndarray,
    y_eval: np.ndarray,
) -> None:
    (unit_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (unit_dir / "labels").mkdir(parents=True, exist_ok=True)
    (unit_dir / "row_order").mkdir(parents=True, exist_ok=True)

    sparse.save_npz(unit_dir / "matrices" / "X_train_char.npz", x_train_char)
    sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_char.npz", x_eval_char)
    sparse.save_npz(unit_dir / "matrices" / "X_train_word.npz", x_train_word)
    sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_word.npz", x_eval_word)
    if x_train_stylo is not None and x_eval_stylo is not None:
        sparse.save_npz(unit_dir / "matrices" / "X_train_stylo.npz", x_train_stylo)
        sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_stylo.npz", x_eval_stylo)

    np.save(unit_dir / "labels" / "y_train_author.npy", y_train)
    np.save(unit_dir / "labels" / f"y_{eval_role}_author.npy", y_eval)

    train_rows = pd.DataFrame(
        {
            "row_idx": np.arange(len(y_train)),
            "id_speech": np.arange(1000, 1000 + len(y_train)),
            "id_person": y_train,
            "fold_id": unit_id,
            "role": "train",
            "author": y_train,
        }
    )
    eval_rows = pd.DataFrame(
        {
            "row_idx": np.arange(len(y_eval)),
            "id_speech": np.arange(2000, 2000 + len(y_eval)),
            "id_person": y_eval,
            "fold_id": unit_id,
            "role": eval_role,
            "author": y_eval,
        }
    )
    train_rows.to_csv(unit_dir / "row_order" / "train_rows.csv", index=False)
    eval_rows.to_csv(unit_dir / "row_order" / f"{eval_role}_rows.csv", index=False)


def _write_project_root(project_root: Path) -> None:
    project_root.joinpath("pyproject.toml").write_text(
        "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
    )
    project_root.joinpath("data").mkdir()


def _selected_direct_candidate_payload() -> dict:
    return {
        "condition_id": "char",
        "condition_label": "char",
        "candidate_id": "char__C=0.1__class_weight=none",
        "feature_set": "char",
        "blocks": ["char"],
        "normalize_rows": True,
        "normalize_each_block": False,
        "block_weights": {"char": 1.0},
        "c_value": 0.1,
        "class_weight": "none",
        "selection_metric": "macro_f1",
        "dev_summary": {"eval_mean_macro_f1": 1.0},
    }


def _write_final_attribution_fixture(
    project_root: Path,
    *,
    config_selected_candidates_path: str = (
        "results/models/toy_split/toy_dev_selection/seed_42/selected_candidates.json"
    ),
) -> tuple[Path, Path]:
    _write_project_root(project_root)

    materialized_root = (
        project_root
        / "data"
        / "splits"
        / "toy_split"
        / "materialized_features"
        / "toy_final_materialization"
    )
    materialized_root.mkdir(parents=True, exist_ok=True)
    _write_upstream_manifests(
        project_root,
        split_name="toy_split",
        row_feature_name="toy_features",
        materialization_name="toy_final_materialization",
    )

    y_train = np.array(["A", "A", "B", "B"], dtype=object)
    y_test = np.array(["A", "B"], dtype=object)
    x_train_char = sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.0],
                [1.5, 0.0],
                [0.0, 2.0],
                [0.0, 1.5],
            ]
        )
    )
    x_test_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    x_train_word = sparse.csr_matrix(np.ones((4, 1)))
    x_test_word = sparse.csr_matrix(np.ones((2, 1)))

    _write_unit(
        materialized_root / "final_test_2013",
        unit_id="final_test_2013",
        eval_role="test",
        x_train_char=x_train_char,
        x_eval_char=x_test_char,
        x_train_word=x_train_word,
        x_eval_word=x_test_word,
        x_train_stylo=None,
        x_eval_stylo=None,
        y_train=y_train,
        y_eval=y_test,
    )

    _write_json(
        materialized_root / "manifest.json",
        {
            "split_name": "toy_split",
            "materialization_name": "toy_final_materialization",
            "row_feature_name": "toy_features",
            "config_path": "data_pipeline/configs/materializations/toy_final_materialization.toml",
            "units": [
                {"unit_id": "final_test_2013", "eval_role": "test"},
            ],
        },
    )

    selection_results_dir = (
        project_root / "results" / "models" / "toy_split" / "toy_dev_selection" / "seed_42"
    )
    selected_candidates_path = selection_results_dir / "selected_candidates.json"
    selected_payload = {
        "selection_scope": "condition",
        "selection_metric": "macro_f1",
        "split_name": "toy_split",
        "materialization_name": "toy_materialization",
        "target": "author",
        "selected_candidates": [_selected_direct_candidate_payload()],
    }
    _write_json(selected_candidates_path, selected_payload)
    _write_json(
        selection_results_dir / "manifest.json",
        {
            "experiment_name": "toy_dev_selection",
            "results_dir": "results/models/toy_split/toy_dev_selection/seed_42",
            "selected_candidates_path": (
                "results/models/toy_split/toy_dev_selection/seed_42/selected_candidates.json"
            ),
        },
    )

    config_path = project_root / "models" / "configs" / "attribution" / "toy_final_eval.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[experiment]",
                "name = 'toy_final_eval'",
                "seed = 42",
                "save_prediction_top_k = 2",
                "",
                "[data]",
                "splits_dir = 'data/splits'",
                "results_dir = 'results/models'",
                "artifacts_dir = 'models/artifacts/attribution'",
                "",
                "[source]",
                "split_name = 'toy_split'",
                "materialization_name = 'toy_final_materialization'",
                "target = 'author'",
                "units = 'all'",
                "",
                "[model]",
                "family = 'linear_svm'",
                "max_iter = 5000",
                "tol = 0.0001",
                "dual = 'auto'",
                "top_k = [2]",
                "",
                "[final_eval]",
                f"selected_candidates_path = '{config_selected_candidates_path}'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path, selected_candidates_path


class TrainAttributionTests(unittest.TestCase):
    def test_run_final_attribution_evaluation_prefers_override_selected_candidates_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path, selected_candidates_path = _write_final_attribution_fixture(
                project_root,
                config_selected_candidates_path=(
                    "results/models/toy_split/does_not_exist/seed_42/selected_candidates.json"
                ),
            )

            manifest = run_final_attribution_evaluation(
                config_path,
                selected_candidates_path_override=selected_candidates_path,
            )

            self.assertEqual(
                manifest["selection_source"]["selected_candidates_path"],
                "results/models/toy_split/toy_dev_selection/seed_42/selected_candidates.json",
            )

    def test_run_final_attribution_evaluation_from_config_uses_preloaded_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path, selected_candidates_path = _write_final_attribution_fixture(
                project_root,
                config_selected_candidates_path=(
                    "results/models/toy_split/does_not_exist/seed_42/selected_candidates.json"
                ),
            )
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            candidates, payload, source = load_selected_direct_candidates(
                project_root,
                config,
                selected_candidates_path_override=selected_candidates_path,
            )

            manifest = run_final_attribution_evaluation_from_config(
                config,
                project_root=project_root,
                config_path=config_path,
                preloaded_candidates=candidates,
                preloaded_selection_payload=payload,
                preloaded_selection_source=source,
            )

            self.assertEqual(
                manifest["selection_source"]["selected_candidates_path"],
                "results/models/toy_split/toy_dev_selection/seed_42/selected_candidates.json",
            )

    def test_run_final_attribution_evaluation_uses_selected_candidates_and_writes_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            config_path, _ = _write_final_attribution_fixture(project_root)

            manifest = run_final_attribution_evaluation(config_path)

            results_dir = project_root / manifest["results_dir"]
            artifacts_dir = project_root / manifest["artifacts_dir"]
            self.assertEqual(manifest["run_type"], "condition_final_evaluation")
            self.assertEqual(
                manifest["selection_source"]["selected_candidates_path"],
                "results/models/toy_split/toy_dev_selection/seed_42/selected_candidates.json",
            )
            self.assertEqual(
                manifest["selection_source"]["selection_results_dir"],
                "results/models/toy_split/toy_dev_selection/seed_42",
            )
            self.assertEqual(
                manifest["condition_results"][0]["candidate_id"],
                "char__C=0.1__class_weight=none",
            )
            self.assertEqual(manifest["provenance"]["split_name"], "toy_split")
            self.assertEqual(manifest["provenance"]["feature_set_name"], "toy_features")
            self.assertEqual(
                manifest["provenance"]["feature_config_path"],
                "data_pipeline/configs/features/toy_features.toml",
            )
            condition_dir = results_dir / "final_by_condition" / "char"
            self.assertTrue((condition_dir / "final_test_metrics.json").exists())
            self.assertTrue((condition_dir / "final_test_predictions.csv").exists())
            self.assertTrue((condition_dir / "resolved_candidate.json").exists())
            self.assertTrue((results_dir / "final_condition_summary.csv").exists())
            self.assertTrue((artifacts_dir / "final_by_condition" / "char" / "model.joblib").exists())

    def test_run_attribution_experiment_rejects_test_eval_units_during_candidate_search(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)
            _write_upstream_manifests(
                project_root,
                split_name="toy_split",
                row_feature_name="toy_features",
                materialization_name="toy_materialization",
            )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "units": [
                        {"unit_id": "final_test_2013", "eval_role": "test"},
                    ],
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [1.0]",
                        "class_weights = ['none']",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "eval_role='val'"):
                run_attribution_experiment(config_path)

    def test_run_attribution_experiment_reconstructs_all_block_from_enabled_matrices(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)
            _write_upstream_manifests(
                project_root,
                split_name="toy_split",
                row_feature_name="toy_features",
                materialization_name="toy_materialization",
            )

            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            y_eval = np.array(["A", "B"], dtype=object)
            x_train_char = sparse.csr_matrix(np.array([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]))
            x_eval_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
            x_train_word = sparse.csr_matrix(np.ones((4, 1)))
            x_eval_word = sparse.csr_matrix(np.ones((2, 1)))

            _write_unit(
                materialized_root / "fold_01",
                unit_id="fold_01",
                eval_role="val",
                x_train_char=x_train_char,
                x_eval_char=x_eval_char,
                x_train_word=x_train_word,
                x_eval_word=x_eval_word,
                x_train_stylo=None,
                x_eval_stylo=None,
                y_train=y_train,
                y_eval=y_eval,
            )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "row_feature_name": "toy_features",
                    "config_path": "data_pipeline/configs/materializations/toy_materialization.toml",
                    "units": [
                        {
                            "unit_id": "fold_01",
                            "eval_role": "val",
                            "enabled_blocks": ["char", "word"],
                            "combined_available": False,
                        },
                    ],
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [1.0]",
                        "class_weights = ['none']",
                        "",
                        "[[conditions]]",
                        "id = 'all_blocks'",
                        "label = 'all_blocks'",
                        "feature_set = 'all_blocks'",
                        "blocks = ['all']",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_attribution_experiment(config_path)
            selected = json.loads(
                (project_root / manifest["selected_candidates_path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selected["selected_candidates"][0]["feature_set"], "all_blocks")
            self.assertEqual(selected["selected_candidates"][0]["blocks"], ["all"])

    def test_run_attribution_experiment_selects_one_candidate_per_condition_and_writes_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)
            _write_upstream_manifests(
                project_root,
                split_name="toy_split",
                row_feature_name="toy_features",
                materialization_name="toy_materialization",
            )

            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            y_eval = np.array(["A", "B"], dtype=object)

            x_train_char = sparse.csr_matrix(
                np.array(
                    [
                        [2.0, 0.0],
                        [1.5, 0.0],
                        [0.0, 2.0],
                        [0.0, 1.5],
                    ]
                )
            )
            x_eval_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

            # Constant word features make the "word" candidate weak.
            x_train_word = sparse.csr_matrix(np.ones((4, 1)))
            x_eval_word = sparse.csr_matrix(np.ones((2, 1)))

            for unit_id in ["fold_01", "fold_02"]:
                _write_unit(
                    materialized_root / unit_id,
                    unit_id=unit_id,
                    eval_role="val",
                    x_train_char=x_train_char,
                    x_eval_char=x_eval_char,
                    x_train_word=x_train_word,
                    x_eval_word=x_eval_word,
                    x_train_stylo=None,
                    x_eval_stylo=None,
                    y_train=y_train,
                    y_eval=y_eval,
                )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "row_feature_name": "toy_features",
                    "config_path": "data_pipeline/configs/materializations/toy_materialization.toml",
                    "units": [
                        {"unit_id": "fold_01", "eval_role": "val"},
                        {"unit_id": "fold_02", "eval_role": "val"},
                    ],
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "seed = 42",
                        "selection_metric = 'macro_f1'",
                        "save_prediction_top_k = 2",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [0.1, 1.0]",
                        "class_weights = ['none']",
                        "max_iter = 5000",
                        "top_k = [2]",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                        "normalize_rows = true",
                        "",
                        "[[conditions]]",
                        "id = 'word'",
                        "label = 'word'",
                        "feature_set = 'word'",
                        "blocks = ['word']",
                        "normalize_rows = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_attribution_experiment(config_path)

            self.assertEqual(manifest["run_type"], "dev_condition_selection")
            self.assertEqual(manifest["condition_count"], 2)
            self.assertEqual(manifest["provenance"]["split_config_path"], "data_pipeline/configs/splits/toy_split.toml")
            self.assertEqual(manifest["provenance"]["feature_set_name"], "toy_features")
            self.assertEqual(
                manifest["provenance"]["materialization_config_path"],
                "data_pipeline/configs/materializations/toy_materialization.toml",
            )

            results_dir = project_root / manifest["results_dir"]
            self.assertTrue((results_dir / "candidate_summary.csv").exists())
            self.assertTrue((results_dir / "condition_summary.csv").exists())
            self.assertTrue((results_dir / "selected_candidates.json").exists())
            self.assertTrue((results_dir / "fold_metrics.csv").exists())
            self.assertFalse((results_dir / "predictions").exists())

            summary = pd.read_csv(results_dir / "candidate_summary.csv")
            self.assertEqual(summary.iloc[0]["feature_set"], "char")
            condition_summary = pd.read_csv(results_dir / "condition_summary.csv")
            self.assertEqual(set(condition_summary["condition_id"]), {"char", "word"})
            selected = json.loads((results_dir / "selected_candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selected["selected_candidates"]), 2)

    def test_run_attribution_experiment_supports_weighted_f1_selection_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)
            _write_upstream_manifests(
                project_root,
                split_name="toy_split",
                row_feature_name="toy_features",
                materialization_name="toy_materialization",
            )

            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            y_eval = np.array(["A", "B"], dtype=object)

            x_train_char = sparse.csr_matrix(
                np.array(
                    [
                        [2.0, 0.0],
                        [1.5, 0.0],
                        [0.0, 2.0],
                        [0.0, 1.5],
                    ]
                )
            )
            x_eval_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

            x_train_word = sparse.csr_matrix(np.ones((4, 1)))
            x_eval_word = sparse.csr_matrix(np.ones((2, 1)))

            for unit_id in ["fold_01", "fold_02"]:
                _write_unit(
                    materialized_root / unit_id,
                    unit_id=unit_id,
                    eval_role="val",
                    x_train_char=x_train_char,
                    x_eval_char=x_eval_char,
                    x_train_word=x_train_word,
                    x_eval_word=x_eval_word,
                    x_train_stylo=None,
                    x_eval_stylo=None,
                    y_train=y_train,
                    y_eval=y_eval,
                )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "row_feature_name": "toy_features",
                    "config_path": "data_pipeline/configs/materializations/toy_materialization.toml",
                    "units": [
                        {"unit_id": "fold_01", "eval_role": "val"},
                        {"unit_id": "fold_02", "eval_role": "val"},
                    ],
                },
            )

            config_path = project_root / "models" / "configs" / "attribution" / "toy_weighted_f1.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_weighted_f1'",
                        "seed = 42",
                        "selection_metric = 'weighted_f1'",
                        "save_prediction_top_k = 2",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [0.1, 1.0]",
                        "class_weights = ['none']",
                        "max_iter = 5000",
                        "top_k = [2]",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                        "normalize_rows = true",
                        "",
                        "[[conditions]]",
                        "id = 'word'",
                        "label = 'word'",
                        "feature_set = 'word'",
                        "blocks = ['word']",
                        "normalize_rows = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_attribution_experiment(config_path)
            selected = json.loads(
                (project_root / manifest["selected_candidates_path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selected["selection_metric"], "weighted_f1")

            results_dir = project_root / manifest["results_dir"]
            summary = pd.read_csv(results_dir / "candidate_summary.csv")
            self.assertIn("eval_mean_weighted_f1", summary.columns)

    def test_run_attribution_experiment_only_attaches_convergence_warnings_to_train_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)

            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            y_eval = np.array(["A", "B"], dtype=object)

            x_train_char = sparse.csr_matrix(
                np.array(
                    [
                        [2.0, 0.0],
                        [1.5, 0.0],
                        [0.0, 2.0],
                        [0.0, 1.5],
                    ]
                )
            )
            x_eval_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
            x_train_word = sparse.csr_matrix(np.ones((4, 1)))
            x_eval_word = sparse.csr_matrix(np.ones((2, 1)))

            _write_unit(
                materialized_root / "fold_01",
                unit_id="fold_01",
                eval_role="val",
                x_train_char=x_train_char,
                x_eval_char=x_eval_char,
                x_train_word=x_train_word,
                x_eval_word=x_eval_word,
                x_train_stylo=None,
                x_eval_stylo=None,
                y_train=y_train,
                y_eval=y_eval,
            )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "units": [
                        {"unit_id": "fold_01", "eval_role": "val"},
                    ],
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "seed = 42",
                        "selection_metric = 'macro_f1'",
                        "save_prediction_top_k = 2",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [0.1]",
                        "class_weights = ['none']",
                        "max_iter = 1",
                        "top_k = [2]",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                        "normalize_rows = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_attribution_experiment(config_path)
            metrics_df = pd.read_csv(project_root / manifest["results_dir"] / "fold_metrics.csv")

            train_rows = metrics_df[metrics_df["split"] == "train"].reset_index(drop=True)
            eval_rows = metrics_df[metrics_df["split"] == "val"].reset_index(drop=True)
            self.assertGreater(int(train_rows.loc[0, "convergence_warning_count"]), 0)
            self.assertEqual(int(eval_rows.loc[0, "convergence_warning_count"]), 0)

    def test_run_attribution_experiment_rejects_duplicate_condition_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [1.0]",
                        "class_weights = ['none']",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['word']",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate \\[\\[conditions\\]\\]\\.id"):
                run_attribution_experiment(config_path)

    def test_run_attribution_experiment_rejects_unsupported_model_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'rbf_svm'",
                        "C_values = [1.0]",
                        "class_weights = ['none']",
                        "",
                        "[[conditions]]",
                        "id = 'char'",
                        "label = 'char'",
                        "feature_set = 'char'",
                        "blocks = ['char']",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported model\\.family"):
                run_attribution_experiment(config_path)

    def test_run_attribution_experiment_records_normalize_each_block_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            materialized_root.mkdir(parents=True, exist_ok=True)

            y_train = np.array(["A", "A", "B", "B"], dtype=object)
            y_eval = np.array(["A", "B"], dtype=object)
            x_train_char = sparse.csr_matrix(np.array([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]))
            x_eval_char = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
            x_train_word = sparse.csr_matrix(np.ones((4, 1)))
            x_eval_word = sparse.csr_matrix(np.ones((2, 1)))
            x_train_stylo = sparse.csr_matrix(np.array([[0.2], [0.1], [1.8], [1.9]]))
            x_eval_stylo = sparse.csr_matrix(np.array([[0.1], [1.9]]))

            _write_unit(
                materialized_root / "fold_01",
                unit_id="fold_01",
                eval_role="val",
                x_train_char=x_train_char,
                x_eval_char=x_eval_char,
                x_train_word=x_train_word,
                x_eval_word=x_eval_word,
                x_train_stylo=x_train_stylo,
                x_eval_stylo=x_eval_stylo,
                y_train=y_train,
                y_eval=y_eval,
            )
            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "materialization_name": "toy_materialization",
                    "units": [{"unit_id": "fold_01", "eval_role": "val"}],
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "seed = 42",
                        "selection_metric = 'macro_f1'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [0.1]",
                        "class_weights = ['none']",
                        "max_iter = 5000",
                        "",
                        "[[conditions]]",
                        "id = 'char_word_stylo_sepnorm'",
                        "label = 'char_word_stylo_sepnorm'",
                        "feature_set = 'char_word_stylo_sepnorm'",
                        "blocks = ['char', 'word', 'stylo']",
                        "normalize_rows = true",
                        "normalize_each_block = true",
                        "",
                        "[[conditions]]",
                        "id = 'char_word'",
                        "label = 'char_word'",
                        "feature_set = 'char_word'",
                        "blocks = ['char', 'word']",
                        "normalize_rows = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_attribution_experiment(config_path)

            results_dir = project_root / manifest["results_dir"]
            selected = json.loads((results_dir / "selected_candidates.json").read_text(encoding="utf-8"))
            summary_df = pd.read_csv(results_dir / "candidate_summary.csv")

            self.assertIn("normalize_each_block", summary_df.columns)
            sepnorm_rows = summary_df[summary_df["feature_set"] == "char_word_stylo_sepnorm"]
            self.assertEqual(len(sepnorm_rows), 1)
            self.assertTrue(bool(sepnorm_rows.iloc[0]["normalize_each_block"]))
            selected_by_condition = {
                item["condition_id"]: item for item in selected["selected_candidates"]
            }
            self.assertIn("normalize_each_block", selected_by_condition["char_word_stylo_sepnorm"])
            self.assertTrue(
                bool(selected_by_condition["char_word_stylo_sepnorm"]["normalize_each_block"])
            )

    def test_run_attribution_experiment_rejects_normalize_each_block_for_all_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_linear_svm.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_linear_svm'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        "split_name = 'toy_split'",
                        "materialization_name = 'toy_materialization'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'linear_svm'",
                        "C_values = [1.0]",
                        "class_weights = ['none']",
                        "",
                        "[[conditions]]",
                        "id = 'all_block'",
                        "label = 'all_block'",
                        "feature_set = 'all_block'",
                        "blocks = ['all']",
                        "normalize_each_block = true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "normalize_each_block=true"):
                run_attribution_experiment(config_path)


if __name__ == "__main__":
    unittest.main()
