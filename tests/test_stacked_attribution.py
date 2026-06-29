from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from models.SVM.training import train_stacked_attribution as stacked_module
from models.SVM.training.train_stacked_attribution import (
    _stacked_candidate_grid,
    run_final_stacked_evaluation,
    run_stacked_experiment,
    stacked_search_profiling_blocks,
)


def _base_stacked_config() -> dict:
    return {
        "source": {"target": "author"},
        "model": {
            "family": "stacked",
            "base_c_values": [1.0],
            "class_weights": ["balanced"],
            "top_c_values": [1.0],
        },
        "families": [
            {"name": "char", "blocks": ["char"]},
            {"name": "word", "blocks": ["word"]},
            {"name": "stylo", "blocks": ["stylo"]},
        ],
        "conditions": [
            {
                "id": "char_word_stylo",
                "label": "char_word_stylo",
                "families": ["char", "word", "stylo"],
                "profiling_blocks": [],
            },
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stacked_selected_candidate_payload(profiling_blocks: list[str] | None = None) -> dict:
    profiling_blocks = profiling_blocks or []
    return {
        "condition_id": "char_word",
        "condition_label": "char_word",
        "candidate_id": "char_word__baseC=1__topC=1__class_weight=balanced",
        "family_set": "char_word",
        "families": [
            {"name": "char", "blocks": ["char"]},
            {"name": "word", "blocks": ["word"]},
        ],
        "base_c": 1.0,
        "class_weight": "balanced",
        "top_c": 1.0,
        "profiling_blocks": profiling_blocks,
        "selection_metric": "macro_f1",
        "dev_summary": {
            "eval_mean_macro_f1": 0.75,
            "eval_std_macro_f1": 0.01,
            "eval_mean_accuracy": 0.8,
        },
    }


def _write_minimal_final_stacked_config(
    config_path: Path,
    *,
    split_name: str,
    materialization_name: str,
    selected_candidates_path: Path,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[experiment]",
                "name = 'toy_final_stacked'",
                "seed = 42",
                "save_prediction_top_k = 5",
                "",
                "[data]",
                "splits_dir = 'data/splits'",
                "results_dir = 'results/models'",
                "artifacts_dir = 'models/artifacts/attribution'",
                "",
                "[source]",
                f"split_name = '{split_name}'",
                f"materialization_name = '{materialization_name}'",
                "target = 'author'",
                "units = 'all'",
                "",
                "[model]",
                "family = 'stacked'",
                "inner_cv = 2",
                "max_iter = 5000",
                "tol = 0.0001",
                "dual = 'auto'",
                "top_max_iter = 500",
                "top_k = [3, 5]",
                "",
                "[final_eval]",
                f"selected_candidates_path = '{selected_candidates_path}'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stacked_upstream_manifests(
    project_root: Path,
    *,
    split_name: str,
    row_feature_name: str,
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


def _write_stacked_unit(
    unit_dir: Path,
    *,
    unit_id: str,
    eval_role: str,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    x_train: sparse.csr_matrix,
    x_eval: sparse.csr_matrix,
) -> None:
    (unit_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (unit_dir / "labels").mkdir(parents=True, exist_ok=True)
    (unit_dir / "row_order").mkdir(parents=True, exist_ok=True)

    sparse.save_npz(unit_dir / "matrices" / "X_train_char.npz", x_train)
    sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_char.npz", x_eval)
    sparse.save_npz(unit_dir / "matrices" / "X_train_word.npz", x_train)
    sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_word.npz", x_eval)
    sparse.save_npz(unit_dir / "matrices" / "X_train_stylo.npz", x_train)
    sparse.save_npz(unit_dir / "matrices" / f"X_{eval_role}_stylo.npz", x_eval)
    np.save(unit_dir / "labels" / "y_train_author.npy", y_train)
    np.save(unit_dir / "labels" / f"y_{eval_role}_author.npy", y_eval)

    pd.DataFrame(
        {
            "row_idx": np.arange(len(y_train)),
            "id_speech": np.arange(1000, 1000 + len(y_train)),
            "id_person": y_train,
            "fold_id": unit_id,
            "role": "train",
        }
    ).to_csv(unit_dir / "row_order" / "train_rows.csv", index=False)
    pd.DataFrame(
        {
            "row_idx": np.arange(len(y_eval)),
            "id_speech": np.arange(2000, 2000 + len(y_eval)),
            "id_person": y_eval,
            "fold_id": unit_id,
            "role": eval_role,
        }
    ).to_csv(unit_dir / "row_order" / f"{eval_role}_rows.csv", index=False)


class StackedAttributionConfigTests(unittest.TestCase):
    def test_oof_family_train_meta_preserves_row_order_and_aligns_missing_classes(self) -> None:
        y_train = np.array(["A", "B", "C"] * 3, dtype=object)
        x_train = sparse.csr_matrix(np.arange(len(y_train), dtype=float).reshape(-1, 1))
        global_classes = np.array(["A", "B", "C"], dtype=object)

        class FakeCalibratedModel:
            classes_ = np.array(["A", "B"], dtype=object)

            def predict_proba(self, x_eval: sparse.csr_matrix) -> np.ndarray:
                row_ids = x_eval.toarray().ravel()
                return np.column_stack([row_ids + 10.0, row_ids + 20.0])

        with mock.patch.object(
            stacked_module,
            "_fit_calibrated_base",
            return_value=FakeCalibratedModel(),
        ):
            train_meta = stacked_module._build_oof_family_train_meta(
                x_train,
                y_train,
                global_classes,
                base_c=1.0,
                class_weight="balanced",
                model_cfg={"inner_cv": 3},
                seed=42,
            )

        row_ids = np.arange(len(y_train), dtype=float)
        np.testing.assert_allclose(train_meta[:, 0], row_ids + 10.0)
        np.testing.assert_allclose(train_meta[:, 1], row_ids + 20.0)
        np.testing.assert_allclose(train_meta[:, 2], np.zeros(len(y_train)))

    def test_assemble_stacked_meta_features_uses_resolved_family_order(self) -> None:
        char_output = stacked_module.ReusableFamilyOutput(
            train_meta=np.array([[1.0, 2.0], [3.0, 4.0]]),
            val_meta=np.array([[5.0, 6.0]]),
        )
        word_output = stacked_module.ReusableFamilyOutput(
            train_meta=np.array([[10.0], [11.0]]),
            val_meta=np.array([[12.0]]),
        )
        reusable_outputs = {
            stacked_module.ReusableFamilyKey("char", 1.0, "balanced"): char_output,
            stacked_module.ReusableFamilyKey("word", 1.0, "balanced"): word_output,
        }
        candidate = _stacked_candidate_grid(_base_stacked_config())[0]
        resolved_outputs = tuple(
            reusable_outputs[
                stacked_module.ReusableFamilyKey(
                    family.name,
                    candidate.base_c,
                    candidate.class_weight,
                )
            ]
            for family in candidate.families[:2]
        )
        profiling_train = np.array([[100.0], [200.0]])
        profiling_val = np.array([[300.0]])

        dev_train, dev_val = stacked_module._assemble_stacked_meta_features(
            family_outputs=resolved_outputs,
            profiling_train=profiling_train,
            profiling_eval=profiling_val,
        )
        final_train, final_val = stacked_module._assemble_stacked_meta_features(
            family_outputs=(char_output, word_output),
            profiling_train=profiling_train,
            profiling_eval=profiling_val,
        )

        expected_train = np.array(
            [
                [1.0, 2.0, 10.0, 100.0],
                [3.0, 4.0, 11.0, 200.0],
            ]
        )
        expected_val = np.array([[5.0, 6.0, 12.0, 300.0]])
        np.testing.assert_allclose(dev_train, expected_train)
        np.testing.assert_allclose(dev_val, expected_val)
        np.testing.assert_allclose(final_train, expected_train)
        np.testing.assert_allclose(final_val, expected_val)

    def test_profiling_blocks_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\[conditions\]\]\.profiling_blocks must be a list"):
            stacked_search_profiling_blocks(
                {"conditions": [{"id": "bad", "profiling_blocks": "profiling_party"}]}
            )

        with self.assertRaisesRegex(ValueError, r"entries must be non-empty strings"):
            stacked_search_profiling_blocks(
                {"conditions": [{"id": "bad", "profiling_blocks": ["profiling_party", ""]}]}
            )

    def test_stacked_candidate_grid_expands_tuned_family_and_c_values(self) -> None:
        config = _base_stacked_config()
        config["model"] = {
            "family": "stacked",
            "base_c_values": [1.0, 10.0],
            "class_weights": ["balanced"],
            "top_c_values": [0.1, 1.0, 10.0],
        }
        config["conditions"] = [
            {
                "id": "char_word__profiled",
                "label": "char_word__profiled",
                "families": ["char", "word"],
                "profiling_blocks": ["profiling_female", "profiling_age_bin"],
            },
            {
                "id": "char_word_stylo__profiled",
                "label": "char_word_stylo__profiled",
                "families": ["char", "word", "stylo"],
                "profiling_blocks": ["profiling_female", "profiling_age_bin"],
            },
        ]

        candidates = _stacked_candidate_grid(config)

        self.assertEqual(len(candidates), 12)
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        self.assertIn(
            "char_word__profiled__baseC=10__topC=1__class_weight=balanced",
            candidate_ids,
        )
        self.assertIn(
            "char_word_stylo__profiled__baseC=1__topC=0.1__class_weight=balanced",
            candidate_ids,
        )

    def test_hard_profiling_condition_ids_distinguish_stacked_candidates(self) -> None:
        config = _base_stacked_config()
        config["conditions"] = [
            {
                "id": "char_word__profiling_party",
                "label": "char_word__profiling_party",
                "families": ["char", "word"],
                "profiling_blocks": ["profiling_party"],
            },
            {
                "id": "char_word__hard_profiling_party",
                "label": "char_word__hard_profiling_party",
                "families": ["char", "word"],
                "profiling_blocks": ["profiling_hard_party"],
            },
        ]

        candidates = _stacked_candidate_grid(config)
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        condition_labels = {candidate.condition_label for candidate in candidates}

        self.assertEqual(len(candidates), 2)
        self.assertIn(
            "char_word__profiling_party__baseC=1__topC=1__class_weight=balanced",
            candidate_ids,
        )
        self.assertIn(
            "char_word__hard_profiling_party__baseC=1__topC=1__class_weight=balanced",
            candidate_ids,
        )
        self.assertIn("char_word__hard_profiling_party", condition_labels)

    def test_stacked_candidate_grid_rejects_unknown_condition_family(self) -> None:
        config = _base_stacked_config()
        config["conditions"] = [
            {
                "id": "bad",
                "label": "bad",
                "families": ["char", "missing"],
                "profiling_blocks": [],
            },
        ]

        with self.assertRaisesRegex(ValueError, "references unknown families"):
            _stacked_candidate_grid(config)

    def test_run_stacked_experiment_writes_dev_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n",
                encoding="utf-8",
            )
            (project_root / "data").mkdir()

            split_name = "toy_split"
            materialization_name = "toy_dev_materialization"
            row_feature_name = "toy_features"
            _write_stacked_upstream_manifests(
                project_root,
                split_name=split_name,
                row_feature_name=row_feature_name,
            )

            labels = np.array([f"A{i}" for i in range(4)], dtype=object)
            y_train = np.repeat(labels, 6)
            y_val = labels.copy()
            eye = np.eye(len(labels))
            x_train = sparse.csr_matrix(
                np.vstack([eye[i] for i in range(len(labels)) for _ in range(6)])
            )
            x_val = sparse.csr_matrix(eye)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / split_name
                / "materialized_features"
                / materialization_name
            )
            units = []
            for unit_id in ["fold_01", "fold_02"]:
                _write_stacked_unit(
                    materialized_root / unit_id,
                    unit_id=unit_id,
                    eval_role="val",
                    y_train=y_train,
                    y_eval=y_val,
                    x_train=x_train,
                    x_eval=x_val,
                )
                units.append(
                    {
                        "unit_id": unit_id,
                        "eval_role": "val",
                        "enabled_blocks": ["char", "word"],
                    }
                )

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": split_name,
                    "materialization_name": materialization_name,
                    "row_feature_name": row_feature_name,
                    "config_path": f"data_pipeline/configs/materializations/{materialization_name}.toml",
                    "units": units,
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_dev_stacked.toml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        "name = 'toy_dev_stacked'",
                        "seed = 42",
                        "selection_metric = 'macro_f1'",
                        "save_prediction_top_k = 2",
                        "n_jobs = 1",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/models'",
                        "artifacts_dir = 'models/artifacts/attribution'",
                        "",
                        "[source]",
                        f"split_name = '{split_name}'",
                        f"materialization_name = '{materialization_name}'",
                        "target = 'author'",
                        "units = 'all'",
                        "",
                        "[model]",
                        "family = 'stacked'",
                        "inner_cv = 2",
                        "base_c_values = [1.0]",
                        "class_weights = ['balanced']",
                        "max_iter = 5000",
                        "tol = 0.0001",
                        "dual = 'auto'",
                        "top_c_values = [1.0]",
                        "top_max_iter = 500",
                        "top_k = [2]",
                        "",
                        "[[families]]",
                        "name = 'char'",
                        "blocks = ['char']",
                        "",
                        "[[families]]",
                        "name = 'word'",
                        "blocks = ['word']",
                        "",
                        "[[conditions]]",
                        "id = 'char_word'",
                        "label = 'char_word'",
                        "families = ['char', 'word']",
                        "profiling_blocks = []",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_stacked_experiment(config_path)

            results_dir = project_root / manifest["results_dir"]
            artifacts_dir = project_root / manifest["artifacts_dir"]
            self.assertEqual(manifest["run_type"], "stacked_condition_selection")
            self.assertEqual(manifest["selection_scope"], "condition")
            self.assertEqual(manifest["n_jobs"], 1)
            self.assertEqual(manifest["unit_count"], 2)
            self.assertEqual(manifest["condition_count"], 1)
            self.assertTrue((results_dir / "candidate_summary.csv").exists())
            self.assertTrue((results_dir / "condition_summary.csv").exists())
            self.assertTrue((results_dir / "selected_candidates.json").exists())
            self.assertFalse((results_dir / "best_candidate.json").exists())
            self.assertFalse((results_dir / "predictions").exists())
            self.assertFalse((artifacts_dir / "models").exists())

            selected = json.loads(
                (results_dir / "selected_candidates.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(selected["selected_candidates"]), 1)
            self.assertEqual(
                selected["selected_candidates"][0]["candidate_id"],
                "char_word__baseC=1__topC=1__class_weight=balanced",
            )

    def test_run_final_stacked_evaluation_writes_condition_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n",
                encoding="utf-8",
            )
            (project_root / "data").mkdir()

            split_name = "toy_split"
            materialization_name = "toy_final_materialization"
            row_feature_name = "toy_features"
            _write_stacked_upstream_manifests(
                project_root,
                split_name=split_name,
                row_feature_name=row_feature_name,
            )

            labels = np.array([f"A{i}" for i in range(6)], dtype=object)
            y_train = np.repeat(labels, 6)
            y_test = labels.copy()
            eye = np.eye(len(labels))
            x_train = sparse.csr_matrix(
                np.vstack([eye[i] for i in range(len(labels)) for _ in range(6)])
            )
            x_test = sparse.csr_matrix(eye)

            materialized_root = (
                project_root
                / "data"
                / "splits"
                / split_name
                / "materialized_features"
                / materialization_name
            )
            _write_stacked_unit(
                materialized_root / "final_test_2013",
                unit_id="final_test_2013",
                eval_role="test",
                y_train=y_train,
                y_eval=y_test,
                x_train=x_train,
                x_eval=x_test,
            )
            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": split_name,
                    "materialization_name": materialization_name,
                    "row_feature_name": row_feature_name,
                    "config_path": f"data_pipeline/configs/materializations/{materialization_name}.toml",
                    "units": [
                        {
                            "unit_id": "final_test_2013",
                            "eval_role": "test",
                            "enabled_blocks": ["char", "word"],
                        },
                    ],
                },
            )

            selection_dir = (
                project_root
                / "results"
                / "models"
                / split_name
                / "toy_stacked_dev"
                / "seed_42"
            )
            selected_candidate = _stacked_selected_candidate_payload()
            selected_payload = {
                "selection_scope": "condition",
                "selection_metric": "macro_f1",
                "split_name": split_name,
                "materialization_name": materialization_name,
                "target": "author",
                "selected_candidates": [selected_candidate],
            }
            _write_json(selection_dir / "selected_candidates.json", selected_payload)
            _write_json(
                selection_dir / "manifest.json",
                {
                    "run_type": "stacked_condition_selection",
                    "selection_scope": "condition",
                    "experiment_name": "toy_stacked_dev",
                    "condition_count": 1,
                    "results_dir": f"results/models/{split_name}/toy_stacked_dev/seed_42",
                    "selected_candidates_path": (
                        f"results/models/{split_name}/toy_stacked_dev/seed_42/selected_candidates.json"
                    ),
                },
            )

            config_path = (
                project_root
                / "models"
                / "configs"
                / "attribution"
                / "toy_final_stacked.toml"
            )
            _write_minimal_final_stacked_config(
                config_path,
                split_name=split_name,
                materialization_name=materialization_name,
                selected_candidates_path=selection_dir.relative_to(project_root) / "selected_candidates.json",
            )

            manifest = run_final_stacked_evaluation(config_path)

            results_dir = project_root / manifest["results_dir"]
            artifacts_dir = project_root / manifest["artifacts_dir"]
            condition_dir = results_dir / "final_by_condition" / "char_word"
            condition_artifacts_dir = (
                artifacts_dir / "final_by_condition" / "char_word" / "final_model"
            )
            self.assertEqual(manifest["run_type"], "stacked_condition_final_evaluation")
            self.assertEqual(manifest["selection_scope"], "condition")
            self.assertEqual(manifest["condition_count"], 1)
            self.assertTrue((results_dir / "selected_candidates.json").exists())
            self.assertTrue((results_dir / "final_condition_summary.csv").exists())
            self.assertTrue((condition_dir / "final_test_metrics.json").exists())
            self.assertTrue((condition_dir / "final_test_predictions.csv").exists())
            self.assertTrue((condition_dir / "resolved_candidate.json").exists())
            self.assertFalse((results_dir / "predictions").exists())
            self.assertTrue((condition_artifacts_dir / "top_model.joblib").exists())
            self.assertTrue((condition_artifacts_dir / "family_char.joblib").exists())
            self.assertTrue((condition_artifacts_dir / "family_word.joblib").exists())

            metrics = json.loads(
                (condition_dir / "final_test_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("top3_accuracy", metrics["final_test_metrics"])
            self.assertIn("top5_accuracy", metrics["final_test_metrics"])

            predictions = pd.read_csv(condition_dir / "final_test_predictions.csv")
            self.assertIn("top5_label", predictions.columns)
            self.assertEqual(len(predictions), len(y_test))

            resolved = json.loads(
                (condition_dir / "resolved_candidate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resolved["candidate_id"], selected_candidate["candidate_id"])
            self.assertEqual(
                manifest["selection_source"]["selected_candidates_path"],
                f"results/models/{split_name}/toy_stacked_dev/seed_42/selected_candidates.json",
            )
            self.assertEqual(
                manifest["condition_results"][0]["model_dir"],
                f"models/artifacts/attribution/{split_name}/toy_final_stacked/seed_42/final_by_condition/char_word/final_model",
            )

    def test_load_selected_stacked_candidates_rejects_manifest_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n",
                encoding="utf-8",
            )
            (project_root / "data").mkdir()

            split_name = "toy_split"
            selection_dir = (
                project_root
                / "results"
                / "models"
                / split_name
                / "toy_stacked_dev"
                / "seed_42"
            )
            selected_payload = {
                "selection_scope": "condition",
                "selection_metric": "macro_f1",
                "split_name": split_name,
                "materialization_name": "toy_materialization",
                "target": "author",
                "selected_candidates": [_stacked_selected_candidate_payload()],
            }
            _write_json(selection_dir / "selected_candidates.json", selected_payload)
            _write_json(
                selection_dir / "manifest.json",
                {
                    "run_type": "stacked_condition_selection",
                    "selection_scope": "condition",
                    "condition_count": 1,
                    "selected_candidates_path": (
                        f"results/models/{split_name}/other_dev/seed_42/selected_candidates.json"
                    ),
                },
            )
            config = {
                "final_eval": {
                    "selected_candidates_path": (
                        f"results/models/{split_name}/toy_stacked_dev/seed_42/selected_candidates.json"
                    )
                }
            }

            with self.assertRaisesRegex(ValueError, "selected_candidates_path does not match"):
                stacked_module.load_selected_stacked_candidates(project_root, config)


if __name__ == "__main__":
    unittest.main()
