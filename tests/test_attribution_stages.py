from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest import mock

from models.SVM.training import attribution_stages
from models.SVM.training.attribution_stages import (
    resolve_model_stage_config,
    run_attribution_model,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_project(project_root: Path) -> None:
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='toy'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    (project_root / "data").mkdir()


def _write_materialization_config(project_root: Path) -> Path:
    config_path = project_root / "data_pipeline" / "configs" / "materializations" / "toy_features.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[materialization]",
                "split_name = 'toy_split'",
                "row_feature_name = 'toy_rows'",
                "",
                "[data]",
                "splits_dir = 'data/splits'",
                "",
                "[stages.dev]",
                "name = 'toy_dev_materialization'",
                "selector = 'all'",
                "",
                "[stages.final]",
                "name = 'toy_final_materialization'",
                "selector = 'final'",
                "",
                "[blocks]",
                "enabled = ['char', 'word', 'stylo']",
                "",
                "[word_tfidf]",
                "min_n = 1",
                "max_n = 1",
                "min_df = 1",
                "max_df = 1.0",
                "max_features = 10",
                "",
                "[char_tfidf]",
                "min_n = 2",
                "max_n = 2",
                "min_df = 1",
                "max_df = 1.0",
                "max_features = 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_baseline_model_config(project_root: Path, materialization_config: Path) -> Path:
    config_path = project_root / "models" / "configs" / "attribution" / "toy_baseline.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "[experiment]",
                "kind = 'baseline'",
                "seed = 42",
                "",
                "[materialization]",
                f"config_path = '{materialization_config.relative_to(project_root)}'",
                "",
                "[dev]",
                "experiment_name = 'toy_dev_linear_svm'",
                "selection_metric = 'macro_f1'",
                "save_prediction_top_k = 3",
                "n_jobs = 1",
                "",
                "[final]",
                "experiment_name = 'toy_final_linear_svm'",
                "save_prediction_top_k = 3",
                "n_jobs = 1",
                "selected_candidates_path = 'results/models/toy_split/toy_dev_linear_svm/seed_42/selected_candidates.json'",
                "",
                "[fit]",
                "max_iter = 5000",
                "tol = 0.0001",
                "dual = 'auto'",
                "top_k = [3]",
                "",
                "[search]",
                "C_values = [1.0]",
                "class_weights = ['none']",
                "",
                "[profiling_source]",
                "profiling_split_name = 'toy_profiling'",
                "profiling_materialization_name = 'toy_profiling_mat'",
                "profiling_experiment_name = 'toy_profiler'",
                "profiling_seed = 42",
                "targets = ['party', 'female', 'age_bin']",
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
    return config_path


def _write_stacked_model_config(project_root: Path, materialization_config: Path) -> Path:
    config_path = project_root / "models" / "configs" / "attribution" / "toy_stacked.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        dedent(
            f"""
            [experiment]
            kind = 'stacked'
            seed = 42

            [materialization]
            config_path = '{materialization_config.relative_to(project_root)}'

            [final]
            experiment_name = 'toy_final_stacked'
            save_prediction_top_k = 3
            n_jobs = 7
            selected_candidates_path = 'results/models/toy_split/toy_dev_stacked/seed_42/selected_candidates.json'
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_oracle_source_config(project_root: Path) -> Path:
    config_path = project_root / "models" / "configs" / "profiling" / "toy_oracle_source.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        dedent(
            """
            [data]
            splits_dir = 'data/splits'

            [source]
            attribution_split_name = 'toy_split'
            targets = ['female', 'party']

            [stages.dev]
            attribution_materialization_name = 'toy_dev_materialization'

            [stages.final]
            attribution_materialization_name = 'toy_final_materialization'
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_oracle_model_config(
    project_root: Path,
    materialization_config: Path,
    oracle_source_config: Path | None,
) -> Path:
    config_path = project_root / "models" / "configs" / "attribution" / "toy_oracle.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_section = ""
    if oracle_source_config is not None:
        oracle_section = dedent(
            f"""
            [oracle_source]
            config_path = '{oracle_source_config.relative_to(project_root)}'

            """
        )
    config_path.write_text(
        dedent(
            f"""
            [experiment]
            kind = 'baseline'
            seed = 42

            [materialization]
            config_path = '{materialization_config.relative_to(project_root)}'

            {oracle_section}[dev]
            experiment_name = 'toy_dev_oracle'
            selection_metric = 'macro_f1'
            save_prediction_top_k = 3
            n_jobs = 1

            [[conditions]]
            id = 'char_word_oracle'
            label = 'char_word_oracle'
            feature_set = 'char_word_oracle'
            blocks = ['char', 'word', 'profiling_oracle']
            normalize_rows = true
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


class AttributionStageTests(unittest.TestCase):
    def test_resolve_model_stage_uses_requested_materialization_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            model_config = _write_baseline_model_config(project_root, materialization_config)

            dev = resolve_model_stage_config(project_root, model_config, stage="dev")
            final = resolve_model_stage_config(project_root, model_config, stage="final")

            self.assertEqual(dev["config"]["experiment"]["name"], "toy_dev_linear_svm")
            self.assertEqual(dev["config"]["source"]["materialization_name"], "toy_dev_materialization")
            self.assertEqual(final["config"]["experiment"]["name"], "toy_final_linear_svm")
            self.assertEqual(final["config"]["source"]["materialization_name"], "toy_final_materialization")
            expected_selection_path = (
                project_root
                / "results"
                / "models"
                / "toy_split"
                / "toy_dev_linear_svm"
                / "seed_42"
                / "selected_candidates.json"
            )
            self.assertEqual(
                project_root / final["config"]["final_eval"]["selected_candidates_path"],
                expected_selection_path,
            )

    def test_resolve_stacked_final_uses_final_n_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            model_config = _write_stacked_model_config(project_root, materialization_config)

            final = resolve_model_stage_config(project_root, model_config, stage="final")

            self.assertEqual(final["config"]["experiment"]["n_jobs"], 7)

    def test_model_stage_config_rejects_mixed_profiling_representations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            model_config = _write_baseline_model_config(project_root, materialization_config)
            with model_config.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "",
                            "[[conditions]]",
                            "id = 'char_word_probability'",
                            "label = 'char_word_probability'",
                            "feature_set = 'char_word_probability'",
                            "blocks = ['char', 'word', 'profiling_party']",
                            "normalize_rows = true",
                            "",
                            "[[conditions]]",
                            "id = 'char_word_hard'",
                            "label = 'char_word_hard'",
                            "feature_set = 'char_word_hard'",
                            "blocks = ['char', 'word', 'profiling_hard_party']",
                            "normalize_rows = true",
                        ]
                    )
                    + "\n"
                )

            with self.assertRaisesRegex(ValueError, "mix profiling representations"):
                resolve_model_stage_config(project_root, model_config, stage="dev")

    def test_run_attribution_model_rejects_mixed_profiling_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            model_config = _write_baseline_model_config(project_root, materialization_config)
            with model_config.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "",
                            "[[conditions]]",
                            "id = 'char_word_probability'",
                            "label = 'char_word_probability'",
                            "feature_set = 'char_word_probability'",
                            "blocks = ['char', 'word', 'profiling_party']",
                            "normalize_rows = true",
                            "",
                            "[[conditions]]",
                            "id = 'char_word_hard'",
                            "label = 'char_word_hard'",
                            "feature_set = 'char_word_hard'",
                            "blocks = ['char', 'word', 'profiling_hard_party']",
                            "normalize_rows = true",
                        ]
                    )
                    + "\n"
                )

            with mock.patch.object(attribution_stages, "run_materialization") as materialization:
                with self.assertRaisesRegex(ValueError, "mix profiling representations"):
                    run_attribution_model(model_config, stage="dev")

            materialization.assert_not_called()

    def test_final_profiling_extraction_uses_selected_candidate_target_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            model_config = _write_baseline_model_config(project_root, materialization_config)
            selected_candidates_path = (
                project_root
                / "results"
                / "models"
                / "toy_split"
                / "toy_dev_linear_svm"
                / "seed_42"
                / "selected_candidates.json"
            )
            _write_json(
                selected_candidates_path,
                {
                    "selection_scope": "condition",
                    "selection_metric": "macro_f1",
                    "split_name": "toy_split",
                    "materialization_name": "toy_dev_materialization",
                    "target": "author",
                    "selected_candidates": [
                        {
                            "condition_id": "char_word_female",
                            "condition_label": "char_word_female",
                            "candidate_id": "char_word_female__C=1__class_weight=none",
                            "feature_set": "char_word_female",
                            "blocks": ["char", "word", "profiling_female"],
                            "normalize_rows": True,
                            "normalize_each_block": False,
                            "block_weights": {"char": 1.0, "word": 1.0, "profiling_female": 1.0},
                            "c_value": 1.0,
                            "class_weight": "none",
                        },
                        {
                            "condition_id": "char_word_party",
                            "condition_label": "char_word_party",
                            "candidate_id": "char_word_party__C=1__class_weight=none",
                            "feature_set": "char_word_party",
                            "blocks": ["char", "word", "profiling_party"],
                            "normalize_rows": True,
                            "normalize_each_block": False,
                            "block_weights": {"char": 1.0, "word": 1.0, "profiling_party": 1.0},
                            "c_value": 1.0,
                            "class_weight": "none",
                        },
                        {
                            "condition_id": "char_word_hard_age_bin",
                            "condition_label": "char_word_hard_age_bin",
                            "candidate_id": "char_word_hard_age_bin__C=1__class_weight=none",
                            "feature_set": "char_word_hard_age_bin",
                            "blocks": ["char", "word", "profiling_hard_age_bin"],
                            "normalize_rows": True,
                            "normalize_each_block": False,
                            "block_weights": {"char": 1.0, "word": 1.0, "profiling_hard_age_bin": 1.0},
                            "c_value": 1.0,
                            "class_weight": "none",
                        },
                    ],
                },
            )

            def fake_final_eval(config: dict, **kwargs: object) -> dict:
                del config
                preloaded_payload = kwargs["preloaded_selection_payload"]
                preloaded_source = kwargs["preloaded_selection_source"]
                self.assertEqual(len(kwargs["preloaded_candidates"]), 3)
                self.assertEqual(
                    preloaded_payload["selected_candidates"][0]["candidate_id"],
                    "char_word_female__C=1__class_weight=none",
                )
                self.assertEqual(
                    preloaded_source["selected_candidates_path"],
                    "results/models/toy_split/toy_dev_linear_svm/seed_42/selected_candidates.json",
                )
                return {
                    "results_dir": "results/models/toy_split/toy_final_linear_svm/seed_42",
                    "selection_source": {
                        "selected_candidates_path": "results/models/toy_split/toy_dev_linear_svm/seed_42/selected_candidates.json",
                        "selection_results_dir": "results/models/toy_split/toy_dev_linear_svm/seed_42",
                        "selection_manifest_path": None,
                    },
                    "selected_candidates_path": "results/models/toy_split/toy_final_linear_svm/seed_42/selected_candidates.json",
                }

            with (
                mock.patch.object(attribution_stages, "run_materialization", return_value={}),
                mock.patch.object(
                    attribution_stages,
                    "run_profiling_signal_extraction",
                    return_value={"targets": ["female"]},
                ) as extraction,
                mock.patch.object(
                    attribution_stages,
                    "load_selected_direct_candidates",
                    wraps=attribution_stages.load_selected_direct_candidates,
                ) as load_selection,
                mock.patch.object(attribution_stages, "run_final_attribution_evaluation", side_effect=fake_final_eval),
            ):
                manifest = run_attribution_model(model_config, stage="final")

            load_selection.assert_called_once()
            extraction.assert_called_once()
            extraction_config_path = extraction.call_args.args[0]
            self.assertEqual(extraction.call_args.kwargs["stage"], "final")
            extraction_text = extraction_config_path.read_text(encoding="utf-8")
            self.assertIn('targets = ["party", "female", "age_bin"]', extraction_text)
            self.assertIn("[stages.final]", extraction_text)
            final_manifest = manifest["stages"]["final"]
            self.assertEqual(
                final_manifest["selection_source"]["selected_candidates_path"],
                "results/models/toy_split/toy_dev_linear_svm/seed_42/selected_candidates.json",
            )
            resolved_spec_path = project_root / final_manifest["resolved_spec_path"]
            self.assertTrue(resolved_spec_path.exists())
            resolved_spec = json.loads(resolved_spec_path.read_text(encoding="utf-8"))
            self.assertEqual(resolved_spec["kind"], "baseline")
            self.assertEqual(resolved_spec["stage"], "final")
            self.assertEqual(
                resolved_spec["config"]["experiment"]["name"],
                "toy_final_linear_svm",
            )

    def test_oracle_metadata_and_injection_use_source_target_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            _write_project(project_root)
            materialization_config = _write_materialization_config(project_root)
            oracle_source = _write_oracle_source_config(project_root)
            model_config = _write_oracle_model_config(
                project_root,
                materialization_config,
                oracle_source,
            )

            def fake_dev_train(config: dict, **_: object) -> dict:
                del config
                return {
                    "results_dir": "results/models/toy_split/toy_dev_oracle/seed_42",
                    "selected_candidates_path": "results/models/toy_split/toy_dev_oracle/seed_42/selected_candidates.json",
                }

            with (
                mock.patch.object(attribution_stages, "run_materialization", return_value={}),
                mock.patch.object(
                    attribution_stages,
                    "run_ground_truth_signal_injection",
                    return_value={"targets": ["female", "party"]},
                ) as injection,
                mock.patch.object(attribution_stages, "run_attribution_experiment", side_effect=fake_dev_train),
            ):
                manifest = run_attribution_model(model_config, stage="dev")

            injection.assert_called_once()
            injection_config_path = injection.call_args.args[0]
            injection_text = injection_config_path.read_text(encoding="utf-8")
            self.assertIn('targets = ["female", "party"]', injection_text)
            self.assertEqual(manifest["profiling"]["targets"], ["female", "party"])


if __name__ == "__main__":
    unittest.main()
