from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipelines import tracks


def _model_manifest(dev_name: str, final_name: str | None = "final") -> dict:
    """Build the subset of an attribution model manifest needed by track tests."""
    manifest = {
        "stages": {
            "dev": {
                "results_dir": f"results/models/toy_split/{dev_name}/seed_42",
                "selected_candidates_path": f"results/models/toy_split/{dev_name}/seed_42/selected_candidates.json",
            },
        }
    }
    if final_name is not None:
        manifest["stages"]["final"] = {
            "results_dir": f"results/models/toy_split/{final_name}/seed_42",
            "final_condition_summary_path": f"results/models/toy_split/{final_name}/seed_42/final_condition_summary.csv",
        }
    return manifest


def _write_data_pipeline_configs(
    project_root: Path, *, final_stage: bool = True
) -> tuple[Path, Path, Path]:
    """Write minimal split, feature, and materialization configs for track tests."""
    split_config = (
        project_root / "data_pipeline" / "configs" / "splits" / "toy_temporal.toml"
    )
    feature_config = (
        project_root / "data_pipeline" / "configs" / "features" / "toy_temporal.toml"
    )
    materialization_config = (
        project_root
        / "data_pipeline"
        / "configs"
        / "materializations"
        / "toy_temporal.toml"
    )
    split_config.parent.mkdir(parents=True, exist_ok=True)
    feature_config.parent.mkdir(parents=True, exist_ok=True)
    materialization_config.parent.mkdir(parents=True, exist_ok=True)
    split_config.write_text(
        "\n".join(
            [
                "[split]",
                "name = 'toy_temporal'",
                "",
                "[data]",
                "splits_dir = 'data/splits'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    feature_config.write_text(
        "\n".join(
            [
                "[feature]",
                "name = 'toy_rows'",
                "split_name = 'toy_temporal'",
                "",
                "[data]",
                "splits_dir = 'data/splits'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    materialization_lines = [
        "[materialization]",
        "split_name = 'toy_temporal'",
        "row_feature_name = 'toy_rows'",
        "",
        "[data]",
        "splits_dir = 'data/splits'",
        "",
        "[stages.dev]",
        "name = 'toy_dev_features'",
        "selector = 'all'",
    ]
    if final_stage:
        materialization_lines.extend(
            [
                "",
                "[stages.final]",
                "name = 'toy_final_features'",
                "selector = 'final'",
            ]
        )
    materialization_lines.extend(
        [
            "",
            "[blocks]",
            "enabled = ['char']",
            "",
            "[char_tfidf]",
            "min_n = 2",
            "max_n = 5",
            "min_df = 1",
            "max_df = 1.0",
            "max_features = 10",
        ]
    )
    materialization_config.write_text(
        "\n".join(materialization_lines) + "\n", encoding="utf-8"
    )
    return split_config, feature_config, materialization_config


class PipelineTrackTests(unittest.TestCase):
    def test_data_pipeline_runs_split_features_and_all_materialization_stages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            split_config, feature_config, materialization_config = (
                _write_data_pipeline_configs(project_root)
            )

            def fake_run_json_stage(
                *,
                project_root: Path,
                label: str,
                config_path: Path,
                manifest_path: Path,
                runner,
                rebuild: bool,
                reuse_validator=None,
            ):
                del (
                    project_root,
                    config_path,
                    manifest_path,
                    runner,
                    rebuild,
                    reuse_validator,
                )
                return {
                    "status": "executed",
                    "manifest_path": f"{label}.json",
                    "summary": {},
                }

            with (
                mock.patch.object(
                    tracks, "_run_json_stage", side_effect=fake_run_json_stage
                ) as run_json_stage,
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_data_pipeline(
                    project_root=project_root,
                    split_config=split_config,
                    feature_config=feature_config,
                    materialization_config=materialization_config,
                    materialization_stage="all",
                    pipeline_name="data_temporal",
                )

            self.assertEqual(manifest["pipeline_name"], "data_temporal")
            self.assertEqual(manifest["materialization_stages"], ["dev", "final"])
            self.assertEqual(run_json_stage.call_count, 4)

    def test_dev_prerequisites_fail_before_features_when_split_has_no_folds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            split_config, feature_config, _ = _write_data_pipeline_configs(project_root)
            split_dir = project_root / "data" / "splits" / "toy_temporal"
            split_dir.mkdir(parents=True)
            (split_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "split_name": "toy_temporal",
                        "fold_count": 0,
                        "fold_ids": [],
                        "dropped_folds": [{"fold_id": "fold_01"}],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                tracks,
                "run_feature_generation",
                side_effect=AssertionError("feature generation should not run"),
            ):
                with self.assertRaisesRegex(ValueError, "produced zero dev folds"):
                    tracks._ensure_split_and_features(
                        project_root,
                        split_config=split_config,
                        feature_config=feature_config,
                        label="Toy",
                        rebuild=False,
                        require_folds=True,
                    )

    def test_data_pipeline_all_uses_only_defined_materialization_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            split_config, feature_config, materialization_config = (
                _write_data_pipeline_configs(
                    project_root,
                    final_stage=False,
                )
            )

            with (
                mock.patch.object(
                    tracks,
                    "_run_json_stage",
                    return_value={
                        "status": "executed",
                        "manifest_path": "manifest.json",
                        "summary": {},
                    },
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_data_pipeline(
                    project_root=project_root,
                    split_config=split_config,
                    feature_config=feature_config,
                    materialization_config=materialization_config,
                    materialization_stage="all",
                )

            self.assertEqual(manifest["materialization_stages"], ["dev"])

    def test_phase1a_writes_selected_and_final_artifact_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                del config_path, stage
                return _model_manifest("phase1a_dev", "phase1a_final")

            def fake_inline_stage(label: str, runner):
                del runner
                return {"status": "executed", "summary": {"label": label}}

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks, "_run_inline_stage", side_effect=fake_inline_stage
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase1a_track(project_root=project_root)

            self.assertEqual(manifest["pipeline_name"], "phase1a_baseline")
            self.assertIn("dev_selection_diagnostics", manifest["stages"])
            self.assertIn("final_diagnostics", manifest["stages"])
            self.assertIn("selected_candidates_path", manifest["artifacts"])
            self.assertIn("final_condition_summary_path", manifest["artifacts"])

    def test_selected_candidates_manifest_path_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            manifest = {"results_dir": "results/models/toy_split/toy_dev/seed_42"}

            with self.assertRaisesRegex(KeyError, "selected_candidates_path"):
                tracks._selected_candidates_path_from_manifest(project_root, manifest)

    def test_phase1b_writes_stacked_pipeline_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                del config_path, stage
                return _model_manifest("phase1b_dev", "phase1b_final")

            def fake_inline_stage(label: str, runner):
                del runner
                return {"status": "executed", "summary": {"label": label}}

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks, "_run_inline_stage", side_effect=fake_inline_stage
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase1b_track(project_root=project_root)

            self.assertEqual(manifest["pipeline_name"], "phase1b_stacked")

    def test_phase3a_smoke_auto_bootstraps_profiling_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                del config_path, stage
                return _model_manifest("phase3a_smoke", final_name=None)

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_profiling_smoke",
                    return_value={"pipeline_name": "phase2_smoke"},
                ) as profiling_smoke,
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase3a_track(
                    project_root=project_root,
                    stage="dev",
                    smoke=True,
                    rebuild=False,
                    skip_diagnostics=True,
                )

            profiling_smoke.assert_called_once_with(
                project_root=project_root, rebuild=False
            )
            self.assertEqual(manifest["pipeline_name"], "phase3a_smoke")
            self.assertEqual(
                manifest["stages"]["profiling_smoke"]["pipeline_name"], "phase2_smoke"
            )

    def test_phase3a_hard_representation_sets_pipeline_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            captured_stage: dict[str, object] = {}

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                del config_path
                captured_stage["stage"] = stage
                return _model_manifest("phase3a_hard", "phase3a_hard_final")

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase3a_track(
                    project_root=project_root,
                    stage="all",
                    profiling_representation="hard",
                    skip_diagnostics=True,
                )

            self.assertEqual(
                manifest["pipeline_name"], "phase3a_baseline_with_hard_profiling"
            )
            self.assertEqual(manifest["phase_label"], "Phase 3A hard")
            self.assertEqual(captured_stage["stage"], "all")

    def test_phase3a_single_signal_scope_sets_pipeline_name_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            captured: dict[str, object] = {}

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                captured["config_path"] = config_path
                captured["stage"] = stage
                return _model_manifest(
                    "phase3a_single_signal",
                    "phase3a_single_signal_final",
                )

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase3a_track(
                    project_root=project_root,
                    stage="all",
                    profiling_scope="single_signal",
                    skip_diagnostics=True,
                )

            self.assertEqual(
                manifest["pipeline_name"],
                "phase3a_baseline_with_single_signal_profiling",
            )
            self.assertEqual(manifest["phase_label"], "Phase 3A single-signal")
            self.assertEqual(captured["stage"], "all")
            self.assertEqual(
                Path(captured["config_path"]).name,
                "bokmal_authorwise_linear_svm_with_single_signal_profiling.toml",
            )

    def test_phase3b_oracle_single_signal_scope_sets_pipeline_name_and_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            captured_config: dict[str, Path] = {}

            def fake_run_attribution_model(
                config_path: Path, *, stage: str, **_: object
            ) -> dict:
                del stage
                captured_config["path"] = config_path
                return _model_manifest(
                    "phase3b_oracle_single_signal",
                    "phase3b_oracle_single_signal_final",
                )

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_attribution_model",
                    side_effect=fake_run_attribution_model,
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase3b_oracle_track(
                    project_root=project_root,
                    stage="all",
                    profiling_scope="single_signal",
                    skip_diagnostics=True,
                )

            self.assertEqual(
                manifest["pipeline_name"],
                "phase3b_oracle_stacked_with_single_signal_oracle_profiling",
            )
            self.assertEqual(manifest["phase_label"], "Phase 3B oracle single-signal")
            self.assertEqual(
                captured_config["path"].name,
                "bokmal_authorwise_stacked_with_single_signal_oracle_profiling.toml",
            )

    def test_phase2_track_runs_transfer_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            def fake_run_json_stage(
                *,
                project_root: Path,
                label: str,
                config_path: Path,
                manifest_path: Path,
                runner,
                rebuild: bool,
                reuse_validator=None,
            ):
                del project_root, config_path, manifest_path, rebuild, reuse_validator
                runner()
                if label.endswith("profiling transfer diagnostics"):
                    return {
                        "status": "executed",
                        "manifest_path": "profiling_quality/manifest.json",
                        "summary": {
                            "outputs": {
                                "profiling_signal_decision": (
                                    "results/profiling_quality/toy/prof/seed_42/"
                                    "profiling_signal_decision.json"
                                ),
                            },
                            "decision": {
                                "selected_targets": ["female", "age_bin"],
                                "excluded_targets": ["party", "left_center_right"],
                                "decision_basis": "attribution_train_profile_metrics",
                            },
                        },
                    }
                return {
                    "status": "executed",
                    "manifest_path": f"{label}.json",
                    "summary": {},
                }

            with (
                mock.patch.object(
                    tracks, "_ensure_split_and_features", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks, "_ensure_materialization", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks, "run_profiling_experiment", return_value={"status": "ok"}
                ),
                mock.patch.object(
                    tracks,
                    "run_final_profiling_training",
                    return_value={"status": "ok"},
                ),
                mock.patch.object(
                    tracks,
                    "_model_manifest_path",
                    side_effect=lambda root, config: root
                    / "results"
                    / "models"
                    / config.name
                    / "manifest.json",
                ),
                mock.patch.object(
                    tracks,
                    "_profiling_final_manifest_path",
                    side_effect=lambda root, config: root
                    / "results"
                    / "models"
                    / config.name
                    / "final_manifest.json",
                ),
                mock.patch.object(
                    tracks,
                    "_extraction_manifest_path",
                    side_effect=lambda root, config, stage="dev": (
                        root
                        / "results"
                        / "extraction"
                        / stage
                        / config.name
                        / "manifest.json"
                    ),
                ),
                mock.patch.object(
                    tracks,
                    "profiling_quality_manifest_path",
                    side_effect=lambda config, project_root: project_root
                    / "results"
                    / "profiling_quality"
                    / config.name
                    / "manifest.json",
                ),
                mock.patch.object(
                    tracks,
                    "run_profiling_signal_extraction",
                    return_value={"status": "ok"},
                ) as extraction,
                mock.patch.object(
                    tracks,
                    "run_profiling_transfer_diagnostics",
                    return_value={"status": "ok"},
                ) as diagnostics,
                mock.patch.object(
                    tracks, "_run_json_stage", side_effect=fake_run_json_stage
                ),
                mock.patch.object(
                    tracks,
                    "_write_pipeline_manifest",
                    side_effect=lambda _root, pipeline_name, payload: {
                        **payload,
                        "pipeline_name": pipeline_name,
                    },
                ),
            ):
                manifest = tracks.run_phase2_track(
                    project_root=project_root, rebuild=False
                )

            self.assertEqual(manifest["pipeline_name"], "phase2_profiling")
            self.assertEqual(extraction.call_count, 2)
            self.assertEqual(
                {call.kwargs["stage"] for call in extraction.call_args_list},
                {"dev", "final"},
            )
            diagnostics.assert_called_once()
            self.assertEqual(
                manifest["artifacts"]["selected_targets"], ["female", "age_bin"]
            )
            self.assertEqual(
                manifest["artifacts"]["decision_basis"],
                "attribution_train_profile_metrics",
            )


if __name__ == "__main__":
    unittest.main()
