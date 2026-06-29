from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from models.SVM.diagnostics.profiling_transfer_diagnostics import run_profiling_transfer_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ProfilingTransferDiagnosticsTests(unittest.TestCase):
    def test_transfer_diagnostics_writes_train_test_metrics_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text("[project]\nname='toy'\n", encoding="utf-8")
            (project_root / "data").mkdir()

            config_path = project_root / "models" / "configs" / "profiling" / "final_extraction.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                """
[data]
splits_dir = "data/splits"
profiling_results_dir = "results/models"

[source]
attribution_split_name = "toy_authorwise"
profiling_split_name = "toy_profiling"
profiling_materialization_name = "toy_profiling_mat"
profiling_experiment_name = "toy_profiling_svm"
profiling_seed = 42
targets = ["female"]

[stages.dev]
attribution_materialization_name = "toy_dev"

[stages.final]
attribution_materialization_name = "toy_final"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            final_root = (
                project_root
                / "data"
                / "splits"
                / "toy_authorwise"
                / "materialized_features"
                / "toy_final"
            )
            unit_dir = final_root / "final_test_2021"
            (unit_dir / "labels").mkdir(parents=True)
            (unit_dir / "matrices").mkdir()
            (unit_dir / "row_order").mkdir()

            _write_json(
                final_root / "manifest.json",
                {"units": [{"unit_id": "final_test_2021", "eval_role": "test"}]},
            )
            _write_json(final_root / "profiling_extraction_manifest.json", {"targets": ["female"]})
            _write_json(
                final_root / "profiling_feature_columns.json",
                {"targets": ["female"], "columns": ["female_0", "female_1"]},
            )

            np.save(unit_dir / "labels" / "y_train_female.npy", np.array(["0", "1", "1", "0"]))
            np.save(unit_dir / "labels" / "y_test_female.npy", np.array(["1", "0"]))
            sparse.save_npz(
                unit_dir / "matrices" / "X_train_profiling_female.npz",
                sparse.csr_matrix(
                    np.array(
                        [
                            [0.90, 0.10],
                            [0.20, 0.80],
                            [0.30, 0.70],
                            [0.85, 0.15],
                        ]
                    )
                ),
            )
            sparse.save_npz(
                unit_dir / "matrices" / "X_test_profiling_female.npz",
                sparse.csr_matrix(np.array([[0.25, 0.75], [0.70, 0.30]])),
            )
            pd.DataFrame(
                {
                    "row_idx": [0, 1, 2, 3],
                    "id_speech": ["s1", "s2", "s3", "s4"],
                    "id_person": ["a1", "a2", "a2", "a1"],
                }
            ).to_csv(unit_dir / "row_order" / "train_rows.csv", index=False)
            pd.DataFrame(
                {
                    "row_idx": [0, 1],
                    "id_speech": ["s5", "s6"],
                    "id_person": ["a3", "a4"],
                }
            ).to_csv(unit_dir / "row_order" / "test_rows.csv", index=False)

            target_results = (
                project_root
                / "results"
                / "models"
                / "toy_profiling"
                / "toy_profiling_svm"
                / "seed_42"
                / "female"
            )
            target_results.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "char_word__C=0.1__class_weight=balanced",
                        "feature_set": "char_word",
                        "eval_mean_macro_f1": 0.91,
                        "eval_mean_accuracy": 0.92,
                        "train_mean_macro_f1": 0.95,
                    }
                ]
            ).to_csv(target_results / "candidate_summary.csv", index=False)
            _write_json(
                target_results / "best_candidate.json",
                {
                    "candidate_id": "char_word__C=0.1__class_weight=balanced",
                    "feature_set": "char_word",
                },
            )

            manifest = run_profiling_transfer_diagnostics(
                config_path,
                output_dir=project_root / "results" / "profiling_quality",
                min_attribution_train_macro_f1=0.80,
                min_profiling_cv_macro_f1=0.80,
                show_progress=False,
            )

            quality_dir = (
                project_root
                / "results"
                / "profiling_quality"
                / "toy_authorwise"
                / "toy_profiling_svm"
                / "seed_42"
            )
            self.assertEqual(manifest["decision"]["selected_targets"], ["female"])
            self.assertTrue((quality_dir / "attribution_train_profile_metrics.csv").exists())
            self.assertTrue((quality_dir / "attribution_test_profile_metrics.csv").exists())
            self.assertTrue((quality_dir / "profiling_signal_decision.json").exists())

            target_summary = pd.read_csv(quality_dir / "target_summary.csv")
            self.assertEqual(target_summary.loc[0, "decision"], "include")
            self.assertEqual(target_summary.loc[0, "decision_basis"], "attribution_train_profile_metrics")
            self.assertGreaterEqual(float(target_summary.loc[0, "attribution_train_macro_f1"]), 0.80)


if __name__ == "__main__":
    unittest.main()
