from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.utils import write_json as _write_json
from models.SVM.diagnostics.stylometry_audit import run_stylometry_audit


class StylometryAuditTests(unittest.TestCase):
    def test_run_stylometry_audit_writes_decision_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n", encoding="utf-8"
            )
            (project_root / "data").mkdir()

            results_dir = (
                project_root
                / "results"
                / "models"
                / "toy_split"
                / "toy_linear_svm"
                / "seed_42"
            )
            materialized_root = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "materialized_features"
                / "toy_materialization"
            )
            row_feature_dir = (
                project_root
                / "data"
                / "splits"
                / "toy_split"
                / "row_features"
                / "toy_features"
            )
            results_dir.mkdir(parents=True, exist_ok=True)
            materialized_root.mkdir(parents=True, exist_ok=True)
            row_feature_dir.mkdir(parents=True, exist_ok=True)

            _write_json(
                results_dir / "manifest.json",
                {
                    "selection_metric": "macro_f1",
                    "materialized_root": "data/splits/toy_split/materialized_features/toy_materialization",
                },
            )
            pd.DataFrame(
                [
                    {
                        "candidate_id": "char_word__C=0.1__class_weight=none",
                        "feature_set": "char_word",
                        "blocks": "char+word",
                        "normalize_rows": True,
                        "normalize_each_block": False,
                        "c_value": 0.1,
                        "class_weight": "none",
                        "eval_mean_macro_f1": 0.70,
                        "eval_mean_accuracy": 0.72,
                    },
                    {
                        "candidate_id": "char_word_stylo__C=0.1__class_weight=none",
                        "feature_set": "char_word_stylo",
                        "blocks": "char+word+stylo",
                        "normalize_rows": True,
                        "normalize_each_block": True,
                        "c_value": 0.1,
                        "class_weight": "none",
                        "eval_mean_macro_f1": 0.62,
                        "eval_mean_accuracy": 0.64,
                    },
                ]
            ).to_csv(results_dir / "candidate_summary.csv", index=False)

            _write_json(
                materialized_root / "manifest.json",
                {
                    "split_name": "toy_split",
                    "row_feature_name": "toy_features",
                },
            )
            pd.DataFrame(
                [
                    {
                        "split": "train",
                        "n_rows": 10,
                        "n_features": 2,
                        "total_substitutions": 0,
                        "nonfinite_output_cells": 0,
                        "all_zero_feature_count": 0,
                        "low_variance_feature_count": 0,
                    }
                ]
            ).to_csv(row_feature_dir / "stylometry_quality_report.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "split": "train",
                        "feature": "stylo_a",
                        "variance": 0.0,
                        "is_all_zero": True,
                        "is_zero_variance": True,
                        "is_low_variance": True,
                    }
                ]
            ).to_csv(row_feature_dir / "stylometry_low_variance_report.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "unit_id": "fold_01",
                        "eval_role": "val",
                        "kept_feature_count": 1,
                        "mean_standardized_mean_gap": 0.75,
                        "median_standardized_mean_gap": 0.75,
                        "max_standardized_mean_gap": 1.20,
                        "features_over_gap_0_5": 1,
                        "features_over_gap_1_0": 1,
                    }
                ]
            ).to_csv(materialized_root / "stylometry_drift_summary.csv", index=False)

            audit_manifest = run_stylometry_audit(results_dir)

            audit_dir = project_root / audit_manifest["audit_dir"]
            self.assertEqual(audit_manifest["decision"]["go_decision"], "no_go")
            self.assertLess(
                float(audit_manifest["decision"]["stylometry_delta_vs_nonstylometry"]),
                0.0,
            )
            self.assertTrue((audit_dir / "ablation_summary.csv").exists())
            self.assertTrue((audit_dir / "quality_summary.csv").exists())
            self.assertTrue((audit_dir / "drift_summary.csv").exists())
            self.assertTrue((audit_dir / "decision.json").exists())
            self.assertTrue((audit_dir / "go_no_go.md").exists())


if __name__ == "__main__":
    unittest.main()
