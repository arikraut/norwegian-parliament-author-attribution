from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from models.SVM.diagnostics.attribution_diagnostics import (
    run_dev_attribution_selection_diagnostics,
    run_final_attribution_diagnostics,
)


class AttributionDiagnosticsTests(unittest.TestCase):
    def _write_dev_selection_fixture(
        self,
        run_dir: Path,
        *,
        run_type: str,
        fold_metrics: pd.DataFrame,
        candidate_summary: pd.DataFrame,
        condition_summary: pd.DataFrame,
        selected_candidates: list[dict[str, object]],
    ) -> None:
        """Write a minimal dev-selection artifact bundle for diagnostics tests."""
        run_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
        candidate_summary.to_csv(run_dir / "candidate_summary.csv", index=False)
        condition_summary.to_csv(run_dir / "condition_summary.csv", index=False)
        (run_dir / "selected_candidates.json").write_text(
            json.dumps(
                {
                    "selection_scope": "condition",
                    "selection_metric": "macro_f1",
                    "selected_candidates": selected_candidates,
                }
            ),
            encoding="utf-8",
        )
        prefix = "results/models/split/dev_exp/seed_1"
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_type": run_type,
                    "selection_scope": "condition",
                    "selection_metric": "macro_f1",
                    "results_dir": prefix,
                    "fold_metrics_path": f"{prefix}/fold_metrics.csv",
                    "candidate_summary_path": f"{prefix}/candidate_summary.csv",
                    "condition_summary_path": f"{prefix}/condition_summary.csv",
                    "selected_candidates_path": f"{prefix}/selected_candidates.json",
                }
            ),
            encoding="utf-8",
        )

    def test_final_diagnostics_iterates_condition_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "models" / "split" / "final_exp" / "seed_1"
            condition_rows = {
                "char_word": pd.DataFrame(
                    {
                        "y_true": ["A", "A", "B", "B"],
                        "y_pred": ["A", "B", "B", "B"],
                        "top1_label": ["A", "B", "B", "B"],
                    }
                ),
                "char_word__profiling_all": pd.DataFrame(
                    {
                        "y_true": ["A", "A", "B", "B"],
                        "y_pred": ["A", "A", "B", "A"],
                        "top1_label": ["A", "A", "B", "A"],
                    }
                ),
            }
            condition_results = []
            for condition_id, frame in condition_rows.items():
                condition_dir = run_dir / "final_by_condition" / condition_id
                condition_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(condition_dir / "final_test_predictions.csv", index=False)
                condition_results.append(
                    {
                        "condition_id": condition_id,
                        "condition_label": condition_id,
                        "candidate_id": f"{condition_id}__candidate",
                        "predictions_path": (
                            f"results/models/split/final_exp/seed_1/"
                            f"final_by_condition/{condition_id}/final_test_predictions.csv"
                        ),
                    }
                )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_type": "condition_final_evaluation",
                        "results_dir": "results/models/split/final_exp/seed_1",
                        "split_name": "split",
                        "condition_results": condition_results,
                    }
                ),
                encoding="utf-8",
            )

            summary = run_final_attribution_diagnostics(run_dir, top_confusions=3)
            diagnostics_dir = Path(summary["diagnostics_dir"])

            self.assertEqual(summary["condition_count"], 2)
            comparison = pd.read_csv(diagnostics_dir / "final_condition_comparison.csv")
            self.assertEqual(set(comparison["condition_id"]), set(condition_rows))
            for condition_id in condition_rows:
                self.assertTrue(
                    (
                        run_dir
                        / "final_by_condition"
                        / condition_id
                        / "diagnostics"
                        / "manifest.json"
                    ).exists()
                )

    def test_dev_selection_diagnostics_use_selection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "models" / "split" / "dev_exp" / "seed_1"
            self._write_dev_selection_fixture(
                run_dir,
                run_type="dev_condition_selection",
                fold_metrics=pd.DataFrame(
                    [
                        {
                            "candidate_id": "base__C=1__class_weight=none",
                            "condition_id": "base",
                            "condition_label": "Base",
                            "blocks": "char+word",
                            "split": "val",
                            "unit_id": "fold_a",
                            "macro_f1": 0.66,
                            "accuracy": 0.70,
                            "convergence_warning_count": 0,
                        },
                        {
                            "candidate_id": "base__C=1__class_weight=none",
                            "condition_id": "base",
                            "condition_label": "Base",
                            "blocks": "char+word",
                            "split": "train",
                            "unit_id": "fold_b",
                            "macro_f1": 0.82,
                            "accuracy": 0.84,
                            "convergence_warning_count": 1,
                        },
                        {
                            "candidate_id": "base__C=1__class_weight=none",
                            "condition_id": "base",
                            "condition_label": "Base",
                            "blocks": "char+word",
                            "split": "val",
                            "unit_id": "fold_b",
                            "macro_f1": 0.74,
                            "accuracy": 0.76,
                            "convergence_warning_count": 0,
                        },
                        {
                            "candidate_id": "profile__C=1__class_weight=none",
                            "condition_id": "profile",
                            "condition_label": "Profile",
                            "blocks": "char+word+profiling_female",
                            "split": "val",
                            "unit_id": "fold_a",
                            "macro_f1": 0.71,
                            "accuracy": 0.73,
                            "convergence_warning_count": 0,
                        },
                        {
                            "candidate_id": "profile__C=1__class_weight=none",
                            "condition_id": "profile",
                            "condition_label": "Profile",
                            "blocks": "char+word+profiling_female",
                            "split": "val",
                            "unit_id": "fold_b",
                            "macro_f1": 0.77,
                            "accuracy": 0.79,
                            "convergence_warning_count": 0,
                        },
                    ]
                ),
                candidate_summary=pd.DataFrame(
                    [
                        {
                            "candidate_id": "base__C=1__class_weight=none",
                            "condition_id": "base",
                            "condition_label": "Base",
                            "feature_set": "char_word",
                            "blocks": "char+word",
                            "c_value": 1.0,
                            "class_weight": "none",
                            "eval_mean_macro_f1": 0.70,
                            "eval_std_macro_f1": 0.04,
                            "eval_mean_accuracy": 0.73,
                            "n_eval_units": 2,
                        },
                        {
                            "candidate_id": "base__C=0.1__class_weight=none",
                            "condition_id": "base",
                            "condition_label": "Base",
                            "feature_set": "char_word",
                            "blocks": "char+word",
                            "c_value": 0.1,
                            "class_weight": "none",
                            "eval_mean_macro_f1": 0.65,
                            "eval_std_macro_f1": 0.02,
                            "eval_mean_accuracy": 0.68,
                            "n_eval_units": 2,
                        },
                        {
                            "candidate_id": "profile__C=1__class_weight=none",
                            "condition_id": "profile",
                            "condition_label": "Profile",
                            "feature_set": "char_word_profile",
                            "blocks": "char+word+profiling_female",
                            "c_value": 1.0,
                            "class_weight": "none",
                            "eval_mean_macro_f1": 0.74,
                            "eval_std_macro_f1": 0.03,
                            "eval_mean_accuracy": 0.76,
                            "n_eval_units": 2,
                        },
                    ]
                ),
                condition_summary=pd.DataFrame(
                    [
                        {
                            "condition_id": "base",
                            "condition_label": "Base",
                            "selected_candidate_id": "base__C=1__class_weight=none",
                            "selection_metric": "macro_f1",
                        },
                        {
                            "condition_id": "profile",
                            "condition_label": "Profile",
                            "selected_candidate_id": "profile__C=1__class_weight=none",
                            "selection_metric": "macro_f1",
                        },
                    ]
                ),
                selected_candidates=[
                    {
                        "condition_id": "base",
                        "candidate_id": "base__C=1__class_weight=none",
                    },
                    {
                        "condition_id": "profile",
                        "candidate_id": "profile__C=1__class_weight=none",
                    },
                ],
            )

            summary = run_dev_attribution_selection_diagnostics(run_dir)
            diagnostics_dir = Path(summary["diagnostics_dir"])

            self.assertEqual(summary["condition_count"], 2)
            self.assertTrue((diagnostics_dir / "manifest.json").exists())
            self.assertIn("fold_metrics_path", summary["inputs"])

            condition_summary = pd.read_csv(diagnostics_dir / "condition_selection_summary.csv")
            base_row = condition_summary.set_index("condition_id").loc["base"]
            self.assertEqual(base_row["selected_candidate_id"], "base__C=1__class_weight=none")
            self.assertAlmostEqual(base_row["selection_margin"], 0.05)
            self.assertEqual(base_row["profiling_representation"], "base_only")

            rankings = pd.read_csv(diagnostics_dir / "candidate_rankings.csv")
            base_rank = rankings[rankings["candidate_id"] == "base__C=1__class_weight=none"].iloc[0]
            self.assertEqual(int(base_rank["rank_within_condition"]), 1)

            stability = pd.read_csv(diagnostics_dir / "fold_stability.csv")
            base_stability = stability.set_index("condition_id").loc["base"]
            self.assertAlmostEqual(base_stability["min_metric"], 0.66)
            self.assertAlmostEqual(base_stability["max_metric"], 0.74)
            self.assertEqual(base_stability["worst_unit_id"], "fold_a")
            self.assertEqual(base_stability["best_unit_id"], "fold_b")
            self.assertEqual(int(base_stability["convergence_warning_count"]), 1)

            comparison = pd.read_csv(diagnostics_dir / "profiling_block_comparison.csv")
            self.assertEqual(
                set(comparison["profiling_representation"]),
                {"base_only", "profiling_probability"},
            )

    def test_dev_selection_diagnostics_accept_stacked_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "models" / "split" / "dev_exp" / "seed_1"
            self._write_dev_selection_fixture(
                run_dir,
                run_type="stacked_condition_selection",
                fold_metrics=pd.DataFrame(
                    [
                        {
                            "candidate_id": "stack_base__baseC=1__topC=1__class_weight=none",
                            "condition_id": "stack_base",
                            "condition_label": "Stack base",
                            "profiling_blocks": "none",
                            "split": "val",
                            "unit_id": "fold_a",
                            "macro_f1": 0.61,
                            "accuracy": 0.65,
                        },
                        {
                            "candidate_id": "stack_hard__baseC=1__topC=1__class_weight=none",
                            "condition_id": "stack_hard",
                            "condition_label": "Stack hard",
                            "profiling_blocks": "profiling_hard_female",
                            "split": "val",
                            "unit_id": "fold_a",
                            "macro_f1": 0.64,
                            "accuracy": 0.67,
                        },
                    ]
                ),
                candidate_summary=pd.DataFrame(
                    [
                        {
                            "candidate_id": "stack_base__baseC=1__topC=1__class_weight=none",
                            "condition_id": "stack_base",
                            "condition_label": "Stack base",
                            "family_set": "char_word",
                            "families": "char+word",
                            "base_c": 1.0,
                            "top_c": 1.0,
                            "class_weight": "none",
                            "profiling_blocks": "none",
                            "eval_mean_macro_f1": 0.61,
                            "eval_std_macro_f1": 0.0,
                            "eval_mean_accuracy": 0.65,
                            "n_eval_units": 1,
                        },
                        {
                            "candidate_id": "stack_hard__baseC=1__topC=1__class_weight=none",
                            "condition_id": "stack_hard",
                            "condition_label": "Stack hard",
                            "family_set": "char_word",
                            "families": "char+word",
                            "base_c": 1.0,
                            "top_c": 1.0,
                            "class_weight": "none",
                            "profiling_blocks": "profiling_hard_female",
                            "eval_mean_macro_f1": 0.64,
                            "eval_std_macro_f1": 0.0,
                            "eval_mean_accuracy": 0.67,
                            "n_eval_units": 1,
                        },
                    ]
                ),
                condition_summary=pd.DataFrame(
                    [
                        {
                            "condition_id": "stack_base",
                            "condition_label": "Stack base",
                            "selected_candidate_id": "stack_base__baseC=1__topC=1__class_weight=none",
                            "selection_metric": "macro_f1",
                        },
                        {
                            "condition_id": "stack_hard",
                            "condition_label": "Stack hard",
                            "selected_candidate_id": "stack_hard__baseC=1__topC=1__class_weight=none",
                            "selection_metric": "macro_f1",
                        },
                    ]
                ),
                selected_candidates=[
                    {
                        "condition_id": "stack_base",
                        "candidate_id": "stack_base__baseC=1__topC=1__class_weight=none",
                    },
                    {
                        "condition_id": "stack_hard",
                        "candidate_id": "stack_hard__baseC=1__topC=1__class_weight=none",
                    },
                ],
            )

            summary = run_dev_attribution_selection_diagnostics(run_dir)
            diagnostics_dir = Path(summary["diagnostics_dir"])

            self.assertEqual(summary["run_type"], "stacked_condition_selection")
            condition_summary = pd.read_csv(diagnostics_dir / "condition_selection_summary.csv")
            self.assertIn("base_c", condition_summary.columns)
            self.assertIn("top_c", condition_summary.columns)
            self.assertEqual(
                set(condition_summary["profiling_representation"]),
                {"base_only", "profiling_hard"},
            )


if __name__ == "__main__":
    unittest.main()
