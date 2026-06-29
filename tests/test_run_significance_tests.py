from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from thesis_reporting.significance import run_comparison


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a prediction CSV in the format expected by the significance script."""
    pd.DataFrame(rows).to_csv(path, index=False)


class SignificanceComparisonTests(unittest.TestCase):
    """Regression tests for prediction-file alignment before significance testing."""

    def test_comparison_requires_same_id_speech_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            _write_predictions(
                path_a,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                    {"id_speech": 2, "y_true": "b", "y_pred": "a", "correct": False},
                ],
            )
            _write_predictions(
                path_b,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                    {"id_speech": 3, "y_true": "c", "y_pred": "c", "correct": True},
                ],
            )

            with self.assertRaisesRegex(ValueError, "same id_speech set"):
                run_comparison(path_a, path_b, "A", "B", n_bootstrap=10)

    def test_comparison_requires_same_y_true_per_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            _write_predictions(
                path_a,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                    {"id_speech": 2, "y_true": "b", "y_pred": "b", "correct": True},
                ],
            )
            _write_predictions(
                path_b,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                    {"id_speech": 2, "y_true": "c", "y_pred": "c", "correct": True},
                ],
            )

            with self.assertRaisesRegex(ValueError, "same y_true"):
                run_comparison(path_a, path_b, "A", "B", n_bootstrap=10)

    def test_comparison_rejects_duplicate_id_speech_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            _write_predictions(
                path_a,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                    {"id_speech": 1, "y_true": "a", "y_pred": "b", "correct": False},
                ],
            )
            _write_predictions(
                path_b,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": True},
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate id_speech"):
                run_comparison(path_a, path_b, "A", "B", n_bootstrap=10)

    def test_comparison_derives_correctness_from_predictions(self) -> None:
        """Ensure stale correctness columns cannot change McNemar counts."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            _write_predictions(
                path_a,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "a", "correct": False},
                    {"id_speech": 2, "y_true": "b", "y_pred": "a", "correct": True},
                ],
            )
            _write_predictions(
                path_b,
                [
                    {"id_speech": 1, "y_true": "a", "y_pred": "b", "correct": True},
                    {"id_speech": 2, "y_true": "b", "y_pred": "b", "correct": False},
                ],
            )

            result = run_comparison(path_a, path_b, "A", "B", n_bootstrap=10)

            self.assertEqual(
                result["mcnemar"]["system_a_correct_system_b_wrong"],
                1,
            )
            self.assertEqual(
                result["mcnemar"]["system_a_wrong_system_b_correct"],
                1,
            )

    def test_uncorrected_mcnemar_is_default_and_corrected_is_selectable(self) -> None:
        """Use uncorrected McNemar by default while retaining the corrected variant."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            rows_a = [
                {"id_speech": i, "y_true": "a", "y_pred": "a"}
                for i in range(1, 5)
            ]
            rows_b = [
                {"id_speech": i, "y_true": "a", "y_pred": "b"}
                for i in range(1, 5)
            ]
            _write_predictions(path_a, rows_a)
            _write_predictions(path_b, rows_b)

            uncorrected = run_comparison(path_a, path_b, "A", "B", n_bootstrap=10)
            corrected = run_comparison(
                path_a,
                path_b,
                "A",
                "B",
                n_bootstrap=10,
                mcnemar_method="continuity_corrected",
            )

            self.assertEqual(
                uncorrected["mcnemar"]["method"],
                "asymptotic_uncorrected",
            )
            self.assertEqual(uncorrected["mcnemar"]["chi2_stat"], 4.0)
            self.assertEqual(corrected["mcnemar"]["method"], "continuity_corrected")
            self.assertEqual(corrected["mcnemar"]["chi2_stat"], 2.25)


if __name__ == "__main__":
    unittest.main()
