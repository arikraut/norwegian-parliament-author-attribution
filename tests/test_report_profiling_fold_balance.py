from __future__ import annotations

import unittest

import pandas as pd

from thesis_reporting.fold_balance import (
    _balance_summary,
    _merge_complete_targets,
    _with_inverse_author_weights,
)


class ProfilingFoldBalanceTests(unittest.TestCase):
    """Protect the report's weighting semantics and input-join boundary."""

    def test_author_weighting_gives_each_author_equal_total_weight(self) -> None:
        """Verify prolific authors do not dominate weighted fold percentages."""
        folded = pd.DataFrame(
            [
                {
                    "fold_id": 0,
                    "fold_role": "val",
                    "id_speech": 1,
                    "id_person": "A",
                    "party": "left",
                },
                {
                    "fold_id": 0,
                    "fold_role": "val",
                    "id_speech": 2,
                    "id_person": "A",
                    "party": "left",
                },
                {
                    "fold_id": 0,
                    "fold_role": "val",
                    "id_speech": 3,
                    "id_person": "B",
                    "party": "right",
                },
            ]
        )
        weighted = _with_inverse_author_weights(
            folded,
            group_cols=["fold_id", "fold_role"],
        )

        unweighted_rows, _ = _balance_summary(
            folded,
            folded,
            ["party"],
            unit_col="id_speech",
            scope="speech",
            roles=["val"],
        )
        weighted_rows, _ = _balance_summary(
            weighted,
            weighted,
            ["party"],
            unit_col="id_person",
            scope="author_weighted_speech",
            roles=["val"],
            weight_col="author_weight",
        )

        self.assertEqual(unweighted_rows[0]["majority_pct"], 66.67)
        self.assertEqual(weighted_rows[0]["majority_pct"], 50.0)
        self.assertEqual(weighted_rows[0]["effective_author_count"], 2.0)

    def test_fold_memberships_require_corresponding_target_rows(self) -> None:
        """Ensure missing target rows cannot be reported as ordinary missing labels."""
        memberships = pd.DataFrame([{"id_speech": 1, "id_person": "A"}])
        targets = pd.DataFrame(columns=["id_speech", "id_person", "party"])

        with self.assertRaisesRegex(ValueError, "no target row"):
            _merge_complete_targets(
                memberships,
                targets,
                on=["id_speech", "id_person"],
                context="fixture",
            )


if __name__ == "__main__":
    unittest.main()
