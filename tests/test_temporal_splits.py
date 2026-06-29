from __future__ import annotations

import unittest

import pandas as pd

from data_pipeline.split.temporal import (
    build_fold_definitions,
    build_temporal_outer_membership,
    filter_authors_by_temporal_fold_coverage,
)
from data_pipeline.split.selection import apply_temporal_test_eligibility_filters
from data_pipeline.split.stats import build_author_stats_from_membership


def _make_synthetic_temporal_df() -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    author_specs = [
        (
            101,
            "A",
            {
                2001: [1200, 1300],
                2005: [1400, 1500],
                2009: [1600],
                2013: [1700, 1800],
            },
        ),
        (
            202,
            "B",
            {
                2001: [1100, 1200],
                2005: [1300, 1400],
                2009: [1500],
                2013: [1600],
            },
        ),
        (303, "C", {2001: [800], 2005: [900], 2013: [1200]}),
    ]

    for author_id, party, elections in author_specs:
        for election, char_counts in elections.items():
            for idx, char_count in enumerate(char_counts):
                rows.append(
                    {
                        "id_speech": speech_id,
                        "id_person": author_id,
                        "name": f"Author {author_id}",
                        "female": int(author_id % 2 == 0),
                        "party": party,
                        "partyname": f"Party {party}",
                        "language": "Bokmål",
                        "age": 45 + (author_id % 10),
                        "election": election,
                        "date": f"{election}-01-{idx + 1:02d}",
                        "time": f"09:{idx:02d}:00",
                        "char_count": int(char_count),
                        "word_count": max(1, int(char_count) // 5),
                    }
                )
                speech_id += 1

    return pd.DataFrame(rows)


class TemporalSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _make_synthetic_temporal_df()
        self.train_elections = [2001, 2005, 2009]
        self.test_elections = [2013]
        self.folds_cfg = {
            "mode": "expanding",
            "source": "train_only",
            "min_train_periods": 1,
        }

    def test_filter_authors_removes_authors_missing_fold_elections(self) -> None:
        outer_membership = build_temporal_outer_membership(
            self.df,
            split_name="temporal",
            train_elections=self.train_elections,
            test_elections=self.test_elections,
        )
        author_stats = build_author_stats_from_membership(self.df, outer_membership)
        fold_defs = build_fold_definitions(
            self.train_elections,
            self.folds_cfg,
        )

        filtered = filter_authors_by_temporal_fold_coverage(
            self.df,
            author_stats,
            fold_defs,
            min_chars_per_fold_election=500,
        )

        self.assertEqual(set(filtered["id_person"]), {101, 202})
        self.assertNotIn(303, set(filtered["id_person"]))

    def test_temporal_test_support_filter_uses_configured_test_speech_count(self) -> None:
        outer_membership = build_temporal_outer_membership(
            self.df,
            split_name="temporal",
            train_elections=self.train_elections,
            test_elections=self.test_elections,
            n_test_speeches=2,
        )
        author_stats = build_author_stats_from_membership(self.df, outer_membership)

        filtered = apply_temporal_test_eligibility_filters(
            author_stats,
            {"min_test_speeches_per_author": 2},
        )
        unfiltered = apply_temporal_test_eligibility_filters(author_stats, {})

        self.assertEqual(set(filtered["id_person"]), {101})
        self.assertEqual(set(unfiltered["id_person"]), {101, 202, 303})


if __name__ == "__main__":
    unittest.main()
