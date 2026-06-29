from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.split.selection import (
    apply_author_split_eligibility_filters,
    select_authors,
)
from data_pipeline.split.authorwise import (
    filter_authors_by_authorwise_fold_coverage,
    build_authorwise_fold_membership,
    build_outer_membership_by_author,
)
from data_pipeline.split.stats import (
    build_author_fold_stats,
    build_author_stats_from_membership,
)
from data_pipeline.split.writer import write_membership_split


def _make_synthetic_author_df() -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    author_specs = [
        # Authors 101 and 202 support both outer and fold-level minima.
        (101, "A", [1500] * 12),
        (202, "B", [1000] * 12),
        # Author 303 passes outer minima but should fail stricter fold char minima.
        (303, "C", [200] * 12),
    ]

    for author_id, party, char_counts in author_specs:
        for idx, char_count in enumerate(char_counts):
            year = 2001 if idx < 6 else 2005
            rows.append(
                {
                    "id_speech": speech_id,
                    "id_person": author_id,
                    "name": f"Author {author_id}",
                    "female": int(author_id % 2 == 0),
                    "party": party,
                    "partyname": f"Party {party}",
                    "language": "Bokmål",
                    "age": 50 + (author_id % 10),
                    "election": year,
                    "date": f"{year}-01-{(idx % 28) + 1:02d}",
                    "time": f"10:{idx:02d}:00",
                    "char_count": int(char_count),
                    "word_count": max(1, int(char_count) // 5),
                }
            )
            speech_id += 1

    return pd.DataFrame(rows)


class AuthorwiseSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _make_synthetic_author_df()
        self.outer_cfg = {
            "strategy": "author_percentage",
            "train": 0.80,
            "test": 0.20,
        }
        self.outer_eligibility_cfg = {
            "min_train_chars_per_author": 3000,
        }
        self.folds_cfg = {
            "mode": "kfold",
            "source": "train_only",
            "n_splits": 3,
            "min_train_chars_per_author": 1500,
            "min_val_chars_per_author": 700,
        }

    def test_outer_split_is_deterministic(self) -> None:
        df = self.df[self.df["id_person"].isin([101, 202])].copy()

        outer_first = build_outer_membership_by_author(
            df,
            self.outer_cfg,
            split_name="outer",
        )
        outer_second = build_outer_membership_by_author(
            df,
            self.outer_cfg,
            split_name="outer",
        )

        stats_first = build_author_stats_from_membership(df, outer_first)
        valid_first = apply_author_split_eligibility_filters(
            stats_first,
            self.outer_eligibility_cfg,
        )
        self.assertTrue(outer_first.equals(outer_second))
        self.assertEqual(set(valid_first["id_person"]), {101, 202})
        self.assertNotIn("val", set(outer_first["outer_role"]))

    def test_outer_split_targets_character_share_with_whole_speeches(self) -> None:
        df = pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4, 5, 6],
                "id_person": [101] * 6,
                "name": ["Author 101"] * 6,
                "female": [0] * 6,
                "party": ["A"] * 6,
                "partyname": ["Party A"] * 6,
                "language": ["Bokmål"] * 6,
                "age": [50] * 6,
                "election": [2001] * 6,
                "date": [f"2001-01-{day:02d}" for day in range(1, 7)],
                "time": [f"10:0{i}:00" for i in range(6)],
                "char_count": [100, 100, 100, 100, 70, 70],
                "word_count": [20, 20, 20, 20, 14, 14],
            }
        )
        outer = build_outer_membership_by_author(df, self.outer_cfg, split_name="char_share")
        stats = build_author_stats_from_membership(df, outer)
        row = stats.loc[stats["id_person"] == 101].iloc[0]
        total_chars = int(row["total_chars_all"])

        self.assertGreaterEqual(row["test_chars"], self.outer_cfg["test"] * total_chars)
        self.assertEqual(int(row["test_chars"]), 140)

    def test_fold_coverage_filter_respects_char_minima(self) -> None:
        probe = build_outer_membership_by_author(self.df, self.outer_cfg, split_name="probe")
        probe_stats = build_author_stats_from_membership(self.df, probe)
        eligible = apply_author_split_eligibility_filters(
            probe_stats,
            self.outer_eligibility_cfg,
        )

        filtered = filter_authors_by_authorwise_fold_coverage(
            self.df,
            probe,
            eligible,
            self.folds_cfg,
            split_name="probe",
        )

        self.assertEqual(set(filtered["id_person"]), {101, 202})
        self.assertNotIn(303, set(filtered["id_person"]))

    def test_final_fold_membership_meets_fold_minima(self) -> None:
        df = self.df[self.df["id_person"].isin([101, 202])].copy()
        outer = build_outer_membership_by_author(
            df,
            self.outer_cfg,
            split_name="final_outer",
        )
        fold_defs, fold_membership = build_authorwise_fold_membership(
            outer,
            self.folds_cfg,
            split_name="final_outer",
            df=df,
        )
        fold_stats = build_author_fold_stats(df, fold_membership)

        self.assertEqual([fd["fold_id"] for fd in fold_defs], ["fold_01", "fold_02", "fold_03"])
        self.assertTrue((fold_stats["train_chars"] >= self.folds_cfg["min_train_chars_per_author"]).all())
        self.assertTrue((fold_stats["val_chars"] >= self.folds_cfg["min_val_chars_per_author"]).all())

    def test_materialization_writes_fold_diagnostics(self) -> None:
        df = self.df[self.df["id_person"].isin([101, 202])].copy()
        outer = build_outer_membership_by_author(
            df,
            self.outer_cfg,
            split_name="materialize",
        )
        author_stats = build_author_stats_from_membership(df, outer)
        fold_defs, fold_membership = build_authorwise_fold_membership(
            outer,
            self.folds_cfg,
            split_name="materialize",
            df=df,
        )

        selected_authors = pd.DataFrame(
            {
                "party": ["A", "B"],
                "id_person": [101, 202],
                "selection_metric_value": [25200, 12000],
                "rank_in_party": [1, 1],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            split_dir = project_root / "data" / "splits" / "materialize"
            corpus_dir = split_dir / "corpus"
            results_dir = project_root / "results" / "splits" / "materialize"

            write_membership_split(
                df=df,
                authors_subset=selected_authors,
                author_stats_full=author_stats,
                split_name="materialize",
                experiment_name="test",
                project_root=project_root,
                split_dir=split_dir,
                corpus_dir=corpus_dir,
                results_dir=results_dir,
                config_path=Path(__file__),
                source_dataset_path=Path(__file__),
                selection_seed=7,
                outer_membership=outer,
                fold_membership=fold_membership,
                fold_defs=fold_defs,
                split_strategy="author_chronological",
                strategy_config={"outer_split": self.outer_cfg, "folds": self.folds_cfg},
            )

            self.assertTrue((results_dir / "author_fold_stats.csv").exists())
            self.assertTrue((results_dir / "per_author_support_summary.csv").exists())
            self.assertTrue((results_dir / "author_level_reliability_summary.csv").exists())
            self.assertTrue((results_dir / "split_summary.md").exists())
            self.assertFalse((split_dir / "eligible_authors.csv").exists())
            self.assertFalse((split_dir / "filtered_pool_authors.csv").exists())

    def test_author_selection_rejects_unknown_ranking_metric(self) -> None:
        outer = build_outer_membership_by_author(
            self.df,
            self.outer_cfg,
            split_name="selection_metric",
        )
        author_stats = build_author_stats_from_membership(self.df, outer)
        selection_cfg = {
            "mode": "total",
            "n_authors_total": 2,
            "ranking_metric": "nonexistent_metric_xyz",
            "strategy": "alternate_parties",
        }
        pool_cfg = {
            "party_mode": "all",
            "language_mode": "bokmal_only",
            "bokmal_codes": ["Bokmål"],
            "min_authors_per_party": 1,
        }

        with self.assertRaises(KeyError):
            select_authors(author_stats, pool_cfg, selection_cfg, selection_seed=7)

    def test_materialization_flags_low_support_cells_without_dropping_folds(self) -> None:
        df = self.df[self.df["id_person"].isin([101, 202])].copy()
        outer = build_outer_membership_by_author(
            df,
            self.outer_cfg,
            split_name="support_policy",
        )
        author_stats = build_author_stats_from_membership(df, outer)
        fold_defs, fold_membership = build_authorwise_fold_membership(
            outer,
            self.folds_cfg,
            split_name="support_policy",
            df=df,
        )

        selected_authors = pd.DataFrame(
            {
                "party": ["A", "B"],
                "id_person": [101, 202],
                "selection_metric_value": [25200, 12000],
                "rank_in_party": [1, 1],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            split_dir = project_root / "data" / "splits" / "support_policy"
            corpus_dir = split_dir / "corpus"
            results_dir = project_root / "results" / "splits" / "support_policy"

            write_membership_split(
                df=df,
                authors_subset=selected_authors,
                author_stats_full=author_stats,
                split_name="support_policy",
                experiment_name="test",
                project_root=project_root,
                split_dir=split_dir,
                corpus_dir=corpus_dir,
                results_dir=results_dir,
                config_path=Path(__file__),
                source_dataset_path=Path(__file__),
                selection_seed=7,
                outer_membership=outer,
                fold_membership=fold_membership,
                fold_defs=fold_defs,
                split_strategy="author_chronological",
                strategy_config={
                    "outer_split": self.outer_cfg,
                    "folds": {
                        **self.folds_cfg,
                        "min_val_chars_per_author_fold_cell": 3500,
                        "min_val_speeches_per_author_fold_cell": 3,
                    },
                },
            )

            manifest = json.loads((split_dir / "manifest.json").read_text())
            self.assertEqual(manifest["fold_count"], 3)

            support_summary = pd.read_csv(results_dir / "per_author_support_summary.csv")
            excluded_cells = support_summary[
                (support_summary["support_scope"] == "fold")
                & (support_summary["role"] == "val")
                & (support_summary["author_level_reliability_status"] == "excluded")
            ].sort_values(["id_person", "fold_id"])
            self.assertEqual(len(excluded_cells), 3)
            self.assertEqual(set(excluded_cells["id_person"]), {202})
            self.assertEqual(
                set(excluded_cells["reliability_exclusion_reason"]),
                {"val_chars_below_threshold"},
            )

            reliability_summary = pd.read_csv(results_dir / "author_level_reliability_summary.csv")
            author_202 = reliability_summary.loc[reliability_summary["id_person"] == 202].iloc[0]
            self.assertEqual(int(author_202["n_total_val_cells"]), 3)
            self.assertEqual(int(author_202["n_reliable_val_cells"]), 0)
            self.assertEqual(int(author_202["n_excluded_val_cells"]), 3)
            self.assertTrue(bool(author_202["fully_excluded_from_author_level_reliability"]))

            exclusion_summary = manifest["support_policy"]
            self.assertEqual(exclusion_summary["totals"]["excluded_val_cells"], 3)
            self.assertEqual(exclusion_summary["totals"]["authors_with_any_excluded_cells"], 1)
            self.assertEqual(
                exclusion_summary["totals"]["authors_fully_excluded_from_reliability_summary"],
                1,
            )
            self.assertEqual(
                exclusion_summary["excluded_by_reason"]["val_chars_below_threshold"]["excluded_cells"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
