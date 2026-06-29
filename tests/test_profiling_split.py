from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.split.profiling import (
    build_grouped_author_fold_membership,
    build_grouped_author_outer_membership,
)
from data_pipeline.split.selection import (
    apply_exclusion_filter,
    build_author_base_stats,
    select_authors,
)
from data_pipeline.split.stats import build_author_stats_from_membership
from data_pipeline.split.writer import write_membership_split


def _make_profiling_df() -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    author_specs = [
        (101, "Ap", 4),
        (102, "Ap", 3),
        (103, "H", 4),
        (104, "H", 2),
        (105, "Sp", 3),
        (106, "Sp", 1),
        (107, "SV", 2),
        (108, "SV", 1),
    ]
    for author_id, party, n_speeches in author_specs:
        for idx in range(n_speeches):
            rows.append(
                {
                    "id_speech": speech_id,
                    "id_person": author_id,
                    "name": f"Author {author_id}",
                    "female": int(author_id % 2 == 0),
                    "party": party,
                    "partyname": f"Party {party}",
                    "language": "Bokmål",
                    "age": 40 + (author_id % 20),
                    "election": 2001 + (idx % 3) * 4,
                    "date": f"{2001 + (idx % 3) * 4}-01-{idx + 1:02d}",
                    "time": f"10:{idx:02d}:00",
                    "char_count": 50 + idx,
                    "word_count": 10 + idx,
                }
            )
            speech_id += 1
    return pd.DataFrame(rows)


def _select_profiling_authors(df: pd.DataFrame) -> pd.DataFrame:
    author_stats = build_author_base_stats(df)
    author_stats = apply_exclusion_filter(author_stats, [101, 102])
    _, selected = select_authors(
        author_stats,
        {
            "party_mode": "all",
            "language_mode": "bokmal_only",
            "bokmal_codes": ["Bokmål"],
            "min_authors_per_party": 0,
        },
        {
            "mode": "all_eligible",
            "ranking_metric": "total_chars_all",
            "strategy": "top_chars",
        },
        selection_seed=42,
    )
    return selected


class ProfilingSplitTests(unittest.TestCase):
    def test_grouped_outer_split_keeps_all_non_excluded_authors_in_train(self) -> None:
        df = _make_profiling_df()
        selected = _select_profiling_authors(df)

        outer = build_grouped_author_outer_membership(
            df,
            selected,
            "profiling",
        )

        selected_ids = set(selected["id_person"])
        self.assertEqual(selected_ids, {103, 104, 105, 106, 107, 108})
        self.assertNotIn(101, selected_ids)
        self.assertNotIn(102, selected_ids)

        roles_per_author = outer.groupby("id_person")["outer_role"].nunique()
        self.assertTrue((roles_per_author == 1).all())
        self.assertEqual(set(outer["outer_role"]), {"train"})
        self.assertEqual(set(outer["id_speech"]), set(df[df["id_person"].isin(selected_ids)]["id_speech"]))

    def test_grouped_folds_are_author_disjoint_and_use_all_profiling_authors(self) -> None:
        df = _make_profiling_df()
        selected = _select_profiling_authors(df)
        outer = build_grouped_author_outer_membership(
            df,
            selected,
            "profiling",
        )

        fold_defs, folds = build_grouped_author_fold_membership(
            df,
            outer,
            {"mode": "stratified_group_kfold", "n_splits": 2, "stratify_by": "party"},
            "profiling",
            seed=42,
        )

        self.assertEqual([fd["fold_id"] for fd in fold_defs], ["fold_01", "fold_02"])
        self.assertEqual(set(folds["id_person"]), set(selected["id_person"]))

        for _, fold in folds.groupby("fold_id"):
            train_authors = set(fold.loc[fold["fold_role"] == "train", "id_person"])
            val_authors = set(fold.loc[fold["fold_role"] == "val", "id_person"])
            self.assertTrue(train_authors)
            self.assertTrue(val_authors)
            self.assertTrue(train_authors.isdisjoint(val_authors))
            self.assertEqual(train_authors | val_authors, set(selected["id_person"]))

    def test_writer_keeps_grouped_folds_without_full_author_coverage(self) -> None:
        df = _make_profiling_df()
        selected = _select_profiling_authors(df)
        outer = build_grouped_author_outer_membership(
            df,
            selected,
            "profiling",
        )
        fold_defs, folds = build_grouped_author_fold_membership(
            df,
            outer,
            {"mode": "stratified_group_kfold", "n_splits": 2, "stratify_by": "party"},
            "profiling",
            seed=42,
        )
        stats = build_author_stats_from_membership(df, outer)
        stats = stats[stats["id_person"].isin(selected["id_person"])].copy()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "split.toml"
            source_path = root / "source.csv"
            config_path.write_text("[split]\nname = 'profiling'\n", encoding="utf-8")
            df.to_csv(source_path, index=False)

            write_membership_split(
                df=df,
                authors_subset=selected,
                author_stats_full=stats,
                split_name="profiling",
                experiment_name="profiling_test",
                project_root=root,
                split_dir=root / "data" / "splits" / "profiling",
                corpus_dir=root / "data" / "splits" / "profiling" / "corpus",
                results_dir=root / "results" / "splits" / "profiling",
                config_path=config_path,
                source_dataset_path=source_path,
                selection_seed=42,
                outer_membership=outer,
                fold_membership=folds,
                fold_defs=fold_defs,
                split_strategy="grouped_authors",
                strategy_config={
                    "outer_split": {},
                    "folds": {
                        "mode": "stratified_group_kfold",
                        "stratify_by": "party",
                    },
                },
                require_fold_author_coverage=False,
            )

            written_folds = pd.read_csv(
                root / "data" / "splits" / "profiling" / "memberships" / "folds.csv"
            )
            self.assertEqual(set(written_folds["fold_id"]), {"fold_01", "fold_02"})


if __name__ == "__main__":
    unittest.main()
