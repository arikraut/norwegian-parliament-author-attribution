from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from data_pipeline.split.context import load_split_run_context
from data_pipeline.split.creation import (
    run_split_creation,
)


def _make_authorwise_df() -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    for author_id, party, char_count in [(101, "A", 700), (202, "B", 650)]:
        for idx in range(8):
            year = 2001 if idx < 4 else 2005
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


def _make_temporal_df() -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    for author_id, party, election_map in [
        (
            101,
            "A",
            {
                2001: [1200, 1300],
                2005: [1400, 1500],
                2009: [1600, 1700],
            },
        ),
        (202, "B", {2001: [1100, 1200], 2005: [1300, 1400], 2009: [1500]}),
    ]:
        for election, char_counts in election_map.items():
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


def _make_profiling_df(author_ids: list[int] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    speech_id = 1
    for author_id in author_ids or [101, 102, 103, 104]:
        party = "A" if author_id % 2 else "B"
        for idx in range(2):
            rows.append(
                {
                    "id_speech": speech_id,
                    "id_person": author_id,
                    "name": f"Author {author_id}",
                    "female": int(author_id % 2 == 0),
                    "party": party,
                    "partyname": f"Party {party}",
                    "language": "Bokmål",
                    "age": 40 + (author_id % 10),
                    "election": 2001 + idx * 4,
                    "date": f"{2001 + idx * 4}-01-01",
                    "time": f"09:{idx:02d}:00",
                    "char_count": 1000 + author_id,
                    "word_count": 200,
                }
            )
            speech_id += 1
    return pd.DataFrame(rows)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


class SplitCreationEntrypointTests(unittest.TestCase):
    def test_load_split_run_context_resolves_project_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "data" / "clean").mkdir(parents=True, exist_ok=True)
            (project_root / "results").mkdir(parents=True, exist_ok=True)
            _write_text(project_root / "pyproject.toml", "[project]\nname='toy'\nversion='0.0.0'")

            _make_authorwise_df().to_csv(project_root / "data" / "clean" / "toy_authorwise.csv", index=False)
            _write_text(
                project_root / "configs" / "toy_authorwise.toml",
                """
                [split]
                name = "toy_authorwise"
                experiment_name = "toy_authorwise"
                selection_seed = 7

                [data]
                source_dataset = "data/clean/toy_authorwise.csv"
                splits_dir = "data/splits"
                results_dir = "results/splits"

                [outer_split]
                strategy = "author_percentage"
                train = 0.80
                test = 0.20

                [folds]
                mode = "none"

                [eligibility]
                min_train_chars_per_author = 500

                [pool]
                party_mode = "all"
                language_mode = "bokmal_only"
                bokmal_codes = ["Bokmål"]
                min_authors_per_party = 1

                [selection]
                mode = "all_eligible"
                ranking_metric = "train_chars"
                """,
            )

            context = load_split_run_context(
                project_root,
                config_path="configs/toy_authorwise.toml",
            )

            self.assertEqual(
                context["config_path"],
                (project_root / "configs" / "toy_authorwise.toml").resolve(),
            )
            self.assertTrue(pd.api.types.is_integer_dtype(context["df"]["election"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(context["df"]["word_count"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(context["df"]["char_count"]))

    def test_run_split_creation_writes_authorwise_bundle_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "data" / "clean").mkdir(parents=True, exist_ok=True)
            _write_text(project_root / "pyproject.toml", "[project]\nname='toy'\nversion='0.0.0'")

            _make_authorwise_df().to_csv(project_root / "data" / "clean" / "toy_authorwise.csv", index=False)
            config_path = project_root / "configs" / "toy_authorwise.toml"
            _write_text(
                config_path,
                """
                [split]
                name = "toy_authorwise"
                experiment_name = "toy_authorwise"
                selection_seed = 7

                [data]
                source_dataset = "data/clean/toy_authorwise.csv"
                splits_dir = "data/splits"
                results_dir = "results/splits"

                [outer_split]
                strategy = "author_percentage"
                train = 0.80
                test = 0.20

                [folds]
                mode = "none"

                [eligibility]
                min_train_chars_per_author = 500

                [pool]
                party_mode = "all"
                language_mode = "bokmal_only"
                bokmal_codes = ["Bokmål"]
                min_authors_per_party = 1

                [selection]
                mode = "all_eligible"
                ranking_metric = "train_chars"
                """,
            )

            summary = run_split_creation(config_path)

            self.assertEqual(summary["split_name"], "toy_authorwise")
            self.assertEqual(summary["split_strategy"], "author_percentage")
            self.assertTrue((project_root / "data" / "splits" / "toy_authorwise" / "authors.csv").exists())
            self.assertTrue((project_root / "results" / "splits" / "toy_authorwise" / "split_summary.md").exists())

    def test_run_split_creation_writes_temporal_bundle_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "data" / "clean").mkdir(parents=True, exist_ok=True)
            _write_text(project_root / "pyproject.toml", "[project]\nname='toy'\nversion='0.0.0'")

            _make_temporal_df().to_csv(project_root / "data" / "clean" / "toy_temporal.csv", index=False)
            config_path = project_root / "configs" / "toy_temporal.toml"
            _write_text(
                config_path,
                """
                [split]
                name = "toy_temporal"
                experiment_name = "toy_temporal"
                selection_seed = 11

                [data]
                source_dataset = "data/clean/toy_temporal.csv"
                splits_dir = "data/splits"
                results_dir = "results/splits"

                [outer_split]
                strategy = "election_based"
                train = [2001, 2005]
                test = [2009]

                [folds]
                mode = "expanding"
                source = "train_only"
                min_train_periods = 1
                coverage_policy = "filter_authors"
                min_chars_per_fold_election = 500

                [eligibility]
                min_train_chars_per_author = 500
                min_test_speeches_per_author = 2

                [pool]
                party_mode = "all"
                language_mode = "bokmal_only"
                bokmal_codes = ["Bokmål"]
                min_authors_per_party = 1

                [selection]
                mode = "all_eligible"
                ranking_metric = "train_chars"
                """,
            )

            summary = run_split_creation(config_path)

            self.assertEqual(summary["split_name"], "toy_temporal")
            self.assertEqual(summary["split_strategy"], "election_based")
            manifest_path = project_root / "data" / "splits" / "toy_temporal" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["split_strategy"], "election_based")
            selected_authors = pd.read_csv(
                project_root / "data" / "splits" / "toy_temporal" / "authors.csv"
            )
            self.assertEqual(set(selected_authors["id_person"]), {101})
            self.assertTrue(
                (
                    project_root
                    / "results"
                    / "splits"
                    / "toy_temporal"
                    / "split_summary.md"
                ).exists()
            )

    def test_run_split_creation_writes_profiling_bundle_with_attribution_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "data" / "clean").mkdir(parents=True, exist_ok=True)
            _write_text(project_root / "pyproject.toml", "[project]\nname='toy'\nversion='0.0.0'")

            _make_profiling_df().to_csv(project_root / "data" / "clean" / "toy_profiling.csv", index=False)
            attribution_authors_path = project_root / "data" / "splits" / "toy_authorwise" / "authors.csv"
            attribution_authors_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id_person": [101, 102]}).to_csv(attribution_authors_path, index=False)
            config_path = project_root / "configs" / "toy_profiling.toml"
            _write_text(
                config_path,
                """
                [split]
                name = "toy_profiling"
                experiment_name = "toy_profiling"
                selection_seed = 11

                [data]
                source_dataset = "data/clean/toy_profiling.csv"
                splits_dir = "data/splits"
                results_dir = "results/splits"

                [exclusion]
                attribution_authors_path = "data/splits/toy_authorwise/authors.csv"

                [outer_split]
                strategy = "grouped_authors"

                [folds]
                mode = "stratified_group_kfold"
                n_splits = 2
                stratify_by = "party"
                coverage_policy = "grouped_author_holdout"

                [pool]
                party_mode = "all"
                language_mode = "bokmal_only"
                bokmal_codes = ["Bokmål"]
                min_authors_per_party = 0

                [selection]
                mode = "all_eligible"
                ranking_metric = "total_chars_all"
                strategy = "top_chars"
                """,
            )

            summary = run_split_creation(config_path)

            self.assertEqual(summary["split_strategy"], "grouped_authors")
            selected_authors = pd.read_csv(
                project_root / "data" / "splits" / "toy_profiling" / "authors.csv"
            )
            self.assertEqual(set(selected_authors["id_person"]), {103, 104})
            manifest = json.loads(
                (project_root / "data" / "splits" / "toy_profiling" / "manifest.json").read_text()
            )
            profiling_policy = manifest["strategy_config"]["profiling"]
            self.assertEqual(profiling_policy["attribution_authors_path"], "data/splits/toy_authorwise/authors.csv")
            self.assertEqual(profiling_policy["excluded_author_count"], 2)
            self.assertEqual(profiling_policy["post_selection_overlap_count"], 0)

    def test_run_split_creation_fails_when_all_profiling_authors_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "data" / "clean").mkdir(parents=True, exist_ok=True)
            _write_text(project_root / "pyproject.toml", "[project]\nname='toy'\nversion='0.0.0'")

            _make_profiling_df([101, 102]).to_csv(
                project_root / "data" / "clean" / "toy_profiling.csv",
                index=False,
            )
            attribution_authors_path = project_root / "data" / "splits" / "toy_authorwise" / "authors.csv"
            attribution_authors_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id_person": [101, 102]}).to_csv(attribution_authors_path, index=False)
            config_path = project_root / "configs" / "toy_profiling.toml"
            _write_text(
                config_path,
                """
                [split]
                name = "toy_profiling"
                experiment_name = "toy_profiling"
                selection_seed = 11

                [data]
                source_dataset = "data/clean/toy_profiling.csv"
                splits_dir = "data/splits"
                results_dir = "results/splits"

                [exclusion]
                attribution_authors_path = "data/splits/toy_authorwise/authors.csv"

                [outer_split]
                strategy = "grouped_authors"

                [folds]
                mode = "stratified_group_kfold"
                n_splits = 2
                stratify_by = "party"
                coverage_policy = "grouped_author_holdout"

                [pool]
                party_mode = "all"
                language_mode = "bokmal_only"
                bokmal_codes = ["Bokmål"]
                min_authors_per_party = 0

                [selection]
                mode = "all_eligible"
                ranking_metric = "total_chars_all"
                strategy = "top_chars"
                """,
            )

            with self.assertRaisesRegex(ValueError, "No authors left"):
                run_split_creation(config_path)


if __name__ == "__main__":
    unittest.main()
