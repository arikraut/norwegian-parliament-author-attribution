from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data_pipeline.row_features.extraction import (
    _safe_float,
    build_stylometry_quality_reports,
    extract_spacy_stylometry_from_df,
    load_bokmal_function_words,
    run_feature_generation,
    textdescriptives_features_from_doc,
    validate_feature_corpus_columns,
)


class _FakeNLP:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def pipe(self, texts, batch_size: int = 64):
        self.seen_texts = list(texts)
        for text in self.seen_texts:
            yield SimpleNamespace(text=text)


class FeatureExtractionTests(unittest.TestCase):
    def test_safe_float_substitutes_and_counts_problem_values(self) -> None:
        counters = Counter()

        self.assertEqual(_safe_float(None, counters), 0.0)
        self.assertEqual(_safe_float("not-a-number", counters), 0.0)
        self.assertEqual(_safe_float(float("nan"), counters), 0.0)
        self.assertEqual(_safe_float(float("inf"), counters), 0.0)
        self.assertEqual(_safe_float("1.25", counters), 1.25)

        self.assertEqual(counters["missing_value_substitutions"], 1)
        self.assertEqual(counters["non_numeric_substitutions"], 1)
        self.assertEqual(counters["nan_substitutions"], 1)
        self.assertEqual(counters["inf_substitutions"], 1)

    def test_textdescriptives_extractor_reads_dict_based_extensions(self) -> None:
        doc = SimpleNamespace(
            _=SimpleNamespace(
                counts={
                    "n_tokens": 10,
                    "n_unique_tokens": 7,
                    "n_characters": 42,
                    "n_sentences": 2,
                },
                token_length={
                    "token_length_mean": 4.2,
                    "token_length_median": 4.0,
                    "token_length_std": 1.1,
                },
                sentence_length={
                    "sentence_length_mean": 5.0,
                    "sentence_length_median": 5.0,
                    "sentence_length_std": 0.5,
                },
                readability={
                    "flesch_reading_ease": 55.0,
                    "flesch_kincaid_grade": 9.0,
                    "lix": 30.0,
                    "rix": 4.0,
                },
                dependency_distance={
                    "dependency_distance_mean": 2.5,
                    "prop_adjacent_dependency_relation_mean": 0.4,
                },
                pos_proportions={
                    "pos_prop_NOUN": 0.4,
                    "pos_prop_VERB": 0.2,
                },
            )
        )

        counters = Counter()
        features = textdescriptives_features_from_doc(doc, quality_counters=counters)

        self.assertEqual(features["counts_n_tokens"], 10.0)
        self.assertEqual(features["counts_n_characters"], 42.0)
        self.assertEqual(features["token_length_mean"], 4.2)
        self.assertEqual(features["sentence_length_std"], 0.5)
        self.assertEqual(features["readability_lix"], 30.0)
        self.assertEqual(features["dep_dist_mean_dependency_distance"], 2.5)
        self.assertEqual(
            features["dep_dist_proportion_adjacent_dependency_relation"], 0.4
        )
        self.assertEqual(features["pos_prop_NOUN"], 0.4)
        self.assertEqual(counters["missing_value_substitutions"], 0)
        self.assertEqual(counters["total_substitutions__counts"], 0)

    def test_textdescriptives_extractor_counts_family_substitutions(self) -> None:
        doc = SimpleNamespace(
            _=SimpleNamespace(
                counts={"n_tokens": 5},
                token_length={"token_length_mean": float("nan")},
            )
        )

        counters = Counter()
        features = textdescriptives_features_from_doc(doc, quality_counters=counters)

        self.assertEqual(features["counts_n_unique_tokens"], 0.0)
        self.assertEqual(features["counts_n_characters"], 0.0)
        self.assertEqual(features["counts_n_sentences"], 0.0)
        self.assertEqual(features["token_length_mean"], 0.0)
        self.assertEqual(features["token_length_median"], 0.0)
        self.assertEqual(features["token_length_std"], 0.0)
        self.assertEqual(counters["missing_value_substitutions"], 5)
        self.assertEqual(counters["nan_substitutions"], 1)
        self.assertEqual(counters["total_substitutions__counts"], 3)
        self.assertEqual(counters["total_substitutions__token_length"], 3)

    def test_load_bokmal_function_words_uses_no_code(self) -> None:
        seen_codes: list[str] = []

        def _fake_stopwords(code: str):
            seen_codes.append(code)
            return {"og", "men"} if code == "no" else {"should_not_be_used"}

        with patch(
            "data_pipeline.row_features.stylometry.stopwords_iso.stopwords",
            side_effect=_fake_stopwords,
        ):
            words = load_bokmal_function_words()

        self.assertEqual(seen_codes, ["no"])
        self.assertEqual(words, ["men", "og"])

    def test_feature_corpus_validation_requires_derived_target_sources(self) -> None:
        with self.assertRaisesRegex(KeyError, "train corpus.*age"):
            validate_feature_corpus_columns(
                {
                    "train": pd.DataFrame(
                        {"id_speech": [1], "id_person": [101], "party": ["Ap"]}
                    )
                },
                profiling_labels=["age_bin"],
                metadata_columns=[],
            )

        with self.assertRaisesRegex(KeyError, "train corpus.*party"):
            validate_feature_corpus_columns(
                {
                    "train": pd.DataFrame(
                        {"id_speech": [1], "id_person": [101], "age": [44]}
                    )
                },
                profiling_labels=["left_center_right"],
                metadata_columns=[],
            )

    def test_batch_extractor_uses_same_normalized_texts_for_doc_and_char_features(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "id_speech": [1, 2],
                "id_person": [101, 102],
                "text": [None, "Alpha"],
            }
        )
        fake_nlp = _FakeNLP()
        char_calls: list[str] = []
        function_word_doc_texts: list[str] = []

        def _char_features(text: str) -> dict[str, float]:
            char_calls.append(text)
            return {"char_signal": float(len(text))}

        def _function_word_features(doc, function_words: list[str]) -> dict[str, float]:
            function_word_doc_texts.append(doc.text)
            return {"fw_signal": float(len(doc.text))}

        with (
            patch(
                "data_pipeline.row_features.stylometry.textdescriptives_features_from_doc",
                return_value={"td_signal": 0.0},
            ),
            patch(
                "data_pipeline.row_features.stylometry.function_word_features_from_doc",
                side_effect=_function_word_features,
            ),
            patch(
                "data_pipeline.row_features.stylometry.char_distribution_features",
                side_effect=_char_features,
            ),
        ):
            stylometry_df, quality = extract_spacy_stylometry_from_df(
                df,
                fake_nlp,
                function_words=["og"],
                return_quality=True,
                desc="toy_split",
            )

        self.assertEqual(fake_nlp.seen_texts, ["", "Alpha"])
        self.assertEqual(char_calls, fake_nlp.seen_texts)
        self.assertEqual(function_word_doc_texts, fake_nlp.seen_texts)
        self.assertEqual(stylometry_df["char_signal"].tolist(), [0.0, 5.0])
        self.assertEqual(quality["n_rows"], 2)
        self.assertEqual(quality["total_substitutions"], 0)

    def test_quality_report_adds_family_level_columns(self) -> None:
        stylometry_frames = {
            "train": pd.DataFrame(
                {
                    "id_speech": [1, 2],
                    "counts_n_tokens": [0.0, 0.0],
                    "fw_total_prop": [0.1, 0.2],
                    "charfreq_a": [0.3, 0.4],
                }
            )
        }
        split_quality = {
            "train": {
                "split": "train",
                "n_rows": 2,
                "n_features": 3,
                "missing_value_substitutions": 2,
                "nan_substitutions": 0,
                "inf_substitutions": 0,
                "non_numeric_substitutions": 0,
                "total_substitutions": 2,
                "nonfinite_output_cells": 0,
                "total_substitutions__counts": 2,
                "total_substitutions__token_length": 0,
                "total_substitutions__sentence_length": 0,
                "total_substitutions__readability": 0,
                "total_substitutions__dependency_distance": 0,
                "total_substitutions__pos_proportions": 0,
                "total_substitutions__function_words": 0,
                "total_substitutions__character_distribution": 0,
                "total_substitutions__other": 0,
            }
        }

        quality_report, low_variance_report = build_stylometry_quality_reports(
            stylometry_frames,
            split_quality,
        )

        train_row = quality_report.loc[quality_report["split"] == "train"].iloc[0]
        all_splits_row = quality_report.loc[
            quality_report["split"] == "all_splits"
        ].iloc[0]

        self.assertEqual(int(train_row["all_zero_feature_count__counts"]), 1)
        self.assertEqual(int(train_row["all_zero_feature_count__function_words"]), 0)
        self.assertEqual(int(train_row["total_substitutions__counts"]), 2)
        self.assertEqual(int(all_splits_row["total_substitutions__counts"]), 2)

        counts_row = low_variance_report.loc[
            low_variance_report["feature"] == "counts_n_tokens"
        ].iloc[0]
        self.assertEqual(counts_row["feature_family"], "counts")
        self.assertTrue(bool(counts_row["is_all_zero"]))

    def test_run_feature_generation_can_skip_stylometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "pyproject.toml").write_text(
                "[project]\nname='toy'\nversion='0.1.0'\n",
                encoding="utf-8",
            )
            corpus_dir = project_root / "data" / "splits" / "toy_split" / "corpus"
            corpus_dir.mkdir(parents=True)
            rows = pd.DataFrame(
                {
                    "id_speech": [1, 2],
                    "id_person": [101, 102],
                    "name": ["A", "B"],
                    "party": ["Ap", "H"],
                    "female": [0, 1],
                    "age": [45, 61],
                    "language": ["Bokmal", "Bokmal"],
                    "text": ["alpha beta", "gamma delta"],
                    "word_count": [2, 2],
                    "char_count": [10, 11],
                }
            )
            rows.iloc[[0]].to_csv(corpus_dir / "train.csv", index=False)
            rows.iloc[[1]].to_csv(corpus_dir / "test.csv", index=False)

            config_path = project_root / "toy_features.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[feature]",
                        "name = 'toy_rows'",
                        "split_name = 'toy_split'",
                        "spacy_model = 'unused'",
                        "",
                        "[data]",
                        "splits_dir = 'data/splits'",
                        "results_dir = 'results/features'",
                        "",
                        "[targets]",
                        "save_author_labels = true",
                        "profiling_labels = ['party', 'age_bin', 'female', 'left_center_right']",
                        "",
                        "[age_bins]",
                        "edges = [0, 50, 200]",
                        "labels = ['<50', '50+']",
                        "",
                        "[party_axis]",
                        "left = ['Ap']",
                        "center = []",
                        "right = ['H']",
                        "",
                        "[stylometry]",
                        "enabled = false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "data_pipeline.row_features.extraction._load_stylometry_pipeline"
            ) as load_pipeline:
                summary = run_feature_generation(config_path)

            load_pipeline.assert_not_called()
            row_feature_dir = project_root / summary["row_feature_dir"]
            manifest = json.loads(
                (row_feature_dir / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertFalse(manifest["stylometry"]["generated"])
            self.assertEqual(manifest["stylometry"]["n_features"], 0)
            self.assertNotIn("stylometry_raw", manifest["artifacts"])
            self.assertTrue((row_feature_dir / "row_meta.csv").exists())
            self.assertTrue((row_feature_dir / "targets.csv").exists())
            self.assertFalse((row_feature_dir / "stylometry_raw.csv.gz").exists())
            self.assertFalse(summary["stylometry_generated"])
            self.assertEqual(summary["n_stylometry_features"], 0)

            feature_results_dir = project_root / summary["feature_results_dir"]
            target_summary = pd.read_csv(feature_results_dir / "target_summary.csv")
            author_summary = target_summary[target_summary["target"] == "author"]
            self.assertEqual(author_summary["split"].tolist(), ["test", "train"])
            self.assertEqual(author_summary["n_classes"].tolist(), [1, 1])
            self.assertTrue(
                (feature_results_dir / "target_distributions" / "author.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
