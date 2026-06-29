from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.preprocessing import (
    add_age_feature,
    add_length_features,
    apply_manual_birthyear_fixes,
    build_candidate_fragment,
    build_redaction_regex,
    combine_language_filtered_views,
    collect_name_redaction_values,
    collect_party_redaction_values,
    drop_duplicate_speeches,
    drop_party_switcher_minority_rows,
    drop_procedural_rows,
    filter_language_view_by_small_party_pct,
    filter_short_speeches,
    fix_missing_spacing_after_punctuation,
    keep_majority_language,
    keep_supported_elections,
    normalize_text_artifacts,
    redact_text,
    remove_small_parties_by_word_pct,
    resolve_preprocessing_config,
)

DEFAULT_CONFIG = resolve_preprocessing_config()


class ResolvePreprocessingConfigTests(unittest.TestCase):
    def test_reads_custom_config_values_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config_path = project_root / "preprocessing.toml"
            config_path.write_text(
                """
[paths]
input = "raw.csv"
majority_output = "clean/majority.csv"
dataset_results_dir = "results/dataset"

[pipeline]
min_words = 42
min_election_year = 2005
small_party_min_word_pct = 2.5
supported_languages = ["Bokmål"]
majority_language_tie_breaker = "Bokmål"

[[language_outputs]]
name = "bokmal_custom"
language = "Bokmål"
path = "clean/bokmal.csv"
dataset_results_dir = "results/bokmal"

[schema]
final_columns = ["date", "text"]
profile_required_columns = ["party"]

[procedural]
status_column = "status"
procedural_column = "procedural"
excluded_statuses = ["chair"]

[deduplication]
key_columns = ["id_person", "text"]

[redaction]
name_token = "<PERSON>"
party_token = "<GROUP>"
name_columns = ["name"]
name_part_min_length = 3
party_codes = ["Ap"]

[redaction.party_aliases_by_code]
ap = ["Arbeiderpartiet"]

[metadata.manual_party_fixes]
"arbeiderpartiet" = "Ap"
""",
                encoding="utf-8",
            )

            config = resolve_preprocessing_config(
                project_root=project_root,
                config_path=config_path,
            )

        self.assertEqual(config.min_words, 42)
        self.assertEqual(config.min_election_year, 2005)
        self.assertEqual(config.small_party_min_word_pct, 2.5)
        self.assertEqual(config.supported_languages, ("Bokmål",))
        self.assertEqual(config.name_redaction_token, "<PERSON>")
        self.assertEqual(config.party_redaction_token, "<GROUP>")
        self.assertEqual(
            config.output_columns,
            ("date", "word_count", "char_count", "text"),
        )
        self.assertEqual(config.manual_party_fixes, {"arbeiderpartiet": "Ap"})
        self.assertEqual(config.language_outputs[0].name, "bokmal_custom")
        self.assertEqual(
            config.language_outputs[0].output_path,
            project_root.resolve() / "clean" / "bokmal.csv",
        )


class PreprocessingTests(unittest.TestCase):
    def test_add_age_feature_imputes_birthyear_from_other_rows(self) -> None:
        df = pd.DataFrame(
            {
                "id_person": [1, 1, 2],
                "birthyear": [1970, None, None],
                "date": pd.to_datetime(["2005-01-01", "2006-01-01", "2005-01-01"]),
            }
        )

        updated_df, details = add_age_feature(df)

        self.assertEqual(int(updated_df.loc[1, "birthyear"]), 1970)
        self.assertEqual(details["birthyear_imputed"], 1)
        self.assertEqual(details["missing_age_after"], 1)

    def test_normalize_text_artifacts_and_redact_text_keep_expected_boundaries(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    'Hei""""Verden Kari Nordmann møter Ap.',
                    "sier ap i dag uten kodegrense",
                ]
            }
        )

        cleaned_df, artifact_details = normalize_text_artifacts(df)
        redacted_names_df, _ = redact_text(cleaned_df, ["Kari Nordmann"], "<NAME>")
        redacted_df, redact_details = redact_text(
            redacted_names_df, ["Ap"], "<PARTY NAME>"
        )

        self.assertNotIn('"""', redacted_df.loc[0, "text"])
        self.assertIn("<NAME>", redacted_df.loc[0, "text"])
        self.assertIn("<PARTY NAME>", redacted_df.loc[0, "text"])
        self.assertIn("ap i dag", redacted_df.loc[1, "text"])
        self.assertGreater(artifact_details["quote_runs_removed"], 0)
        self.assertEqual(redact_details["rows_changed"], 1)

    def test_filter_short_speeches_and_keep_majority_language_follow_expected_contract(
        self,
    ) -> None:
        df = pd.DataFrame(
            {
                "id_person": [1, 1, 1, 2, 2],
                "language": ["Bokmål", "Bokmål", "Nynorsk", "Nynorsk", "Bokmål"],
                "text": [
                    "ett to tre fire",
                    "fem seks sju åtte",
                    "ni ti elleve tolv",
                    "ein to tre fire",
                    "ett to",
                ],
                "date": pd.to_datetime(["2005-01-01"] * 5),
                "time": ["10:00:00"] * 5,
                "election": [2005] * 5,
                "id_speech": [10, 11, 12, 20, 21],
                "name": ["A", "A", "A", "B", "B"],
                "age": [40, 40, 40, 50, 50],
                "female": [0, 0, 0, 1, 1],
                "party": ["Ap", "Ap", "Ap", "H", "H"],
                "partyname": [
                    "Arbeiderpartiet",
                    "Arbeiderpartiet",
                    "Arbeiderpartiet",
                    "Høyre",
                    "Høyre",
                ],
            }
        )

        counted_df, _ = add_length_features(df)
        filtered_df, details = filter_short_speeches(
            counted_df,
            min_words=3,
            output_columns=DEFAULT_CONFIG.output_columns,
        )
        majority_df, majority_details = keep_majority_language(
            filtered_df,
            DEFAULT_CONFIG.majority_language_tie_breaker,
        )

        self.assertEqual(details["rows_below_threshold"], 1)
        self.assertTrue((filtered_df["word_count"] >= 3).all())
        self.assertEqual(set(majority_df["id_person"]), {1, 2})
        self.assertEqual(
            majority_df.loc[majority_df["id_person"] == 1, "language"]
            .unique()
            .tolist(),
            ["Bokmål"],
        )
        self.assertEqual(majority_details["multilingual_authors_before"], 1)


class KeepSupportedElectionsTests(unittest.TestCase):
    def _make_df(self, elections: list) -> pd.DataFrame:
        return pd.DataFrame({"election": elections, "value": range(len(elections))})

    def test_drops_pre_2001_rows(self) -> None:
        df = self._make_df([1999, 2001, 2005])
        result, details = keep_supported_elections(df, min_year=2001)
        self.assertNotIn(1999, result["election"].tolist())
        self.assertEqual(details["rows_before_min_election_year"], 1)

    def test_drops_unparseable_rows(self) -> None:
        df = self._make_df([2005, "not_a_year", None])
        result, details = keep_supported_elections(df, min_year=2001)
        self.assertEqual(len(result), 1)
        self.assertEqual(details["unparseable_election_rows"], 2)

    def test_normalizes_election_column_to_int(self) -> None:
        df = self._make_df(["2001-2005", "2005-2009"])
        result, _ = keep_supported_elections(df, min_year=2001)
        self.assertTrue(pd.api.types.is_integer_dtype(result["election"]))
        self.assertListEqual(result["election"].tolist(), [2001, 2005])

    def test_configured_min_year_changes_temporal_cutoff(self) -> None:
        df = self._make_df([2001, 2005, 2009])
        result, details = keep_supported_elections(df, min_year=2005)
        self.assertEqual(result["election"].tolist(), [2005, 2009])
        self.assertEqual(details["rows_before_min_election_year"], 1)


class DropProceduralRowsTests(unittest.TestCase):
    def test_drops_president_status_and_non_empty_procedural_column_rows(self) -> None:
        df = pd.DataFrame(
            {
                "id_person": [1, 2, 3, 4],
                "status": ["representative", "president", "minister", "presidenten"],
                "procedural": [None, None, "ordfører for saken", ""],
            }
        )

        result, details = drop_procedural_rows(
            df,
            DEFAULT_CONFIG.procedural_status_column,
            DEFAULT_CONFIG.procedural_column,
            DEFAULT_CONFIG.procedural_excluded_statuses,
        )

        self.assertEqual(result["id_person"].tolist(), [1])
        self.assertEqual(details["excluded_status_rows"], 2)
        self.assertEqual(details["procedural_column_rows"], 1)
        self.assertEqual(details["non_status_procedural_column_rows"], 1)

    def test_blank_procedural_values_are_kept(self) -> None:
        df = pd.DataFrame(
            {
                "id_person": [1, 2, 3],
                "status": ["representative", "minister", "deputy"],
                "procedural": [None, "", "   "],
            }
        )

        result, details = drop_procedural_rows(
            df,
            DEFAULT_CONFIG.procedural_status_column,
            DEFAULT_CONFIG.procedural_column,
            DEFAULT_CONFIG.procedural_excluded_statuses,
        )

        self.assertEqual(result["id_person"].tolist(), [1, 2, 3])
        self.assertEqual(details["procedural_column_rows"], 0)


class KeepMajorityLanguageTests(unittest.TestCase):
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_keeps_majority_language_per_author(self) -> None:
        df = self._make_df(
            [
                {"id_person": 1, "language": "Bokmål"},
                {"id_person": 1, "language": "Bokmål"},
                {"id_person": 1, "language": "Bokmål"},
                {"id_person": 1, "language": "Nynorsk"},
            ]
        )
        result, _ = keep_majority_language(
            df,
            DEFAULT_CONFIG.majority_language_tie_breaker,
        )
        self.assertTrue((result["language"] == "Bokmål").all())
        self.assertEqual(len(result), 3)

    def test_tie_defaults_to_nynorsk(self) -> None:
        df = self._make_df(
            [
                {"id_person": 1, "language": "Bokmål"},
                {"id_person": 1, "language": "Nynorsk"},
            ]
        )
        result, details = keep_majority_language(
            df,
            DEFAULT_CONFIG.majority_language_tie_breaker,
        )
        self.assertTrue((result["language"] == "Nynorsk").all())
        self.assertEqual(details["tied_authors_using_tie_breaker"], 1)

    def test_configured_tie_breaker_changes_tied_majority_language(self) -> None:
        df = self._make_df(
            [
                {"id_person": 1, "language": "Bokmål"},
                {"id_person": 1, "language": "Nynorsk"},
            ]
        )
        result, details = keep_majority_language(df, tie_break_language="Bokmål")
        self.assertTrue((result["language"] == "Bokmål").all())
        self.assertEqual(details["majority_language_tie_breaker"], "Bokmål")


class CollectNameRedactionValuesTests(unittest.TestCase):
    def test_includes_full_name_and_individual_parts(self) -> None:
        df = pd.DataFrame({"name": ["Jens Stoltenberg"]})
        result = collect_name_redaction_values(
            df,
            DEFAULT_CONFIG.name_redaction_columns,
            DEFAULT_CONFIG.name_part_min_length,
        )
        self.assertIn("Jens Stoltenberg", result)
        self.assertIn("Stoltenberg", result)
        self.assertIn("Jens", result)

    def test_excludes_single_character_parts(self) -> None:
        df = pd.DataFrame({"name": ["J. Hagen"]})
        result = collect_name_redaction_values(
            df,
            DEFAULT_CONFIG.name_redaction_columns,
            DEFAULT_CONFIG.name_part_min_length,
        )
        self.assertNotIn("J", result)
        self.assertIn("Hagen", result)

    def test_deduplicates_across_name_and_name_original(self) -> None:
        df = pd.DataFrame(
            {"name": ["Kari Nordmann"], "name_original": ["Kari Nordmann"]}
        )
        result = collect_name_redaction_values(
            df,
            DEFAULT_CONFIG.name_redaction_columns,
            DEFAULT_CONFIG.name_part_min_length,
        )
        self.assertEqual(result.count("Kari Nordmann"), 1)

    def test_sorted_longest_first(self) -> None:
        df = pd.DataFrame({"name": ["Kari Nordmann"]})
        result = collect_name_redaction_values(
            df,
            DEFAULT_CONFIG.name_redaction_columns,
            DEFAULT_CONFIG.name_part_min_length,
        )
        full_name_index = result.index("Kari Nordmann")
        self.assertLess(full_name_index, result.index("Nordmann"))
        self.assertLess(full_name_index, result.index("Kari"))


class CollectPartyRedactionValuesTests(unittest.TestCase):
    def test_includes_bokmal_and_nynorsk_party_names_from_party_codes(self) -> None:
        df = pd.DataFrame(
            {
                "party": ["Ap", "FrP", "H", "KrF", "MDG", "R"],
                "partyname": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            }
        )

        result = collect_party_redaction_values(
            df,
            DEFAULT_CONFIG.party_name_redaction_aliases_by_code,
            DEFAULT_CONFIG.party_codes_for_redaction,
        )

        for party_name in [
            "Arbeiderpartiet",
            "Arbeidarpartiet",
            "Fremskrittspartiet",
            "Framstegspartiet",
            "Høyre",
            "Høgre",
            "Kristelig Folkeparti",
            "Kristeleg Folkeparti",
            "Miljøpartiet De Grønne",
            "Miljøpartiet Dei Grøne",
            "Rødt",
            "Raudt",
        ]:
            self.assertIn(party_name, result)

        redacted_df, details = redact_text(
            pd.DataFrame(
                {"text": ["Arbeidarpartiet møtte Høgre og Raudt i debatten."]}
            ),
            result,
            "<PARTY NAME>",
        )
        self.assertEqual(
            redacted_df.loc[0, "text"],
            "<PARTY NAME> møtte <PARTY NAME> og <PARTY NAME> i debatten.",
        )
        self.assertEqual(details["matches_replaced"], 3)

    def test_configured_party_codes_control_code_redaction_candidates(self) -> None:
        df = pd.DataFrame({"party": ["Ap", "H"], "partyname": [pd.NA, pd.NA]})
        result = collect_party_redaction_values(
            df,
            {"h": ("Høyre",)},
            frozenset({"H"}),
        )
        self.assertIn("H", result)
        self.assertIn("Høyre", result)
        self.assertNotIn("Ap", result)


class BuildCandidateFragmentTests(unittest.TestCase):
    def test_requires_uppercase_start(self) -> None:
        fragment = build_candidate_fragment("Hagen")
        pattern = re.compile(rf"(?<![\w>])(?:{fragment})(?![\w<])")
        self.assertIsNone(pattern.search("hagen er her"))
        self.assertIsNotNone(pattern.search("Hagen er her"))

    def test_rest_of_name_is_case_insensitive(self) -> None:
        fragment = build_candidate_fragment("Hagen")
        pattern = re.compile(rf"(?<![\w>])(?:{fragment})(?![\w<])")
        self.assertIsNotNone(pattern.search("HAGEN er her"))

    def test_non_alpha_prefix_is_matched_literally(self) -> None:
        # A name fragment starting with a non-alpha char: the first letter is still the anchor.
        fragment = build_candidate_fragment("-Berg")
        self.assertTrue(fragment.startswith(re.escape("-")))


class BuildRedactionRegexTests(unittest.TestCase):
    def test_matches_possessive_forms(self) -> None:
        pattern, _ = build_redaction_regex(["Hagen"])
        self.assertEqual(pattern.sub("<NAME>", "Hagen"), "<NAME>")
        self.assertEqual(pattern.sub("<NAME>", "Hagens"), "<NAME>")
        self.assertEqual(pattern.sub("<NAME>", "Hagen's"), "<NAME>")
        self.assertEqual(pattern.sub("<NAME>", "Hagen\u2019s"), "<NAME>")

    def test_does_not_match_inside_word(self) -> None:
        pattern, _ = build_redaction_regex(["Hagen"])
        self.assertIsNone(pattern.search("unhagen"))

    def test_returns_none_for_empty_values(self) -> None:
        pattern, count = build_redaction_regex([])
        self.assertIsNone(pattern)
        self.assertEqual(count, 0)


class ApplyManualBirthyearFixesTests(unittest.TestCase):
    def _base_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_corrects_oddmund_birthyear(self) -> None:
        df = self._base_df(
            [
                {
                    "id_person": 1,
                    "name": "Oddmund Hoel",
                    "birthyear": 1950,
                    "date": "2005-01-01",
                },
                {
                    "id_person": 2,
                    "name": "Kari Nordmann",
                    "birthyear": 1970,
                    "date": "2005-01-01",
                },
            ]
        )
        result, details = apply_manual_birthyear_fixes(df)
        self.assertEqual(
            int(result.loc[result["id_person"] == 1, "birthyear"].iloc[0]), 1968
        )
        self.assertEqual(details["birthyear_corrections"], 1)

    def test_does_not_touch_unrelated_rows(self) -> None:
        df = self._base_df(
            [
                {
                    "id_person": 2,
                    "name": "Kari Nordmann",
                    "birthyear": 1970,
                    "date": "2005-01-01",
                },
            ]
        )
        result, details = apply_manual_birthyear_fixes(df)
        self.assertEqual(int(result["birthyear"].iloc[0]), 1970)
        self.assertEqual(details["birthyear_corrections"], 0)


class DropPartySwitcherMinorityRowsTests(unittest.TestCase):
    def _base_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_drops_minority_party_rows_for_switcher(self) -> None:
        df = self._base_df(
            [
                {
                    "id_person": 1,
                    "name": "A",
                    "birthyear": 1970,
                    "date": f"200{i}-01-01",
                    "party": "Ap",
                }
                for i in range(5)
            ]
            + [
                {
                    "id_person": 1,
                    "name": "A",
                    "birthyear": 1970,
                    "date": f"201{i}-01-01",
                    "party": "H",
                }
                for i in range(2)
            ]
        )
        result, details = drop_party_switcher_minority_rows(df)
        self.assertTrue((result["party"] == "Ap").all())
        self.assertEqual(details["party_switcher_rows_dropped"], 2)

    def test_tie_in_party_speech_count_keeps_later_party(self) -> None:
        df = self._base_df(
            [
                {
                    "id_person": 1,
                    "name": "A",
                    "birthyear": 1970,
                    "date": f"200{i}-01-01",
                    "party": "Ap",
                }
                for i in range(3)
            ]
            + [
                {
                    "id_person": 1,
                    "name": "A",
                    "birthyear": 1970,
                    "date": f"201{i}-01-01",
                    "party": "H",
                }
                for i in range(3)
            ]
        )
        result, details = drop_party_switcher_minority_rows(df)
        self.assertTrue((result["party"] == "H").all())
        self.assertEqual(details["party_switcher_rows_dropped"], 3)

    def test_non_switching_authors_unaffected(self) -> None:
        df = self._base_df(
            [
                {
                    "id_person": 1,
                    "name": "A",
                    "birthyear": 1970,
                    "date": f"200{i}-01-01",
                    "party": "Ap",
                }
                for i in range(4)
            ]
        )
        result, details = drop_party_switcher_minority_rows(df)
        self.assertEqual(len(result), 4)
        self.assertEqual(details["party_switcher_rows_dropped"], 0)


class SmallPartyRemovalTests(unittest.TestCase):
    def test_keeps_party_at_word_pct_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "party": ["Ap", "H", "Kp"],
                "id_speech": [1, 2, 3],
                "id_person": [10, 11, 12],
                "word_count": [500, 450, 50],
            }
        )

        result, details = remove_small_parties_by_word_pct(df, min_word_pct=5.0)

        self.assertEqual(result["party"].tolist(), ["Ap", "H", "Kp"])
        self.assertEqual(details["small_parties_removed"], [])

    def test_remove_small_parties_uses_word_pct_not_speech_pct(self) -> None:
        df = pd.DataFrame(
            {
                "party": ["Ap", "Ap", "H", "Kp", "Kp"],
                "id_speech": [1, 2, 3, 4, 5],
                "id_person": [10, 11, 12, 13, 14],
                "word_count": [450, 50, 460, 20, 20],
            }
        )

        result, details = remove_small_parties_by_word_pct(df, min_word_pct=5.0)

        self.assertEqual(result["party"].tolist(), ["Ap", "Ap", "H"])
        self.assertEqual(details["small_parties_removed"], ["Kp"])
        self.assertEqual(details["small_party_speeches_removed"], 2)
        self.assertEqual(details["small_party_words_removed"], 40)
        self.assertAlmostEqual(details["largest_removed_word_pct"], 4.0)

    def test_language_view_filter_computes_threshold_within_each_language(self) -> None:
        df = pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4, 5, 6],
                "id_person": [10, 11, 12, 13, 14, 15],
                "language": [
                    "Bokmål",
                    "Bokmål",
                    "Bokmål",
                    "Nynorsk",
                    "Nynorsk",
                    "Nynorsk",
                ],
                "party": ["Ap", "Kp", "MDG", "Ap", "Kp", "MDG"],
                "word_count": [760, 40, 200, 760, 200, 40],
            }
        )

        bokmal_df, bokmal_details = filter_language_view_by_small_party_pct(
            df, "Bokmål", min_word_pct=5.0
        )
        nynorsk_df, nynorsk_details = filter_language_view_by_small_party_pct(
            df, "Nynorsk", min_word_pct=5.0
        )
        combined_df, _ = combine_language_filtered_views(df, [bokmal_df, nynorsk_df])

        self.assertEqual(bokmal_df["party"].tolist(), ["Ap", "MDG"])
        self.assertEqual(bokmal_details["small_parties_removed"], ["Kp"])
        self.assertEqual(nynorsk_df["party"].tolist(), ["Ap", "Kp"])
        self.assertEqual(nynorsk_details["small_parties_removed"], ["MDG"])
        self.assertEqual(combined_df["id_speech"].tolist(), [1, 3, 4, 5])


class DuplicateSpeechRemovalTests(unittest.TestCase):
    def test_drops_duplicate_speeches_with_distinct_ids(self) -> None:
        df = pd.DataFrame(
            {
                "id_speech": [100, 101, 102],
                "id_person": [1, 1, 2],
                "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-01"]),
                "time": ["10:00:00", "10:00:00", "11:00:00"],
                "text": ["same speech", "same speech", "same speech"],
            }
        )

        result, details = drop_duplicate_speeches(
            df,
            DEFAULT_CONFIG.duplicate_key_columns,
        )

        self.assertEqual(result["id_speech"].tolist(), [100, 102])
        self.assertEqual(details["duplicate_speeches_removed"], 1)
        self.assertEqual(details["duplicate_groups"], 1)

    def test_keeps_same_speaker_text_at_different_times(self) -> None:
        df = pd.DataFrame(
            {
                "id_speech": [100, 101],
                "id_person": [1, 1],
                "date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
                "time": ["10:00:00", "10:05:00"],
                "text": ["same speech", "same speech"],
            }
        )

        result, details = drop_duplicate_speeches(
            df,
            DEFAULT_CONFIG.duplicate_key_columns,
        )

        self.assertEqual(result["id_speech"].tolist(), [100, 101])
        self.assertEqual(details["duplicate_speeches_removed"], 0)


class FixMissingSpacingAfterPunctuationTests(unittest.TestCase):
    def test_inserts_space_after_sentence_punctuation_before_uppercase(self) -> None:
        df = pd.DataFrame(
            {"text": ["Aksjekursene har gått ned, og oljeprisen falt.Norge er rikt."]}
        )

        result, details = fix_missing_spacing_after_punctuation(df)

        self.assertEqual(
            result.loc[0, "text"],
            "Aksjekursene har gått ned, og oljeprisen falt. Norge er rikt.",
        )
        self.assertEqual(details["sentence_spacing_fixes"], 1)
        self.assertEqual(details["rows_with_punctuation_spacing_changes"], 1)

    def test_leaves_decimal_numbers_without_inserted_spaces(self) -> None:
        df = pd.DataFrame({"text": ["Veksten var 2.5 pst.Norge fulgte etter."]})

        result, details = fix_missing_spacing_after_punctuation(df)

        self.assertEqual(
            result.loc[0, "text"],
            "Veksten var 2.5 pst. Norge fulgte etter.",
        )
        self.assertEqual(details["sentence_spacing_fixes"], 1)

    def test_inserts_space_after_year_before_sentence_start(self) -> None:
        df = pd.DataFrame({"text": ["I 2002.Norge fulgte etter."]})

        result, details = fix_missing_spacing_after_punctuation(df)

        self.assertEqual(result.loc[0, "text"], "I 2002. Norge fulgte etter.")
        self.assertEqual(details["sentence_spacing_fixes"], 1)

    def test_leaves_lowercase_web_domains_without_inserted_spaces(self) -> None:
        df = pd.DataFrame(
            {"text": ["Se regjeringen.no, example.com og portal.net for detaljer."]}
        )

        result, details = fix_missing_spacing_after_punctuation(df)

        self.assertEqual(
            result.loc[0, "text"],
            "Se regjeringen.no, example.com og portal.net for detaljer.",
        )
        self.assertEqual(details["sentence_spacing_fixes"], 0)
        self.assertEqual(details["rows_with_punctuation_spacing_changes"], 0)

    def test_inserts_space_after_year_colon_before_clause_start(self) -> None:
        df = pd.DataFrame({"text": ["I 2002:Norge fulgte etter."]})

        result, details = fix_missing_spacing_after_punctuation(df)

        self.assertEqual(result.loc[0, "text"], "I 2002: Norge fulgte etter.")
        self.assertEqual(details["clause_spacing_fixes"], 1)


class NormalizeTextArtifactsTests(unittest.TestCase):
    def test_removes_invisible_characters(self) -> None:
        df = pd.DataFrame(
            {"text": ["Hei\u00adVer\u200bden"]}
        )  # soft hyphen + zero-width space
        result, details = normalize_text_artifacts(df)
        self.assertEqual(result["text"].iloc[0], "HeiVerden")
        self.assertEqual(details["rows_changed"], 1)

    def test_leaves_normal_text_unchanged(self) -> None:
        df = pd.DataFrame({"text": ["Vanlig tekst uten problemer"]})
        result, details = normalize_text_artifacts(df)
        self.assertEqual(result["text"].iloc[0], "Vanlig tekst uten problemer")
        self.assertEqual(details["rows_changed"], 0)
        self.assertEqual(details["quote_runs_removed"], 0)

    def test_removes_long_quote_runs_but_not_normal_double_quotes(self) -> None:
        df = pd.DataFrame({"text": ['Han sa ""ordet"" høyt', 'Her er """""" en blokk']})
        result, details = normalize_text_artifacts(df)
        self.assertEqual(result["text"].iloc[0], 'Han sa ""ordet"" høyt')
        self.assertEqual(result["text"].iloc[1], "Her er en blokk")
        self.assertEqual(details["quote_runs_removed"], 1)


if __name__ == "__main__":
    unittest.main()
