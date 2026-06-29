"""Preprocessing pipeline for the raw parliamentary corpus."""

from __future__ import annotations

import argparse
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_pipeline.utils import find_project_root
from data_pipeline.reporting import (
    save_speech_length_threshold_figure,
)

# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESSING_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "preprocessing" / "default.toml"
)


@dataclass(frozen=True)
class LanguageOutputConfig:
    """Resolved output settings for one language-specific cleaned corpus view."""

    name: str
    language: str
    output_path: Path
    dataset_results_dir: Path | None = None


MANUAL_BIRTHYEAR_NAME_COLUMNS = ("name_original", "name")
MANUAL_BIRTHYEAR_FIXES = {"oddmund hoel": (pd.Timestamp("2001-01-01"), 1968)}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Resolved paths and methodological choices used by preprocessing."""

    project_root: Path
    config_path: Path
    input_path: Path
    majority_output_path: Path
    dataset_results_dir: Path
    language_outputs: tuple[LanguageOutputConfig, ...]
    min_words: int
    min_election_year: int
    small_party_min_word_pct: float
    supported_languages: tuple[str, ...]
    majority_language_tie_breaker: str
    final_columns: tuple[str, ...]
    profile_required_columns: tuple[str, ...]
    duplicate_key_columns: tuple[str, ...]
    procedural_status_column: str
    procedural_column: str
    procedural_excluded_statuses: frozenset[str]
    name_redaction_columns: tuple[str, ...]
    name_part_min_length: int
    name_redaction_token: str
    party_redaction_token: str
    manual_party_fixes: dict[str, str]
    party_name_redaction_aliases_by_code: dict[str, tuple[str, ...]]
    party_codes_for_redaction: frozenset[str]

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the final output schema with length columns before text."""
        return self.final_columns[:-1] + (
            "word_count",
            "char_count",
            self.final_columns[-1],
        )


def _resolve_project_path(project_root: Path, path_value: str) -> Path:
    """Resolve a config path relative to the project root unless it is absolute."""
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load_language_outputs(
    project_root: Path, rows: Sequence[dict[str, Any]]
) -> tuple[LanguageOutputConfig, ...]:
    """Resolve configured language-specific output views."""
    return tuple(
        LanguageOutputConfig(
            name=row["name"],
            language=row["language"],
            output_path=_resolve_project_path(project_root, row["path"]),
            dataset_results_dir=(
                _resolve_project_path(project_root, row["dataset_results_dir"])
                if "dataset_results_dir" in row
                else None
            ),
        )
        for row in rows
    )


def resolve_preprocessing_config(
    project_root: Path | None = None,
    config_path: Path | None = None,
) -> PreprocessingConfig:
    """Build the preprocessing config for the project corpus from TOML."""
    resolved_root = (project_root or find_project_root()).resolve()
    if config_path is None:
        resolved_config_path = DEFAULT_PREPROCESSING_CONFIG_PATH
    else:
        resolved_config_path = _resolve_project_path(resolved_root, str(config_path))
    with resolved_config_path.open("rb") as config_file:
        config_data = tomllib.load(config_file)
    paths = config_data["paths"]
    pipeline = config_data["pipeline"]
    schema = config_data["schema"]
    procedural = config_data["procedural"]
    deduplication = config_data["deduplication"]
    redaction = config_data["redaction"]
    metadata = config_data["metadata"]
    party_aliases = redaction["party_aliases_by_code"]
    return PreprocessingConfig(
        project_root=resolved_root,
        config_path=resolved_config_path,
        input_path=_resolve_project_path(resolved_root, paths["input"]),
        majority_output_path=_resolve_project_path(resolved_root, paths["majority_output"]),
        dataset_results_dir=_resolve_project_path(resolved_root, paths["dataset_results_dir"]),
        language_outputs=_load_language_outputs(
            resolved_root, config_data["language_outputs"]
        ),
        min_words=pipeline["min_words"],
        min_election_year=pipeline["min_election_year"],
        small_party_min_word_pct=pipeline["small_party_min_word_pct"],
        supported_languages=tuple(pipeline["supported_languages"]),
        majority_language_tie_breaker=pipeline["majority_language_tie_breaker"],
        final_columns=tuple(schema["final_columns"]),
        profile_required_columns=tuple(schema["profile_required_columns"]),
        duplicate_key_columns=tuple(deduplication["key_columns"]),
        procedural_status_column=procedural["status_column"],
        procedural_column=procedural["procedural_column"],
        procedural_excluded_statuses=frozenset(
            status.casefold() for status in procedural["excluded_statuses"]
        ),
        name_redaction_columns=tuple(redaction["name_columns"]),
        name_part_min_length=redaction["name_part_min_length"],
        name_redaction_token=redaction["name_token"],
        party_redaction_token=redaction["party_token"],
        manual_party_fixes={
            party_name.casefold(): party_code
            for party_name, party_code in metadata["manual_party_fixes"].items()
        },
        party_name_redaction_aliases_by_code={
            party_code.casefold(): tuple(aliases)
            for party_code, aliases in party_aliases.items()
        },
        party_codes_for_redaction=frozenset(redaction["party_codes"]),
    )


# ---------------------------------------------------------------------------
# Candidate preparation and reporting helpers
# ---------------------------------------------------------------------------


def normalize_redaction_text(value: object) -> str:
    """Normalize redaction candidates before regex construction."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u00ad\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def collect_name_redaction_values(
    df: pd.DataFrame,
    columns: Sequence[str],
    min_part_length: int,
) -> list[str]:
    """Collect speaker names and name parts that could leak author identity."""
    values: list[object] = []
    for column in columns:
        if column in df.columns:
            values.extend(df[column].dropna().tolist())
    cleaned_values = []
    for value in values:
        normalized_value = normalize_redaction_text(value)
        if normalized_value:
            cleaned_values.append(normalized_value)
            for part in normalized_value.split():
                if len(part) >= min_part_length:
                    cleaned_values.append(part)
    return sorted(set(cleaned_values), key=lambda value: (-len(value), value))


def collect_party_redaction_values(
    df: pd.DataFrame,
    party_name_aliases_by_code: dict[str, tuple[str, ...]],
    party_codes_for_redaction: frozenset[str],
) -> list[str]:
    """Collect party names and safe party codes that could leak profile labels."""
    values: list[object] = []
    if "partyname" in df.columns:
        values.extend(df["partyname"].dropna().tolist())
    if "party" in df.columns:
        party_codes = df["party"].dropna().astype(str).str.strip()
        values.extend(party_codes[party_codes.isin(party_codes_for_redaction)].tolist())
        for party_code in party_codes:
            values.extend(party_name_aliases_by_code.get(party_code.casefold(), ()))
    cleaned_values = []
    for value in values:
        normalized_value = normalize_redaction_text(value)
        if normalized_value:
            cleaned_values.append(normalized_value)
    return sorted(set(cleaned_values), key=lambda value: (-len(value), value))


def _format_stage_detail_value(value: object) -> object:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return value


def _is_informative_stage_detail(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def record_stage(
    stage_summary_rows: list[dict[str, Any]],
    stage_detail_rows: list[dict[str, Any]],
    step: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    details: dict[str, Any] | None = None,
    *,
    report_rows_removed: bool = False,
    report_authors_removed: bool = False,
) -> None:
    """Record row/author counts and details for one preprocessing stage."""
    details = details or {}
    before_rows = len(before_df)
    after_rows = len(after_df)
    removed_rows = before_rows - after_rows
    before_authors = (
        int(before_df["id_person"].nunique())
        if "id_person" in before_df.columns
        else None
    )
    after_authors = (
        int(after_df["id_person"].nunique())
        if "id_person" in after_df.columns
        else None
    )
    removed_authors = (
        before_authors - after_authors
        if before_authors is not None and after_authors is not None
        else None
    )
    informative_details = {
        key: value
        for key, value in details.items()
        if _is_informative_stage_detail(value)
    }

    summary_row: dict[str, Any] = {"step": step}
    if report_rows_removed:
        summary_row["rows_after"] = after_rows
        summary_row["rows_removed"] = removed_rows
    if report_authors_removed:
        summary_row["authors_after"] = after_authors
        summary_row["authors_removed"] = removed_authors or 0
    if len(summary_row) > 1:
        stage_summary_rows.append(summary_row)

    for key, value in informative_details.items():
        stage_detail_rows.append(
            {"step": step, "metric": key, "value": _format_stage_detail_value(value)}
        )

    message_parts = [step]
    if report_rows_removed:
        message_parts.append(f"rows_removed={removed_rows:,}")
    if report_authors_removed:
        message_parts.append(f"authors_removed={(removed_authors or 0):,}")
    message_parts.extend(
        f"{key}={_format_stage_detail_value(value)}"
        for key, value in informative_details.items()
    )
    print(" | ".join(message_parts))


def run_stage(
    stage_summary_rows: list[dict[str, Any]],
    stage_detail_rows: list[dict[str, Any]],
    df: pd.DataFrame,
    step: str,
    transform,
    *args,
    report_rows_removed: bool = False,
    report_authors_removed: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Run a transform and add its metadata to the stage report."""
    updated_df, details = transform(df, *args, **kwargs)
    record_stage(
        stage_summary_rows,
        stage_detail_rows,
        step,
        df,
        updated_df,
        details,
        report_rows_removed=report_rows_removed,
        report_authors_removed=report_authors_removed,
    )
    return updated_df


def record_output(
    output_reports: list[dict[str, Any]],
    label: str,
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Append one written corpus view to the preprocessing output report."""
    report = {
        "dataset": label,
        "rows": len(df),
        "authors": int(df["id_person"].nunique()) if "id_person" in df.columns else 0,
        "path": str(path),
    }
    output_reports.append(report)
    print(
        f"saved {label}: {report['rows']:,} rows, {report['authors']:,} authors -> {path}"
    )


# ---------------------------------------------------------------------------
# Row-level preprocessing stages
# ---------------------------------------------------------------------------


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw NPD CSV and normalize column-name whitespace."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def clean_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse speech dates and drop rows with invalid dates."""
    parsed_dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    updated_df = df.loc[parsed_dates.notna()].copy()
    updated_df["date"] = parsed_dates.loc[updated_df.index]
    return updated_df, {"invalid_dates": invalid_dates}


def keep_supported_elections(
    df: pd.DataFrame, min_year: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep elections at or after the configured minimum year and normalize labels."""
    election = df["election"]
    if pd.api.types.is_numeric_dtype(election):
        election_year = pd.to_numeric(election, errors="coerce")
    else:
        election_year = pd.to_numeric(
            election.astype(str).str.extract(r"((?:19|20)\d{2})", expand=False),
            errors="coerce",
        )
    unparseable_rows = election_year.isna()
    old_rows = election_year < min_year
    keep_rows = ~unparseable_rows & ~old_rows
    updated_df = df.loc[keep_rows].copy()
    updated_df["election"] = election_year.loc[keep_rows].astype(int)
    return updated_df, {
        "min_election_year": min_year,
        "rows_before_min_election_year": int(old_rows.sum()),
        "unparseable_election_rows": int(unparseable_rows.sum()),
    }


def drop_procedural_rows(
    df: pd.DataFrame,
    status_column: str,
    procedural_column: str,
    excluded_statuses: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove president-status rows and rows marked as procedural."""
    president_rows = (
        df[status_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(excluded_statuses)
    )
    if procedural_column in df.columns:
        procedural_rows = df[procedural_column].notna() & df[procedural_column].astype(
            str
        ).str.strip().ne("")
    else:
        procedural_rows = pd.Series(False, index=df.index)
    rows_to_remove = president_rows | procedural_rows
    updated_df = df.loc[~rows_to_remove].copy()
    return updated_df, {
        "excluded_status_rows": int(president_rows.sum()),
        "procedural_column_rows": int(procedural_rows.sum()),
        "non_status_procedural_column_rows": int(
            (procedural_rows & ~president_rows).sum()
        ),
    }


def add_age_feature(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill stable missing birthyears within author and derive age at speech year."""
    if "id_person" not in df.columns or "birthyear" not in df.columns:
        return df.copy(), {"birthyear_imputed": 0, "missing_age_after": 0}

    updated_df = df.copy()
    missing_birthyear_before = int(updated_df["birthyear"].isna().sum())
    updated_df["birthyear"] = pd.to_numeric(updated_df["birthyear"], errors="coerce")

    unique_birthyears = (
        updated_df[["id_person", "birthyear"]]
        .dropna(subset=["birthyear"])
        .groupby("id_person")["birthyear"]
        .agg(lambda values: values.iloc[0] if values.nunique() == 1 else np.nan)
        .dropna()
    )
    missing_birthyear = updated_df["birthyear"].isna() & updated_df["id_person"].isin(
        unique_birthyears.index
    )
    updated_df.loc[missing_birthyear, "birthyear"] = updated_df.loc[
        missing_birthyear, "id_person"
    ].map(unique_birthyears)

    updated_df["age"] = updated_df["date"].dt.year - pd.to_numeric(
        updated_df["birthyear"], errors="coerce"
    )
    return updated_df, {
        "missing_birthyear_before": missing_birthyear_before,
        "birthyear_imputed": int(missing_birthyear.sum()),
        "missing_age_after": int(updated_df["age"].isna().sum()),
    }


def select_final_columns(
    df: pd.DataFrame, final_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project the raw frame to the stable downstream preprocessing schema."""
    df_work = df.copy()
    if "partyname" not in df_work.columns:
        df_work["partyname"] = pd.NA
    selected_columns = [column for column in final_columns if column in df_work.columns]
    final_column_set = set(final_columns)
    updated_df = df_work.loc[:, selected_columns].copy()
    return updated_df, {
        "columns_kept": int(len(updated_df.columns)),
        "columns_dropped": int(sum(1 for c in df.columns if c not in final_column_set)),
    }


def drop_incomplete_profiles(
    df: pd.DataFrame, required_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows that lack the metadata required for profiling labels."""
    complete_rows = df.loc[:, required_columns].notna().all(axis=1)
    updated_df = df.loc[complete_rows].copy()
    details = {
        f"rows_with_missing_{column}": int(df[column].isna().sum())
        for column in required_columns
    }
    return updated_df, details


def fill_missing_party_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backfill missing party names from the dominant name for each party code."""
    updated_df = df.copy()
    missing_before = int(updated_df["partyname"].isna().sum())
    party_key = updated_df["party"].astype(str).str.strip().str.upper()
    known = updated_df.loc[
        updated_df["partyname"].notna(), ["party", "partyname"]
    ].copy()
    known["party_key"] = known["party"].astype(str).str.strip().str.upper()
    mapping = (
        known.groupby(["party_key", "partyname"])
        .size()
        .reset_index(name="count")
        .sort_values(["party_key", "count"], ascending=[True, False])
        .drop_duplicates("party_key")
        .set_index("party_key")["partyname"]
    )
    missing_partyname = updated_df["partyname"].isna()
    fills = party_key.map(mapping)
    filled_rows = missing_partyname & fills.notna()
    updated_df.loc[filled_rows, "partyname"] = fills[filled_rows]
    return updated_df, {
        "missing_partyname_before": missing_before,
        "filled_partyname_rows": int(filled_rows.sum()),
        "missing_partyname_after": int(updated_df["partyname"].isna().sum()),
    }


def apply_manual_party_fixes(
    df: pd.DataFrame, manual_party_fixes: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize reviewed full-name party-code edge cases."""
    updated_df = df.copy()
    party_before = updated_df["party"].copy()
    partyname = updated_df["partyname"].astype(str).str.strip().str.lower()
    for party_name, abbreviation in manual_party_fixes.items():
        updated_df.loc[partyname == party_name, "party"] = abbreviation
    return updated_df, {
        "party_codes_normalized": int(updated_df["party"].ne(party_before).sum())
    }


def _normalized_party_codes(df: pd.DataFrame) -> pd.Series:
    """Return stripped party codes with empty placeholder values masked out."""
    if "party" not in df.columns:
        raise KeyError("Expected a 'party' column when normalizing party codes.")
    party_codes = df["party"].fillna("").astype(str).str.strip()
    missing_codes = party_codes.eq("") | party_codes.str.casefold().isin(
        {"nan", "none", "<na>"}
    )
    return party_codes.mask(missing_codes)


def _party_word_percentages(df: pd.DataFrame) -> pd.Series:
    """Compute each party's share of words within one language view."""
    if "word_count" not in df.columns:
        raise KeyError("Expected a 'word_count' column when checking small parties.")

    party_codes = _normalized_party_codes(df)
    valid_party_rows = party_codes.notna()
    if not valid_party_rows.any():
        return pd.Series(dtype=float)

    word_counts = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)
    party_words = (
        word_counts.loc[valid_party_rows]
        .groupby(party_codes.loc[valid_party_rows], sort=True)
        .sum()
    )
    total_party_words = float(party_words.sum())
    if total_party_words <= 0:
        return pd.Series(0.0, index=party_words.index, dtype=float)
    return (party_words / total_party_words * 100).round(6)


def remove_small_parties_by_word_pct(
    df: pd.DataFrame,
    min_word_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove parties whose word share is below the configured threshold."""
    party_word_pct = _party_word_percentages(df)
    removed_parties = sorted(
        party_word_pct.loc[party_word_pct < float(min_word_pct)].index.astype(str)
    )
    details: dict[str, Any] = {
        "small_party_min_word_pct": float(min_word_pct),
        "small_parties_removed": removed_parties,
    }
    if not removed_parties:
        return df.copy(), details

    party_codes = _normalized_party_codes(df)
    remove_mask = party_codes.isin(removed_parties)
    removed_df = df.loc[remove_mask]
    updated_df = df.loc[~remove_mask].copy()
    details.update(
        {
            "small_party_speeches_removed": int(
                removed_df["id_speech"].nunique()
                if "id_speech" in removed_df.columns
                else remove_mask.sum()
            ),
            "small_party_words_removed": int(
                pd.to_numeric(removed_df["word_count"], errors="coerce").fillna(0).sum()
            ),
            "largest_removed_word_pct": round(
                float(party_word_pct.loc[removed_parties].max()), 6
            ),
        }
    )
    return updated_df, details


def filter_language_view_by_small_party_pct(
    df: pd.DataFrame,
    language: str,
    min_word_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the small-party threshold inside one language-specific corpus view."""
    language_rows = df["language"] == language
    language_df = df.loc[language_rows].copy()
    filtered_df, details = remove_small_parties_by_word_pct(language_df, min_word_pct)
    return filtered_df, {
        "language": language,
        "language_rows": int(language_rows.sum()),
        **details,
    }


def _find_party_switching_authors(df: pd.DataFrame) -> pd.Series:
    """Identify authors who have speech rows under more than one party code."""
    sort_cols = [
        col for col in ["id_person", "date", "time", "id_speech"] if col in df.columns
    ]
    ordered_df = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    ordered_df = ordered_df.assign(party_clean=_normalized_party_codes(ordered_df))
    ordered_df = ordered_df[ordered_df["party_clean"].notna()].copy()
    if ordered_df.empty:
        return pd.Series(dtype=object)
    party_sequence = ordered_df.groupby("id_person")["party_clean"].agg(
        lambda values: list(dict.fromkeys(values.tolist()))
    )
    return party_sequence[party_sequence.apply(len) > 1]


def apply_manual_birthyear_fixes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply hand-reviewed birthyear corrections before age is derived."""
    updated_df = df.copy()
    birthyear_fixed = 0
    name_column = next(
        (c for c in MANUAL_BIRTHYEAR_NAME_COLUMNS if c in updated_df.columns), None
    )
    if name_column is not None:
        for normalized_name, (
            from_date,
            correct_birthyear,
        ) in MANUAL_BIRTHYEAR_FIXES.items():
            mask = updated_df[name_column].map(
                lambda v: normalize_redaction_text(v).casefold()
            ).eq(normalized_name) & (updated_df["date"] >= from_date)
            updated_df.loc[mask, "birthyear"] = correct_birthyear
            birthyear_fixed += int(mask.sum())
    return updated_df, {"birthyear_corrections": birthyear_fixed}


def drop_party_switcher_minority_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only majority-party rows for authors who changed party over time."""
    updated_df = df.copy()

    # For authors who spoke under more than one party, keep only the majority-party rows.
    # Ties are broken by keeping the later party (the one they ended up in).
    party_switchers = _find_party_switching_authors(updated_df)
    rows_dropped = 0
    if not party_switchers.empty:
        sort_cols = [
            col
            for col in ["id_person", "date", "time", "id_speech"]
            if col in updated_df.columns
        ]
        ordered_df = updated_df.sort_values(sort_cols) if sort_cols else updated_df
        speech_counts = (
            updated_df[updated_df["id_person"].isin(party_switchers.index)]
            .groupby(["id_person", "party"])
            .size()
            .reset_index(name="n_speeches")
        )
        # Sort by n_speeches ascending then by last appearance date descending so that for ties
        # the later party sorts last — drop_duplicates keeping "last" then gives us the winner.
        last_date = (
            ordered_df[ordered_df["id_person"].isin(party_switchers.index)]
            .groupby(["id_person", "party"])["date"]
            .max()
            .reset_index(name="last_date")
        )
        speech_counts = speech_counts.merge(last_date, on=["id_person", "party"])
        speech_counts = speech_counts.sort_values(
            ["id_person", "n_speeches", "last_date"]
        )
        majority_party = speech_counts.drop_duplicates("id_person", keep="last")[
            ["id_person", "party"]
        ]

        before = len(updated_df)
        switcher_rows = updated_df["id_person"].isin(party_switchers.index)
        keep_mask = (
            ~switcher_rows
            | updated_df.merge(
                majority_party, on=["id_person", "party"], how="left", indicator=True
            )["_merge"]
            .eq("both")
            .values
        )
        updated_df = updated_df.loc[keep_mask].copy()
        rows_dropped = before - len(updated_df)

    return updated_df, {
        "party_switcher_authors": int(len(party_switchers)),
        "party_switcher_rows_dropped": rows_dropped,
    }


# ---------------------------------------------------------------------------
# Text cleanup and redaction
# ---------------------------------------------------------------------------


def fix_missing_spacing_after_punctuation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Insert missing spaces after punctuation without touching decimal numbers."""
    updated_df = df.copy()
    text = updated_df["text"].fillna("").astype(str)
    sentence_spacing_pattern = r"([.!?])(?![\s\d])(?=[A-ZÆØÅ<])"
    clause_spacing_pattern = r"(?<!\d)([,;])(?!\s)(?=[A-ZÆØÅa-zæøå<])"
    colon_spacing_pattern = r"(:)(?![\s\d])(?=[A-ZÆØÅa-zæøå<])"
    sentence_fixes = text.str.count(sentence_spacing_pattern)
    clause_fixes = text.str.count(clause_spacing_pattern) + text.str.count(
        colon_spacing_pattern
    )
    updated_text = text.str.replace(sentence_spacing_pattern, r"\1 ", regex=True)
    updated_text = updated_text.str.replace(clause_spacing_pattern, r"\1 ", regex=True)
    updated_text = updated_text.str.replace(colon_spacing_pattern, r"\1 ", regex=True)
    updated_df["text"] = updated_text
    return updated_df, {
        "rows_with_punctuation_spacing_changes": int(updated_text.ne(text).sum()),
        "sentence_spacing_fixes": int(sentence_fixes.sum()),
        "clause_spacing_fixes": int(clause_fixes.sum()),
    }


def normalize_text_artifacts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove invisible characters, long quote runs, and repeated whitespace."""
    updated_df = df.copy()
    text = updated_df["text"].fillna("").astype(str)
    cleaned_text = (
        text.str.replace(r"[\u00ad\u200b\u200c\u200d\ufeff]", "", regex=True)
        .str.replace("\xa0", " ", regex=False)
        .str.replace("\u202f", " ", regex=False)
        .str.replace(r'"{3,}', "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    updated_df["text"] = cleaned_text
    return updated_df, {
        "rows_changed": int(cleaned_text.ne(text).sum()),
        "soft_hyphens_removed": int(text.str.count("\u00ad").sum()),
        "quote_runs_removed": int(text.str.count(r'"{3,}').sum()),
    }


def build_candidate_fragment(value: str) -> str:
    """Build a redaction regex fragment that requires an uppercase first letter."""
    for index, char in enumerate(value):
        if char.isalpha():
            prefix = re.escape(value[:index])
            first_char = re.escape(char.upper())
            suffix = value[index + 1 :]
            if suffix:
                return prefix + first_char + f"(?i:{re.escape(suffix)})"
            return prefix + first_char
    return re.escape(value)


def build_redaction_regex(values: list[str]) -> tuple[re.Pattern[str] | None, int]:
    """Combine normalized candidates into one boundary-aware redaction regex."""
    # Values are expected to be pre-normalized via normalize_redaction_text.
    if not values:
        return None, 0
    sorted_values = sorted(set(values), key=lambda value: (-len(value), value))
    body = "|".join(build_candidate_fragment(value) for value in sorted_values)
    apostrophes = (
        "\x27\u2018\u2019"  # ASCII apostrophe (U+0027), left/right curly single quotes
    )
    pattern = re.compile(rf"(?<![\w>])(?:{body})(?:[{apostrophes}]s|s)?(?![\w<])")
    return pattern, len(sorted_values)


def redact_text(
    df: pd.DataFrame,
    values: list[str],
    replacement: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace matched name or party mentions with a redaction token."""
    pattern, candidate_count = build_redaction_regex(values)
    if pattern is None:
        return df.copy(), {
            "candidates": 0,
            "rows_changed": 0,
            "matches_replaced": 0,
            "replacement": replacement,
        }
    updated_df = df.copy()
    match_counts = updated_df["text"].fillna("").astype(str).str.count(pattern)
    updated_df["text"] = (
        updated_df["text"]
        .fillna("")
        .astype(str)
        .str.replace(pattern, replacement, regex=True)
    )
    return updated_df, {
        "candidates": candidate_count,
        "rows_changed": int(match_counts.gt(0).sum()),
        "matches_replaced": int(match_counts.sum()),
        "replacement": replacement,
    }


# ---------------------------------------------------------------------------
# Dataset shaping and output views
# ---------------------------------------------------------------------------


def add_length_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute length features from the current text column."""
    updated_df = df.copy()
    text = updated_df["text"].fillna("").astype(str)
    updated_df["word_count"] = text.str.split().str.len()
    updated_df["char_count"] = text.str.len()
    return updated_df, {}


def filter_short_speeches(
    df: pd.DataFrame,
    min_words: int,
    output_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop speeches shorter than the threshold using existing length features."""
    updated_df = df.copy()
    short_rows = updated_df["word_count"] < min_words
    updated_df = updated_df.loc[~short_rows].copy()
    updated_df = updated_df.loc[
        :, [column for column in output_columns if column in updated_df.columns]
    ]
    return updated_df, {
        "min_words": min_words,
        "rows_below_threshold": int(short_rows.sum()),
    }


def drop_duplicate_speeches(
    df: pd.DataFrame, key_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop duplicate speech records using speaker, time, and text identity."""
    key_columns = list(key_columns)
    duplicate_rows = df.duplicated(subset=key_columns, keep="first")
    duplicate_group_rows = df.duplicated(subset=key_columns, keep=False)
    duplicate_groups = int(
        df.loc[duplicate_group_rows, key_columns].drop_duplicates().shape[0]
    )
    updated_df = df.loc[~duplicate_rows].copy()
    return updated_df, {
        "duplicate_key_columns": key_columns,
        "duplicate_groups": duplicate_groups,
        "duplicate_speeches_removed": int(duplicate_rows.sum()),
    }


def keep_supported_languages(
    df: pd.DataFrame, supported_languages: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only Bokmal and Nynorsk rows in the cleaned corpus."""
    supported_rows = df["language"].isin(supported_languages)
    updated_df = df.loc[supported_rows].copy()
    unsupported_labels = sorted(
        df.loc[~supported_rows, "language"].dropna().astype(str).unique().tolist()
    )
    return updated_df, {
        "unsupported_language_rows": int((~supported_rows).sum()),
        "unsupported_language_labels": unsupported_labels,
    }


def keep_majority_language(
    df: pd.DataFrame, tie_break_language: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep each author's dominant language for the majority-language output."""
    language_counts = (
        df.groupby(["id_person", "language"]).size().reset_index(name="n_speeches")
    )
    multilingual_authors = int(
        language_counts.groupby("id_person")["language"].nunique().gt(1).sum()
    )

    author_max = language_counts.groupby("id_person")["n_speeches"].transform("max")
    top_counts = language_counts[language_counts["n_speeches"] == author_max]
    tied_authors = int((top_counts.groupby("id_person")["language"].count() > 1).sum())

    language_counts["tie_break_rank"] = language_counts["language"].eq(
        tie_break_language
    )
    sorted_counts = language_counts.sort_values(
        ["id_person", "n_speeches", "tie_break_rank", "language"],
        ascending=[True, False, False, True],
    )
    keep_pairs = sorted_counts.drop_duplicates("id_person")[["id_person", "language"]]
    updated_df = df.merge(keep_pairs, on=["id_person", "language"], how="inner")
    return updated_df, {
        "multilingual_authors_before": multilingual_authors,
        "majority_language_tie_breaker": tie_break_language,
        "tied_authors_using_tie_breaker": tied_authors,
    }


def combine_language_filtered_views(
    df: pd.DataFrame, language_views: list[pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recombine rows retained by the language-specific party filters."""
    kept_arrays = [
        view_df.index.to_numpy() for view_df in language_views if len(view_df)
    ]
    kept_index = (
        pd.Index(np.concatenate(kept_arrays)).drop_duplicates()
        if kept_arrays
        else pd.Index([])
    )
    updated_df = df.loc[df.index.isin(kept_index)].copy()
    return updated_df, {"language_filtered_rows": int(len(updated_df))}


def run_preprocessing(config: PreprocessingConfig | None = None) -> dict[str, Any]:
    """Run preprocessing, write cleaned corpora, and return audit tables."""
    # Imported lazily so helper tests and small scripts can import this module
    # without triggering Matplotlib setup through dataset_stats.
    from data_pipeline.dataset_stats import save_dataset_stats

    cfg = config or resolve_preprocessing_config()
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.input_path}")

    stage_summary_rows: list[dict[str, Any]] = []
    stage_detail_rows: list[dict[str, Any]] = []
    output_reports: list[dict[str, Any]] = []

    df = load_raw_data(cfg.input_path)
    record_stage(
        stage_summary_rows,
        stage_detail_rows,
        "loaded raw data",
        df,
        df,
        {"columns": int(df.shape[1])},
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "clean dates",
        clean_dates,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        f"keep elections from {cfg.min_election_year} onward",
        keep_supported_elections,
        cfg.min_election_year,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "drop procedural speeches",
        drop_procedural_rows,
        cfg.procedural_status_column,
        cfg.procedural_column,
        cfg.procedural_excluded_statuses,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "apply manual birthyear fixes",
        apply_manual_birthyear_fixes,
    )
    df = run_stage(
        stage_summary_rows, stage_detail_rows, df, "add age feature", add_age_feature
    )

    # Collect before select_final_columns — name_original is dropped by that stage
    name_redaction_values = collect_name_redaction_values(
        df,
        cfg.name_redaction_columns,
        cfg.name_part_min_length,
    )
    record_stage(
        stage_summary_rows,
        stage_detail_rows,
        "prepare name redaction values",
        df,
        df,
        {"name_candidates": len(name_redaction_values)},
    )

    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "select final columns",
        select_final_columns,
        cfg.final_columns,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "fill missing party names",
        fill_missing_party_names,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "apply manual party fixes",
        apply_manual_party_fixes,
        cfg.manual_party_fixes,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "drop minority-party rows for switchers",
        drop_party_switcher_minority_rows,
        report_rows_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "drop incomplete profiles",
        drop_incomplete_profiles,
        cfg.profile_required_columns,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "normalize text artifacts",
        normalize_text_artifacts,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "fix missing spacing after punctuation",
        fix_missing_spacing_after_punctuation,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "drop duplicate speeches",
        drop_duplicate_speeches,
        cfg.duplicate_key_columns,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "redact speaker names",
        redact_text,
        name_redaction_values,
        cfg.name_redaction_token,
    )

    party_redaction_values = collect_party_redaction_values(
        df,
        cfg.party_name_redaction_aliases_by_code,
        cfg.party_codes_for_redaction,
    )
    redaction_party_codes = [
        value
        for value in party_redaction_values
        if value in cfg.party_codes_for_redaction
    ]
    record_stage(
        stage_summary_rows,
        stage_detail_rows,
        "prepare party redaction values",
        df,
        df,
        {
            "party_candidates": len(party_redaction_values),
            "redaction_party_codes": redaction_party_codes,
        },
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "redact party names",
        redact_text,
        party_redaction_values,
        cfg.party_redaction_token,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "add post-redaction length features",
        add_length_features,
    )
    speech_length_threshold_path = (
        cfg.dataset_results_dir / "figures" / "speech_length_threshold_prefilter.png"
    )
    threshold_figure_info = save_speech_length_threshold_figure(
        df,
        speech_length_threshold_path,
        threshold=cfg.min_words,
        quantile=0.99,
    )
    record_stage(
        stage_summary_rows,
        stage_detail_rows,
        "write pre-filter speech length threshold figure",
        df,
        df,
        threshold_figure_info,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        f"keep speeches with at least {cfg.min_words} words",
        filter_short_speeches,
        cfg.min_words,
        cfg.output_columns,
        report_rows_removed=True,
        report_authors_removed=True,
    )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "keep supported language rows",
        keep_supported_languages,
        cfg.supported_languages,
        report_rows_removed=True,
        report_authors_removed=True,
    )

    language_output_dfs: dict[str, pd.DataFrame] = {}
    for language_output in cfg.language_outputs:
        language_output_dfs[language_output.name] = run_stage(
            stage_summary_rows,
            stage_detail_rows,
            df,
            f"filter {language_output.name} by small-party threshold",
            filter_language_view_by_small_party_pct,
            language_output.language,
            cfg.small_party_min_word_pct,
            report_rows_removed=True,
            report_authors_removed=True,
        )
    df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "combine language-filtered views",
        combine_language_filtered_views,
        list(language_output_dfs.values()),
        report_rows_removed=True,
        report_authors_removed=True,
    )

    majority_df = run_stage(
        stage_summary_rows,
        stage_detail_rows,
        df,
        "keep each author's majority language",
        keep_majority_language,
        cfg.majority_language_tie_breaker,
        report_rows_removed=True,
    )

    output_paths = [cfg.majority_output_path] + [
        language_output.output_path for language_output in cfg.language_outputs
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    majority_df.to_csv(cfg.majority_output_path, index=False, encoding="utf-8")
    record_output(output_reports, "majority", majority_df, cfg.majority_output_path)
    for language_output in cfg.language_outputs:
        language_df = language_output_dfs[language_output.name]
        language_df.to_csv(language_output.output_path, index=False, encoding="utf-8")
        record_output(
            output_reports,
            language_output.name,
            language_df,
            language_output.output_path,
        )

    stage_summary_df = pd.DataFrame(stage_summary_rows)
    for column in ["rows_after", "rows_removed", "authors_after", "authors_removed"]:
        if column in stage_summary_df.columns:
            stage_summary_df[column] = stage_summary_df[column].astype("Int64")
    # astype(object) converts Int64 pd.NA to None so fillna("") can replace it without
    # raising TypeError: Invalid value '' for dtype Int64.
    stage_summary_df = stage_summary_df.astype(object).fillna("")
    stage_detail_tables: dict[str, pd.DataFrame] = {}
    ordered_steps = list(
        dict.fromkeys(
            [row["step"] for row in stage_detail_rows]
            + [row["step"] for row in stage_summary_rows]
        )
    )
    for step in ordered_steps:
        step_rows = [row for row in stage_detail_rows if row["step"] == step]
        if step_rows:
            stage_detail_tables[step] = pd.DataFrame(step_rows)[
                ["metric", "value"]
            ].fillna("")
    output_report_df = pd.DataFrame(output_reports)
    tables = save_dataset_stats(df, cfg.dataset_results_dir, cfg.project_root)
    language_dataset_tables = {}
    for language_output in cfg.language_outputs:
        if language_output.dataset_results_dir is not None:
            language_dataset_tables[language_output.name] = save_dataset_stats(
                language_output_dfs[language_output.name],
                language_output.dataset_results_dir,
                cfg.project_root,
            )

    return {
        "config": cfg,
        "preprocessed_df": df,
        "majority_df": majority_df,
        "language_output_dfs": language_output_dfs,
        "name_redaction_values": name_redaction_values,
        "party_redaction_values": party_redaction_values,
        "stage_summary_df": stage_summary_df,
        "stage_detail_tables": stage_detail_tables,
        "output_report_df": output_report_df,
        "dataset_tables": tables,
        "language_dataset_tables": language_dataset_tables,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an intentional preprocessing run."""
    parser = argparse.ArgumentParser(
        description="Run raw NPD preprocessing with a TOML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Preprocessing config TOML. Defaults to "
            "data_pipeline/configs/preprocessing/default.toml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the requested preprocessing config and run the pipeline."""
    args = _parse_args()
    run_preprocessing(resolve_preprocessing_config(config_path=args.config))


if __name__ == "__main__":
    main()
