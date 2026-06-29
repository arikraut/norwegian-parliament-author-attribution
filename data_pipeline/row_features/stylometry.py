"""spaCy/TextDescriptives stylometry feature extraction."""

from __future__ import annotations

import shutil
import sys
import warnings
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd
import spacy

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import stopwordsiso as stopwords_iso
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*no word vectors.*", category=UserWarning)

NORWEGIAN_LETTERS = list("abcdefghijklmnopqrstuvwxyzæøå")
STYLOMETRY_FEATURE_FAMILIES = (
    "counts",
    "token_length",
    "sentence_length",
    "readability",
    "dependency_distance",
    "pos_proportions",
    "function_words",
    "character_distribution",
    "other",
)
SUBSTITUTION_COUNTER_KEYS = (
    "missing_value_substitutions",
    "nan_substitutions",
    "inf_substitutions",
    "non_numeric_substitutions",
)


def _progress_bar_width() -> int:
    """Return a conservative tqdm width that avoids wrapped progress lines."""
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    return max(40, min(100, terminal_width - 1))


def _make_stylometry_progress_bar(total: int, desc: str) -> tqdm:
    """Create a capped-width progress bar for one feature-generation role."""
    return tqdm(
        total=total,
        desc=desc,
        unit="speech",
        ncols=_progress_bar_width(),
        dynamic_ncols=False,
        mininterval=2.0,
        maxinterval=10.0,
        leave=True,
        disable=not sys.stderr.isatty(),
    )


def _load_stylometry_pipeline(spacy_model: str):
    """Load a spaCy pipeline with TextDescriptives attached for stylometry extraction."""
    import textdescriptives  # noqa: F401

    nlp = spacy.load(spacy_model)
    nlp.add_pipe("textdescriptives/all")
    return nlp


def coerce_to_text(value: object) -> str:
    """Return value as-is if it is a string, otherwise an empty string."""
    if isinstance(value, str):
        return value
    return ""


def _safe_float(
    value: object,
    quality_counters: Counter | None = None,
    *,
    family: str | None = None,
) -> float:
    """Cast value to float, returning 0.0 for invalid values and counting substitutions."""
    if value is None:
        if quality_counters is not None:
            quality_counters["missing_value_substitutions"] += 1
        _count_family_substitution(quality_counters, family)
        return 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        if quality_counters is not None:
            quality_counters["non_numeric_substitutions"] += 1
        _count_family_substitution(quality_counters, family)
        return 0.0

    if np.isnan(numeric_value):
        if quality_counters is not None:
            quality_counters["nan_substitutions"] += 1
        _count_family_substitution(quality_counters, family)
        return 0.0
    if not np.isfinite(numeric_value):
        if quality_counters is not None:
            quality_counters["inf_substitutions"] += 1
        _count_family_substitution(quality_counters, family)
        return 0.0
    return numeric_value


def stylometry_feature_family(feature_name: str) -> str:
    """Map a stylometry column name to its feature family group based on its prefix."""
    if feature_name.startswith("counts_"):
        return "counts"
    if feature_name.startswith("token_length_"):
        return "token_length"
    if feature_name.startswith("sentence_length_"):
        return "sentence_length"
    if feature_name.startswith("readability_"):
        return "readability"
    if feature_name.startswith("dep_dist_"):
        return "dependency_distance"
    if feature_name.startswith("pos_"):
        return "pos_proportions"
    if feature_name.startswith("fw_"):
        return "function_words"
    if feature_name.startswith("charfreq_") or feature_name.startswith("prop_"):
        return "character_distribution"
    return "other"


def _count_family_substitution(
    quality_counters: Counter | None, family: str | None
) -> None:
    """Increment the per-family substitution counter used by quality reports."""
    if quality_counters is None or not family:
        return
    quality_counters[f"total_substitutions__{family}"] += 1


def _total_base_substitutions(quality_counters: Counter) -> int:
    """Sum the four base substitution counter keys into a single total."""
    return int(
        sum(int(quality_counters.get(key, 0)) for key in SUBSTITUTION_COUNTER_KEYS)
    )


def _mapping_from_extension(value: object) -> Mapping[str, object] | None:
    """Extract a dict-like mapping from a spaCy extension object, plain dict, or named tuple."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "_asdict"):
        mapped = value._asdict()
        if isinstance(mapped, Mapping):
            return mapped
    if hasattr(value, "__dict__"):
        mapped = vars(value)
        if isinstance(mapped, Mapping):
            return mapped
    return None


def _safe_metric_lookup(
    source: object,
    key: str,
    *,
    quality_counters: Counter | None = None,
    family: str | None = None,
    alternative_keys: tuple[str, ...] = (),
) -> float:
    """Read a TextDescriptives metric from its accepted extension key names."""
    mapping = _mapping_from_extension(source)
    candidates = (key, *alternative_keys)

    if mapping is not None:
        for candidate in candidates:
            if candidate in mapping:
                return _safe_float(mapping[candidate], quality_counters, family=family)
        return _safe_float(None, quality_counters, family=family)

    for candidate in candidates:
        if hasattr(source, candidate):
            return _safe_float(
                getattr(source, candidate), quality_counters, family=family
            )

    return _safe_float(None, quality_counters, family=family)


def load_bokmal_function_words() -> list[str]:
    """Return the Norwegian stopword inventory used for Bokmal stylometry."""
    sw_no = stopwords_iso.stopwords("no") or set()
    return sorted(set(sw_no))


def function_word_features_from_doc(doc, function_words: list[str]) -> dict:
    """Return per-function-word frequency proportions from one spaCy doc."""
    tokens = [t.text.lower() for t in doc if not t.is_space]
    n_tokens = len(tokens)
    feats: dict = {}

    if n_tokens == 0:
        for w in function_words:
            feats[f"fw_{w.replace(' ', '_')}"] = 0.0
        feats["fw_total_prop"] = 0.0
        return feats

    token_counts = Counter(tokens)
    fw_count = 0
    for w in function_words:
        count = token_counts.get(w, 0)
        feats[f"fw_{w.replace(' ', '_')}"] = count / n_tokens
        fw_count += count

    feats["fw_total_prop"] = fw_count / n_tokens
    return feats


def char_distribution_features(text: str) -> dict:
    """Return character-frequency and proportion features from a raw text string."""
    if not isinstance(text, str) or not text:
        feats = {f"charfreq_{ch}": 0.0 for ch in NORWEGIAN_LETTERS}
        feats.update(
            {"prop_alpha": 0.0, "prop_digit": 0.0, "prop_space": 0.0, "prop_other": 0.0}
        )
        return feats

    n_total = len(text)
    lower_text = text.lower()

    feats = {
        f"charfreq_{ch}": lower_text.count(ch) / n_total for ch in NORWEGIAN_LETTERS
    }
    feats["prop_alpha"] = sum(1 for c in text if c.isalpha()) / n_total
    feats["prop_digit"] = sum(1 for c in text if c.isdigit()) / n_total
    feats["prop_space"] = sum(1 for c in text if c.isspace()) / n_total
    feats["prop_other"] = (
        1.0 - feats["prop_alpha"] - feats["prop_digit"] - feats["prop_space"]
    )
    return feats


def textdescriptives_features_from_doc(
    doc, quality_counters: Counter | None = None
) -> dict:
    """Extract TextDescriptives metrics from one spaCy doc."""
    feats: dict = {}

    if hasattr(doc._, "counts") and doc._.counts is not None:
        counts = doc._.counts
        for attr in ("n_tokens", "n_unique_tokens", "n_characters", "n_sentences"):
            feats[f"counts_{attr}"] = _safe_metric_lookup(
                counts,
                attr,
                quality_counters=quality_counters,
                family="counts",
            )

    if hasattr(doc._, "token_length") and doc._.token_length is not None:
        tl = doc._.token_length
        for attr in ("mean", "median", "std"):
            feats[f"token_length_{attr}"] = _safe_metric_lookup(
                tl,
                f"token_length_{attr}",
                quality_counters=quality_counters,
                family="token_length",
                alternative_keys=(attr,),
            )

    if hasattr(doc._, "sentence_length") and doc._.sentence_length is not None:
        sl = doc._.sentence_length
        for attr in ("mean", "median", "std"):
            feats[f"sentence_length_{attr}"] = _safe_metric_lookup(
                sl,
                f"sentence_length_{attr}",
                quality_counters=quality_counters,
                family="sentence_length",
                alternative_keys=(attr,),
            )

    if hasattr(doc._, "readability") and doc._.readability is not None:
        r = doc._.readability
        for attr in ("flesch_reading_ease", "flesch_kincaid_grade", "lix", "rix"):
            feats[f"readability_{attr}"] = _safe_metric_lookup(
                r,
                attr,
                quality_counters=quality_counters,
                family="readability",
            )

    if hasattr(doc._, "dependency_distance") and doc._.dependency_distance is not None:
        dd = doc._.dependency_distance
        dependency_distance_keys = {
            "mean_dependency_distance": ("dependency_distance_mean",),
            "proportion_adjacent_dependency_relation": (
                "prop_adjacent_dependency_relation_mean",
            ),
        }
        for output_name, candidates in dependency_distance_keys.items():
            feats[f"dep_dist_{output_name}"] = _safe_metric_lookup(
                dd,
                candidates[0],
                quality_counters=quality_counters,
                family="dependency_distance",
                alternative_keys=(output_name,),
            )

    if hasattr(doc._, "pos_proportions") and doc._.pos_proportions is not None:
        pp = doc._.pos_proportions
        if isinstance(pp, dict):
            for k, v in pp.items():
                feats[k] = _safe_float(v, quality_counters)
        elif hasattr(pp, "__dict__"):
            for k, v in vars(pp).items():
                feats[f"pos_{k}"] = _safe_float(v, quality_counters)
        elif hasattr(pp, "_asdict"):
            for k, v in pp._asdict().items():
                feats[f"pos_{k}"] = _safe_float(v, quality_counters)

    return feats


def extract_spacy_stylometry_from_df(
    df: pd.DataFrame,
    nlp,
    function_words: list[str],
    text_col: str = "text",
    batch_size: int = 64,
    desc: str = "stylometry",
    return_quality: bool = False,
    progress_bar: tqdm | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Run spaCy + TextDescriptives over a DataFrame and return a feature DataFrame."""
    texts = [coerce_to_text(value) for value in df[text_col].tolist()]
    quality_counters: Counter = Counter()
    records = []
    local_progress_bar = progress_bar is None
    progress = progress_bar or _make_stylometry_progress_bar(len(df), desc)
    if progress_bar is not None:
        progress.set_description_str(desc, refresh=False)
    try:
        for text, doc in zip(texts, nlp.pipe(texts, batch_size=batch_size)):
            feats: dict = {}
            feats.update(
                textdescriptives_features_from_doc(
                    doc, quality_counters=quality_counters
                )
            )
            feats.update(function_word_features_from_doc(doc, function_words))
            feats.update(char_distribution_features(text))
            records.append(feats)
            progress.update(1)
    finally:
        if local_progress_bar:
            progress.close()

    stylo_df = pd.DataFrame(records)
    for col in ("id_speech", "id_person"):
        if col in df.columns:
            stylo_df.insert(0, col, df[col].values)
    if not return_quality:
        return stylo_df

    feature_cols = [
        col for col in stylo_df.columns if col not in {"id_speech", "id_person"}
    ]
    numeric = (
        stylo_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        if feature_cols
        else pd.DataFrame()
    )
    nonfinite_cells = (
        int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
        if not numeric.empty
        else 0
    )
    quality_summary: dict[str, int | float | str] = {
        "split": desc,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "missing_value_substitutions": int(
            quality_counters.get("missing_value_substitutions", 0)
        ),
        "nan_substitutions": int(quality_counters.get("nan_substitutions", 0)),
        "inf_substitutions": int(quality_counters.get("inf_substitutions", 0)),
        "non_numeric_substitutions": int(
            quality_counters.get("non_numeric_substitutions", 0)
        ),
        "total_substitutions": _total_base_substitutions(quality_counters),
        "nonfinite_output_cells": nonfinite_cells,
    }
    for family in STYLOMETRY_FEATURE_FAMILIES:
        quality_summary[f"total_substitutions__{family}"] = int(
            quality_counters.get(f"total_substitutions__{family}", 0)
        )
    return stylo_df, quality_summary
