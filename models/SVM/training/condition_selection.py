"""Shared condition-local candidate selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import pandas as pd


CandidateT = TypeVar("CandidateT")


def select_candidates_by_condition(
    summary_df: pd.DataFrame,
    candidates: Sequence[CandidateT],
    *,
    selection_metric: str,
    sort_cols: list[str],
    ascending: list[bool],
    candidate_id: Callable[[CandidateT], str],
    condition_id: Callable[[CandidateT], str],
    condition_label: Callable[[CandidateT], str],
    selected_payload: Callable[[CandidateT, str, dict[str, Any]], dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select the best candidate independently within each declared condition."""
    candidates_by_id = {candidate_id(candidate): candidate for candidate in candidates}
    condition_order = list(dict.fromkeys(condition_id(candidate) for candidate in candidates))

    condition_rows: list[dict[str, Any]] = []
    selected_payloads: list[dict[str, Any]] = []
    for current_condition_id in condition_order:
        condition_df = summary_df[summary_df["condition_id"] == current_condition_id]
        selected_row = condition_df.sort_values(
            by=sort_cols,
            ascending=ascending,
            kind="stable",
        ).iloc[0]
        selected_candidate_id = str(selected_row["candidate_id"])
        candidate = candidates_by_id[selected_candidate_id]
        selected_summary = selected_row.to_dict()
        condition_row = {
            "condition_id": current_condition_id,
            "condition_label": condition_label(candidate),
            "selected_candidate_id": selected_candidate_id,
            "selection_metric": selection_metric,
        }
        for key, value in selected_summary.items():
            if key not in condition_row and key != "candidate_id":
                condition_row[key] = value
        condition_rows.append(condition_row)
        selected_payloads.append(
            selected_payload(candidate, selection_metric, selected_summary)
        )

    return pd.DataFrame(condition_rows), selected_payloads
