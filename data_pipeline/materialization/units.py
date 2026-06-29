"""Materialization unit selection."""

from __future__ import annotations

import pandas as pd

from data_pipeline.materialization.config import MaterializationUnit


def _select_units(
    config: dict,
    outer_membership: pd.DataFrame,
    fold_membership: pd.DataFrame,
    *,
    split_strategy: str = "",
) -> list[MaterializationUnit]:
    """Translate folds.selector into materialization units."""
    selector = str(config.get("folds", {}).get("selector", "all")).lower()

    if selector == "all":
        if fold_membership.empty or "fold_id" not in fold_membership.columns:
            raise ValueError(
                "Materialization selector 'all' requires memberships/folds.csv with fold_id values."
            )
        units = []
        for fold_id in sorted(fold_membership["fold_id"].dropna().astype(str).unique()):
            membership = fold_membership[
                fold_membership["fold_id"].astype(str) == fold_id
            ].copy()
            units.append(
                MaterializationUnit(
                    unit_id=fold_id, eval_role="val", membership=membership
                )
            )
        return units

    if selector == "final":
        if "outer_role" not in outer_membership.columns:
            raise ValueError(
                "Materialization selector 'final' requires outer memberships with an outer_role column."
            )
        membership = outer_membership[
            outer_membership["outer_role"].isin(["train", "test"])
        ].copy()
        if membership.empty:
            raise ValueError(
                "Materialization selector 'final' requires outer train/test memberships."
            )
        if str(split_strategy).lower() == "election_based":
            test_elections = sorted(
                membership.loc[membership["outer_role"] == "test", "election"]
                .dropna()
                .astype(int)
                .unique()
            )
            suffix = (
                "_".join(str(year) for year in test_elections)
                if test_elections
                else "unknown"
            )
            unit_id = f"final_test_{suffix}"
        else:
            unit_id = "final_test"
        return [
            MaterializationUnit(unit_id=unit_id, eval_role="test", membership=membership)
        ]

    raise ValueError(f"Unsupported folds.selector: {selector}")
