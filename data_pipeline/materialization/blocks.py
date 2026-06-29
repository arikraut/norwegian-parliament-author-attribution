"""Feature-block matrix and label writers for materialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from data_pipeline.materialization.constants import _PREFERRED_TARGETS
from data_pipeline.utils import write_json


def _fit_stylometry_scaler(
    config: dict, train_values: np.ndarray, eval_values: np.ndarray
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit StandardScaler on train rows only, then transform both train and eval."""
    stylo_cfg = config.get("stylometry", {})
    scaler_name = str(stylo_cfg.get("scaler", "standard")).lower()
    if scaler_name != "standard":
        raise ValueError(f"Unsupported stylometry scaler: {scaler_name}")

    scaler = StandardScaler(
        with_mean=bool(stylo_cfg.get("with_mean", True)),
        with_std=bool(stylo_cfg.get("with_std", True)),
    )
    if train_values.shape[1] == 0:
        return scaler, train_values.copy(), eval_values.copy()
    return scaler, scaler.fit_transform(train_values), scaler.transform(eval_values)


def _make_row_order(df_role: pd.DataFrame, unit_id: str, role: str) -> pd.DataFrame:
    """Build a row-order frame mapping matrix row indices back to speech IDs."""
    row_order = df_role[["id_speech", "id_person"]].copy()
    row_order.insert(0, "row_idx", np.arange(len(row_order), dtype=int))
    row_order["fold_id"] = unit_id
    row_order["role"] = role
    for col_name in ["election", "party", "language", "author"]:
        if col_name in df_role.columns and col_name not in row_order.columns:
            row_order[col_name] = df_role[col_name].values
    return row_order


def _save_label_arrays(
    labels_dir: Path, train_df: pd.DataFrame, eval_df: pd.DataFrame, eval_role: str
) -> list[str]:
    """Save raw label arrays for every preferred target present in train and eval."""
    saved_targets: list[str] = []
    for target_name in _PREFERRED_TARGETS:
        if target_name not in train_df.columns or target_name not in eval_df.columns:
            continue
        np.save(labels_dir / f"y_train_{target_name}.npy", train_df[target_name].values)
        np.save(
            labels_dir / f"y_{eval_role}_{target_name}.npy",
            eval_df[target_name].values,
        )
        saved_targets.append(target_name)

    return saved_targets


def _save_sparse_matrix(path: Path, matrix: sparse.spmatrix) -> None:
    """Write a sparse feature matrix to disk in .npz format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, matrix)


def _save_feature_columns(
    unit_dir: Path,
    *,
    char_vectorizer: TfidfVectorizer | None = None,
    word_vectorizer: TfidfVectorizer | None = None,
    stylo_feature_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """Save per-block feature column names to feature_columns.json."""
    payload: dict[str, list[str]] = {}
    if char_vectorizer is not None:
        payload["char"] = char_vectorizer.get_feature_names_out().tolist()
    if word_vectorizer is not None:
        payload["word"] = word_vectorizer.get_feature_names_out().tolist()
    if stylo_feature_cols is not None:
        payload["stylo"] = list(stylo_feature_cols)
    write_json(unit_dir / "feature_columns.json", payload)
    return payload
