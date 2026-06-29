"""BERT profiling signal extractor.

Applies saved BERT profiling models to all attribution speeches in one run,
writing probability matrices in the same format as the SVM profiling signal
extractor so train_stacked_attribution.py can consume them unchanged.

Output layout
-------------
  data/splits/{attribution_split}/materialized_features/{materialization_name}/
      manifest.json
      fold_01/ … fold_N/                 one unit per CV fold
      final_test/                        one unit for the held-out test set
          row_order/
              train_rows.csv
              {val|test}_rows.csv
          matrices/
              X_train_profiling.npz               all targets combined (soft probs)
              X_{val|test}_profiling.npz
              X_train_profiling_hard.npz           all targets combined (one-hot argmax)
              X_{val|test}_profiling_hard.npz
              X_train_profiling_{target}.npz       per-target soft probs
              X_{val|test}_profiling_{target}.npz
              X_train_profiling_hard_{target}.npz  per-target one-hot argmax
              X_{val|test}_profiling_hard_{target}.npz

Config keys
-----------
[data]
    splits_dir    string
    artifacts_dir string

[source]
    attribution_split_name       string  — split whose speeches to predict on
    profiling_split_name         string  — split the models were trained on
    profiling_experiment_name    string  — experiment name for artifact path
    profiling_seed               int
    targets                      list    — must match the training targets
    materialization_name         string  — output materialization name

[model]
    family          string  must be "bert_profiling"
    pretrained_name string
    max_length      int     (default: 512)

[truncation]
    pairs = [[head, tail]]

[age_bins]            (optional — required only if "age_bin" in targets)
    edges  list[int]
    labels list[str]

[[party_axes]]        (optional — one entry per axis scheme)
    name   string
    <label> list[str]
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        p
        for p in [_THIS_FILE.parent, *_THIS_FILE.parents]
        if (p / "pyproject.toml").exists() and (p / "data").exists()
    ),
    _THIS_FILE.parents[3],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils import (  # noqa: E402
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.bert.bert_utils import (  # noqa: E402
    SpeechDataset,
    TruncationConfig,
    add_derived_columns,
    build_party_axes,
)

SUPPORTED_MODEL_FAMILY = "bert_profiling"


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT profiling signal extractor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


# ── Config ─────────────────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _validate_config(cfg: dict[str, Any]) -> None:
    family = str(cfg.get("model", {}).get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(f"Unsupported model.family: {family!r}.")
    source = cfg.get("source", {})
    for key in ("attribution_split_name", "profiling_split_name",
                "profiling_experiment_name", "profiling_seed",
                "targets", "materialization_name"):
        if not source.get(key) and source.get(key) != 0:
            raise ValueError(f"source.{key} must be defined.")
    if not isinstance(source["targets"], list) or not source["targets"]:
        raise ValueError("source.targets must be a non-empty list.")


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict_probs(
    model: Any,
    texts: list[str],
    tokenizer: Any,
    truncation: TruncationConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return (n, n_classes) float32 softmax probabilities."""
    model.eval()
    encodings = [SpeechDataset._encode(t, tokenizer, truncation) for t in texts]
    all_probs: list[np.ndarray] = []
    for start in range(0, len(encodings), batch_size):
        batch_enc = encodings[start : start + batch_size]
        input_ids = torch.tensor(
            [e["input_ids"] for e in batch_enc], dtype=torch.long
        ).to(device)
        attention_mask = torch.tensor(
            [e["attention_mask"] for e in batch_enc], dtype=torch.long
        ).to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        all_probs.append(probs)
    if not all_probs:
        return np.zeros((0, model.config.num_labels), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def _hard_label_matrix(probs: np.ndarray) -> np.ndarray:
    """Convert softmax probabilities to one-hot argmax matrix."""
    n, k = probs.shape
    hard = np.zeros((n, k), dtype=np.float32)
    hard[np.arange(n), np.argmax(probs, axis=1)] = 1.0
    return hard


# ── Writing helpers ────────────────────────────────────────────────────────────


def _write_rows_csv(path: Path, df: pd.DataFrame, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_idx":   range(len(df)),
        "id_speech": df["id_speech"].values,
        "id_person": df["id_person"].values,
        "role":      role,
    }).to_csv(path, index=False)


def _write_unit(
    unit_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_role: str,
    targets: list[str],
    models: dict[str, Any],
    label_maps: dict[str, dict[str, str]],
    tokenizer: Any,
    truncation: TruncationConfig,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> dict[str, Any]:
    """Run inference for one unit (fold or final_test) and write all matrices."""
    matrices_dir = unit_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    _write_rows_csv(unit_dir / "row_order" / "train_rows.csv",     train_df, "train")
    _write_rows_csv(unit_dir / "row_order" / f"{eval_role}_rows.csv", eval_df, eval_role)

    train_texts = train_df["text"].tolist()
    eval_texts  = eval_df["text"].tolist()

    train_per_target: dict[str, np.ndarray] = {}
    eval_per_target:  dict[str, np.ndarray] = {}
    column_names: list[str] = []

    for target in targets:
        train_probs = _predict_probs(models[target], train_texts, tokenizer, truncation, batch_size, device)
        eval_probs  = _predict_probs(models[target], eval_texts,  tokenizer, truncation, batch_size, device)

        train_per_target[target] = train_probs
        eval_per_target[target]  = eval_probs

        lmap = label_maps.get(target, {})
        classes = [lmap.get(str(i), str(i)) for i in range(train_probs.shape[1])]
        column_names.extend(f"{target}_{cls}" for cls in classes)

        sparse.save_npz(matrices_dir / f"X_train_profiling_{target}.npz",
                        sparse.csr_matrix(train_probs))
        sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling_{target}.npz",
                        sparse.csr_matrix(eval_probs))
        sparse.save_npz(matrices_dir / f"X_train_profiling_hard_{target}.npz",
                        sparse.csr_matrix(_hard_label_matrix(train_probs)))
        sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling_hard_{target}.npz",
                        sparse.csr_matrix(_hard_label_matrix(eval_probs)))

        if show_progress:
            print(f"  {target}: train={train_probs.shape}  {eval_role}={eval_probs.shape}")

    # Combined matrices (all targets horizontally stacked).
    train_combined      = np.hstack([train_per_target[t] for t in targets])
    eval_combined       = np.hstack([eval_per_target[t]  for t in targets])
    train_hard_combined = np.hstack([_hard_label_matrix(train_per_target[t]) for t in targets])
    eval_hard_combined  = np.hstack([_hard_label_matrix(eval_per_target[t])  for t in targets])

    sparse.save_npz(matrices_dir / "X_train_profiling.npz",            sparse.csr_matrix(train_combined))
    sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling.npz",     sparse.csr_matrix(eval_combined))
    sparse.save_npz(matrices_dir / "X_train_profiling_hard.npz",       sparse.csr_matrix(train_hard_combined))
    sparse.save_npz(matrices_dir / f"X_{eval_role}_profiling_hard.npz", sparse.csr_matrix(eval_hard_combined))

    return {
        "train_rows":     len(train_df),
        "eval_rows":      len(eval_df),
        "profiling_dim":  train_combined.shape[1],
        "column_names":   column_names,
    }


# ── Public API ─────────────────────────────────────────────────────────────────


def run_extraction(
    config_path: Path,
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Apply trained BERT profiling models to all attribution speeches.

    Writes one unit per CV fold (train + val) and one final_test unit
    (train + test), each containing combined and per-target probability
    matrices plus one-hot hard-label variants.
    """
    config_path  = config_path.resolve()
    project_root = find_project_root(config_path.parent, Path.cwd(), _THIS_FILE.parent)
    cfg          = _load_config(config_path)
    _validate_config(cfg)

    source_cfg             = cfg["source"]
    attribution_split_name = str(source_cfg["attribution_split_name"])
    profiling_split_name   = str(source_cfg["profiling_split_name"])
    profiling_exp_name     = str(source_cfg["profiling_experiment_name"])
    profiling_seed         = int(source_cfg["profiling_seed"])
    targets: list[str]     = [str(t) for t in source_cfg["targets"]]
    materialization_name   = str(source_cfg["materialization_name"])

    age_bin_cfg    = cfg.get("age_bins", {})
    age_bin_edges  = [int(e) for e in age_bin_cfg.get("edges", [0, 50, 200])]
    age_bin_labels = [str(l) for l in age_bin_cfg.get("labels", ["<50", "50+"])]
    party_axes     = build_party_axes(cfg.get("party_axes", []))

    model_cfg       = cfg["model"]
    pretrained_name = str(model_cfg.get("pretrained_name", "NbAiLab/nb-bert-base"))
    max_length      = int(model_cfg.get("max_length", 512))

    head_tokens, tail_tokens = [int(v) for v in cfg["truncation"]["pairs"][0]]
    truncation = TruncationConfig(head_tokens=head_tokens, tail_tokens=tail_tokens, max_length=max_length)

    data_cfg       = cfg.get("data", {})
    splits_root    = resolve_project_path(project_root, data_cfg.get("splits_dir", "data/splits"))
    artifacts_root = resolve_project_path(project_root, data_cfg.get("artifacts_dir", "models/artifacts/profiling"))

    attribution_split_dir = splits_root / attribution_split_name
    mat_root = attribution_split_dir / "materialized_features" / materialization_name
    mat_root.mkdir(parents=True, exist_ok=True)

    artifacts_final_dir = (
        artifacts_root / profiling_split_name / profiling_exp_name
        / f"seed_{profiling_seed}_final" / "final"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if show_progress:
        print(f"Device            : {device}")
        print(f"Attribution split : {attribution_split_name}")
        print(f"Targets           : {targets}")
        print(f"Materialization   : {materialization_name}")

    # Load models and label maps once.
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)
    models: dict[str, Any] = {}
    label_maps: dict[str, dict[str, str]] = {}
    for target in targets:
        model_path = artifacts_final_dir / target / "saved_model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found for '{target}': {model_path}\n"
                "Run bert_profiling.py --mode final first."
            )
        models[target] = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(device)
        label_map_path = model_path / "label_map.json"
        if label_map_path.exists():
            with label_map_path.open(encoding="utf-8") as fh:
                label_maps[target] = json.load(fh)
        if show_progress:
            print(f"  Loaded: {target}")

    batch_size = 32

    # Load and prepare the attribution corpus. `train.csv` holds only the outer
    # train rows; `test.csv` is the held-out outer test set and must never be
    # mixed into any unit's training data.
    corpus = pd.read_csv(attribution_split_dir / "corpus" / "train.csv", dtype={"id_speech": str})
    corpus = add_derived_columns(corpus, age_bin_edges, age_bin_labels, party_axes)
    folds  = pd.read_csv(attribution_split_dir / "memberships" / "folds.csv", dtype={"id_speech": str})
    fold_ids = sorted(folds["fold_id"].unique())

    unit_metas: list[dict[str, Any]] = []

    # ── CV fold units ──────────────────────────────────────────────────────────
    for fold_id in fold_ids:
        if show_progress:
            print(f"\n[{fold_id}]")
        fold_rows = folds[folds["fold_id"] == fold_id]
        val_ids   = set(fold_rows.loc[fold_rows["fold_role"] == "val", "id_speech"])
        val_df    = corpus[corpus["id_speech"].isin(val_ids)].reset_index(drop=True)
        train_df  = corpus[~corpus["id_speech"].isin(val_ids)].reset_index(drop=True)

        meta = _write_unit(
            unit_dir      = mat_root / fold_id,
            train_df      = train_df,
            eval_df       = val_df,
            eval_role     = "val",
            targets       = targets,
            models        = models,
            label_maps    = label_maps,
            tokenizer     = tokenizer,
            truncation    = truncation,
            batch_size    = batch_size,
            device        = device,
            show_progress = show_progress,
        )
        unit_metas.append({"unit_id": fold_id, "eval_role": "val", **meta})

    # ── Final test unit ────────────────────────────────────────────────────────
    if show_progress:
        print("\n[final_test]")
    test_corpus  = pd.read_csv(attribution_split_dir / "corpus" / "test.csv", dtype={"id_speech": str})

    meta = _write_unit(
        unit_dir      = mat_root / "final_test",
        train_df      = corpus,
        eval_df       = test_corpus,
        eval_role     = "test",
        targets       = targets,
        models        = models,
        label_maps    = label_maps,
        tokenizer     = tokenizer,
        truncation    = truncation,
        batch_size    = batch_size,
        device        = device,
        show_progress = show_progress,
    )
    unit_metas.append({"unit_id": "final_test", "eval_role": "test", **meta})

    manifest: dict[str, Any] = {
        "run_type":             "bert_profiling_extraction",
        "attribution_split":    attribution_split_name,
        "materialization_name": materialization_name,
        "targets":              targets,
        "units":                unit_metas,
        "config_path":          relative_to_project(project_root, config_path),
    }
    write_json(mat_root / "manifest.json", manifest)

    if show_progress:
        print(f"\nExtraction complete: {relative_to_project(project_root, mat_root)}")

    return manifest


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args        = _parse_args()
    config_path = args.config.resolve()
    manifest    = run_extraction(config_path, show_progress=not args.no_progress)
    print(f"Materialization : {manifest['materialization_name']}")
    print(f"Units written   : {len(manifest['units'])}")


if __name__ == "__main__":
    main()
