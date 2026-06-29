"""
BERT profiling classifier trainer.

Trains one fine-tuned BERT sequence classifier per profiling target
(party, female, age_bin, left_senter_right, …) on the profiling split.

Dev mode:   trains each (target, fold) once for `epochs` epochs, logging
            validation metrics after every epoch. The best epoch per target is
            selected by the mean selection_metric across folds at each epoch.
Final mode: re-trains on all profiling data for the per-target best epoch and
            saves each model for use by the extractor.

Public API
----------
run_dev_search(config_path) -> dict
run_final_training(config_path) -> dict

Config keys used in the config files:
-----------
[experiment]
    name             string
    seed             int       (default 42)
    selection_metric string    (default "macro_f1")

[data]
    splits_dir    string
    results_dir   string
    artifacts_dir string

[source]
    split_name string  — profiling split name
    targets    list    — e.g. ["party", "female", "age_bin", "left_senter_right"]

[age_bins]
    edges  list[int]   e.g. [0, 50, 200]
    labels list[str]   e.g. ["<50", "50+"]

[[party_axes]]           (zero or more entries; each becomes a derived column)
    name   string       — column name added to the corpus (used as a target)
    <label> list[str]   — one key per axis label, value is the list of parties

[model]
    family          string  must be "bert_profiling"
    pretrained_name string
    max_length      int     (default 512)

[truncation]
    pairs = [[head, tail]]  — one pair expected

[training]
    learning_rate             float
    epochs                    int      — train for this many epochs; best epoch selected from curve
    batch_size                int
    weight_decay              float    (default 0.01)
    hidden_dropout            float    (default 0.1)
    attention_dropout         float    (default 0.1)
    max_speeches_per_author   int      (default 0 = disabled) — randomly sample at most this
                                        many speeches per author in the training set, using the
                                        experiment seed for reproducibility
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
    copy_config_outputs,
    find_project_root,
    relative_to_project,
    resolve_project_path,
    write_json,
)
from models.bert.bert_utils import (  # noqa: E402
    SpeechDataset,
    TruncationConfig,
    add_derived_columns,
    build_label_map,
    build_party_axes,
    device_info,
    evaluate,
    make_optimizer,
    sample_per_author,
    set_seed,
    train_one_epoch,
)

SUPPORTED_MODEL_FAMILY = "bert_profiling"


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT profiling classifier trainer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config.")
    parser.add_argument(
        "--mode",
        choices=["dev", "final"],
        default="dev",
        help="'dev' to train and select best epoch, 'final' for full training + model save.",
    )
    return parser.parse_args()


# ── Config loading and validation ──────────────────────────────────────────────


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _validate_config(config: dict[str, Any]) -> None:
    family = str(config.get("model", {}).get("family", "")).strip()
    if family != SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            f"Unsupported model.family: {family!r}. Expected {SUPPORTED_MODEL_FAMILY!r}."
        )
    if not config.get("source", {}).get("split_name"):
        raise ValueError("source.split_name must be defined.")
    targets = config.get("source", {}).get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError("source.targets must be a non-empty list.")
    pairs = config.get("truncation", {}).get("pairs", [])
    if not pairs:
        raise ValueError("truncation.pairs must contain at least one [head, tail] pair.")
    epochs = config.get("training", {}).get("epochs")
    if not epochs:
        raise ValueError("training.epochs must be defined.")


# ── Training helpers ───────────────────────────────────────────────────────────


def _make_dataset(
    df: pd.DataFrame,
    target: str,
    label_map: dict[str, int],
    tokenizer: Any,
    truncation: TruncationConfig,
) -> SpeechDataset:
    labels = [label_map[str(v)] for v in df[target]]
    return SpeechDataset(
        texts=df["text"].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        truncation=truncation,
    )


def _build_model(
    pretrained_name: str,
    num_classes: int,
    hidden_dropout: float,
    attention_dropout: float,
    device: torch.device,
) -> AutoModelForSequenceClassification:
    """Load a classification model with overridden dropout values."""
    model_config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_classes, trust_remote_code=True)
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = hidden_dropout
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = attention_dropout
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_name, config=model_config, ignore_mismatched_sizes=True, trust_remote_code=True
    ).to(device)


def _run_fold(
    *,
    target: str,
    fold_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_map: dict[str, int],
    tokenizer: Any,
    truncation: TruncationConfig,
    epochs: int,
    pretrained_name: str,
    lr: float,
    batch_size: int,
    weight_decay: float,
    hidden_dropout: float,
    attention_dropout: float,
    max_speeches_per_author: int,
    seed: int,
    device: torch.device,
    fold_dir: Path,
) -> list[dict[str, Any]]:
    """Train for `epochs` epochs, evaluate on val after each epoch.

    Returns a list of per-epoch dicts with keys:
        epoch, train_loss, val_accuracy, val_macro_f1, majority_accuracy, majority_macro_f1

    Skips if epoch_metrics.csv already exists AND the saved params.json matches
    the current hyperparameters. Re-runs (overwriting) if any param changed.
    """
    params: dict[str, Any] = {
        "pretrained_name": pretrained_name,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "hidden_dropout": hidden_dropout,
        "attention_dropout": attention_dropout,
        "max_speeches_per_author": max_speeches_per_author,
        "seed": seed,
    }
    metrics_path = fold_dir / "epoch_metrics.csv"
    params_path = fold_dir / "params.json"

    if metrics_path.exists():
        if params_path.exists():
            saved = json.loads(params_path.read_text(encoding="utf-8"))
            if saved == params:
                print(f"    [{fold_id}] already done — skipping")
                return pd.read_csv(metrics_path).to_dict("records")
            changed = [k for k in params if params[k] != saved.get(k)]
            print(f"    [{fold_id}] params changed {changed} — re-running")
        else:
            print(f"    [{fold_id}] no params.json found — re-running")

    set_seed(seed)
    if max_speeches_per_author > 0:
        train_df = sample_per_author(train_df, max_speeches_per_author, seed)
        print(f"    [{fold_id}] sampled to {len(train_df)} speeches (max {max_speeches_per_author}/author)")

    train_dataset = _make_dataset(train_df, target, label_map, tokenizer, truncation)
    val_dataset = _make_dataset(val_df, target, label_map, tokenizer, truncation)

    model = _build_model(pretrained_name, len(label_map), hidden_dropout, attention_dropout, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    optimizer = make_optimizer(model, lr, weight_decay)

    val_labels = val_df[target].astype(str).tolist()
    maj_label = max(set(val_labels), key=val_labels.count)
    majority_accuracy = val_labels.count(maj_label) / len(val_labels)
    majority_macro_f1 = float(
        f1_score(val_labels, [maj_label] * len(val_labels), average="macro", zero_division=0)
    )

    epoch_log: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_accuracy": round(val["accuracy"], 6),
            "val_macro_f1": round(val["macro_f1"], 6),
            "majority_accuracy": round(majority_accuracy, 6),
            "majority_macro_f1": round(majority_macro_f1, 6),
        }
        epoch_log.append(row)
        print(
            f"    [{fold_id}] ep={epoch}/{epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_acc={val['accuracy']:.4f}"
            f"  val_f1={val['macro_f1']:.4f}"
            f"  maj_acc={majority_accuracy:.4f}"
            f"  maj_f1={majority_macro_f1:.4f}"
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(params_path, params)
    pd.DataFrame(epoch_log).to_csv(metrics_path, index=False)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return epoch_log


# ── Per-target output helpers ─────────────────────────────────────────────────


def _write_target_epoch_outputs(
    target: str,
    target_dir: Path,
    fold_epoch_rows: dict[str, list[dict[str, Any]]],
    fold_ids: list[str],
    selection_metric: str,
    majority_stats: dict[str, float],
    config_path: Path | None,
) -> tuple[int, float]:
    """Write epoch_summary.csv and best_epoch.json for one target.

    epoch_summary.csv has one row per epoch with mean/std across folds.
    Returns (best_epoch, best_mean_metric).
    """
    # Determine the number of epochs (min across folds to be safe).
    n_epochs = min(len(rows) for rows in fold_epoch_rows.values() if rows)
    if n_epochs == 0:
        raise ValueError(f"No epoch data for target '{target}'")

    metric_key = f"val_{selection_metric}"
    summary_rows: list[dict[str, Any]] = []
    for ep_idx in range(n_epochs):
        ep = ep_idx + 1
        fold_vals = {
            fid: fold_epoch_rows[fid][ep_idx]
            for fid in fold_ids
            if fid in fold_epoch_rows and ep_idx < len(fold_epoch_rows[fid])
        }
        if not fold_vals:
            continue
        row: dict[str, Any] = {"epoch": ep}
        for col in ("train_loss", "val_accuracy", "val_macro_f1"):
            vals = [v[col] for v in fold_vals.values() if col in v]
            if vals:
                row[f"mean_{col}"] = round(float(np.mean(vals)), 6)
                row[f"std_{col}"] = round(float(np.std(vals)), 6)
        row["majority_accuracy"] = round(majority_stats.get("accuracy", float("nan")), 6)
        row["majority_macro_f1"] = round(majority_stats.get("macro_f1", float("nan")), 6)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(target_dir / "epoch_summary.csv", index=False)

    # Best epoch: highest mean selection metric.
    mean_col = f"mean_{metric_key}"
    best_idx = int(summary_df[mean_col].idxmax())
    best_ep = int(summary_df.loc[best_idx, "epoch"])
    best_score = float(summary_df.loc[best_idx, mean_col])

    best_row = summary_df.loc[best_idx]
    write_json(
        target_dir / "best_epoch.json",
        {
            "best_epoch": best_ep,
            "selection_metric": selection_metric,
            f"mean_val_{selection_metric}": round(best_score, 6),
            "mean_val_accuracy": round(float(best_row.get("mean_val_accuracy", float("nan"))), 6),
            "majority_accuracy": round(majority_stats.get("accuracy", float("nan")), 6),
            "majority_macro_f1": round(majority_stats.get("macro_f1", float("nan")), 6),
        },
    )
    if config_path is not None:
        shutil.copy2(config_path, target_dir / config_path.name)

    return best_ep, best_score


# ── Directory resolution ───────────────────────────────────────────────────────


def _resolve_dirs(
    project_root: Path,
    config: dict[str, Any],
    mode: str,
) -> tuple[Path, Path]:
    data_cfg = config.get("data", {})
    source_cfg = config.get("source", {})
    experiment_cfg = config.get("experiment", {})

    split_name = str(source_cfg["split_name"])
    experiment_name = str(experiment_cfg.get("name", "bert_profiling"))
    seed = int(experiment_cfg.get("seed", 42))

    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/models")
    )
    artifacts_root = resolve_project_path(
        project_root, data_cfg.get("artifacts_dir", "models/artifacts/profiling")
    )

    suffix = "_final" if mode == "final" else ""
    results_dir = results_root / split_name / experiment_name / f"seed_{seed}{suffix}"
    artifacts_dir = artifacts_root / split_name / experiment_name / f"seed_{seed}{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, artifacts_dir


# ── Public API ─────────────────────────────────────────────────────────────────


def run_dev_search(config_path: Path) -> dict[str, Any]:
    """Train each (target, fold) for `epochs` epochs, evaluate after every epoch.

    Selects the best epoch per target by the mean selection_metric across folds
    at each epoch. Writes per-fold epoch_metrics.csv and per-target
    epoch_summary.csv / best_epoch.json. Returns the run manifest.
    """
    config_path = config_path.resolve()
    project_root = find_project_root(
        config_path.parent, Path.cwd(), Path(__file__).resolve().parent
    )
    cfg = _load_config(config_path)
    _validate_config(cfg)

    experiment_cfg = cfg["experiment"]
    experiment_name = str(experiment_cfg["name"])
    seed = int(experiment_cfg.get("seed", 42))
    selection_metric = str(experiment_cfg.get("selection_metric", "macro_f1"))

    source_cfg = cfg["source"]
    split_name = str(source_cfg["split_name"])
    targets: list[str] = [str(t) for t in source_cfg["targets"]]

    age_bin_cfg = cfg.get("age_bins", {})
    age_bin_edges = [int(e) for e in age_bin_cfg.get("edges", [0, 50, 200])]
    age_bin_labels = [str(la) for la in age_bin_cfg.get("labels", ["<50", "50+"])]
    party_axes = build_party_axes(cfg.get("party_axes", []))

    training_cfg = cfg["training"]
    epochs_raw = training_cfg["epochs"]
    # Accept either a single int or a list (list → use the single value or max).
    if isinstance(epochs_raw, list):
        epochs = int(max(epochs_raw))
        if len(epochs_raw) > 1:
            print(f"[WARNING] training.epochs is a list {epochs_raw}; using max={epochs}.")
    else:
        epochs = int(epochs_raw)

    lr = float(training_cfg["learning_rate"])
    batch_size = int(training_cfg.get("batch_size", 16))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    hidden_dropout = float(training_cfg.get("hidden_dropout", 0.1))
    attention_dropout = float(training_cfg.get("attention_dropout", 0.1))
    max_speeches_per_author = int(training_cfg.get("max_speeches_per_author", 0))

    model_cfg = cfg["model"]
    pretrained_name = str(model_cfg["pretrained_name"])
    max_length = int(model_cfg.get("max_length", 512))

    head_tokens, tail_tokens = [int(v) for v in cfg["truncation"]["pairs"][0]]
    truncation = TruncationConfig(
        head_tokens=head_tokens, tail_tokens=tail_tokens, max_length=max_length
    )

    results_dir, artifacts_dir = _resolve_dirs(project_root, cfg, "dev")

    splits_root = resolve_project_path(
        project_root, cfg.get("data", {}).get("splits_dir", "data/splits")
    )
    split_dir = splits_root / split_name
    corpus = pd.read_csv(split_dir / "corpus" / "train.csv", dtype={"id_speech": str})
    folds_df = pd.read_csv(split_dir / "memberships" / "folds.csv", dtype={"id_speech": str})

    corpus = add_derived_columns(corpus, age_bin_edges, age_bin_labels, party_axes)
    fold_ids = sorted(folds_df["fold_id"].unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device                   : {device}")
    print(f"Experiment               : {experiment_name}")
    print(f"Targets                  : {targets}")
    print(f"Epochs                   : {epochs}")
    print(f"Dropout                  : hidden={hidden_dropout}  attention={attention_dropout}")
    print(f"Max speeches per author  : {max_speeches_per_author or 'unlimited'}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)

    best_epochs: dict[str, int] = {}

    for target in targets:
        print(f"\n=== Target: {target} ===")

        label_map = build_label_map(corpus, target)
        print(f"  Classes ({len(label_map)}): {list(label_map.keys())[:10]}")

        # Pre-split fold DataFrames (NaN rows dropped per target).
        fold_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for fold_id in fold_ids:
            fold_rows = folds_df[folds_df["fold_id"] == fold_id]
            train_ids = set(fold_rows.loc[fold_rows["fold_role"] == "train", "id_speech"])
            val_ids = set(fold_rows.loc[fold_rows["fold_role"] == "val", "id_speech"])
            train_df = (
                corpus[corpus["id_speech"].isin(train_ids)]
                .dropna(subset=[target])
                .reset_index(drop=True)
            )
            val_df = (
                corpus[corpus["id_speech"].isin(val_ids)]
                .dropna(subset=[target])
                .reset_index(drop=True)
            )
            fold_data[fold_id] = (train_df, val_df)
            print(f"  [{fold_id}] train={len(train_df)}  val={len(val_df)}")

        # Majority baseline (averaged over folds).
        maj_acc_list, maj_f1_list = [], []
        for fold_id in fold_ids:
            _, val_df = fold_data[fold_id]
            val_labels = val_df[target].astype(str).tolist()
            if val_labels:
                maj = max(set(val_labels), key=val_labels.count)
                maj_acc_list.append(val_labels.count(maj) / len(val_labels))
                maj_f1_list.append(
                    float(
                        f1_score(
                            val_labels, [maj] * len(val_labels), average="macro", zero_division=0
                        )
                    )
                )
        majority_stats = {
            "accuracy": float(np.mean(maj_acc_list)) if maj_acc_list else float("nan"),
            "macro_f1": float(np.mean(maj_f1_list)) if maj_f1_list else float("nan"),
        }

        # Train each fold once for all epochs.
        fold_epoch_rows: dict[str, list[dict[str, Any]]] = {}
        for fold_id in fold_ids:
            train_df, val_df = fold_data[fold_id]
            fold_epoch_rows[fold_id] = _run_fold(
                target=target,
                fold_id=fold_id,
                train_df=train_df,
                val_df=val_df,
                label_map=label_map,
                tokenizer=tokenizer,
                truncation=truncation,
                epochs=epochs,
                pretrained_name=pretrained_name,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                max_speeches_per_author=max_speeches_per_author,
                seed=seed,
                device=device,
                fold_dir=results_dir / target / fold_id,
            )
        # Write epoch_summary.csv + best_epoch.json, select best epoch.
        best_ep, best_score = _write_target_epoch_outputs(
            target=target,
            target_dir=results_dir / target,
            fold_epoch_rows=fold_epoch_rows,
            fold_ids=fold_ids,
            selection_metric=selection_metric,
            majority_stats=majority_stats,
            config_path=config_path,
        )
        best_epochs[target] = best_ep
        print(
            f"  [{target}] best_epoch={best_ep}"
            f"  mean_val_{selection_metric}={best_score:.4f}"
            f"  majority_{selection_metric}={majority_stats.get(selection_metric, float('nan')):.4f}"
        )

    print(f"\nBest epochs : {best_epochs}")
    print(f"Outputs     : {results_dir}")

    return {
        "run_type": "profiling_dev_search",
        "experiment_name": experiment_name,
        "seed": seed,
        "targets": targets,
        "best_epochs": best_epochs,
        "results_dir": relative_to_project(project_root, results_dir),
    }


def run_final_training(config_path: Path) -> dict[str, Any]:
    """Train one BERT per target on all profiling training data and save models.

    Each model is saved under:
        {artifacts_dir}/final/{target}/saved_model/

    Returns the run manifest dict.
    """
    config_path = config_path.resolve()
    project_root = find_project_root(
        config_path.parent, Path.cwd(), Path(__file__).resolve().parent
    )
    cfg = _load_config(config_path)
    _validate_config(cfg)

    experiment_cfg = cfg["experiment"]
    experiment_name = str(experiment_cfg["name"])
    seed = int(experiment_cfg.get("seed", 42))

    source_cfg = cfg["source"]
    split_name = str(source_cfg["split_name"])
    targets: list[str] = [str(t) for t in source_cfg["targets"]]

    age_bin_cfg = cfg.get("age_bins", {})
    age_bin_edges = [int(e) for e in age_bin_cfg.get("edges", [0, 50, 200])]
    age_bin_labels = [str(la) for la in age_bin_cfg.get("labels", ["<50", "50+"])]
    party_axes = build_party_axes(cfg.get("party_axes", []))

    training_cfg = cfg["training"]
    lr = float(training_cfg["learning_rate"])
    batch_size = int(training_cfg.get("batch_size", 16))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    hidden_dropout = float(training_cfg.get("hidden_dropout", 0.1))
    attention_dropout = float(training_cfg.get("attention_dropout", 0.1))
    max_speeches_per_author = int(training_cfg.get("max_speeches_per_author", 0))

    model_cfg = cfg["model"]
    pretrained_name = str(model_cfg["pretrained_name"])
    max_length = int(model_cfg.get("max_length", 512))

    head_tokens, tail_tokens = [int(v) for v in cfg["truncation"]["pairs"][0]]
    truncation = TruncationConfig(
        head_tokens=head_tokens, tail_tokens=tail_tokens, max_length=max_length
    )

    results_dir, artifacts_dir = _resolve_dirs(project_root, cfg, "final")
    copy_config_outputs(config_path, artifacts_dir / config_path.name)

    # Load best epoch per target from each target's best_epoch.json.
    data_cfg = cfg.get("data", {})
    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/models")
    )
    dev_run_dir = results_root / split_name / experiment_name / f"seed_{seed}"
    best_epochs: dict[str, int] = {}
    for target in targets:
        best_epoch_path = dev_run_dir / target / "best_epoch.json"
        if not best_epoch_path.exists():
            raise FileNotFoundError(
                f"best_epoch.json not found for target '{target}': {best_epoch_path}\n"
                "Run dev search first (--mode dev)."
            )
        best_epochs[target] = int(
            json.loads(best_epoch_path.read_text(encoding="utf-8"))["best_epoch"]
        )

    splits_root = resolve_project_path(
        project_root, cfg.get("data", {}).get("splits_dir", "data/splits")
    )
    split_dir = splits_root / split_name
    corpus = pd.read_csv(split_dir / "corpus" / "train.csv", dtype={"id_speech": str})
    corpus = add_derived_columns(corpus, age_bin_edges, age_bin_labels, party_axes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    print(f"Experiment     : {experiment_name}")
    print(f"Training on    : {len(corpus)} speeches")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)

    targets_summary: dict[str, Any] = {}
    for target in targets:
        best_epoch = best_epochs[target]
        print(f"\n=== Target: {target}  best_epoch={best_epoch} ===")

        label_map = build_label_map(corpus, target)
        train_df = corpus.dropna(subset=[target]).reset_index(drop=True)
        if max_speeches_per_author > 0:
            train_df = sample_per_author(train_df, max_speeches_per_author, seed)
            print(f"  Sampled to {len(train_df)} speeches (max {max_speeches_per_author}/author)")
        train_dataset = _make_dataset(train_df, target, label_map, tokenizer, truncation)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        set_seed(seed)
        model = _build_model(
            pretrained_name, len(label_map), hidden_dropout, attention_dropout, device
        )
        optimizer = make_optimizer(model, lr, weight_decay)

        for epoch in range(1, best_epoch + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"  epoch {epoch}/{best_epoch}  loss={loss:.4f}")

        save_dir = artifacts_dir / "final" / target / "saved_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        label_map_inv = {str(idx): label for label, idx in label_map.items()}
        write_json(save_dir / "label_map.json", label_map_inv)
        print(f"  Model saved: {save_dir}")

        target_manifest: dict[str, Any] = {
            "run_type": "profiling_final_training",
            "experiment_name": experiment_name,
            "target": target,
            "seed": seed,
            "best_epoch": best_epoch,
            "num_classes": len(label_map),
            "label_map": label_map,
            "max_speeches_per_author": max_speeches_per_author,
            "hardware": device_info(device),
            "model_path": relative_to_project(project_root, save_dir),
            "artifacts_dir": relative_to_project(project_root, artifacts_dir),
            "config_path": relative_to_project(project_root, config_path),
        }
        target_results_dir = results_dir / target
        target_results_dir.mkdir(parents=True, exist_ok=True)
        write_json(target_results_dir / "manifest.json", target_manifest)

        targets_summary[target] = {
            "best_epoch": best_epoch,
            "num_classes": len(label_map),
            "model_path": relative_to_project(project_root, save_dir),
        }

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nOutputs written to: {results_dir}")

    manifest = {
        "experiment_name": experiment_name,
        "targets_summary": targets_summary,
        "results_dir": str(results_dir),
    }
    return manifest


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()

    if args.mode == "dev":
        manifest = run_dev_search(config_path)
        print(f"\nFinished profiling dev search : {manifest['experiment_name']}")
        print(f"Best epochs   : {manifest['best_epochs']}")
        print(f"Results       : {manifest['results_dir']}")
    else:
        manifest = run_final_training(config_path)
        print(f"\nFinished profiling final training : {manifest['experiment_name']}")
        for target, summary in manifest["targets_summary"].items():
            print(f"  {target}: best_epoch={summary['best_epoch']}  model={summary['model_path']}")


if __name__ == "__main__":
    main()
