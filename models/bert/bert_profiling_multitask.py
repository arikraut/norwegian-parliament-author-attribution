"""
BERT multi-task profiling classifier trainer.

Trains ONE shared BERT encoder with separate classification heads for each
profiling target simultaneously. The combined cross-entropy loss is
back-propagated through the shared encoder, letting it learn representations
useful across all demographic prediction tasks at once.

Missing labels (NaN for a target on a given speech) are represented as -100
so PyTorch's CrossEntropyLoss ignores them — all speeches contribute to tasks
where they do have labels.

Architecture:
    BERT encoder  →  pooler output  →  head_task1  →  logits_1
                                    →  head_task2  →  logits_2
                                    →  …
    loss = weighted mean of per-task cross-entropy losses

Dev mode:   trains each fold once for `epochs` epochs, logging per-task
            validation metrics after every epoch. Best epoch selected by
            mean selection_metric averaged across tasks and folds.
Final mode: re-trains on all profiling data for the best epoch, then exports
            each task as a standalone AutoModelForSequenceClassification for
            compatibility with bert_profiling_extractor.py.

Public API
----------
run_dev_search(config_path)     -> dict
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
    split_name string
    targets    list    — e.g. ["female", "age_bin", "left_right"]

[age_bins]
    edges  list[int]
    labels list[str]

[[party_axes]]
    name    string
    <label> list[str]

[model]
    family          string  must be "bert_multitask_profiling"
    pretrained_name string
    max_length      int     (default 512)

[truncation]
    pairs = [[head, tail]]   — one pair used

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

[task_weights]           (optional — all default to 1.0)
    <target_name> = float
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
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

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
    make_optimizer,
    sample_per_author,
    set_seed,
)

SUPPORTED_MODEL_FAMILY = "bert_multitask_profiling"


# ── Model ─────────────────────────────────────────────────────────────────────


class MultiTaskBERT(nn.Module):
    """Shared BERT encoder with per-task linear classification heads."""

    def __init__(self, encoder: nn.Module, task_heads: dict[str, nn.Linear]) -> None:
        super().__init__()
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str,
        num_labels_per_task: dict[str, int],
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> "MultiTaskBERT":
        encoder_config = AutoConfig.from_pretrained(pretrained_name, trust_remote_code=True)
        if hasattr(encoder_config, "hidden_dropout_prob"):
            encoder_config.hidden_dropout_prob = hidden_dropout
        if hasattr(encoder_config, "attention_probs_dropout_prob"):
            encoder_config.attention_probs_dropout_prob = attention_dropout
        encoder = AutoModel.from_pretrained(pretrained_name, config=encoder_config, trust_remote_code=True)
        hidden_size = encoder.config.hidden_size
        heads = {
            target: nn.Linear(hidden_size, n)
            for target, n in num_labels_per_task.items()
        }
        return cls(encoder, heads)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_labels: dict[str, torch.Tensor] | None = None,
        task_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output if available (standard BERT), otherwise fall back to
        # the CLS token from last_hidden_state (some models omit the pooler).
        pooled = (
            output.pooler_output
            if hasattr(output, "pooler_output") and output.pooler_output is not None
            else output.last_hidden_state[:, 0, :]
        )  # (batch, hidden_size)

        logits = {name: head(pooled) for name, head in self.task_heads.items()}

        loss: torch.Tensor | None = None
        if task_labels:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            weighted: list[torch.Tensor] = []
            for name, task_logits in logits.items():
                if name in task_labels:
                    w = (task_weights or {}).get(name, 1.0)
                    weighted.append(w * loss_fn(task_logits, task_labels[name]))
            if weighted:
                loss = torch.stack(weighted).mean()

        return loss, logits


# ── Dataset ───────────────────────────────────────────────────────────────────


class MultiTaskSpeechDataset(Dataset):
    """One item per speech; labels_per_task uses -100 to mark missing entries."""

    def __init__(
        self,
        texts: list[str],
        labels_per_task: dict[str, list[int]],
        tokenizer: Any,
        truncation: TruncationConfig,
    ) -> None:
        self.labels = labels_per_task
        self.encodings = [
            SpeechDataset._encode(text, tokenizer, truncation) for text in texts
        ]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(self.encodings[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encodings[idx]["attention_mask"], dtype=torch.long
            ),
        }
        for target, labels in self.labels.items():
            item[f"label_{target}"] = torch.tensor(labels[idx], dtype=torch.long)
        return item


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT multi-task profiling trainer.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
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
    if not isinstance(targets, list) or len(targets) < 2:
        raise ValueError("source.targets must list at least two tasks for multi-task learning.")
    pairs = config.get("truncation", {}).get("pairs", [])
    if not pairs:
        raise ValueError("truncation.pairs must contain at least one [head, tail] pair.")
    if not config.get("training", {}).get("epochs"):
        raise ValueError("training.epochs must be defined.")


# ── Training helpers ───────────────────────────────────────────────────────────


def _make_multitask_dataset(
    df: pd.DataFrame,
    targets: list[str],
    label_maps: dict[str, dict[str, int]],
    tokenizer: Any,
    truncation: TruncationConfig,
) -> MultiTaskSpeechDataset:
    """Build a MultiTaskSpeechDataset; missing labels become -100 (ignored in loss)."""
    texts = df["text"].tolist()
    labels_per_task: dict[str, list[int]] = {}
    for target in targets:
        lmap = label_maps[target]
        task_labels: list[int] = []
        col = df[target] if target in df.columns else pd.Series([None] * len(df))
        for v in col:
            s = str(v) if not pd.isna(v) else None  # type: ignore[arg-type]
            task_labels.append(lmap[s] if s is not None and s in lmap else -100)
        labels_per_task[target] = task_labels
    return MultiTaskSpeechDataset(
        texts=texts,
        labels_per_task=labels_per_task,
        tokenizer=tokenizer,
        truncation=truncation,
    )


def _train_one_epoch_multitask(
    model: MultiTaskBERT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    targets: list[str],
    task_weights: dict[str, float] | None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        task_labels = {
            t: batch[f"label_{t}"].to(device) for t in targets if f"label_{t}" in batch
        }
        loss, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            task_labels=task_labels,
            task_weights=task_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _evaluate_multitask(
    model: MultiTaskBERT,
    loader: DataLoader,
    device: torch.device,
    targets: list[str],
) -> dict[str, dict[str, float]]:
    """Per-task accuracy and macro-F1 (ignoring -100 labels)."""
    model.eval()
    all_labels: dict[str, list[int]] = {t: [] for t in targets}
    all_preds: dict[str, list[int]] = {t: [] for t in targets}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        for target in targets:
            key = f"label_{target}"
            if key not in batch:
                continue
            labels_np = batch[key].cpu().numpy()
            preds_np = logits[target].argmax(dim=-1).cpu().numpy()
            mask = labels_np != -100
            all_labels[target].extend(labels_np[mask].tolist())
            all_preds[target].extend(preds_np[mask].tolist())

    metrics: dict[str, dict[str, float]] = {}
    for target in targets:
        if all_labels[target]:
            metrics[target] = {
                "accuracy": float(accuracy_score(all_labels[target], all_preds[target])),
                "macro_f1": float(
                    f1_score(
                        all_labels[target],
                        all_preds[target],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }

    return metrics


def _run_fold_multitask(
    *,
    fold_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: list[str],
    label_maps: dict[str, dict[str, int]],
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
    task_weights: dict[str, float] | None,
    seed: int,
    device: torch.device,
    fold_dir: Path,
) -> list[dict[str, Any]]:
    """Train the multi-task model for `epochs` epochs, evaluate after every epoch.

    Returns a list of per-epoch dicts with:
        epoch, train_loss,
        <target>_val_accuracy, <target>_val_macro_f1,
        <target>_majority_accuracy, <target>_majority_macro_f1  for each target.

    Skips if epoch_metrics.csv exists AND params.json matches current hyperparameters.
    Re-runs (overwriting) if any param changed.
    """
    params: dict[str, Any] = {
        "pretrained_name": pretrained_name,
        "targets": targets,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "hidden_dropout": hidden_dropout,
        "attention_dropout": attention_dropout,
        "max_speeches_per_author": max_speeches_per_author,
        "task_weights": task_weights,
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

    num_labels_per_task = {t: len(label_maps[t]) for t in targets}
    model = MultiTaskBERT.from_pretrained(
        pretrained_name, num_labels_per_task, hidden_dropout, attention_dropout
    ).to(device)

    train_dataset = _make_multitask_dataset(train_df, targets, label_maps, tokenizer, truncation)
    val_dataset = _make_multitask_dataset(val_df, targets, label_maps, tokenizer, truncation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    optimizer = make_optimizer(model, lr, weight_decay)

    # Per-fold majority baseline, computed once per target from this fold's val set.
    fold_majority: dict[str, dict[str, float]] = {}
    for target in targets:
        if target in val_df.columns:
            val_labels = val_df[target].dropna().astype(str).tolist()
            if val_labels:
                maj = max(set(val_labels), key=val_labels.count)
                fold_majority[target] = {
                    "accuracy": val_labels.count(maj) / len(val_labels),
                    "macro_f1": float(
                        f1_score(val_labels, [maj] * len(val_labels), average="macro", zero_division=0)
                    ),
                }

    epoch_log: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch_multitask(
            model, train_loader, optimizer, device, targets, task_weights
        )
        per_task = _evaluate_multitask(model, val_loader, device, targets)

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
        }
        for target in targets:
            if target in per_task:
                row[f"{target}_val_accuracy"] = round(per_task[target]["accuracy"], 6)
                row[f"{target}_val_macro_f1"] = round(per_task[target]["macro_f1"], 6)
            row[f"{target}_majority_accuracy"] = round(
                fold_majority.get(target, {}).get("accuracy", float("nan")), 6
            )
            row[f"{target}_majority_macro_f1"] = round(
                fold_majority.get(target, {}).get("macro_f1", float("nan")), 6
            )

        epoch_log.append(row)

        task_summary = "  ".join(
            f"{t}: acc={per_task[t]['accuracy']:.3f} (maj={fold_majority.get(t, {}).get('accuracy', float('nan')):.3f})"
            f" f1={per_task[t]['macro_f1']:.3f} (maj={fold_majority.get(t, {}).get('macro_f1', float('nan')):.3f})"
            for t in targets
            if t in per_task
        )
        print(
            f"    [{fold_id}] ep={epoch}/{epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  {task_summary}"
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(params_path, params)
    pd.DataFrame(epoch_log).to_csv(metrics_path, index=False)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return epoch_log


# ── Per-target output helpers ─────────────────────────────────────────────────


def _write_epoch_outputs(
    results_dir: Path,
    fold_epoch_rows: dict[str, list[dict[str, Any]]],
    fold_ids: list[str],
    targets: list[str],
    selection_metric: str,
    majority_stats: dict[str, dict[str, float]],
    config_path: Path | None,
) -> tuple[int, float]:
    """Write epoch_summary.csv + best_epoch.json for the multi-task run.

    Best epoch is chosen by mean of per-task selection_metric across tasks and folds.
    Returns (best_epoch, combined_mean_metric).
    """
    n_epochs = min(len(rows) for rows in fold_epoch_rows.values() if rows)
    if n_epochs == 0:
        raise ValueError("No epoch data found.")

    summary_rows: list[dict[str, Any]] = []
    for ep_idx in range(n_epochs):
        ep = ep_idx + 1
        row: dict[str, Any] = {"epoch": ep}

        # Per-task metrics.
        task_sel_vals: list[float] = []
        for target in targets:
            for metric in ("val_accuracy", "val_macro_f1"):
                col = f"{target}_{metric}"
                vals = [
                    fold_epoch_rows[fid][ep_idx].get(col, float("nan"))
                    for fid in fold_ids
                    if fid in fold_epoch_rows
                ]
                row[f"mean_{col}"] = round(float(np.nanmean(vals)), 6)
                row[f"std_{col}"] = round(float(np.nanstd(vals)), 6)
            sel_col = f"mean_{target}_val_{selection_metric}"
            if sel_col in row:
                task_sel_vals.append(row[sel_col])

            row[f"majority_{target}_accuracy"] = round(
                majority_stats.get(target, {}).get("accuracy", float("nan")), 6
            )
            row[f"majority_{target}_macro_f1"] = round(
                majority_stats.get(target, {}).get("macro_f1", float("nan")), 6
            )

        row[f"combined_mean_val_{selection_metric}"] = round(
            float(np.mean(task_sel_vals)) if task_sel_vals else float("nan"), 6
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "epoch_summary.csv", index=False)

    combined_col = f"combined_mean_val_{selection_metric}"
    best_idx = int(summary_df[combined_col].idxmax())
    best_ep = int(summary_df.loc[best_idx, "epoch"])
    best_score = float(summary_df.loc[best_idx, combined_col])

    write_json(
        results_dir / "best_epoch.json",
        {
            "best_epoch": best_ep,
            "selection_metric": selection_metric,
            f"combined_mean_val_{selection_metric}": round(best_score, 6),
            "per_task_at_best": {
                t: {
                    "mean_val_accuracy": round(
                        float(summary_df.loc[best_idx, f"mean_{t}_val_accuracy"]), 6
                    ),
                    f"mean_val_{selection_metric}": round(
                        float(summary_df.loc[best_idx, f"mean_{t}_val_{selection_metric}"]), 6
                    ),
                    "majority_accuracy": round(
                        majority_stats.get(t, {}).get("accuracy", float("nan")), 6
                    ),
                    "majority_macro_f1": round(
                        majority_stats.get(t, {}).get("macro_f1", float("nan")), 6
                    ),
                }
                for t in targets
                if f"mean_{t}_val_accuracy" in summary_df.columns
            },
        },
    )
    if config_path is not None:
        shutil.copy2(config_path, results_dir / config_path.name)

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
    experiment_name = str(experiment_cfg.get("name", "bert_multitask_profiling"))
    seed = int(experiment_cfg.get("seed", 42))

    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/models")
    )
    artifacts_root = resolve_project_path(
        project_root, data_cfg.get("artifacts_dir", "models/artifacts/profiling")
    )

    targets = sorted(str(t) for t in source_cfg.get("targets", []))
    targets_key = "+".join(targets) if targets else "no_targets"

    suffix = "_final" if mode == "final" else ""
    seed_dir = f"seed_{seed}{suffix}"
    results_dir = results_root / split_name / experiment_name / seed_dir / targets_key
    artifacts_dir = artifacts_root / split_name / experiment_name / seed_dir / targets_key
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, artifacts_dir


# ── Export helper ──────────────────────────────────────────────────────────────


def _export_task_model(
    multitask_model: MultiTaskBERT,
    target: str,
    pretrained_name: str,
    num_classes: int,
    save_dir: Path,
    tokenizer: Any,
    label_map: dict[str, int],
) -> None:
    """Export one task head as a standalone AutoModelForSequenceClassification.

    Copies the shared encoder weights + task head into a format compatible with
    bert_profiling_extractor.py. Assumes a BERT-family architecture where
    AutoModelForSequenceClassification exposes .base_model and .classifier.
    """
    export_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name, num_labels=num_classes, ignore_mismatched_sizes=True, trust_remote_code=True
    )
    export_model.base_model.load_state_dict(multitask_model.encoder.state_dict())
    task_head = multitask_model.task_heads[target]
    export_model.classifier.weight.data.copy_(task_head.weight.data)
    export_model.classifier.bias.data.copy_(task_head.bias.data)

    save_dir.mkdir(parents=True, exist_ok=True)
    export_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    label_map_inv = {str(idx): label for label, idx in label_map.items()}
    write_json(save_dir / "label_map.json", label_map_inv)


# ── Public API ─────────────────────────────────────────────────────────────────


def run_dev_search(config_path: Path) -> dict[str, Any]:
    """Train one MultiTaskBERT per fold for `epochs` epochs, evaluate after every epoch.

    Selects a single best epoch by mean selection_metric averaged across all tasks
    and folds. Returns the run manifest dict.
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
    task_weights: dict[str, float] | None = cfg.get("task_weights") or None

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
    label_maps = {t: build_label_map(corpus, t) for t in targets}
    fold_ids = sorted(folds_df["fold_id"].unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device                   : {device}")
    print(f"Experiment               : {experiment_name}")
    print(f"Targets                  : {targets}")
    print(f"Epochs                   : {epochs}")
    print(f"Dropout                  : hidden={hidden_dropout}  attention={attention_dropout}")
    print(f"Max speeches per author  : {max_speeches_per_author or 'unlimited'}")
    print(f"Task weights             : {task_weights}")
    for target, lmap in label_maps.items():
        print(f"  {target} classes ({len(lmap)}): {list(lmap.keys())[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)

    # Pre-split fold DataFrames once.
    fold_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold_id in fold_ids:
        fold_rows = folds_df[folds_df["fold_id"] == fold_id]
        train_ids = set(fold_rows.loc[fold_rows["fold_role"] == "train", "id_speech"])
        val_ids = set(fold_rows.loc[fold_rows["fold_role"] == "val", "id_speech"])
        train_df = corpus[corpus["id_speech"].isin(train_ids)].reset_index(drop=True)
        val_df = corpus[corpus["id_speech"].isin(val_ids)].reset_index(drop=True)
        fold_data[fold_id] = (train_df, val_df)
        print(f"  [{fold_id}] train={len(train_df)}  val={len(val_df)}")

    # Majority baselines per target (averaged over folds).
    majority_stats: dict[str, dict[str, float]] = {}
    for target in targets:
        maj_acc_list, maj_f1_list = [], []
        for fold_id in fold_ids:
            _, val_df = fold_data[fold_id]
            val_labels = (
                val_df[target].dropna().astype(str).tolist()
                if target in val_df.columns
                else []
            )
            if val_labels:
                maj = max(set(val_labels), key=val_labels.count)
                maj_acc_list.append(val_labels.count(maj) / len(val_labels))
                maj_f1_list.append(
                    float(
                        f1_score(
                            val_labels,
                            [maj] * len(val_labels),
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
        majority_stats[target] = {
            "accuracy": float(np.mean(maj_acc_list)) if maj_acc_list else float("nan"),
            "macro_f1": float(np.mean(maj_f1_list)) if maj_f1_list else float("nan"),
        }

    # Train each fold once for all epochs.
    fold_epoch_rows: dict[str, list[dict[str, Any]]] = {}
    for fold_id in fold_ids:
        train_df, val_df = fold_data[fold_id]
        fold_epoch_rows[fold_id] = _run_fold_multitask(
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            targets=targets,
            label_maps=label_maps,
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
            task_weights=task_weights,
            seed=seed,
            device=device,
            fold_dir=results_dir / fold_id,
        )
    best_ep, best_score = _write_epoch_outputs(
        results_dir=results_dir,
        fold_epoch_rows=fold_epoch_rows,
        fold_ids=fold_ids,
        targets=targets,
        selection_metric=selection_metric,
        majority_stats=majority_stats,
        config_path=config_path,
    )

    print(
        f"\nBest epoch : {best_ep}"
        f"  combined mean {selection_metric}={best_score:.4f}"
    )
    print(f"Outputs    : {results_dir}")

    return {
        "run_type": "multitask_profiling_dev_search",
        "experiment_name": experiment_name,
        "seed": seed,
        "targets": targets,
        "best_epoch": best_ep,
        "results_dir": relative_to_project(project_root, results_dir),
    }


def run_final_training(config_path: Path) -> dict[str, Any]:
    """Train MultiTaskBERT on all profiling data and export per-task models.

    Each task is exported as a standalone AutoModelForSequenceClassification under:
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
    task_weights: dict[str, float] | None = cfg.get("task_weights") or None

    model_cfg = cfg["model"]
    pretrained_name = str(model_cfg["pretrained_name"])
    max_length = int(model_cfg.get("max_length", 512))

    head_tokens, tail_tokens = [int(v) for v in cfg["truncation"]["pairs"][0]]
    truncation = TruncationConfig(
        head_tokens=head_tokens, tail_tokens=tail_tokens, max_length=max_length
    )

    results_dir, artifacts_dir = _resolve_dirs(project_root, cfg, "final")
    copy_config_outputs(config_path, artifacts_dir / config_path.name)

    data_cfg = cfg.get("data", {})
    results_root = resolve_project_path(
        project_root, data_cfg.get("results_dir", "results/models")
    )
    targets_key = "+".join(sorted(targets))
    dev_run_dir = results_root / split_name / experiment_name / f"seed_{seed}" / targets_key
    best_epoch_path = dev_run_dir / "best_epoch.json"
    if not best_epoch_path.exists():
        raise FileNotFoundError(
            f"best_epoch.json not found at {best_epoch_path}. "
            "Run dev search first (--mode dev)."
        )
    best_epoch = int(json.loads(best_epoch_path.read_text(encoding="utf-8"))["best_epoch"])
    print(f"Best epoch (from dev): {best_epoch}")

    splits_root = resolve_project_path(
        project_root, cfg.get("data", {}).get("splits_dir", "data/splits")
    )
    split_dir = splits_root / split_name
    corpus = pd.read_csv(split_dir / "corpus" / "train.csv", dtype={"id_speech": str})
    corpus = add_derived_columns(corpus, age_bin_edges, age_bin_labels, party_axes)

    label_maps = {t: build_label_map(corpus, t) for t in targets}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    print(f"Experiment     : {experiment_name}")
    print(f"Training on    : {len(corpus)} speeches")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)

    train_corpus = corpus
    if max_speeches_per_author > 0:
        train_corpus = sample_per_author(corpus, max_speeches_per_author, seed)
        print(f"  Sampled to {len(train_corpus)} speeches (max {max_speeches_per_author}/author)")
    train_dataset = _make_multitask_dataset(train_corpus, targets, label_maps, tokenizer, truncation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    set_seed(seed)
    num_labels_per_task = {t: len(label_maps[t]) for t in targets}
    model = MultiTaskBERT.from_pretrained(
        pretrained_name, num_labels_per_task, hidden_dropout, attention_dropout
    ).to(device)
    optimizer = make_optimizer(model, lr, weight_decay)

    for epoch in range(1, best_epoch + 1):
        loss = _train_one_epoch_multitask(
            model, train_loader, optimizer, device, targets, task_weights
        )
        print(f"  epoch {epoch}/{best_epoch}  loss={loss:.4f}")

    targets_summary: dict[str, Any] = {}
    for target in targets:
        num_classes = len(label_maps[target])
        save_dir = artifacts_dir / "final" / target / "saved_model"
        _export_task_model(
            model, target, pretrained_name, num_classes, save_dir, tokenizer, label_maps[target]
        )
        print(f"  {target}: exported to {save_dir}")
        targets_summary[target] = {
            "best_epoch": best_epoch,
            "num_classes": num_classes,
            "label_map": label_maps[target],
            "model_path": relative_to_project(project_root, save_dir),
        }

    manifest: dict[str, Any] = {
        "run_type": "multitask_profiling_final_training",
        "experiment_name": experiment_name,
        "seed": seed,
        "max_speeches_per_author": max_speeches_per_author,
        "task_weights": task_weights,
        "targets": targets,
        "best_epoch": best_epoch,
        "targets_summary": targets_summary,
        "hardware": device_info(device),
        "results_dir": relative_to_project(project_root, results_dir),
        "artifacts_dir": relative_to_project(project_root, artifacts_dir),
        "config_path": relative_to_project(project_root, config_path),
    }
    write_json(results_dir / "manifest.json", manifest)
    print(f"\nOutputs written to: {results_dir}")

    return manifest


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()

    if args.mode == "dev":
        manifest = run_dev_search(config_path)
        print(f"\nFinished multi-task profiling dev search : {manifest['experiment_name']}")
        print(f"Best epoch : {manifest['best_epoch']}")
        print(f"Results    : {manifest['results_dir']}")
    else:
        manifest = run_final_training(config_path)
        print(f"\nFinished multi-task profiling final training : {manifest['experiment_name']}")
        for target, summary in manifest["targets_summary"].items():
            print(f"  {target}: epoch={summary['best_epoch']}  model={summary['model_path']}")


if __name__ == "__main__":
    main()
