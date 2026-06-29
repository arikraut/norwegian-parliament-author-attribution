"""
bert_utils.py — Shared utilities for all BERT profiling models.

Public API
----------
Dataset:
    TruncationConfig
    SpeechDataset

Training primitives:
    set_seed(seed)
    device_info(device)                                          -> dict
    CandidateSpec
    make_optimizer(model, learning_rate, weight_decay)           -> AdamW
    sample_per_author(df, max_per_author, seed)                  -> pd.DataFrame
    train_one_epoch(model, loader, optimizer, device)            -> float
    evaluate(model, loader, device)                              -> dict
    evaluate_with_loss(model, loader, device)                    -> dict
    evaluate_with_probs(model, loader, device)  -> (dict, list, list, ndarray)

Label derivation:
    build_party_axes(party_axes_cfg)                             -> list[tuple]
    add_derived_columns(df, edges, labels, axes)                 -> pd.DataFrame
    build_label_map(corpus, target)                              -> dict[str, int]
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


# ── Dataset ────────────────────────────────────────────────────────────────────
#
# For speeches longer than (max_length - 2) tokens, the input is truncated to:
#     [CLS]  first head_tokens tokens  last tail_tokens tokens  [SEP]
# Shorter speeches are used as-is and padded to max_length.


@dataclass
class TruncationConfig:
    head_tokens: int
    tail_tokens: int
    max_length: int = 512

    @property
    def label(self) -> str:
        return f"head{self.head_tokens}_tail{self.tail_tokens}"


class SpeechDataset(Dataset):
    """One item per speech: tokenised, truncated, and padded to max_length."""

    def __init__(self, texts, labels, tokenizer, truncation):
        self.labels = labels
        self.encodings = [self._encode(text, tokenizer, truncation) for text in texts]

    @staticmethod
    def _encode(text, tokenizer, trunc):
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        max_content = trunc.max_length - 2
        if len(token_ids) > max_content:
            token_ids = token_ids[: trunc.head_tokens] + token_ids[-trunc.tail_tokens :]

        input_ids      = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        pad_length      = trunc.max_length - len(input_ids)
        input_ids      += [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.encodings[idx]["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings[idx]["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],                      dtype=torch.long),
        }


# ── Training primitives ───────────────────────────────────────────────────────


def device_info(device: torch.device) -> dict[str, Any]:
    """Return a hardware summary dict suitable for embedding in a manifest."""
    info: dict[str, Any] = {"type": str(device)}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@dataclass
class CandidateSpec:
    """One point in the hyperparameter grid (used for final evaluation)."""
    truncation: TruncationConfig
    learning_rate: float
    epochs: int

    @property
    def name(self) -> str:
        return f"{self.truncation.label}__lr={self.learning_rate:.0e}"


def make_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """AdamW with constant learning rate — no warmup, no decay schedule."""
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def sample_per_author(
    df: pd.DataFrame,
    max_per_author: int,
    seed: int,
) -> pd.DataFrame:
    """Return df with at most max_per_author speeches per author.

    Authors with fewer speeches are kept in full. Selection is deterministic
    given the seed, so the same config always produces the same training set.
    """
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, group in df.groupby("id_person", sort=False):
        idx = group.index.tolist()
        if len(idx) > max_per_author:
            keep.extend(rng.choice(idx, size=max_per_author, replace=False).tolist())
        else:
            keep.extend(idx)
    return df.loc[sorted(keep)].reset_index(drop=True)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Run one training pass and return the mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute accuracy and macro-F1 on a DataLoader."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        all_labels.extend(batch["labels"].cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_with_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute accuracy, macro-F1, and mean cross-entropy loss on a DataLoader."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    total_loss = 0.0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss += outputs.loss.item()
        all_labels.extend(batch["labels"].cpu().numpy())
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "loss": total_loss / len(loader),
    }


@torch.no_grad()
def evaluate_with_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int], np.ndarray]:
    """Compute metrics and return softmax probabilities alongside labels and predictions."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    prob_chunks: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits, dim=-1)
        all_labels.extend(batch["labels"].cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        prob_chunks.append(probs.cpu().numpy())
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }
    return metrics, all_labels, all_preds, np.concatenate(prob_chunks, axis=0)


# ── Label derivation ──────────────────────────────────────────────────────────


def _normalize_party(p: str) -> str:
    return p.strip().upper()


def build_party_axes(
    party_axes_cfg: list[dict[str, Any]],
) -> list[tuple[str, dict[str, str]]]:
    """Convert [[party_axes]] config entries into (column_name, party→label) tuples."""
    result: list[tuple[str, dict[str, str]]] = []
    for axis_cfg in party_axes_cfg:
        col_name = str(axis_cfg.get("name", "party_axis"))
        mapping: dict[str, str] = {}
        for key, parties in axis_cfg.items():
            if key == "name":
                continue
            for p in parties:
                mapping[_normalize_party(str(p))] = key
        result.append((col_name, mapping))
    return result


def add_derived_columns(
    df: pd.DataFrame,
    age_bin_edges: list[int],
    age_bin_labels: list[str],
    party_axes: list[tuple[str, dict[str, str]]],
) -> pd.DataFrame:
    """Add age_bin and one column per party axis to df."""
    df = df.copy()
    if "age" in df.columns:
        df["age_bin"] = (
            pd.cut(
                df["age"],
                bins=age_bin_edges,
                labels=age_bin_labels,
                right=False,
                include_lowest=True,
            )
            .astype(str)
            .replace("nan", "unknown")
        )
    else:
        df["age_bin"] = "unknown"

    for col_name, axis_map in party_axes:
        if "party" in df.columns:
            df[col_name] = df["party"].apply(
                lambda p: axis_map.get(_normalize_party(str(p)), "unknown")
            )
        else:
            df[col_name] = "unknown"

    return df


def build_label_map(corpus: pd.DataFrame, target: str) -> dict[str, int]:
    """Return a sorted {class_string: int_index} mapping for a target column."""
    unique_vals = sorted(corpus[target].dropna().astype(str).unique())
    return {v: i for i, v in enumerate(unique_vals)}
