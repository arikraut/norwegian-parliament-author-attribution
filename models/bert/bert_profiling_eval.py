"""BERT profiling signal evaluation.

Loads profiling probability matrices produced by bert_profiling_extractor.py
and evaluates prediction quality against ground-truth demographic labels.

Reports per-target accuracy, macro F1, and per-class F1
for both CV val folds and the final held-out test set.

Usage
-----
  python -m models.bert.bert_profiling_eval \
      --config models/configs/profiling/bokmal_profiling_bert_signal_extraction.toml \
      --output results/profiling_eval/bert
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
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

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

from data_pipeline.utils import resolve_project_path  # noqa: E402
from models.bert.bert_utils import add_derived_columns, build_party_axes  # noqa: E402


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BERT profiling predictions.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Same extraction config used for bert_profiling_extractor.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to write evaluation results")
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_npz_dense(path: Path) -> np.ndarray:
    return sparse.load_npz(path).toarray()


def _argmax_labels(probs: np.ndarray, label_map: dict[str, str]) -> list[str]:
    return [label_map[str(i)] for i in np.argmax(probs, axis=1)]


def _majority_baseline(true_labels: list[str]) -> tuple[float, float]:
    maj = max(set(true_labels), key=true_labels.count)
    maj_preds = [maj] * len(true_labels)
    acc = accuracy_score(true_labels, maj_preds)
    f1  = float(f1_score(true_labels, maj_preds, average="macro", zero_division=0))
    return round(acc, 4), round(f1, 4)


def _evaluate_split(
    probs: np.ndarray,
    true_labels: list[str],
    label_map: dict[str, str],
    target: str,
    split_label: str,
) -> dict[str, Any]:
    classes        = [label_map[str(i)] for i in range(probs.shape[1])]
    pred_labels    = _argmax_labels(probs, label_map)
    maj_acc, maj_f1 = _majority_baseline(true_labels)

    acc = round(float(accuracy_score(true_labels, pred_labels)), 4)
    macro_f1 = round(float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)), 4)
    per_class_f1 = f1_score(true_labels, pred_labels, labels=classes, average=None, zero_division=0)

    return {
        "split":            split_label,
        "target":           target,
        "n_speeches":       len(true_labels),
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "majority_accuracy": maj_acc,
        "majority_macro_f1": maj_f1,
        **{f"f1_{cls}": round(float(v), 4) for cls, v in zip(classes, per_class_f1)},
    }


# ── Public API ─────────────────────────────────────────────────────────────────


def run_eval(config_path: Path, output_dir: Path) -> None:
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)

    source_cfg             = cfg["source"]
    attribution_split_name = str(source_cfg["attribution_split_name"])
    materialization_name   = str(source_cfg["materialization_name"])
    targets: list[str]     = [str(t) for t in source_cfg["targets"]]

    age_bin_cfg    = cfg.get("age_bins", {})
    age_bin_edges  = [int(e) for e in age_bin_cfg.get("edges", [0, 50, 200])]
    age_bin_labels = [str(l) for l in age_bin_cfg.get("labels", ["<50", "50+"])]
    party_axes     = build_party_axes(cfg.get("party_axes", []))

    data_cfg    = cfg.get("data", {})
    splits_root = resolve_project_path(PROJECT_ROOT, data_cfg.get("splits_dir", "data/splits"))

    attribution_split_dir = splits_root / attribution_split_name
    mat_root = attribution_split_dir / "materialized_features" / materialization_name

    if not mat_root.exists():
        raise FileNotFoundError(
            f"Materialization not found: {mat_root}\n"
            "Run bert_profiling_extractor.py first."
        )

    # Load attribution corpus with derived label columns.
    corpus = pd.read_csv(attribution_split_dir / "corpus" / "all.csv", dtype={"id_speech": str})
    corpus = add_derived_columns(corpus, age_bin_edges, age_bin_labels, party_axes)
    corpus_lookup = corpus.set_index("id_speech")

    test_corpus = pd.read_csv(attribution_split_dir / "corpus" / "test.csv", dtype={"id_speech": str})
    test_corpus = add_derived_columns(test_corpus, age_bin_edges, age_bin_labels, party_axes)
    test_lookup = test_corpus.set_index("id_speech")

    # Load label maps from saved models.
    artifacts_root = resolve_project_path(
        PROJECT_ROOT, data_cfg.get("artifacts_dir", "models/artifacts/profiling")
    )
    profiling_split  = str(source_cfg["profiling_split_name"])
    profiling_exp    = str(source_cfg["profiling_experiment_name"])
    profiling_seed   = int(source_cfg["profiling_seed"])
    artifacts_dir    = (
        artifacts_root / profiling_split / profiling_exp
        / f"seed_{profiling_seed}_final" / "final"
    )
    label_maps: dict[str, dict[str, str]] = {}
    for target in targets:
        lm_path = artifacts_dir / target / "saved_model" / "label_map.json"
        if lm_path.exists():
            with lm_path.open(encoding="utf-8") as fh:
                label_maps[target] = json.load(fh)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    def _eval_role(unit_dir: Path, role: str, row_order_name: str,
                   lookup: pd.DataFrame, split_label: str) -> None:
        row_order = pd.read_csv(
            unit_dir / "row_order" / row_order_name, dtype={"id_speech": str}
        )
        for target in targets:
            if target not in label_maps:
                continue
            npz_path = unit_dir / "matrices" / f"X_{role}_profiling_{target}.npz"
            if not npz_path.exists():
                continue
            probs = _load_npz_dense(npz_path)
            true_labels = [
                str(lookup.at[sid, target])
                if sid in lookup.index and target in lookup.columns
                else "unknown"
                for sid in row_order["id_speech"]
            ]
            if all(l == "unknown" for l in true_labels):
                continue
            row = _evaluate_split(probs, true_labels, label_maps[target],
                                  target, split_label)
            all_rows.append(row)
            print(
                f"[{split_label}] {target}: acc={row['accuracy']:.4f} "
                f"(maj={row['majority_accuracy']:.4f})  "
                f"f1={row['macro_f1']:.4f} (maj={row['majority_macro_f1']:.4f})"
            )

    # ── CV fold train + val evaluation ───────────────────────────────────────
    manifest = json.loads((mat_root / "manifest.json").read_text(encoding="utf-8"))
    fold_units = [u for u in manifest["units"] if u["eval_role"] == "val"]

    for unit in fold_units:
        fold_id  = str(unit["unit_id"])
        unit_dir = mat_root / fold_id
        _eval_role(unit_dir, "train", "train_rows.csv", corpus_lookup, f"train_{fold_id}")
        _eval_role(unit_dir, "val",   "val_rows.csv",   corpus_lookup, f"val_{fold_id}")

    # ── Final test unit: full train + test evaluation ────────────────────────
    test_unit_dir = mat_root / "final_test"
    _eval_role(test_unit_dir, "train", "train_rows.csv", corpus_lookup, "train_final")
    _eval_role(test_unit_dir, "test",  "test_rows.csv",  test_lookup,   "test")

    # ── Aggregate val summary (mean across folds) ─────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "per_split_metrics.csv", index=False)

    summary_parts: list[pd.DataFrame] = []
    numeric_cols = [c for c in df.columns
                    if c not in ("split", "target") and df[c].dtype != object]

    for prefix, label in [("val_", "val_mean"), ("train_fold", "train_fold_mean")]:
        subset = df[df["split"].str.startswith(prefix)]
        if not subset.empty:
            agg = subset.groupby("target")[numeric_cols].mean().round(4).reset_index()
            agg.insert(1, "split", label)
            summary_parts.append(agg)

    for exact_split in ("train_final", "test"):
        subset = df[df["split"] == exact_split]
        if not subset.empty:
            summary_parts.append(subset)

    if summary_parts:
        combined = pd.concat(summary_parts, ignore_index=True)
        combined.to_csv(output_dir / "summary_metrics.csv", index=False)
        print(f"\nSummary written to {output_dir / 'summary_metrics.csv'}")

    print(f"All metrics     : {output_dir / 'per_split_metrics.csv'}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    run_eval(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
