"""Tests for train_profiling_classifiers.py.

Covers:
- author_balance config validation
- inverse-author weight computation
- author-weighted metric computation
- sample weights passed through to SVM fit and calibrated refit
- author-weighted metrics appear in fold_metrics.csv
- author-weighted metrics can drive candidate summary selection
- dev training with a stylo-only and mixed candidate
- final training with a stylo-containing best candidate
- signal extraction with stylo block
- regression: checked-in profiling config validation still works
"""

from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy import sparse

from data_pipeline.utils import write_json as _write_json
from models.SVM.training.train_profiling_classifiers import (
    _author_weighted_metrics,
    _fit_calibrated_family_model,
    _inverse_author_weights,
    _load_author_ids,
    _validate_target_unit_label_coverage,
    _validate_profiling_config,
    run_final_profiling_training,
    run_profiling_experiment,
)
from models.SVM.signals.profiling_signal_extractor import (
    _resolve_profiling_fold,
    run_profiling_signal_extraction,
)
from models.SVM.training.profiling_selection import (
    FeatureLayout,
    ProfilingCandidateSpec,
    summarize_profiling_candidates,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _make_candidate(
    name: str = "char_word",
    blocks: tuple[str, ...] = ("char", "word"),
    c: float = 0.1,
    normalize_rows: bool = True,
    normalize_each_block: bool = False,
    block_weights: dict | None = None,
) -> ProfilingCandidateSpec:
    return ProfilingCandidateSpec(
        feature_layout=FeatureLayout(
            name=name,
            blocks=blocks,
            normalize_rows=normalize_rows,
            normalize_each_block=normalize_each_block,
            block_weights=block_weights or {},
        ),
        c_value=c,
        class_weight="balanced",
    )


def _write_sparse(path: Path, mat: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, mat)


def _write_profiling_unit(
    unit_dir: Path,
    *,
    unit_id: str,
    eval_role: str = "val",
    n_train: int = 20,
    n_eval: int = 10,
    n_features_char: int = 4,
    n_features_word: int = 3,
    n_features_stylo: int = 5,
    n_authors: int = 2,
    target_name: str = "female",
    blocks: list[str] | None = None,
    eval_speech_start: int = 0,
) -> None:
    """Create all files for a single materialized profiling fold unit."""
    if blocks is None:
        blocks = ["char", "word"]

    rng = np.random.default_rng(42)
    labels_train = np.array(["0", "1"] * (n_train // 2), dtype=object)
    labels_eval = np.array(["0", "1"] * (n_eval // 2), dtype=object)

    # Ensure each class has at least CALIBRATION_CV = 3 representatives.
    labels_train = np.tile(["0", "1", "1"], n_train // 3 + 1)[:n_train]
    labels_eval = np.tile(["0", "1", "1"], n_eval // 3 + 1)[:n_eval]

    author_ids = [f"a{i % n_authors}" for i in range(n_train)]
    eval_author_ids = [f"a{i % n_authors}" for i in range(n_eval)]

    (unit_dir / "labels").mkdir(parents=True, exist_ok=True)
    (unit_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (unit_dir / "row_order").mkdir(parents=True, exist_ok=True)
    (unit_dir / "preprocessors").mkdir(parents=True, exist_ok=True)

    np.save(unit_dir / "labels" / f"y_train_{target_name}.npy", labels_train)
    np.save(unit_dir / "labels" / f"y_{eval_role}_{target_name}.npy", labels_eval)

    if "char" in blocks:
        _write_sparse(
            unit_dir / "matrices" / "X_train_char.npz",
            sparse.csr_matrix(rng.random((n_train, n_features_char))),
        )
        _write_sparse(
            unit_dir / "matrices" / f"X_{eval_role}_char.npz",
            sparse.csr_matrix(rng.random((n_eval, n_features_char))),
        )
    if "word" in blocks:
        _write_sparse(
            unit_dir / "matrices" / "X_train_word.npz",
            sparse.csr_matrix(rng.random((n_train, n_features_word))),
        )
        _write_sparse(
            unit_dir / "matrices" / f"X_{eval_role}_word.npz",
            sparse.csr_matrix(rng.random((n_eval, n_features_word))),
        )
    if "stylo" in blocks:
        _write_sparse(
            unit_dir / "matrices" / "X_train_stylo.npz",
            sparse.csr_matrix(rng.random((n_train, n_features_stylo))),
        )
        _write_sparse(
            unit_dir / "matrices" / f"X_{eval_role}_stylo.npz",
            sparse.csr_matrix(rng.random((n_eval, n_features_stylo))),
        )

    pd.DataFrame(
        {
            "row_idx": np.arange(n_train),
            "id_speech": [f"s{i}" for i in range(n_train)],
            "id_person": author_ids,
        }
    ).to_csv(unit_dir / "row_order" / "train_rows.csv", index=False)
    pd.DataFrame(
        {
            "row_idx": np.arange(n_eval),
            "id_speech": [f"sv{eval_speech_start + i}" for i in range(n_eval)],
            "id_person": eval_author_ids,
        }
    ).to_csv(unit_dir / "row_order" / f"{eval_role}_rows.csv", index=False)


def _write_toy_profiling_env(
    project_root: Path,
    *,
    split_name: str = "toy_profiling",
    materialization_name: str = "toy_profiling_mat",
    row_feature_name: str = "toy_profiling_rows",
    n_folds: int = 2,
    blocks: list[str] | None = None,
    target_name: str = "female",
    with_stylo_raw: bool = False,
) -> tuple[Path, list[dict]]:
    """Create a complete toy profiling materialization environment."""
    if blocks is None:
        blocks = ["char", "word"]

    mat_root = (
        project_root
        / "data"
        / "splits"
        / split_name
        / "materialized_features"
        / materialization_name
    )
    mat_root.mkdir(parents=True, exist_ok=True)

    units = []
    n_eval = 10
    for i in range(1, n_folds + 1):
        unit_id = f"fold_{i:02d}"
        unit_dir = mat_root / unit_id
        _write_profiling_unit(
            unit_dir,
            unit_id=unit_id,
            eval_role="val",
            n_eval=n_eval,
            blocks=blocks,
            target_name=target_name,
            eval_speech_start=(i - 1) * n_eval,
        )
        units.append({"unit_id": unit_id, "eval_role": "val", "enabled_blocks": blocks})

    _write_json(
        mat_root / "manifest.json",
        {
            "split_name": split_name,
            "row_feature_name": row_feature_name,
            "materialization_name": materialization_name,
            "enabled_blocks": blocks,
            "units": units,
        },
    )

    # Create corpus for final training.
    corpus_dir = project_root / "data" / "splits" / split_name / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    all_speech_ids = [f"sv{i}" for i in range(n_folds * n_eval)]
    pd.DataFrame(
        {
            "id_speech": all_speech_ids,
            "text": [f"text {i}" for i in range(len(all_speech_ids))],
        }
    ).to_csv(corpus_dir / "all.csv", index=False)

    # Optionally write raw stylometry.
    if with_stylo_raw:
        row_feature_dir = (
            project_root / "data" / "splits" / split_name / "row_features" / row_feature_name
        )
        row_feature_dir.mkdir(parents=True, exist_ok=True)
        # Match the row-feature stylometry artifact structure:
        # id_speech, id_person, outer_role (string), then numeric feature cols.
        n = len(all_speech_ids)
        stylo_data = pd.DataFrame(
            {
                "id_speech": all_speech_ids,
                "id_person": [f"p{i % 5}" for i in range(n)],
                "outer_role": ["train"] * (n * 2 // 3) + ["test"] * (n - n * 2 // 3),
                "feat_a": np.random.default_rng(1).random(n),
                "feat_b": np.random.default_rng(2).random(n),
                "feat_zero": np.zeros(n),  # zero-variance, should be dropped
            }
        )
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            stylo_data.to_csv(io.TextIOWrapper(gz, encoding="utf-8"), index=False)
        (row_feature_dir / "stylometry_raw.csv.gz").write_bytes(buf.getvalue())

    return mat_root, units


# ── Config validation tests ───────────────────────────────────────────────────


class ValidateProfilingConfigTests(unittest.TestCase):
    def _base_config(self) -> dict:
        return {
            "experiment": {"name": "test", "seed": 42},
            "data": {"splits_dir": "data/splits", "results_dir": "results/models", "artifacts_dir": "models/artifacts/profiling"},
            "source": {"split_name": "x", "materialization_name": "y", "targets": ["female"]},
            "model": {"family": "linear_svm", "C_values": [0.1]},
            "feature_sets": [{"name": "char_word", "blocks": ["char", "word"], "normalize_rows": True}],
        }

    def test_rejects_unknown_weighting_mode(self) -> None:
        cfg = self._base_config()
        cfg["author_balance"] = {"train_sample_weighting": "random_resample"}
        with self.assertRaises(ValueError, msg="should reject unknown weighting mode"):
            _validate_profiling_config(cfg)

    def test_rejects_unknown_author_balance_key(self) -> None:
        cfg = self._base_config()
        cfg["author_balance"] = {
            "train_sample_weighting": "none",
            "unknown_key": True,
        }
        with self.assertRaises(ValueError):
            _validate_profiling_config(cfg)

    def test_rejects_unknown_root_key(self) -> None:
        cfg = self._base_config()
        cfg["bad_key"] = "value"
        with self.assertRaises(ValueError):
            _validate_profiling_config(cfg)


class ProfilingLabelCoverageTests(unittest.TestCase):
    def test_target_unit_label_coverage_rejects_eval_only_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            materialized_root = Path(tmpdir)
            labels_dir = materialized_root / "fold_01" / "labels"
            labels_dir.mkdir(parents=True)
            np.save(labels_dir / "y_train_party.npy", np.array(["Ap", "H", "V"]))
            np.save(labels_dir / "y_val_party.npy", np.array(["Ap", "SV"]))

            with self.assertRaisesRegex(ValueError, "SV"):
                _validate_target_unit_label_coverage(
                    "party",
                    [{"unit_id": "fold_01", "eval_role": "val"}],
                    materialized_root,
                )

    def test_target_unit_label_coverage_accepts_eval_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            materialized_root = Path(tmpdir)
            labels_dir = materialized_root / "fold_01" / "labels"
            labels_dir.mkdir(parents=True)
            np.save(labels_dir / "y_train_party.npy", np.array(["Ap", "H", "SV", "V"]))
            np.save(labels_dir / "y_val_party.npy", np.array(["Ap", "SV"]))

            _validate_target_unit_label_coverage(
                "party",
                [{"unit_id": "fold_01", "eval_role": "val"}],
                materialized_root,
            )


# ── Weight computation tests ──────────────────────────────────────────────────


class InverseAuthorWeightTests(unittest.TestCase):
    def test_weight_is_inverse_of_author_count(self) -> None:
        # a0 appears 3 times → weight 1/3 each; a1 appears 1 time → weight 1
        author_ids = np.array(["a0", "a0", "a0", "a1"], dtype=str)
        weights = _inverse_author_weights(author_ids, normalize=False)
        np.testing.assert_allclose(weights[:3], [1 / 3, 1 / 3, 1 / 3], rtol=1e-9)
        np.testing.assert_allclose(weights[3], 1.0, rtol=1e-9)

    def test_normalized_weights_have_mean_one(self) -> None:
        author_ids = np.array(["a0", "a0", "a1", "a2", "a2", "a2"], dtype=str)
        weights = _inverse_author_weights(author_ids, normalize=True)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=9)

    def test_single_author_all_weight_one(self) -> None:
        author_ids = np.array(["a0", "a0", "a0"], dtype=str)
        weights = _inverse_author_weights(author_ids, normalize=True)
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0], rtol=1e-9)


class LoadAuthorIdsTests(unittest.TestCase):
    def test_loads_id_person_in_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            unit_dir = Path(tmpdir) / "fold_01"
            (unit_dir / "row_order").mkdir(parents=True)
            pd.DataFrame(
                {
                    "row_idx": [0, 1, 2],
                    "id_speech": ["s1", "s2", "s3"],
                    "id_person": ["a1", "a2", "a1"],
                }
            ).to_csv(unit_dir / "row_order" / "train_rows.csv", index=False)

            result = _load_author_ids(unit_dir, "train")
            np.testing.assert_array_equal(result, ["a1", "a2", "a1"])


# ── Author-weighted metrics tests ─────────────────────────────────────────────


class AuthorWeightedMetricsTests(unittest.TestCase):
    def test_uniform_weights_match_unweighted(self) -> None:
        from sklearn.metrics import f1_score
        y_true = np.array(["0", "1", "1", "0", "0"])
        y_pred = np.array(["0", "1", "0", "0", "1"])
        w = np.ones(len(y_true))
        metrics = _author_weighted_metrics(y_true, y_pred, w)
        expected_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        self.assertAlmostEqual(metrics["author_weighted_macro_f1"], expected_f1, places=9)


# ── Candidate summary tests ───────────────────────────────────────────────────


class CandidateSummarySelectionTests(unittest.TestCase):
    def test_author_weighted_macro_f1_can_drive_candidate_selection(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "candidate_id": "plain_macro_wins",
                    "feature_set": "char_word",
                    "blocks": "char+word",
                    "normalize_rows": True,
                    "normalize_each_block": False,
                    "c_value": 0.1,
                    "class_weight": "balanced",
                    "unit_id": "fold_01",
                    "eval_role": "val",
                    "split": "val",
                    "n_samples": 20,
                    "n_classes": 2,
                    "accuracy": 0.95,
                    "macro_f1": 0.90,
                    "weighted_f1": 0.94,
                    "macro_precision": 0.90,
                    "macro_recall": 0.90,
                    "fit_seconds": 1.0,
                    "predict_seconds": 0.1,
                    "convergence_warning_count": 0,
                    "author_weighted_macro_f1": 0.60,
                },
                {
                    "candidate_id": "author_weighted_wins",
                    "feature_set": "char_word_stylo",
                    "blocks": "char+word+stylo",
                    "normalize_rows": True,
                    "normalize_each_block": True,
                    "c_value": 0.1,
                    "class_weight": "balanced",
                    "unit_id": "fold_01",
                    "eval_role": "val",
                    "split": "val",
                    "n_samples": 20,
                    "n_classes": 2,
                    "accuracy": 0.80,
                    "macro_f1": 0.75,
                    "weighted_f1": 0.79,
                    "macro_precision": 0.75,
                    "macro_recall": 0.75,
                    "fit_seconds": 1.0,
                    "predict_seconds": 0.1,
                    "convergence_warning_count": 0,
                    "author_weighted_macro_f1": 0.85,
                },
            ]
        )

        summary_df = summarize_profiling_candidates(
            metrics_df, selection_metric="author_weighted_macro_f1"
        )

        self.assertIn("eval_mean_author_weighted_macro_f1", summary_df.columns)
        self.assertEqual(summary_df.iloc[0]["candidate_id"], "author_weighted_wins")


# ── Sample weight passthrough tests ──────────────────────────────────────────


class SampleWeightPassthroughTests(unittest.TestCase):
    """Verify sample_weight is passed to _fit_linear_svm and _fit_calibrated_family_model."""

    def test_fit_linear_svm_passes_sample_weight(self) -> None:
        from models.SVM.linear_svm_common import _fit_linear_svm
        x = sparse.csr_matrix(np.eye(4))
        y = np.array(["0", "1", "0", "1"])
        candidate = _make_candidate()
        weights = np.array([1.0, 2.0, 1.0, 2.0])

        with patch("models.SVM.linear_svm_common.LinearSVC") as MockSVC:
            mock_clf = MagicMock()
            MockSVC.return_value = mock_clf
            _fit_linear_svm(
                x,
                y,
                c_value=candidate.c_value,
                class_weight=candidate.class_weight,
                model_cfg={"max_iter": 100},
                seed=0,
                sample_weight=weights,
            )
            mock_clf.fit.assert_called_once()
            passed_weight = mock_clf.fit.call_args[1].get("sample_weight")
            self.assertIsNotNone(passed_weight)
            np.testing.assert_array_equal(passed_weight, weights)

    def test_fit_calibrated_model_passes_sample_weight(self) -> None:
        candidate = _make_candidate()
        x = sparse.csr_matrix(np.eye(6).astype(float))
        y = np.array(["0", "1", "0", "1", "0", "1"])
        weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

        with patch("models.SVM.training.train_profiling_classifiers.CalibratedClassifierCV") as MockCal:
            mock_cal = MagicMock()
            MockCal.return_value = mock_cal
            _fit_calibrated_family_model(
                x, y, candidate, {"max_iter": 100}, {"method": "sigmoid", "cv": 3}, seed=0,
                sample_weight=weights
            )
            mock_cal.fit.assert_called_once()
            call_kwargs = mock_cal.fit.call_args
            passed_weight = call_kwargs[1].get("sample_weight")
            np.testing.assert_array_equal(passed_weight, weights)


# ── Dev training tests ────────────────────────────────────────────────────────


class ProfilingDevTrainingTests(unittest.TestCase):
    """Integration tests that run run_profiling_experiment on toy data."""

    def _write_env_and_config(
        self,
        project_root: Path,
        *,
        blocks: list[str],
        author_balance: bool = False,
    ) -> Path:
        """Write materialization and config, return config path."""
        mat_root, units = _write_toy_profiling_env(
            project_root,
            split_name="toy_prof",
            materialization_name="toy_mat",
            blocks=blocks,
        )

        # Write upstream manifests.
        _write_json(
            project_root / "data" / "splits" / "toy_prof" / "manifest.json",
            {"split_name": "toy_prof"},
        )
        _write_json(
            project_root / "data" / "splits" / "toy_prof" / "row_features" / "toy_rows" / "manifest.json",
            {"split_name": "toy_prof", "feature_set_name": "toy_rows"},
        )

        feature_set_toml = "\n".join([
            "[[feature_sets]]",
            f'name = "{"_".join(blocks)}"',
            f'blocks = {json.dumps(blocks)}',
            "normalize_rows = true",
        ])

        balance_toml = ""
        if author_balance:
            balance_toml = """
[author_balance]
train_sample_weighting = "inverse_author_speech_count"
normalize_train_weights = true
report_author_weighted_eval = true
"""

        config_content = f"""
[experiment]
name = "toy_prof_exp"
seed = 42
selection_metric = "macro_f1"
n_jobs = 1

[data]
splits_dir = "data/splits"
results_dir = "results/models"
artifacts_dir = "models/artifacts/profiling"

[source]
split_name = "toy_prof"
materialization_name = "toy_mat"
units = "all"
targets = ["female"]

[model]
family = "linear_svm"
C_values = [0.1]
max_iter = 1000
tol = 0.001
dual = "auto"
{balance_toml}
{feature_set_toml}
""".strip()

        config_path = project_root / "models" / "configs" / "profiling" / "toy_config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_char_word_training_produces_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._write_env_and_config(root, blocks=["char", "word"])
            run_profiling_experiment(config_path, show_progress=False)
            results_dir = root / "results" / "models" / "toy_prof" / "toy_prof_exp" / "seed_42"
            self.assertTrue((results_dir / "female" / "fold_metrics.csv").exists())
            self.assertTrue((results_dir / "female" / "best_candidate.json").exists())

    def test_stylo_candidate_produces_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._write_env_and_config(root, blocks=["char", "word", "stylo"])
            run_profiling_experiment(config_path, show_progress=False)
            results_dir = root / "results" / "models" / "toy_prof" / "toy_prof_exp" / "seed_42"
            self.assertTrue((results_dir / "female" / "fold_metrics.csv").exists())
            best = json.loads((results_dir / "female" / "best_candidate.json").read_text())
            # The stylo block must appear in the candidate's blocks list.
            self.assertIn("stylo", best["blocks"])

    def test_author_weighted_metrics_in_fold_metrics_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._write_env_and_config(
                root, blocks=["char", "word"], author_balance=True
            )
            run_profiling_experiment(config_path, show_progress=False)
            results_dir = root / "results" / "models" / "toy_prof" / "toy_prof_exp" / "seed_42"
            metrics_df = pd.read_csv(results_dir / "female" / "fold_metrics.csv")
            # Author-weighted metrics should appear in the val rows only.
            val_rows = metrics_df[metrics_df["split"] != "train"]
            self.assertIn("author_weighted_macro_f1", val_rows.columns)
            self.assertFalse(val_rows["author_weighted_macro_f1"].isna().all())

    def test_no_author_weighted_metrics_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._write_env_and_config(
                root, blocks=["char", "word"], author_balance=False
            )
            run_profiling_experiment(config_path, show_progress=False)
            results_dir = root / "results" / "models" / "toy_prof" / "toy_prof_exp" / "seed_42"
            metrics_df = pd.read_csv(results_dir / "female" / "fold_metrics.csv")
            self.assertNotIn("author_weighted_macro_f1", metrics_df.columns)


# ── Final training tests ──────────────────────────────────────────────────────


class FinalProfilingTrainingTests(unittest.TestCase):
    def _setup_env_with_dev_results(
        self,
        project_root: Path,
        *,
        blocks: list[str],
        with_stylo_raw: bool = False,
    ) -> Path:
        """Set up toy environment and write a fake best_candidate.json for female."""
        mat_root, units = _write_toy_profiling_env(
            project_root,
            split_name="toy_prof",
            materialization_name="toy_mat",
            row_feature_name="toy_rows",
            blocks=blocks,
            with_stylo_raw=with_stylo_raw,
        )

        _write_json(
            project_root / "data" / "splits" / "toy_prof" / "manifest.json",
            {"split_name": "toy_prof"},
        )
        _write_json(
            project_root / "data" / "splits" / "toy_prof" / "row_features" / "toy_rows" / "manifest.json",
            {"split_name": "toy_prof", "feature_set_name": "toy_rows"},
        )

        # Write dev best_candidate.json pointing at the first listed block set.
        results_dir = (
            project_root / "results" / "models" / "toy_prof" / "toy_prof_exp" / "seed_42"
        )
        (results_dir / "female").mkdir(parents=True, exist_ok=True)
        _write_json(
            results_dir / "female" / "best_candidate.json",
            {
                "candidate_id": f"{'_'.join(blocks)}__C=0.1__class_weight=balanced",
                "feature_set": "_".join(blocks),
                "blocks": blocks,
                "normalize_rows": True,
                "normalize_each_block": False,
                "block_weights": {},
                "c_value": 0.1,
                "class_weight": "balanced",
                "selection_metric": "macro_f1",
            },
        )

        feature_set_toml = "\n".join([
            "[[feature_sets]]",
            f'name = "{"_".join(blocks)}"',
            f'blocks = {json.dumps(blocks)}',
            "normalize_rows = true",
        ])

        config_content = f"""
[experiment]
name = "toy_prof_exp"
seed = 42
n_jobs = 1

[data]
splits_dir = "data/splits"
results_dir = "results/models"
artifacts_dir = "models/artifacts/profiling"

[source]
split_name = "toy_prof"
materialization_name = "toy_mat"
units = "all"
targets = ["female"]

[model]
family = "linear_svm"
C_values = [0.1]
max_iter = 1000
tol = 0.001
dual = "auto"

{feature_set_toml}
""".strip()

        config_path = project_root / "models" / "configs" / "profiling" / "toy_config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def _write_char_word_preprocessors(self, project_root: Path, mat_root: Path) -> None:
        """Write toy char+word vectorizers for fold_01 (used by final training to clone)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        preprocessors_dir = mat_root / "fold_01" / "preprocessors"
        preprocessors_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = mat_root.parent.parent / "corpus" / "all.csv"
        texts = pd.read_csv(corpus_path)["text"].astype(str).tolist()
        char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1, max_features=50)
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50)
        char_vec.fit(texts)
        word_vec.fit(texts)
        joblib.dump(char_vec, preprocessors_dir / "char_vectorizer.joblib")
        joblib.dump(word_vec, preprocessors_dir / "word_vectorizer.joblib")

    def _build_final_training_project(
        self,
        tmpdir: str,
        *,
        blocks: list[str],
        with_stylo_raw: bool = False,
    ) -> tuple[Path, Path, Path]:
        """Create the common final-training toy project used by this test group."""
        root = Path(tmpdir)
        (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
        (root / "data").mkdir()
        config_path = self._setup_env_with_dev_results(
            root,
            blocks=blocks,
            with_stylo_raw=with_stylo_raw,
        )
        mat_root = root / "data" / "splits" / "toy_prof" / "materialized_features" / "toy_mat"
        return root, config_path, mat_root

    def test_final_char_word_training_saves_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, mat_root = self._build_final_training_project(
                tmpdir,
                blocks=["char", "word"],
            )
            self._write_char_word_preprocessors(root, mat_root)

            manifest = run_final_profiling_training(config_path, show_progress=False)
            artifacts_dir = root / "models" / "artifacts" / "profiling" / "toy_prof" / "toy_prof_exp" / "seed_42"
            self.assertTrue((artifacts_dir / "female" / "models" / "final" / "model.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "char_vectorizer.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "word_vectorizer.joblib").exists())
            coverage = manifest["training_corpus_coverage"]
            self.assertTrue(coverage["checked_exactly_once"])
            self.assertEqual(coverage["corpus_row_count"], 20)
            self.assertEqual(coverage["collected_eval_row_count"], 20)

    def test_final_training_rejects_duplicate_eval_speech_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, _ = self._build_final_training_project(
                tmpdir,
                blocks=["char", "word"],
            )
            row_path = (
                root
                / "data"
                / "splits"
                / "toy_prof"
                / "materialized_features"
                / "toy_mat"
                / "fold_02"
                / "row_order"
                / "val_rows.csv"
            )
            rows = pd.read_csv(row_path, dtype={"id_speech": str, "id_person": str})
            rows.loc[0, "id_speech"] = "sv0"
            rows.to_csv(row_path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate id_speech"):
                run_final_profiling_training(config_path, show_progress=False)

    def test_final_training_rejects_incomplete_eval_corpus_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, _ = self._build_final_training_project(
                tmpdir,
                blocks=["char", "word"],
            )
            corpus_path = root / "data" / "splits" / "toy_prof" / "corpus" / "all.csv"
            corpus = pd.read_csv(corpus_path, dtype={"id_speech": str})
            corpus.loc[len(corpus)] = {"id_speech": "sv_extra", "text": "extra text"}
            corpus.to_csv(corpus_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing from eval rows"):
                run_final_profiling_training(config_path, show_progress=False)

    def test_final_training_rejects_label_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, _ = self._build_final_training_project(
                tmpdir,
                blocks=["char", "word"],
            )
            label_path = (
                root
                / "data"
                / "splits"
                / "toy_prof"
                / "materialized_features"
                / "toy_mat"
                / "fold_01"
                / "labels"
                / "y_val_female.npy"
            )
            labels = np.load(label_path, allow_pickle=True)
            np.save(label_path, labels[:-1])

            with self.assertRaisesRegex(ValueError, "label/row mismatch"):
                run_final_profiling_training(config_path, show_progress=False)

    def test_final_stylo_training_saves_model_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, _ = self._build_final_training_project(
                tmpdir,
                blocks=["stylo"],
                with_stylo_raw=True,
            )

            run_final_profiling_training(config_path, show_progress=False)
            artifacts_dir = root / "models" / "artifacts" / "profiling" / "toy_prof" / "toy_prof_exp" / "seed_42"
            self.assertTrue((artifacts_dir / "female" / "models" / "final" / "model.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "stylo_scaler.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "stylo_columns.json").exists())
            self.assertTrue((artifacts_dir / "final" / "feature_build_meta.json").exists())

            meta = json.loads((artifacts_dir / "final" / "feature_build_meta.json").read_text())
            self.assertIn("stylo", meta["available_blocks"])
            # Zero-variance column should be dropped.
            self.assertNotIn("feat_zero", meta["stylo_columns"])

    def test_final_char_word_stylo_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root, config_path, mat_root = self._build_final_training_project(
                tmpdir,
                blocks=["char", "word", "stylo"],
                with_stylo_raw=True,
            )
            self._write_char_word_preprocessors(root, mat_root)

            run_final_profiling_training(config_path, show_progress=False)
            artifacts_dir = root / "models" / "artifacts" / "profiling" / "toy_prof" / "toy_prof_exp" / "seed_42"
            self.assertTrue((artifacts_dir / "final" / "char_vectorizer.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "word_vectorizer.joblib").exists())
            self.assertTrue((artifacts_dir / "final" / "stylo_scaler.joblib").exists())
            self.assertTrue((artifacts_dir / "female" / "models" / "final" / "model.joblib").exists())


# ── Signal extraction tests ───────────────────────────────────────────────────


class SignalExtractionTests(unittest.TestCase):
    """Tests for profiling_signal_extractor.py with various block configurations."""

    def test_resolve_profiling_fold_uses_matching_dev_fold(self) -> None:
        fold_id = _resolve_profiling_fold(
            "fold_01",
            ["fold_01", "fold_02"],
            stage="dev",
            eval_role="val",
        )

        self.assertEqual(fold_id, "fold_01")

    def test_resolve_profiling_fold_rejects_unmatched_dev_fold(self) -> None:
        with self.assertRaisesRegex(ValueError, "match a profiling fold"):
            _resolve_profiling_fold(
                "fold_99",
                ["fold_01", "fold_02"],
                stage="dev",
                eval_role="val",
            )

    def test_resolve_profiling_fold_uses_final_for_final_test_unit(self) -> None:
        fold_id = _resolve_profiling_fold(
            "final_test_2021",
            ["fold_01", "fold_02"],
            stage="final",
            eval_role="test",
        )

        self.assertEqual(fold_id, "final")

    def test_resolve_profiling_fold_rejects_final_non_test_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "only for test units"):
            _resolve_profiling_fold(
                "fold_99",
                ["fold_01", "fold_02"],
                stage="final",
                eval_role="val",
            )

    def _build_toy_extraction_env(
        self,
        project_root: Path,
        *,
        blocks: list[str] = None,
        with_attribution_stylo: bool = False,
    ) -> Path:
        """Build a complete toy extraction environment and return config path."""
        if blocks is None:
            blocks = ["char", "word"]

        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        n_train, n_eval = 12, 6

        # ── Attribution side ──────────────────────────────────────────────
        attr_split = "toy_attr"
        attr_mat = "toy_attr_mat"
        attr_row_feature = "toy_attr_rows"

        attr_mat_root = (
            project_root / "data" / "splits" / attr_split / "materialized_features" / attr_mat
        )
        attr_unit_dir = attr_mat_root / "fold_01"
        (attr_unit_dir / "matrices").mkdir(parents=True, exist_ok=True)
        (attr_unit_dir / "row_order").mkdir(parents=True, exist_ok=True)

        attr_labels = np.array(["A", "B"] * (n_train // 2), dtype=object)
        attr_eval_labels = np.array(["A", "B"] * (n_eval // 2), dtype=object)
        np.save(attr_unit_dir / "labels" / "y_train_author.npy", attr_labels) if False else None
        (attr_unit_dir / "labels").mkdir(exist_ok=True)
        np.save(attr_unit_dir / "labels" / "y_train_author.npy", attr_labels)
        np.save(attr_unit_dir / "labels" / "y_val_author.npy", attr_eval_labels)

        pd.DataFrame(
            {
                "row_idx": np.arange(n_train),
                "id_speech": [f"as{i}" for i in range(n_train)],
                "id_person": [f"p{i % 3}" for i in range(n_train)],
            }
        ).to_csv(attr_unit_dir / "row_order" / "train_rows.csv", index=False)
        pd.DataFrame(
            {
                "row_idx": np.arange(n_eval),
                "id_speech": [f"av{i}" for i in range(n_eval)],
                "id_person": [f"p{i % 3}" for i in range(n_eval)],
            }
        ).to_csv(attr_unit_dir / "row_order" / "val_rows.csv", index=False)

        _write_json(
            attr_mat_root / "manifest.json",
            {
                "split_name": attr_split,
                "row_feature_name": attr_row_feature,
                "materialization_name": attr_mat,
                "enabled_blocks": ["char", "word", "stylo"],
                "units": [
                    {
                        "unit_id": "fold_01",
                        "eval_role": "val",
                        "enabled_blocks": ["char", "word", "stylo"],
                        "derived_blocks": ["profiling_oracle"],
                    }
                ],
            },
        )

        # Attribution corpus.
        corpus_dir = project_root / "data" / "splits" / attr_split / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        all_ids = [f"as{i}" for i in range(n_train)] + [f"av{i}" for i in range(n_eval)]
        pd.DataFrame(
            {"id_speech": all_ids, "text": [f"text {x}" for x in all_ids]}
        ).to_csv(corpus_dir / "all.csv", index=False)
        pd.DataFrame({"id_person": ["p0", "p1", "p2"]}).to_csv(
            project_root / "data" / "splits" / attr_split / "authors.csv",
            index=False,
        )

        # Write attribution raw stylometry if needed.
        if with_attribution_stylo or "stylo" in blocks:
            rf_dir = (
                project_root / "data" / "splits" / attr_split / "row_features" / attr_row_feature
            )
            rf_dir.mkdir(parents=True, exist_ok=True)
            n_ids = len(all_ids)
            stylo_df = pd.DataFrame(
                {
                    "id_speech": all_ids,
                    "id_person": [f"p{i % 3}" for i in range(n_ids)],
                    "outer_role": ["train"] * n_train + ["test"] * n_eval,
                    "feat_a": rng.random(n_ids),
                    "feat_b": rng.random(n_ids),
                }
            )
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                stylo_df.to_csv(io.TextIOWrapper(gz, encoding="utf-8"), index=False)
            (rf_dir / "stylometry_raw.csv.gz").write_bytes(buf.getvalue())

        # ── Profiling side ────────────────────────────────────────────────
        prof_split = "toy_prof"
        prof_mat = "toy_prof_mat"
        prof_exp = "toy_prof_exp"
        prof_seed = 42

        prof_mat_root = (
            project_root / "data" / "splits" / prof_split / "materialized_features" / prof_mat
        )
        prof_split_dir = project_root / "data" / "splits" / prof_split
        prof_split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id_person": ["q0", "q1", "q2"]}).to_csv(
            prof_split_dir / "authors.csv",
            index=False,
        )
        fold_dir = prof_mat_root / "fold_01"
        preprocessors_dir = fold_dir / "preprocessors"
        preprocessors_dir.mkdir(parents=True, exist_ok=True)

        all_texts = [f"text as{i}" for i in range(n_train)] + [f"text av{i}" for i in range(n_eval)]

        if "char" in blocks:
            char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1, max_features=20)
            char_vec.fit(all_texts)
            joblib.dump(char_vec, preprocessors_dir / "char_vectorizer.joblib")

        if "word" in blocks:
            word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, max_features=20)
            word_vec.fit(all_texts)
            joblib.dump(word_vec, preprocessors_dir / "word_vectorizer.joblib")

        n_stylo_features = 2
        if "stylo" in blocks:
            scaler = StandardScaler()
            scaler.fit(rng.random((n_train, n_stylo_features)))
            joblib.dump(scaler, preprocessors_dir / "stylo_scaler.joblib")
            # feature_columns.json with stylo key.
            _write_json(
                fold_dir / "feature_columns.json",
                {"char": [f"c{i}" for i in range(20)], "word": [f"w{i}" for i in range(20)], "stylo": ["feat_a", "feat_b"]},
            )
        else:
            _write_json(
                fold_dir / "feature_columns.json",
                {"char": [f"c{i}" for i in range(20)], "word": [f"w{i}" for i in range(20)]},
            )

        _write_json(
            prof_mat_root / "manifest.json",
            {
                "split_name": prof_split,
                "materialization_name": prof_mat,
                "units": [{"unit_id": "fold_01", "eval_role": "val"}],
            },
        )

        # Profiling model artifacts: build a tiny calibrated model.
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        prof_artifacts_dir = (
            project_root / "models" / "artifacts" / "profiling"
            / prof_split / prof_exp / f"seed_{prof_seed}"
        )

        # Build a tiny feature matrix matching the blocks for training a real model.
        n_total = n_train + n_eval
        dummy_x_parts = []
        if "char" in blocks:
            dummy_x_parts.append(char_vec.transform(all_texts))
        if "word" in blocks:
            dummy_x_parts.append(word_vec.transform(all_texts))
        if "stylo" in blocks:
            dummy_x_parts.append(sparse.csr_matrix(rng.random((n_total, n_stylo_features))))
        if dummy_x_parts:
            dummy_x = sparse.hstack(dummy_x_parts, format="csr") if len(dummy_x_parts) > 1 else dummy_x_parts[0]
        else:
            dummy_x = sparse.csr_matrix(np.eye(n_total))
        dummy_y = np.array(["0", "1"] * (n_total // 2))
        dummy_y = np.tile(["0", "1", "1"], n_total // 3 + 1)[:n_total]

        base = LinearSVC(C=0.1, max_iter=1000, random_state=0)
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        cal.fit(dummy_x, dummy_y)

        target_model_dir = prof_artifacts_dir / "female" / "models" / "fold_01"
        target_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal, target_model_dir / "model.joblib")

        # best_candidate.json.
        prof_results_dir = (
            project_root / "results" / "models" / prof_split / prof_exp / f"seed_{prof_seed}" / "female"
        )
        prof_results_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            prof_results_dir / "best_candidate.json",
            {
                "candidate_id": f"{'_'.join(blocks)}__C=0.1__class_weight=balanced",
                "feature_set": "_".join(blocks),
                "blocks": blocks,
                "normalize_rows": True,
                "normalize_each_block": False,
                "block_weights": {},
                "c_value": 0.1,
                "class_weight": "balanced",
                "selection_metric": "macro_f1",
            },
        )

        # Extraction config.
        config_content = f"""
[data]
splits_dir = "data/splits"
artifacts_dir = "models/artifacts/profiling"
profiling_results_dir = "results/models"

[source]
attribution_split_name = "{attr_split}"
profiling_split_name = "{prof_split}"
profiling_materialization_name = "{prof_mat}"
profiling_experiment_name = "{prof_exp}"
profiling_seed = {prof_seed}
targets = ["female"]

[stages.dev]
attribution_materialization_name = "{attr_mat}"
""".strip()

        config_path = project_root / "models" / "configs" / "profiling" / "toy_extract.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content, encoding="utf-8")
        return config_path

    def test_char_word_extraction_writes_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._build_toy_extraction_env(root, blocks=["char", "word"])
            run_profiling_signal_extraction(config_path, show_progress=False)

            attr_mat_root = (
                root / "data" / "splits" / "toy_attr" / "materialized_features" / "toy_attr_mat"
            )
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_train_profiling.npz").exists())
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_val_profiling.npz").exists())
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_train_profiling_female.npz").exists())
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_train_profiling_hard.npz").exists())
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_val_profiling_hard.npz").exists())
            self.assertTrue((attr_mat_root / "fold_01" / "matrices" / "X_train_profiling_hard_female.npz").exists())
            self.assertTrue((attr_mat_root / "profiling_hard_feature_columns.json").exists())

            train_proba = sparse.load_npz(
                attr_mat_root / "fold_01" / "matrices" / "X_train_profiling.npz"
            ).toarray()
            val_proba = sparse.load_npz(
                attr_mat_root / "fold_01" / "matrices" / "X_val_profiling.npz"
            ).toarray()
            train_hard = sparse.load_npz(
                attr_mat_root / "fold_01" / "matrices" / "X_train_profiling_hard.npz"
            ).toarray()
            self.assertEqual(train_proba.shape, (12, 2))
            self.assertEqual(val_proba.shape, (6, 2))
            self.assertEqual(train_proba.shape, train_hard.shape)
            np.testing.assert_allclose(train_hard.sum(axis=1), np.ones(train_hard.shape[0]))
            np.testing.assert_array_equal(train_hard.argmax(axis=1), train_proba.argmax(axis=1))

            extraction_manifest = json.loads(
                (attr_mat_root / "profiling_extraction_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(extraction_manifest["representations"], ["probability", "hard"])
            self.assertEqual(
                extraction_manifest["profiling_dim"],
                extraction_manifest["hard_profiling_dim"],
            )
            self.assertEqual(extraction_manifest["author_disjointness"]["overlap_count"], 0)

            materialization_manifest = json.loads(
                (attr_mat_root / "manifest.json").read_text(encoding="utf-8")
            )
            derived_blocks = materialization_manifest["units"][0]["derived_blocks"]
            self.assertIn("profiling_oracle", derived_blocks)
            self.assertIn("profiling", derived_blocks)
            self.assertIn("profiling_female", derived_blocks)
            self.assertIn("profiling_hard", derived_blocks)
            self.assertIn("profiling_hard_female", derived_blocks)

    def test_signal_extraction_rejects_overlapping_attribution_and_profiling_authors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._build_toy_extraction_env(root, blocks=["char", "word"])
            pd.DataFrame({"id_person": ["p1", "q1"]}).to_csv(
                root / "data" / "splits" / "toy_prof" / "authors.csv",
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "author-disjoint"):
                run_profiling_signal_extraction(config_path, show_progress=False)

    def test_stylo_extraction_writes_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._build_toy_extraction_env(root, blocks=["stylo"])
            run_profiling_signal_extraction(config_path, show_progress=False)

            attr_mat_root = (
                root / "data" / "splits" / "toy_attr" / "materialized_features" / "toy_attr_mat"
            )
            train_mat = sparse.load_npz(
                attr_mat_root / "fold_01" / "matrices" / "X_train_profiling.npz"
            )
            self.assertEqual(train_mat.shape[0], 12)
            self.assertGreater(train_mat.shape[1], 0)

    def test_char_word_stylo_extraction_writes_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            config_path = self._build_toy_extraction_env(
                root, blocks=["char", "word", "stylo"]
            )
            manifest = run_profiling_signal_extraction(config_path, show_progress=False)
            self.assertEqual(manifest["targets"], ["female"])
            self.assertGreater(manifest["profiling_dim"], 0)

    def test_extraction_fails_when_attribution_stylo_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
            (root / "data").mkdir()
            # Build env with stylo block but do NOT write the raw stylometry file.
            config_path = self._build_toy_extraction_env(
                root, blocks=["stylo"], with_attribution_stylo=False
            )
            # Remove the raw stylometry file that _build_toy_extraction_env created.
            stylo_path = (
                root / "data" / "splits" / "toy_attr" / "row_features"
                / "toy_attr_rows" / "stylometry_raw.csv.gz"
            )
            if stylo_path.exists():
                stylo_path.unlink()

            with self.assertRaises(FileNotFoundError):
                run_profiling_signal_extraction(config_path, show_progress=False)

# ── Regression: checked-in profiling config validates ────────────────────────


class ProfilingConfigRegressionTests(unittest.TestCase):
    def test_profiling_linear_svm_config_validates(self) -> None:
        config_path = Path("models/configs/profiling/bokmal_profiling_linear_svm.toml")
        if not config_path.exists():
            self.skipTest("bokmal_profiling_linear_svm.toml not found (expected in repo)")
        with config_path.open("rb") as fh:
            import tomllib
            cfg = tomllib.load(fh)
        _validate_profiling_config(cfg)


if __name__ == "__main__":
    unittest.main()
