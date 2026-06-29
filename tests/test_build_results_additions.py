from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from scripts.build_results_additions import requested_sections
from thesis_reporting.artifacts import ResultArtifacts
from thesis_reporting.author_performance import write_author_performance_outputs
from thesis_reporting.confusions import write_confusion_outputs
from thesis_reporting.profile_quality import (
    build_attribution_vs_profile_correctness,
    build_profile_confusion_summary,
    build_profile_target_confusions,
    load_attribution_prediction_tables,
    load_profile_prediction_tables,
    read_profile_calibration_summary,
    read_profile_target_metrics,
    write_profile_quality_outputs,
)
from thesis_reporting.profiling_effects import write_profiling_effect_outputs
from thesis_reporting.provenance import write_manifest, write_summary
from thesis_reporting.significance import write_significance_outputs
from thesis_reporting.topk_confidence import write_topk_confidence_outputs
from thesis_reporting.config import (
    ProfileQualityRun,
    ProfileTarget,
    ResultSystem,
    SystemComparison,
)


def _system(
    key: str,
    condition_dir: str,
    *,
    label: str | None = None,
    phase: str = "phase",
    architecture: str = "direct",
    representation: str = "none",
    scope: str = "baseline",
) -> ResultSystem:
    """Create a compact result-system fixture."""

    return ResultSystem(
        key=key,
        label=label or key,
        phase=phase,
        split="fixture_split",
        architecture=architecture,
        representation=representation,
        scope=scope,
        condition_id=condition_dir,
        condition_dir=Path(condition_dir),
    )


def _author_row(
    label: str,
    display: str,
    party: str,
    *,
    support: int,
    pred_count: int,
    correct_count: int,
    precision: float,
    recall: float,
    f1: float,
) -> dict[str, object]:
    """Create one per-author metrics row."""

    return {
        "author_label": label,
        "author_display": display,
        "author_name": display,
        "author_party": party,
        "support": support,
        "pred_count": pred_count,
        "correct_count": correct_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_share": support / 20,
        "accuracy_within_true_class": recall,
    }


def _scored_author(
    label: str,
    *,
    support: int = 10,
    correct_count: int = 5,
) -> dict[str, object]:
    """Create a compact per-author row for validation-boundary cases."""

    score = correct_count / support
    return _author_row(
        label,
        f"Author {label}",
        f"Party {label}",
        support=support,
        pred_count=support,
        correct_count=correct_count,
        precision=score,
        recall=score,
        f1=score,
    )


def _write_per_author_metrics(
    remote_dir: Path,
    system: ResultSystem,
    rows: list[dict[str, object]],
) -> None:
    """Write a per-author diagnostics fixture."""

    path = remote_dir / system.per_author_metrics_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_confusion_artifacts(
    remote_dir: Path,
    system: ResultSystem,
    rows: list[dict[str, object]],
) -> None:
    """Write confusion diagnostics and copied artifact fixtures."""

    confusion_path = remote_dir / system.confusion_pairs_path
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(confusion_path, index=False)

    matrix_path = remote_dir / system.normalized_confusion_matrix_path
    pd.DataFrame(
        [[1.0, 0.0], [0.25, 0.75]],
        index=["A", "B"],
        columns=["A", "B"],
    ).to_csv(matrix_path)

    figure_path = remote_dir / system.normalized_confusion_heatmap_path
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"fake-png")


def _write_final_predictions(
    remote_dir: Path,
    system: ResultSystem,
    rows: list[dict[str, object]],
) -> None:
    """Write a final prediction fixture."""

    path = remote_dir / system.final_predictions_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_profile_quality_files(
    remote_dir: Path,
    profile_run: ProfileQualityRun,
    target: ProfileTarget,
    prediction_rows: list[dict[str, object]],
) -> None:
    """Write minimal profile-quality metric and prediction fixtures."""

    quality_dir = remote_dir / profile_run.quality_dir
    quality_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "target": target.prediction_file_key,
                "unit_id": "final_test",
                "role": "test",
                "n_samples": len(prediction_rows),
                "n_true_classes": 2,
                "n_model_classes": 2,
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "log_loss": 0.7,
                "brier_score": 0.4,
                "expected_calibration_error": 0.1,
                "mean_max_probability": 0.7,
                "mean_correct_probability": 0.8,
                "mean_incorrect_probability": 0.6,
                "labels_missing_from_model": "",
                "majority_label": (
                    prediction_rows[0].get("y_true", "A") if prediction_rows else "A"
                ),
                "majority_accuracy": 0.5,
                "majority_macro_f1": 0.33,
                "macro_f1_lift_over_majority": 0.17,
                "accuracy_lift_over_majority": 0.0,
            }
        ]
    ).to_csv(quality_dir / "attribution_test_profile_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": target.prediction_file_key,
                "role": "test",
                "n_samples": len(prediction_rows),
                "log_loss": 0.7,
                "brier_score": 0.4,
                "expected_calibration_error": 0.1,
                "mean_max_probability": 0.7,
                "mean_correct_probability": 0.8,
                "mean_incorrect_probability": 0.6,
            }
        ]
    ).to_csv(quality_dir / "calibration_summary.csv", index=False)
    prediction_path = remote_dir / profile_run.prediction_path(target, "test")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)


class ResultsAdditionsTests(unittest.TestCase):
    """Tests for thesis result-addition post-processing scripts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one reusable temporary result tree for this test class."""

        super().setUpClass()
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._result_root = Path(cls._temporary_directory.name)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the shared temporary result tree after all tests finish."""

        cls._temporary_directory.cleanup()
        super().tearDownClass()

    @contextmanager
    def _result_tree(self):
        """Yield one clean result/output workspace backed by the shared root."""

        remote_dir = self._result_root / "remote"
        output_dir = self._result_root / "out"
        for path in (remote_dir, output_dir):
            if path.exists():
                shutil.rmtree(path)
        yield self._result_root, remote_dir, output_dir

    def test_requested_all_sections_include_profile_effects(self) -> None:
        """Verify valid shortcuts and concrete section lists expand exactly."""

        csv_sections = {
            "author_performance",
            "confusions",
            "profiling_effects",
            "topk_confidence",
            "profile_quality",
            "significance",
        }
        cases = (
            ("all", csv_sections),
            (
                "all_with_feature_importance",
                csv_sections | {"feature_importance"},
            ),
            (
                "author_performance,significance",
                {"author_performance", "significance"},
            ),
        )
        for raw_sections, expected in cases:
            with self.subTest(raw_sections=raw_sections):
                self.assertEqual(requested_sections(raw_sections), expected)

    def test_requested_sections_reject_every_invalid_token(self) -> None:
        """Ensure shortcuts cannot hide invalid section tokens."""

        for sections in ("all,typo", "significance,unknown"):
            with self.subTest(sections=sections):
                with self.assertRaisesRegex(ValueError, "Unsupported section"):
                    requested_sections(sections)

    def test_author_performance_outputs_rank_best_and_worst_authors(self) -> None:
        """Verify author-performance outputs rank authors by expected metrics."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system("system_a", "system_a")
            _write_per_author_metrics(
                remote_dir,
                system,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=10,
                        pred_count=9,
                        correct_count=9,
                        precision=1.0,
                        recall=0.9,
                        f1=0.95,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=10,
                        pred_count=10,
                        correct_count=6,
                        precision=0.6,
                        recall=0.6,
                        f1=0.6,
                    ),
                    _author_row(
                        "C",
                        "Author C",
                        "Party 2",
                        support=10,
                        pred_count=11,
                        correct_count=3,
                        precision=0.27,
                        recall=0.3,
                        f1=0.28,
                    ),
                ],
            )

            paths = write_author_performance_outputs(
                (system,),
                results_dir=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                top_n=1,
            )

            rankings = pd.read_csv(paths["per_author_rankings"])
            best = pd.read_csv(paths["best_authors"])
            worst = pd.read_csv(paths["worst_authors"])

            author_a = rankings.loc[rankings["author_label"] == "A"].iloc[0]
            author_c = rankings.loc[rankings["author_label"] == "C"].iloc[0]
            self.assertEqual(author_a["rank_f1_desc"], 1)
            self.assertEqual(author_c["rank_error_count_desc"], 1)
            self.assertEqual(
                best.loc[best["ranking_metric"] == "f1", "author_label"].iloc[0],
                "A",
            )
            self.assertEqual(
                worst.loc[
                    worst["ranking_metric"] == "error_count",
                    "author_label",
                ].iloc[0],
                "C",
            )

    def test_confusion_outputs_aggregate_pair_and_party_counts(self) -> None:
        """Verify confusion outputs aggregate directed errors into thesis tables."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system("system_a", "system_a")
            _write_confusion_artifacts(
                remote_dir,
                system,
                [
                    {
                        "y_true_label": "A",
                        "y_true_display": "Author A",
                        "y_true_party": "Party 1",
                        "y_pred_label": "B",
                        "y_pred_display": "Author B",
                        "y_pred_party": "Party 1",
                        "count": 2,
                        "error_share": 0.33,
                        "p_pred_given_true": 0.2,
                    },
                    {
                        "y_true_label": "B",
                        "y_true_display": "Author B",
                        "y_true_party": "Party 1",
                        "y_pred_label": "A",
                        "y_pred_display": "Author A",
                        "y_pred_party": "Party 1",
                        "count": 3,
                        "error_share": 0.5,
                        "p_pred_given_true": 0.3,
                    },
                    {
                        "y_true_label": "A",
                        "y_true_display": "Author A",
                        "y_true_party": "Party 1",
                        "y_pred_label": "C",
                        "y_pred_display": "Author C",
                        "y_pred_party": "Party 2",
                        "count": 1,
                        "error_share": 0.17,
                        "p_pred_given_true": 0.1,
                    },
                ],
            )

            paths = write_confusion_outputs(
                (system,),
                results_dir=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                top_n=5,
            )

            symmetric = pd.read_csv(paths["symmetric_confusion_pairs"])
            party_summary = pd.read_csv(paths["party_confusion_summary"])
            author_pair = symmetric.loc[
                (symmetric["author_a_label"] == "A")
                & (symmetric["author_b_label"] == "B")
            ].iloc[0]
            same_party = party_summary.loc[
                party_summary["confusion_group"] == "same_party"
            ].iloc[0]
            cross_party = party_summary.loc[
                party_summary["confusion_group"] == "cross_party"
            ].iloc[0]

            self.assertEqual(author_pair["total_pair_confusions"], 5)
            self.assertEqual(same_party["confusion_count"], 5)
            self.assertEqual(cross_party["confusion_count"], 1)
            self.assertTrue(Path(paths["normalized_matrix_system_a"]).exists())
            self.assertEqual(
                Path(paths["normalized_heatmap_system_a"]).read_bytes(),
                b"fake-png",
            )

    def test_confusion_outputs_require_configured_heatmaps(self) -> None:
        """Verify missing heatmaps fail as artifact-integrity errors."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system("system_a", "system_a")
            confusion_path = remote_dir / system.confusion_pairs_path
            confusion_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "y_true_label": "A",
                        "y_true_display": "Author A",
                        "y_true_party": "Party 1",
                        "y_pred_label": "B",
                        "y_pred_display": "Author B",
                        "y_pred_party": "Party 2",
                        "count": 1,
                        "error_share": 1.0,
                        "p_pred_given_true": 0.1,
                    }
                ]
            ).to_csv(confusion_path, index=False)
            matrix_path = remote_dir / system.normalized_confusion_matrix_path
            pd.DataFrame([[1.0]], index=["A"], columns=["A"]).to_csv(matrix_path)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Required normalized confusion heatmap",
            ):
                write_confusion_outputs(
                    (system,),
                    results_dir=ResultArtifacts(remote_dir),
                    output_dir=output_dir,
                    top_n=5,
                )

    def test_profiling_effect_outputs_compute_author_deltas(self) -> None:
        """Verify profiling-effect outputs capture gains, losses, and oracle gaps."""

        with self._result_tree() as (root, remote_dir, output_dir):
            baseline = _system("baseline", "baseline")
            predicted = _system(
                "predicted",
                "predicted",
                representation="probability",
                scope="all_signal",
            )
            oracle = _system(
                "oracle",
                "oracle",
                representation="oracle",
                scope="all_signal",
            )
            _write_per_author_metrics(
                remote_dir,
                baseline,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=10,
                        pred_count=10,
                        correct_count=5,
                        precision=0.5,
                        recall=0.5,
                        f1=0.5,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=10,
                        pred_count=10,
                        correct_count=8,
                        precision=0.8,
                        recall=0.8,
                        f1=0.8,
                    ),
                ],
            )
            _write_per_author_metrics(
                remote_dir,
                predicted,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=10,
                        pred_count=10,
                        correct_count=7,
                        precision=0.7,
                        recall=0.7,
                        f1=0.7,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=10,
                        pred_count=10,
                        correct_count=6,
                        precision=0.6,
                        recall=0.6,
                        f1=0.6,
                    ),
                ],
            )
            _write_per_author_metrics(
                remote_dir,
                oracle,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=10,
                        pred_count=10,
                        correct_count=9,
                        precision=0.9,
                        recall=0.9,
                        f1=0.9,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=10,
                        pred_count=10,
                        correct_count=7,
                        precision=0.7,
                        recall=0.7,
                        f1=0.7,
                    ),
                ],
            )
            comparisons = (
                SystemComparison(
                    key="baseline_to_predicted",
                    label="Baseline to predicted",
                    source_system_key="baseline",
                    target_system_key="predicted",
                    purpose="Predicted profile effect",
                    comparison_group="predicted_profile_effect",
                ),
                SystemComparison(
                    key="predicted_to_oracle",
                    label="Predicted to oracle",
                    source_system_key="predicted",
                    target_system_key="oracle",
                    purpose="Oracle gap",
                    comparison_group="oracle_predicted_gap",
                ),
            )

            paths = write_profiling_effect_outputs(
                (baseline, predicted, oracle),
                comparisons,
                results_dir=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                top_n=1,
            )

            deltas = pd.read_csv(paths["per_author_profile_deltas"])
            summary = pd.read_csv(paths["profile_delta_summary"])
            top_gains = pd.read_csv(paths["top_profile_gains"])
            top_losses = pd.read_csv(paths["top_profile_losses"])
            oracle_gap = pd.read_csv(paths["oracle_predicted_gap_by_author"])

            author_a = deltas.loc[
                (deltas["comparison_key"] == "baseline_to_predicted")
                & (deltas["author_label"] == "A")
            ].iloc[0]
            author_b = deltas.loc[
                (deltas["comparison_key"] == "baseline_to_predicted")
                & (deltas["author_label"] == "B")
            ].iloc[0]
            baseline_summary = summary.loc[
                summary["comparison_key"] == "baseline_to_predicted"
            ].iloc[0]

            self.assertAlmostEqual(author_a["f1_delta"], 0.2)
            self.assertEqual(author_a["correct_count_delta"], 2)
            self.assertEqual(author_a["error_count_delta"], -2)
            self.assertEqual(author_a["f1_delta_direction"], "improved")
            self.assertEqual(author_a["error_count_delta_direction"], "fewer_errors")
            self.assertAlmostEqual(author_b["f1_delta"], -0.2)
            self.assertEqual(baseline_summary["authors_f1_improved"], 1)
            self.assertEqual(baseline_summary["authors_f1_worse"], 1)
            self.assertEqual(
                top_gains.loc[
                    top_gains["comparison_key"] == "baseline_to_predicted",
                    "author_label",
                ].iloc[0],
                "A",
            )
            self.assertEqual(
                top_losses.loc[
                    top_losses["comparison_key"] == "baseline_to_predicted",
                    "author_label",
                ].iloc[0],
                "B",
            )
            self.assertEqual(
                set(oracle_gap["comparison_key"]),
                {"predicted_to_oracle"},
            )

    def test_profiling_effect_outputs_validate_author_alignment(self) -> None:
        """Verify author comparisons reject missing, duplicate, and shifted support."""

        baseline = _system("baseline", "baseline")
        predicted = _system("predicted", "predicted")
        comparison = SystemComparison(
            key="baseline_to_predicted",
            label="Baseline to predicted",
            source_system_key="baseline",
            target_system_key="predicted",
            purpose="Fixture",
            comparison_group="fixture",
        )
        cases = (
            (
                "missing author",
                [_scored_author("A"), _scored_author("B")],
                [_scored_author("A", correct_count=6)],
                "same '_author_key' set",
            ),
            (
                "duplicate author",
                [_scored_author("A"), _scored_author("A", correct_count=6)],
                [_scored_author("A")],
                "duplicate 'author_label'",
            ),
            (
                "support mismatch",
                [_scored_author("A")],
                [_scored_author("A", support=9)],
                "identical per-author support",
            ),
        )
        for case_name, baseline_rows, predicted_rows, error_pattern in cases:
            with self.subTest(case=case_name):
                with self._result_tree() as (_, remote_dir, output_dir):
                    _write_per_author_metrics(remote_dir, baseline, baseline_rows)
                    _write_per_author_metrics(remote_dir, predicted, predicted_rows)

                    with self.assertRaisesRegex(ValueError, error_pattern):
                        write_profiling_effect_outputs(
                            (baseline, predicted),
                            (comparison,),
                            results_dir=ResultArtifacts(remote_dir),
                            output_dir=output_dir,
                            top_n=1,
                        )

    def test_manifest_and_summary_include_comparisons(self) -> None:
        """Verify provenance files include configured comparison metadata."""

        with self._result_tree() as (root, remote_dir, output_dir):
            output_dir.mkdir()
            system = _system("baseline", "baseline")
            comparison = SystemComparison(
                key="baseline_to_predicted",
                label="Baseline to predicted",
                source_system_key="baseline",
                target_system_key="predicted",
                purpose="Predicted profile effect",
                comparison_group="predicted_profile_effect",
            )
            profile_run = ProfileQualityRun(
                key="profile_quality",
                label="Profile quality",
                quality_dir=Path("profile_quality"),
            )
            profile_target = ProfileTarget(
                key="party",
                label="Party",
                prediction_file_key="party",
            )
            outputs = {
                "profiling_effects": {
                    "per_author_profile_deltas": str(
                        output_dir / "profiling_effects/per_author_profile_deltas.csv"
                    )
                }
            }

            manifest_path = write_manifest(
                project_root=root,
                output_dir=output_dir,
                results_dir=remote_dir,
                data_dir=root / "data",
                sections={"profiling_effects"},
                systems=(system,),
                comparisons=(comparison,),
                profile_run=profile_run,
                profile_targets=(profile_target,),
                outputs=outputs,
            )
            summary_path = write_summary(
                project_root=root,
                output_dir=output_dir,
                results_dir=remote_dir,
                data_dir=root / "data",
                systems=(system,),
                comparisons=(comparison,),
                profile_targets=(profile_target,),
                outputs=outputs,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = summary_path.read_text(encoding="utf-8")

            self.assertEqual(
                manifest["comparisons"][0]["key"],
                "baseline_to_predicted",
            )
            self.assertEqual(manifest["results_dir"], "remote")
            self.assertEqual(manifest["data_dir"], "data")
            self.assertIn("## Profiling Effects", summary)
            self.assertIn("baseline_to_predicted", summary)

    def test_topk_confidence_outputs_compute_rescues_and_slices(self) -> None:
        """Verify top-k rescue and confidence-slice outputs."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system("system_a", "system_a")
            _write_per_author_metrics(
                remote_dir,
                system,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=2,
                        pred_count=1,
                        correct_count=1,
                        precision=1.0,
                        recall=0.5,
                        f1=0.67,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=1,
                        pred_count=1,
                        correct_count=0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                    ),
                    _author_row(
                        "C",
                        "Author C",
                        "Party 3",
                        support=1,
                        pred_count=2,
                        correct_count=1,
                        precision=0.5,
                        recall=1.0,
                        f1=0.67,
                    ),
                ],
            )
            _write_final_predictions(
                remote_dir,
                system,
                [
                    {
                        "row_idx": 0,
                        "id_speech": 1,
                        "election": 2021,
                        "party": "Party 1",
                        "y_true": "A",
                        "y_pred": "A",
                        "top1_label": "A",
                        "top1_score": 0.9,
                        "top2_label": "B",
                        "top2_score": 0.2,
                        "top3_label": "C",
                        "top3_score": 0.1,
                    },
                    {
                        "row_idx": 1,
                        "id_speech": 2,
                        "election": 2021,
                        "party": "Party 1",
                        "y_true": "A",
                        "y_pred": "B",
                        "top1_label": "B",
                        "top1_score": 0.8,
                        "top2_label": "A",
                        "top2_score": 0.7,
                        "top3_label": "C",
                        "top3_score": 0.1,
                    },
                    {
                        "row_idx": 2,
                        "id_speech": 3,
                        "election": 2021,
                        "party": "Party 2",
                        "y_true": "B",
                        "y_pred": "C",
                        "top1_label": "C",
                        "top1_score": 0.9,
                        "top2_label": "A",
                        "top2_score": 0.2,
                        "top3_label": "B",
                        "top3_score": 0.1,
                    },
                    {
                        "row_idx": 3,
                        "id_speech": 4,
                        "election": 2021,
                        "party": "Party 3",
                        "y_true": "C",
                        "y_pred": "C",
                        "top1_label": "C",
                        "top1_score": 0.51,
                        "top2_label": "A",
                        "top2_score": 0.5,
                        "top3_label": "B",
                        "top3_score": 0.2,
                    },
                ],
            )

            paths = write_topk_confidence_outputs(
                (system,),
                results_dir=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                top_n=1,
            )

            overall = pd.read_csv(paths["topk_rescue_summary"])
            per_author = pd.read_csv(paths["topk_rescue_by_author"])
            confident_errors = pd.read_csv(paths["confident_errors"])
            low_margin = pd.read_csv(paths["uncertain_correct"])

            k2 = overall.loc[overall["k"] == 2].iloc[0]
            k3 = overall.loc[overall["k"] == 3].iloc[0]
            author_a_k2 = per_author.loc[
                (per_author["author_label"] == "A")
                & (per_author["k"] == 2)
            ].iloc[0]

            self.assertEqual(k2["topk_correct"], 3)
            self.assertEqual(k2["rescue_count"], 1)
            self.assertEqual(k3["topk_correct"], 4)
            self.assertEqual(k3["rescue_count"], 2)
            self.assertEqual(author_a_k2["topk_correct"], 2)
            self.assertEqual(author_a_k2["rescue_count"], 1)
            self.assertAlmostEqual(author_a_k2["rescue_rate_among_top1_errors"], 1.0)
            self.assertEqual(confident_errors["id_speech"].iloc[0], 3)
            self.assertEqual(low_margin["id_speech"].iloc[0], 4)

    def test_profile_quality_outputs_join_profile_and_attribution_correctness(self) -> None:
        """Verify profile-quality outputs and attribution correctness joins."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system(
                "phase3a_probability_all",
                "phase3a_probability_all",
                representation="probability",
                scope="all_signal",
            )
            profile_run = ProfileQualityRun(
                key="profile_quality",
                label="Profile quality",
                quality_dir=Path("profile_quality"),
            )
            target = ProfileTarget(
                key="party",
                label="Party",
                prediction_file_key="party",
            )
            _write_per_author_metrics(
                remote_dir,
                system,
                [
                    _author_row(
                        "A",
                        "Author A",
                        "Party 1",
                        support=1,
                        pred_count=0,
                        correct_count=0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                    ),
                    _author_row(
                        "B",
                        "Author B",
                        "Party 2",
                        support=1,
                        pred_count=2,
                        correct_count=1,
                        precision=0.5,
                        recall=1.0,
                        f1=0.67,
                    ),
                ],
            )
            _write_final_predictions(
                remote_dir,
                system,
                [
                    {
                        "row_idx": 0,
                        "id_speech": 1,
                        "election": 2021,
                        "party": "Party 1",
                        "y_true": "A",
                        "y_pred": "B",
                        "top1_label": "B",
                        "top1_score": 0.9,
                        "top2_label": "A",
                        "top2_score": 0.2,
                    },
                    {
                        "row_idx": 1,
                        "id_speech": 2,
                        "election": 2021,
                        "party": "Party 2",
                        "y_true": "B",
                        "y_pred": "B",
                        "top1_label": "B",
                        "top1_score": 0.8,
                        "top2_label": "A",
                        "top2_score": 0.3,
                    },
                ],
            )
            _write_profile_quality_files(
                remote_dir,
                profile_run,
                target,
                [
                    {
                        "row_idx": 0,
                        "id_speech": 1,
                        "id_person": "A",
                        "fold_id": "final_test",
                        "role": "test",
                        "election": 2021,
                        "party": "Party 1",
                        "language": "Bokmal",
                        "author": "A",
                        "profile_target": "party",
                        "y_true": "Party 1",
                        "y_pred": "Party 2",
                        "correct": False,
                        "confidence": 0.95,
                    },
                    {
                        "row_idx": 1,
                        "id_speech": 2,
                        "id_person": "B",
                        "fold_id": "final_test",
                        "role": "test",
                        "election": 2021,
                        "party": "Party 2",
                        "language": "Bokmal",
                        "author": "B",
                        "profile_target": "party",
                        "y_true": "Party 2",
                        "y_pred": "Party 2",
                        "correct": True,
                        "confidence": 0.8,
                    },
                ],
            )

            paths = write_profile_quality_outputs(
                (system,),
                profile_run,
                (target,),
                results_dir=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                top_n=1,
            )

            target_metrics = pd.read_csv(paths["profile_target_metrics"])
            confusions = pd.read_csv(paths["profile_confusions_by_target"])
            confusion_summary = pd.read_csv(paths["profile_confusion_summary"])
            confidence_summary = pd.read_csv(paths["profile_confidence_summary"])
            confident_errors = pd.read_csv(paths["profile_confident_errors"])
            correctness_detail = pd.read_csv(paths["attribution_vs_profile_correctness"])
            correctness = pd.read_csv(paths["attribution_vs_profile_correctness_summary"])
            joint_errors = pd.read_csv(
                paths["profile_wrong_attribution_wrong_examples"]
            )

            wrong_wrong = correctness.loc[
                (~correctness["profile_correct"])
                & (~correctness["attribution_correct"])
            ].iloc[0]
            correct_correct = correctness.loc[
                correctness["profile_correct"]
                & correctness["attribution_correct"]
            ].iloc[0]

            self.assertEqual(target_metrics["target_label"].iloc[0], "Party")
            self.assertEqual(confusions["confusion_count"].iloc[0], 1)
            self.assertEqual(
                confusion_summary["most_common_confusion_count"].iloc[0],
                1,
            )
            self.assertEqual(
                set(confidence_summary["profile_correct_group"]),
                {"profile_correct", "profile_wrong"},
            )
            self.assertEqual(confident_errors["id_speech"].iloc[0], 1)
            self.assertEqual(len(correctness_detail), 2)
            self.assertEqual(wrong_wrong["speech_count"], 1)
            self.assertEqual(correct_correct["speech_count"], 1)
            self.assertEqual(joint_errors["id_speech"].iloc[0], 1)

    def test_profile_quality_outputs_canonicalize_historical_target_key(self) -> None:
        """Verify remote left_senter_right metrics are output as left_center_right."""

        with self._result_tree() as (root, remote_dir, output_dir):
            profile_run = ProfileQualityRun(
                key="profile_quality",
                label="Profile quality",
                quality_dir=Path("profile_quality"),
            )
            target = ProfileTarget(
                key="left_center_right",
                label="Left-center-right bloc",
                prediction_file_key="left_senter_right",
            )
            _write_profile_quality_files(
                remote_dir,
                profile_run,
                target,
                [
                    {
                        "row_idx": 0,
                        "id_speech": 1,
                        "id_person": "A",
                        "fold_id": "final_test",
                        "role": "test",
                        "election": 2021,
                        "party": "Sp",
                        "language": "Bokmal",
                        "author": "A",
                        "profile_target": "left_senter_right",
                        "y_true": "senter",
                        "y_pred": "left",
                        "correct": False,
                        "confidence": 0.9,
                    }
                ],
            )

            artifacts = ResultArtifacts(remote_dir)
            metrics = read_profile_target_metrics(
                profile_run,
                (target,),
                results_dir=artifacts,
            )
            calibration = read_profile_calibration_summary(
                profile_run,
                (target,),
                results_dir=artifacts,
            )
            profile_predictions = load_profile_prediction_tables(
                profile_run,
                (target,),
                artifacts,
            )
            summary = build_profile_confusion_summary(
                metrics,
                build_profile_target_confusions(profile_predictions),
            )

            self.assertEqual(metrics["target"].iloc[0], "left_center_right")
            self.assertEqual(
                metrics["target_label"].iloc[0],
                "Left-center-right bloc",
            )
            self.assertEqual(calibration["target"].iloc[0], "left_center_right")
            self.assertEqual(summary["target"].iloc[0], "left_center_right")
            self.assertEqual(metrics["majority_label"].iloc[0], "center")
            self.assertEqual(summary["majority_label"].iloc[0], "center")
            self.assertEqual(summary["most_common_true_profile"].iloc[0], "center")
            self.assertEqual(summary["most_common_pred_profile"].iloc[0], "left")

    def test_profile_attribution_join_rejects_missing_profile_ids(self) -> None:
        """Verify profile/attribution joins require exact speech-id coverage."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system(
                "phase3a_probability_all",
                "phase3a_probability_all",
                representation="probability",
                scope="all_signal",
            )
            profile_run = ProfileQualityRun(
                key="profile_quality",
                label="Profile quality",
                quality_dir=Path("profile_quality"),
            )
            target = ProfileTarget(
                key="party",
                label="Party",
                prediction_file_key="party",
            )
            _write_final_predictions(
                remote_dir,
                system,
                [
                    {
                        "id_speech": 1,
                        "y_true": "A",
                        "y_pred": "A",
                        "top1_label": "A",
                        "top1_score": 0.9,
                        "top2_label": "B",
                        "top2_score": 0.1,
                    },
                    {
                        "id_speech": 2,
                        "y_true": "B",
                        "y_pred": "A",
                        "top1_label": "A",
                        "top1_score": 0.8,
                        "top2_label": "B",
                        "top2_score": 0.7,
                    },
                ],
            )
            _write_profile_quality_files(
                remote_dir,
                profile_run,
                target,
                [
                    {
                        "id_speech": 1,
                        "id_person": "A",
                        "role": "test",
                        "election": 2021,
                        "party": "Party 1",
                        "y_true": "Party 1",
                        "y_pred": "Party 1",
                        "correct": True,
                        "confidence": 0.8,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "same 'id_speech_key' set"):
                artifacts = ResultArtifacts(remote_dir)
                attribution_predictions = load_attribution_prediction_tables(
                    (system,),
                    artifacts,
                )
                build_attribution_vs_profile_correctness(
                    (system,),
                    (target,),
                    attribution_predictions,
                    load_profile_prediction_tables(
                        profile_run,
                        (target,),
                        artifacts,
                    ),
                )

    def test_profile_attribution_join_rejects_duplicate_attribution_ids(self) -> None:
        """Verify profile/attribution joins reject duplicate attribution rows."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system = _system(
                "phase3a_probability_all",
                "phase3a_probability_all",
                representation="probability",
                scope="all_signal",
            )
            profile_run = ProfileQualityRun(
                key="profile_quality",
                label="Profile quality",
                quality_dir=Path("profile_quality"),
            )
            target = ProfileTarget(
                key="party",
                label="Party",
                prediction_file_key="party",
            )
            _write_final_predictions(
                remote_dir,
                system,
                [
                    {
                        "id_speech": 1,
                        "y_true": "A",
                        "y_pred": "A",
                        "top1_label": "A",
                        "top1_score": 0.9,
                        "top2_label": "B",
                        "top2_score": 0.1,
                    },
                    {
                        "id_speech": 1,
                        "y_true": "A",
                        "y_pred": "B",
                        "top1_label": "B",
                        "top1_score": 0.8,
                        "top2_label": "A",
                        "top2_score": 0.7,
                    },
                ],
            )
            _write_profile_quality_files(
                remote_dir,
                profile_run,
                target,
                [
                    {
                        "id_speech": 1,
                        "id_person": "A",
                        "role": "test",
                        "election": 2021,
                        "party": "Party 1",
                        "y_true": "Party 1",
                        "y_pred": "Party 1",
                        "correct": True,
                        "confidence": 0.8,
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate 'id_speech_key'"):
                artifacts = ResultArtifacts(remote_dir)
                attribution_predictions = load_attribution_prediction_tables(
                    (system,),
                    artifacts,
                )
                build_attribution_vs_profile_correctness(
                    (system,),
                    (target,),
                    attribution_predictions,
                    load_profile_prediction_tables(
                        profile_run,
                        (target,),
                        artifacts,
                    ),
                )

    def test_significance_outputs_write_summary_and_json(self) -> None:
        """Verify significance outputs call the comparison helper and write files."""

        with self._result_tree() as (root, remote_dir, output_dir):
            system_a = _system("system_a", "system_a", label="System A")
            system_b = _system("system_b", "system_b", label="System B")
            _write_final_predictions(
                remote_dir,
                system_a,
                [
                    {"id_speech": 1, "y_true": "X", "y_pred": "X", "correct": True},
                    {"id_speech": 2, "y_true": "Y", "y_pred": "Y", "correct": True},
                    {"id_speech": 3, "y_true": "X", "y_pred": "Y", "correct": False},
                    {"id_speech": 4, "y_true": "Y", "y_pred": "Y", "correct": True},
                ],
            )
            _write_final_predictions(
                remote_dir,
                system_b,
                [
                    {"id_speech": 1, "y_true": "X", "y_pred": "X", "correct": True},
                    {"id_speech": 2, "y_true": "Y", "y_pred": "X", "correct": False},
                    {"id_speech": 3, "y_true": "X", "y_pred": "X", "correct": True},
                    {"id_speech": 4, "y_true": "Y", "y_pred": "X", "correct": False},
                ],
            )
            comparison = SystemComparison(
                key="system_a_vs_system_b",
                label="System A versus System B",
                source_system_key="system_a",
                target_system_key="system_b",
                purpose="Fixture comparison",
                comparison_group="fixture",
            )

            paths = write_significance_outputs(
                (system_a, system_b),
                (comparison,),
                artifacts=ResultArtifacts(remote_dir),
                output_dir=output_dir,
                project_root=root,
                n_bootstrap=20,
                seed=7,
            )

            summary = pd.read_csv(paths["comparisons_csv"])
            combined_json_path = Path(paths["comparisons_json"])
            json_path = Path(paths["system_a_vs_system_b_json"])
            result = json.loads(json_path.read_text(encoding="utf-8"))
            combined_result = json.loads(combined_json_path.read_text(encoding="utf-8"))

            self.assertEqual(summary["comparison_key"].iloc[0], "system_a_vs_system_b")
            self.assertEqual(summary["n_speeches"].iloc[0], 4)
            self.assertEqual(summary["bootstrap_n_iterations"].iloc[0], 20)
            self.assertTrue(json_path.exists())
            self.assertTrue(combined_json_path.exists())
            self.assertEqual(result["n_speeches"], 4)
            self.assertEqual(
                combined_result["system_a_vs_system_b"]["n_speeches"],
                4,
            )


if __name__ == "__main__":
    unittest.main()
