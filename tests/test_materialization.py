from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from data_pipeline.materialization import run_materialization


class MaterializationTests(unittest.TestCase):
    def _materialization_config_text(
        self,
        *,
        blocks: list[str] | None = None,
        drop_zero_variance_columns: bool = False,
    ) -> str:
        """Build the smallest configurable materialization TOML needed by the tests in this class."""
        enabled_blocks = ["char", "word", "stylo"] if blocks is None else blocks
        block_values = ", ".join(repr(block) for block in enabled_blocks)
        lines = [
            "[materialization]",
            "split_name = 'toy_split'",
            "row_feature_name = 'toy_features'",
            "",
            "[data]",
            "splits_dir = 'data/splits'",
            "",
            "[stages.dev]",
            "name = 'toy_materialization'",
            "selector = 'all'",
            "",
            "[stages.final]",
            "name = 'toy_final_materialization'",
            "selector = 'final'",
            "",
            "[blocks]",
            f"enabled = [{block_values}]",
            "",
        ]
        lines.extend(
            [
                "[word_tfidf]",
                "min_n = 1",
                "max_n = 1",
                "min_df = 1",
                "max_df = 1.0",
                "max_features = 10",
                "",
                "[char_tfidf]",
                "min_n = 2",
                "max_n = 2",
                "min_df = 1",
                "max_df = 1.0",
                "max_features = 10",
                "",
            ]
        )
        if "stylo" in enabled_blocks:
            lines.extend(
                [
                    "[stylometry]",
                    "scaler = 'standard'",
                    "with_mean = true",
                    "with_std = true",
                ]
            )
            if drop_zero_variance_columns:
                lines.append("drop_zero_variance_columns = true")
            lines.append("")
        return "\n".join(lines) + "\n"

    def _build_toy_project(
        self,
        tmpdir: str,
        *,
        row_stylo: pd.DataFrame,
        config_text: str,
        stylometry_generated: bool = True,
    ) -> tuple[Path, Path]:
        """Create a complete toy split layout so each test only needs to vary the pieces it cares about."""
        project_root = Path(tmpdir)
        (project_root / "pyproject.toml").write_text(
            "[project]\nname='toy'\nversion='0.1.0'\n",
            encoding="utf-8",
        )
        split_dir = project_root / "data" / "splits" / "toy_split"
        corpus_dir = split_dir / "corpus"
        memberships_dir = split_dir / "memberships"
        row_feature_dir = split_dir / "row_features" / "toy_features"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        memberships_dir.mkdir(parents=True, exist_ok=True)
        row_feature_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "manifest.json").write_text(
            json.dumps(
                {"split_name": "toy_split", "split_strategy": "fixed_test_speeches"},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4],
                "id_person": [101, 102, 101, 102],
                "text": ["alpha alpha", "beta beta", "alpha beta", "beta alpha"],
                "election": [2017, 2017, 2021, 2021],
                "word_count": [2, 2, 2, 2],
                "char_count": [11, 9, 10, 10],
            }
        ).to_csv(corpus_dir / "all.csv", index=False)

        pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4],
                "id_person": [101, 102, 101, 102],
                "election": [2017, 2017, 2021, 2021],
                "party": ["A", "B", "A", "B"],
                "language": ["Bokmal"] * 4,
                "split_name": ["toy_split"] * 4,
                "outer_role": ["train", "train", "test", "test"],
            }
        ).to_csv(memberships_dir / "outer.csv", index=False)

        pd.DataFrame(
            {
                "fold_id": ["fold_01"] * 4,
                "id_speech": [1, 2, 3, 4],
                "id_person": [101, 102, 101, 102],
                "election": [2017, 2017, 2021, 2021],
                "party": ["A", "B", "A", "B"],
                "language": ["Bokmal"] * 4,
                "split_name": ["toy_split"] * 4,
                "fold_role": ["train", "train", "val", "val"],
            }
        ).to_csv(memberships_dir / "folds.csv", index=False)

        pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4],
                "id_person": [101, 102, 101, 102],
                "election": [2017, 2017, 2021, 2021],
                "name": ["Author 101", "Author 102", "Author 101", "Author 102"],
                "party": ["A", "B", "A", "B"],
                "female": [0, 1, 0, 1],
                "age": [50, 51, 50, 51],
                "language": ["Bokmal"] * 4,
            }
        ).to_csv(row_feature_dir / "row_meta.csv", index=False)

        pd.DataFrame(
            {
                "id_speech": [1, 2, 3, 4],
                "id_person": [101, 102, 101, 102],
                "author": ["101", "102", "101", "102"],
                "party": ["A", "B", "A", "B"],
                "female": [0, 1, 0, 1],
                "language": ["Bokmal"] * 4,
            }
        ).to_csv(row_feature_dir / "targets.csv", index=False)

        artifacts = {
            "row_meta": "row_meta.csv",
            "targets": "targets.csv",
        }
        if stylometry_generated:
            row_stylo.to_csv(row_feature_dir / "stylometry_raw.csv.gz", index=False, compression="gzip")
            artifacts["stylometry_raw"] = "stylometry_raw.csv.gz"
        (row_feature_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "split_name": "toy_split",
                    "feature_set_name": "toy_features",
                    "row_counts": {"train": 2, "test": 2},
                    "stylometry": {
                        "generated": stylometry_generated,
                        "n_features": int(len(row_stylo.columns) - 2) if stylometry_generated else 0,
                    },
                    "artifacts": artifacts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        config_path = project_root / "toy_materialization.toml"
        config_path.write_text(config_text, encoding="utf-8")
        return project_root, config_path

    def test_run_materialization_writes_feature_columns_manifest_and_raw_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_a": [0.1, 0.2, 0.3, 0.4],
                        "stylo_b": [1.0, 0.0, 1.0, 0.0],
                    }
                ),
                config_text=self._materialization_config_text(),
            )

            summary = run_materialization(config_path, stage="dev")
            materialized_root = project_root / summary["materialized_root"]
            unit_dir = materialized_root / "fold_01"
            feature_columns = json.loads((unit_dir / "feature_columns.json").read_text(encoding="utf-8"))

            self.assertEqual(
                len(feature_columns["char"]),
                sparse.load_npz(unit_dir / "matrices" / "X_train_char.npz").shape[1],
            )
            self.assertEqual(
                len(feature_columns["word"]),
                sparse.load_npz(unit_dir / "matrices" / "X_train_word.npz").shape[1],
            )
            self.assertEqual(
                len(feature_columns["stylo"]),
                sparse.load_npz(unit_dir / "matrices" / "X_train_stylo.npz").shape[1],
            )

            unit_manifest = json.loads((unit_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(unit_manifest["dimensions"]["combined"]["generated"])
            self.assertEqual(summary["stage"], "dev")
            self.assertEqual(summary["selector"], "all")
            self.assertEqual(unit_manifest["dimensions"]["combined"]["cols"], 0)
            self.assertEqual(unit_manifest["enabled_blocks"], ["char", "word", "stylo"])
            self.assertFalse(summary["combined_available"])
            self.assertFalse(summary["units"][0]["combined_available"])
            self.assertFalse((unit_dir / "combined" / "X_train_all.npz").exists())
            self.assertFalse((unit_dir / "combined" / "X_val_all.npz").exists())
            self.assertEqual(
                unit_manifest["feature_columns"]["counts"]["char"],
                len(feature_columns["char"]),
            )
            self.assertEqual(
                unit_manifest["feature_columns"]["counts"]["word"],
                len(feature_columns["word"]),
            )
            self.assertEqual(
                unit_manifest["feature_columns"]["counts"]["stylo"],
                len(feature_columns["stylo"]),
            )

            y_train_author = np.load(unit_dir / "labels" / "y_train_author.npy", allow_pickle=True)
            self.assertEqual(y_train_author.tolist(), [101, 102])
            self.assertEqual(summary["target_summary"]["author"]["classes"], 2)
            self.assertEqual(summary["target_summary"]["author"]["non_null_rows"], 4)

    def test_run_materialization_can_disable_stylometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_a": [0.1, 0.2, 0.3, 0.4],
                        "stylo_b": [1.0, 0.0, 1.0, 0.0],
                    }
                ),
                config_text=self._materialization_config_text(blocks=["char", "word"]),
            )

            summary = run_materialization(config_path, stage="dev")
            unit_dir = project_root / summary["materialized_root"] / "fold_01"
            feature_columns = json.loads((unit_dir / "feature_columns.json").read_text(encoding="utf-8"))
            unit_manifest = json.loads((unit_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(set(feature_columns), {"char", "word"})
            self.assertFalse((unit_dir / "matrices" / "X_train_stylo.npz").exists())
            self.assertFalse((unit_dir / "matrices" / "X_val_stylo.npz").exists())
            self.assertFalse((unit_dir / "preprocessors" / "stylo_scaler.joblib").exists())
            self.assertFalse((unit_dir / "stylometry_column_report.csv").exists())
            self.assertEqual(unit_manifest["enabled_blocks"], ["char", "word"])
            self.assertFalse(unit_manifest["dimensions"]["stylo"]["generated"])
            self.assertEqual(unit_manifest["dimensions"]["stylo"]["cols"], 0)
            self.assertFalse(unit_manifest["dimensions"]["combined"]["generated"])
            self.assertEqual(unit_manifest["dimensions"]["combined"]["cols"], 0)
            self.assertNotIn("stylometry_column_selection", unit_manifest)
            self.assertEqual(summary["units"][0]["stylo_dim"], 0)

    def test_run_materialization_rejects_stylo_when_row_features_skipped_stylometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_a": [0.1, 0.2, 0.3, 0.4],
                        "stylo_b": [1.0, 0.0, 1.0, 0.0],
                    }
                ),
                config_text=self._materialization_config_text(),
                stylometry_generated=False,
            )

            with self.assertRaisesRegex(ValueError, "did not generate stylometry"):
                run_materialization(config_path, stage="dev")

    def test_run_materialization_drops_zero_variance_stylometry_columns_and_writes_drift_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_keep": [0.1, 0.2, 1.1, 1.2],
                        "stylo_zero": [0.0, 0.0, 0.0, 0.0],
                    }
                ),
                config_text=self._materialization_config_text(drop_zero_variance_columns=True),
            )

            summary = run_materialization(config_path, stage="dev")
            materialized_root = project_root / summary["materialized_root"]
            unit_dir = materialized_root / "fold_01"
            feature_columns = json.loads((unit_dir / "feature_columns.json").read_text(encoding="utf-8"))
            column_report = pd.read_csv(unit_dir / "stylometry_column_report.csv")
            drift_summary = pd.read_csv(materialized_root / "stylometry_drift_summary.csv")

            self.assertEqual(feature_columns["stylo"], ["stylo_keep"])
            dropped_row = column_report[column_report["feature"] == "stylo_zero"].iloc[0]
            self.assertEqual(dropped_row["drop_reason"], "zero_variance")
            self.assertFalse(bool(dropped_row["kept"]))
            self.assertEqual(len(drift_summary), 1)
            self.assertEqual(drift_summary.iloc[0]["unit_id"], "fold_01")
            self.assertEqual(int(drift_summary.iloc[0]["kept_feature_count"]), 1)

    def test_run_materialization_rejects_nonfinite_stylometry_with_role_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_bad": [0.1, np.inf, np.nan, 0.4],
                    }
                ),
                config_text=self._materialization_config_text(),
            )

            with self.assertRaisesRegex(ValueError, "Non-finite stylometry values") as caught:
                run_materialization(config_path, stage="dev")

            message = str(caught.exception)
            self.assertIn("train:stylo_bad=1", message)
            self.assertIn("val:stylo_bad=1", message)

    def test_run_materialization_resolves_final_stage_and_writes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_a": [0.1, 0.2, 0.3, 0.4],
                    }
                ),
                config_text=self._materialization_config_text(blocks=["char", "word"]),
            )

            summary = run_materialization(config_path, stage="final")
            materialized_root = project_root / summary["materialized_root"]

            self.assertEqual(summary["stage"], "final")
            self.assertEqual(summary["selector"], "final")
            self.assertEqual(summary["materialization_name"], "toy_final_materialization")
            self.assertEqual(len(summary["units"]), 1)
            self.assertEqual(summary["units"][0]["eval_role"], "test")
            snapshot = materialized_root / "materialization_config.toml"
            self.assertTrue(snapshot.exists())
            snapshot_text = snapshot.read_text(encoding="utf-8")
            self.assertIn('name = "toy_final_materialization"', snapshot_text)
            self.assertIn('selector = "final"', snapshot_text)

    def test_run_materialization_reuses_same_name_manifest_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root, config_path = self._build_toy_project(
                tmpdir,
                row_stylo=pd.DataFrame(
                    {
                        "id_speech": [1, 2, 3, 4],
                        "id_person": [101, 102, 101, 102],
                        "stylo_a": [0.1, 0.2, 0.3, 0.4],
                    }
                ),
                config_text=self._materialization_config_text(blocks=["char", "word"]),
            )

            first = run_materialization(config_path, stage="dev")
            materialized_root = project_root / first["materialized_root"]
            marker = materialized_root / "fold_01" / "matrices" / "reuse_marker.txt"
            marker.write_text("kept", encoding="utf-8")

            second = run_materialization(config_path, stage="dev")

            self.assertEqual(second["materialization_name"], "toy_materialization")
            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
