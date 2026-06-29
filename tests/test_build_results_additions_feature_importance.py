from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_reporting.feature_importance import (
    DIRECT_OUTPUT_NAMES,
    STACKED_OUTPUT_NAMES,
    condition_manifest_path,
    copy_importance_outputs,
    run_system_importance,
)
from thesis_reporting.config import ResultSystem


def _system(*, key: str, architecture: str) -> ResultSystem:
    """Create one configured-system fixture in the canonical results layout."""
    return ResultSystem(
        key=key,
        label=key,
        phase="phase",
        split="split",
        architecture=architecture,
        representation="none",
        scope="baseline",
        condition_id="char_word",
        condition_dir=Path(
            f"models/split/{key}/seed_42/final_by_condition/char_word"
        ),
    )


class FeatureImportanceAdditionTests(unittest.TestCase):
    """Tests for local feature-importance result collection."""

    def test_condition_manifest_uses_local_results_root(self) -> None:
        """Resolve final manifests below the caller-provided results directory."""
        system = _system(key="direct", architecture="direct")

        path = condition_manifest_path(system, Path("results"))

        self.assertEqual(path, Path("results/models/split/direct/seed_42/manifest.json"))

    def test_system_dispatch_uses_public_condition_helper(self) -> None:
        """Dispatch direct and stacked systems through their public helpers."""

        cases = (
            (
                "direct",
                "run_condition_importance_analysis",
                {
                    "top_n": 12,
                    "condition_id": "char_word",
                },
            ),
            (
                "stacked",
                "run_stacked_condition_importance_analysis",
                {"condition_id": "char_word"},
            ),
        )
        for architecture, helper_name, expected_kwargs in cases:
            with self.subTest(architecture=architecture):
                system = _system(key=architecture, architecture=architecture)
                patch_path = f"thesis_reporting.feature_importance.{helper_name}"
                with patch(patch_path, return_value={"outputs": {}}) as helper:
                    result = run_system_importance(
                        system,
                        results_dir=Path("results"),
                        top_n=12,
                    )

                self.assertEqual(result, {"outputs": {}})
                helper.assert_called_once_with(
                    Path(
                        f"results/models/split/{architecture}/seed_42/manifest.json"
                    ),
                    **expected_kwargs,
                )

    def test_copy_outputs_resolves_paths_from_project_root(self) -> None:
        """Copy helper outputs from canonical project-relative result paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            output_dir = project_root / "thesis_outputs"
            for architecture, output_names in (
                ("direct", DIRECT_OUTPUT_NAMES),
                ("stacked", STACKED_OUTPUT_NAMES),
            ):
                system = _system(key=architecture, architecture=architecture)
                helper_dir = project_root / "results" / architecture
                helper_dir.mkdir(parents=True)
                summary_outputs: dict[str, str] = {}
                for output_name in output_names:
                    suffix = ".png" if output_name.endswith("_plot") else ".csv"
                    source = helper_dir / f"{output_name}{suffix}"
                    source.write_text(output_name, encoding="utf-8")
                    summary_outputs[output_name] = str(source.relative_to(project_root))

                copied = copy_importance_outputs(
                    system,
                    {"outputs": summary_outputs},
                    project_root=project_root,
                    output_dir=output_dir,
                )

                self.assertEqual(set(copied), set(output_names))
                for copied_path in copied.values():
                    self.assertTrue(Path(copied_path).exists())


if __name__ == "__main__":
    unittest.main()
