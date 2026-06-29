"""Build research-result analysis tables from existing project artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.utils import find_project_root, resolve_project_path
from thesis_reporting.artifacts import ResultArtifacts
from thesis_reporting.author_performance import write_author_performance_outputs
from thesis_reporting.config import (
    configured_comparisons,
    configured_profile_quality_run,
    configured_profile_targets,
    configured_systems,
)
from thesis_reporting.confusions import write_confusion_outputs
from thesis_reporting.feature_importance import write_feature_importance_outputs
from thesis_reporting.profile_quality import write_profile_quality_outputs
from thesis_reporting.profiling_effects import write_profiling_effect_outputs
from thesis_reporting.provenance import write_manifest, write_summary
from thesis_reporting.significance import write_significance_outputs
from thesis_reporting.topk_confidence import write_topk_confidence_outputs


CSV_SECTIONS = {
    "author_performance",
    "confusions",
    "profiling_effects",
    "topk_confidence",
    "profile_quality",
    "significance",
}
CONCRETE_SECTIONS = CSV_SECTIONS | {"feature_importance"}
SECTION_SHORTCUTS = {"all", "all_with_feature_importance"}


def parse_args() -> argparse.Namespace:
    """Parse result-addition CLI arguments without resolving project paths."""

    parser = argparse.ArgumentParser(
        description="Build research-result analyses from existing project results.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Result root containing model and profiling outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/result_additions"),
        help="Directory where result-addition outputs are written.",
    )
    parser.add_argument(
        "--sections",
        default="all",
        help=(
            "Comma-separated concrete sections, or exactly one shortcut: "
            "all or all_with_feature_importance."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of authors retained in best/worst tables.",
    )
    parser.add_argument(
        "--significance-bootstrap",
        type=int,
        default=10_000,
        help="Bootstrap iterations for significance comparisons.",
    )
    parser.add_argument(
        "--mcnemar-method",
        choices=("continuity_corrected", "asymptotic_uncorrected"),
        default="asymptotic_uncorrected",
        help="McNemar variant used for significance comparisons.",
    )
    return parser.parse_args()


def requested_sections(raw_sections: str) -> set[str]:
    """Validate every section token before expanding an optional shortcut."""

    tokens = {
        token.strip()
        for token in raw_sections.split(",")
        if token.strip()
    }
    unsupported = tokens - CONCRETE_SECTIONS - SECTION_SHORTCUTS
    if unsupported:
        raise ValueError(f"Unsupported section(s): {sorted(unsupported)}")

    shortcuts = tokens & SECTION_SHORTCUTS
    if shortcuts and len(tokens) != 1:
        raise ValueError(
            "Section shortcuts cannot be combined with other section tokens: "
            f"{sorted(tokens)}"
        )
    if tokens == {"all"}:
        return set(CSV_SECTIONS)
    if tokens == {"all_with_feature_importance"}:
        return set(CONCRETE_SECTIONS)
    return tokens


def main() -> None:
    """Resolve project paths and orchestrate the selected report sections."""

    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    results_dir = resolve_project_path(project_root, args.results_dir)
    data_dir = resolve_project_path(project_root, Path("data"))
    output_dir = resolve_project_path(project_root, args.output_dir)
    sections = requested_sections(args.sections)

    systems = configured_systems()
    comparisons = configured_comparisons()
    profile_run = configured_profile_quality_run()
    profile_targets = configured_profile_targets()
    artifacts = ResultArtifacts(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    if "author_performance" in sections:
        outputs["author_performance"] = write_author_performance_outputs(
            systems,
            results_dir=artifacts,
            output_dir=output_dir,
            top_n=args.top_n,
        )
    if "confusions" in sections:
        outputs["confusions"] = write_confusion_outputs(
            systems,
            results_dir=artifacts,
            output_dir=output_dir,
            top_n=args.top_n,
        )
    if "profiling_effects" in sections:
        outputs["profiling_effects"] = write_profiling_effect_outputs(
            systems,
            comparisons,
            results_dir=artifacts,
            output_dir=output_dir,
            top_n=args.top_n,
        )
    if "topk_confidence" in sections:
        outputs["topk_confidence"] = write_topk_confidence_outputs(
            systems,
            results_dir=artifacts,
            output_dir=output_dir,
            top_n=args.top_n,
        )
    if "profile_quality" in sections:
        outputs["profile_quality"] = write_profile_quality_outputs(
            systems,
            profile_run,
            profile_targets,
            results_dir=artifacts,
            output_dir=output_dir,
            top_n=args.top_n,
        )
    if "significance" in sections:
        outputs["significance"] = write_significance_outputs(
            systems,
            comparisons,
            artifacts=artifacts,
            output_dir=output_dir,
            project_root=project_root,
            n_bootstrap=args.significance_bootstrap,
            mcnemar_method=args.mcnemar_method,
        )
    if "feature_importance" in sections:
        outputs["feature_importance"] = write_feature_importance_outputs(
            systems,
            project_root=project_root,
            results_dir=results_dir,
            output_dir=output_dir,
            top_n=args.top_n,
        )

    provenance_outputs = dict(outputs)
    provenance_outputs["summary"] = {
        "manifest": str(output_dir / "manifest.json"),
        "summary": str(output_dir / "summary.md"),
    }
    write_manifest(
        project_root=project_root,
        output_dir=output_dir,
        results_dir=results_dir,
        data_dir=data_dir,
        sections=sections,
        systems=systems,
        comparisons=comparisons,
        profile_run=profile_run,
        profile_targets=profile_targets,
        outputs=provenance_outputs,
    )
    write_summary(
        project_root=project_root,
        output_dir=output_dir,
        results_dir=results_dir,
        data_dir=data_dir,
        systems=systems,
        comparisons=comparisons,
        profile_targets=profile_targets,
        outputs=provenance_outputs,
    )


if __name__ == "__main__":
    main()
