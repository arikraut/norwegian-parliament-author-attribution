"""Run a paired bootstrap macro-F1 and McNemar comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.utils import find_project_root, resolve_project_path
from thesis_reporting.significance import run_comparison


def parse_args() -> argparse.Namespace:
    """Parse standalone significance-comparison arguments."""

    parser = argparse.ArgumentParser(
        description="Bootstrap CI and McNemar test for two attribution systems.",
    )
    parser.add_argument("--system-a", type=Path, required=True)
    parser.add_argument("--system-b", type=Path, required=True)
    parser.add_argument("--label-a", default="System A")
    parser.add_argument("--label-b", default="System B")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mcnemar-method",
        choices=("continuity_corrected", "asymptotic_uncorrected"),
        default="asymptotic_uncorrected",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def format_result(result: dict[str, Any]) -> str:
    """Render one comparison result for concise terminal output."""

    system_a = result["system_a"]
    system_b = result["system_b"]
    mcnemar = result["mcnemar"]
    return "\n".join(
        [
            f"{result['label_a']} vs {result['label_b']}",
            (
                f"{result['label_a']}: {system_a['macro_f1']:.4f} "
                f"[{system_a['ci_95_lower']:.4f}, {system_a['ci_95_upper']:.4f}]"
            ),
            (
                f"{result['label_b']}: {system_b['macro_f1']:.4f} "
                f"[{system_b['ci_95_lower']:.4f}, {system_b['ci_95_upper']:.4f}]"
            ),
            f"McNemar: {mcnemar['significance']}; {mcnemar['conclusion']}",
        ]
    )


def main() -> None:
    """Resolve project paths, calculate one comparison, and optionally write JSON."""

    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    system_a = resolve_project_path(project_root, args.system_a)
    system_b = resolve_project_path(project_root, args.system_b)
    result = run_comparison(
        system_a,
        system_b,
        args.label_a,
        args.label_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        mcnemar_method=args.mcnemar_method,
    )
    print(format_result(result))
    if args.output is not None:
        output_path = resolve_project_path(project_root, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
