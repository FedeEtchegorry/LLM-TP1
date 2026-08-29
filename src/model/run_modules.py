"""The module alternatives, each measured against the base configuration L4.

    .venv/bin/python -m src.model.run_modules
    .venv/bin/python -m src.model.run_modules --axis C

Every axis point changes exactly one thing from L4, so the difference it produces is
attributable to that one thing. The table reports the change, not just the level.
"""

from __future__ import annotations

import argparse

from src.eda.loading import load_dataset
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    axis_runs,
    ladder_runs,
    load_parameters,
)
from src.model.experiment import describe, partition, run_one
from src.model.protocol import EvaluationResult
from src.model.results import RESULTS_DIR

BASE = "L4"
"""The rung every axis point is compared against."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--axis", type=str, default="", help="run one axis only, by its letter"
    )
    return parser.parse_args(argv)


def comparison_table(base: EvaluationResult, results: list[EvaluationResult]) -> str:
    """Each alternative beside the base, with the change it produced."""
    lines = [
        "| Axis | Alternative | ROC-AUC | PR-AUC (AP) | Δ AP vs base |",
        "|---|---|---:|---:|---:|",
        f"| — | **{base.name}** (base) "
        f"| {base.roc_auc_mean:.3f} ± {base.roc_auc_std:.3f} "
        f"| **{base.average_precision_mean:.3f} ± {base.average_precision_std:.3f}** | — |",
    ]
    for result in results:
        axis, _, alternative = result.name.partition(" ")
        delta = result.average_precision_mean - base.average_precision_mean
        lines.append(
            f"| {axis} | {alternative} "
            f"| {result.roc_auc_mean:.3f} ± {result.roc_auc_std:.3f} "
            f"| {result.average_precision_mean:.3f} ± {result.average_precision_std:.3f} "
            f"| {delta:+.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    points = axis_runs(declared)
    if args.axis:
        points = {
            name: run for name, run in points.items()
            if name.startswith(f"{args.axis.upper()} ")
        }
    base_config = next(
        run for name, run in ladder_runs(declared).items() if name.startswith(BASE)
    )

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\n=== BASE ===")
    base, note = run_one(
        base_config, frame, partitions, directory=args.results, force=args.force
    )
    print(f"  {base_config.digest}  {base.summary_row()}   [{note}]")

    print(f"\n=== MODULE ALTERNATIVES ({len(points)}) ===")
    results: list[EvaluationResult] = []
    for name, config in points.items():
        result, note = run_one(
            config, frame, partitions, directory=args.results, force=args.force
        )
        results.append(result)
        delta = result.average_precision_mean - base.average_precision_mean
        print(f"  {config.digest}  {result.summary_row()}   {delta:+.4f}   [{note}]")

    print()
    print(comparison_table(base, results))
    print(f"\nrecorded in {args.results}/ -- read them with src.model.results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
