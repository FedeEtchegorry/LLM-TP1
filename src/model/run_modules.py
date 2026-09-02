"""The module alternatives, each measured against one ladder base configuration.

    .venv/bin/python -m src.model.run_modules
    .venv/bin/python -m src.model.run_modules --axis C

Every axis point changes exactly one thing from the base, so the difference it produces is
attributable to that one thing. The table reports the change, not just the level.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from dataclasses import asdict

from src.eda.loading import load_dataset
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    RunConfig,
    axis_runs,
    ladder_runs,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one, sweep_note
from src.model.protocol import EvaluationResult
from src.model.results import RESULTS_DIR

BASE = "L4"
"""The rung every axis point is compared against."""


def differences(run: RunConfig, base: RunConfig) -> list[str]:
    """The fields where ``run`` departs from ``base``, ignoring the name."""
    left, right = asdict(run), asdict(base)
    return sorted(key for key in left if key != "name" and left[key] != right[key])


def axis_points(
    declared: dict[str, RunConfig], base: RunConfig
) -> tuple[dict[str, RunConfig], dict[str, list[str]]]:
    """Partition declared axes into single-field points and rejected runs."""
    kept, dropped = {}, {}
    for name, run in axis_runs(declared).items():
        moved = differences(run, base)
        if len(moved) == 1:
            kept[name] = run
        else:
            dropped[name] = moved
    return kept, dropped


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument(
        "--base",
        type=str,
        default=BASE,
        help="ladder prefix used as the one-factor comparison base",
    )
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
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    matches = [
        run for name, run in ladder_runs(declared).items()
        if name.startswith(args.base)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one ladder run starting with {args.base!r}, found {len(matches)}"
        )
    base_config = matches[0]
    points, dropped = axis_points(declared, base_config)
    for name, moved in dropped.items():
        print(
            f"  skipping [{name}]: moves {len(moved)} fields from "
            f"{args.base}, not one"
        )
    if args.axis:
        points = {
            name: run for name, run in points.items()
            if name.startswith(f"{args.axis.upper()} ")
        }

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
    trained_seconds, trained = 0.0, 0
    for position, (name, config) in enumerate(points.items(), start=1):
        started = time.perf_counter()
        print(f"  [{position}/{len(points)}] {name}", flush=True)
        result, note = run_one(
            config, frame, partitions, directory=args.results, force=args.force
        )
        if note != "recorded":
            trained_seconds += time.perf_counter() - started
            trained += 1
        results.append(result)
        delta = result.average_precision_mean - base.average_precision_mean
        print(
            f"  {config.digest}  {result.summary_row()}   {delta:+.4f}   [{note}]"
            + sweep_note(trained_seconds, trained, len(points) - position)
        )

    print()
    print(comparison_table(base, results))
    print(f"\nrecorded in {args.results}/ -- read them with src.model.results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
