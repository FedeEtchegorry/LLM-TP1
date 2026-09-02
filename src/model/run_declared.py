"""Run an explicitly declared family of configurations selected by prefix."""

from __future__ import annotations

import argparse
import time

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import PARAMETERS_PATH, PROTOCOL, RunConfig, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one, sweep_note
from src.model.protocol import EvaluationResult, markdown_table
from src.model.results import RESULTS_DIR


def selected_runs(
    declared: dict[str, RunConfig], prefix: str
) -> dict[str, RunConfig]:
    """Return the declared runs whose names begin with ``prefix``."""
    selected = {name: run for name, run in declared.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError(f"no declared runs start with {prefix!r}")
    return selected


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument(
        "--force", action="store_true", help="retrain even when a result is recorded"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    runs = selected_runs(declared, args.prefix)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\n=== DECLARED ({len(runs)} runs starting with {args.prefix!r}) ===")
    results: list[EvaluationResult] = []
    trained_seconds, trained = 0.0, 0
    for position, (name, config) in enumerate(runs.items(), start=1):
        started = time.perf_counter()
        print(f"  [{position}/{len(runs)}] {name}", flush=True)
        result, note = run_one(
            config, frame, partitions, directory=args.results, force=args.force
        )
        if note != "recorded":
            trained_seconds += time.perf_counter() - started
            trained += 1
        results.append(result)
        print(
            f"  {config.digest}  {result.summary_row()}   [{note}]"
            + sweep_note(trained_seconds, trained, len(runs) - position)
        )

    print()
    print(markdown_table(results, float(target_of(frame).mean())))
    print(f"\nrecorded in {args.results}/ -- read them with src.model.results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
