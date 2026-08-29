"""The ladder: the linear bar, then each component the Transformer adds to reach it.

    .venv/bin/python -m src.model.run_ladder

Every rung is declared in ``parameters.txt`` and is one run that doubles as a slide and
as a point in the ablation table:

    L1 -> L2   axis B, does attention buy anything over an average?
    L2 -> L3   axis G, do the tabular columns add to the text?
    L3 -> L4   axis A, is the bucket term worth the +0.137 AP the EDA predicts?
"""

from __future__ import annotations

import argparse
import time

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import PARAMETERS_PATH, PROTOCOL, ladder_runs, load_parameters
from src.model.experiment import describe, partition, run_one, sweep_note
from src.model.protocol import EvaluationResult, markdown_table
from src.model.results import RESULTS_DIR


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument(
        "--force", action="store_true", help="retrain even when a result is recorded"
    )
    parser.add_argument("--only", type=str, default="", help="run just this section")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = ladder_runs(load_parameters(args.parameters))
    if args.only:
        runs = {name: run for name, run in runs.items() if args.only in name}

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\n=== LADDER ({len(runs)} rungs from {args.parameters}) ===")
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

    if results:
        print()
        print(markdown_table(results, float(target_of(frame).mean())))
        print(f"\nrecorded in {args.results}/ -- read them with src.model.results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
