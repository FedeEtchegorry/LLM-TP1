"""Run the whole Ejercicio 2 experimental pipeline, in the order the report depends on.

    .venv/bin/python -m scripts.run_all_experiments
    .venv/bin/python -m scripts.run_all_experiments --skip-transfer
    .venv/bin/python -m scripts.run_all_experiments --from run_architecture

Each step is a subprocess call to an existing entrypoint under ``src.model``. The
order matters and is not configurable: representations before the ladder, the ladder
before the architecture search, the architecture search before its own greedy-order
validation, the audit before the stability seeds, and the seeds before the holdout.
Transfer learning and the figures are appended at the end, since neither one feeds a
decision upstream of it.

The script stops at the first step that exits non-zero and prints exactly which step
failed and the command that failed, rather than limping on with a partial pipeline. A
summary of what ran (and what did not) is printed at the end either way.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

PARAMETERS = "parameters-eda.txt"
RESULTS = "results/eda-contract"
FINAL_RESULTS = f"{RESULTS}/final"
AUDIT_OUTPUT = f"{RESULTS}/audit.md"
GREEDY_OUTPUT = f"{RESULTS}/architecture/greedy-validation.json"
FIGURES = "figures/eda-contract"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    optional: bool = False


def python() -> str:
    """The interpreter running this script -- so the orchestrator honours whatever
    virtualenv (or lack of one) the caller already activated."""
    return sys.executable


def steps(args: argparse.Namespace) -> list[Step]:
    py = python()
    all_steps = [
        Step(
            "run_embeddings",
            [py, "-m", "scripts.run_embeddings", "--results", RESULTS],
        ),
        Step(
            "run_ladder",
            [py, "-m", "scripts.run_ladder", "--parameters", PARAMETERS, "--results", RESULTS],
        ),
        Step(
            "run_architecture",
            [py, "-m", "scripts.run_architecture", "--parameters", PARAMETERS, "--results", RESULTS],
        ),
        Step(
            "run_greedy_validation",
            [
                py, "-m", "scripts.run_greedy_validation",
                "--parameters", PARAMETERS, "--results", RESULTS, "--output", GREEDY_OUTPUT,
            ],
        ),
        Step(
            "run_eda_audit",
            [
                py, "-m", "scripts.run_eda_audit",
                "--parameters", PARAMETERS, "--results", RESULTS, "--output", AUDIT_OUTPUT, "--strict",
            ],
        ),
        Step(
            "run_declared (stability seeds)",
            [
                py, "-m", "scripts.run_declared",
                "--parameters", PARAMETERS, "--results", RESULTS, "--prefix", "S selected",
            ],
        ),
        Step(
            "run_final",
            [
                py, "-m", "scripts.run_final",
                "--parameters", PARAMETERS, "--results", RESULTS, "--final-results", FINAL_RESULTS,
            ],
        ),
        Step(
            "run_transfer",
            [py, "-m", "scripts.run_transfer", "--parameters", PARAMETERS, "--results", RESULTS],
            optional=True,
        ),
        Step(
            "run_embedding_figures",
            [py, "-m", "scripts.run_embedding_figures", "--results", RESULTS, "--figures", FIGURES],
            optional=True,
        ),
        Step(
            "run_figures",
            [
                py, "-m", "scripts.run_figures",
                "--parameters", PARAMETERS, "--results", RESULTS, "--figures", FIGURES,
            ],
            optional=True,
        ),
    ]
    if args.skip_transfer:
        all_steps = [s for s in all_steps if s.name != "run_transfer"]
    if args.from_step:
        names = [s.name for s in all_steps]
        matches = [n for n in names if args.from_step in n]
        if not matches:
            raise SystemExit(f"--from {args.from_step!r} matches no step: {names}")
        start = names.index(matches[0])
        all_steps = all_steps[start:]
    return all_steps


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="skip run_transfer (needs the sentence-transformers checkpoint available)",
    )
    parser.add_argument(
        "--from", dest="from_step", type=str, default="",
        help="resume from the first step whose name contains this substring",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = steps(args)

    print(f"=== EJERCICIO 2: {len(plan)} steps ===")
    for position, step in enumerate(plan, start=1):
        print(f"  [{position}/{len(plan)}] {step.name}")
    print()

    completed: list[str] = []
    failed: str | None = None
    for position, step in enumerate(plan, start=1):
        print(f"\n--- [{position}/{len(plan)}] {step.name} ---")
        print("  $ " + " ".join(step.command), flush=True)
        started = time.perf_counter()
        result = subprocess.run(step.command)
        seconds = time.perf_counter() - started
        if result.returncode != 0:
            print(f"\n[{step.name}] FAILED after {seconds:.0f}s (exit {result.returncode})")
            if step.optional:
                print(f"  {step.name} is optional (does not gate later steps) -- continuing")
                continue
            failed = step.name
            break
        completed.append(step.name)
        print(f"[{step.name}] done in {seconds:.0f}s")

    print("\n=== SUMMARY ===")
    for step in plan:
        mark = "x" if step.name in completed else (" " if step.name != failed else "!")
        print(f"  [{mark}] {step.name}")

    if failed:
        print(f"\nstopped at {failed!r} -- fix the error above and rerun; earlier steps are cached")
        return 1

    print("\nall steps completed. See docs/informe-ejercicio-2.md and "
          f"{AUDIT_OUTPUT} for the evidence trail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
