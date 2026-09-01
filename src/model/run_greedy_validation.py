"""Task 11: check whether the greedy depth -> width -> heads walk hid a better point.

    .venv/bin/python -m src.model.run_greedy_validation --parameters parameters-eda.txt \
        --results results/eda-contract \
        --output results/eda-contract/architecture/greedy-validation.json

Must run **after** Task 6 (M chosen) and **before** Task 7 Step 6 (the holdout opens).
Two biases are checked, in order:

1. **Conditioning bias** -- width and heads were only ever tried at the depth the
   greedy walk kept; ``depth_probe`` reopens the depth it discarded and tries the two
   widths and two head counts there.
2. **Order bias** -- the final point depends on having walked depth before width;
   ``capacity_neighbourhood`` tries every single-coordinate move on the three capacity
   axes from the point the greedy walk actually landed on.

The validation only ever demands ``improves`` -- never ``tie-break`` -- to promote a
probe over the selected configuration: a tie-break is a cheap, reversible move during
the search, but M is already frozen here, and flipping the finalist on a difference
the margin cannot resolve would be changing candidates by noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import PARAMETERS_PATH, PROTOCOL, RunConfig, changed_fields, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one
from src.model.representation_selection import compare, paired_margin
from src.model.results import RESULTS_DIR
from src.model.run_architecture import DEPTHS, HEADS, WIDTHS, depth_candidates, head_candidates, width_candidates

M_NAME = "M selected from directed comparisons"


def alternate_depth(selected: RunConfig) -> int:
    """The depth the greedy never combined with any other width or head count."""
    return 1 if selected.n_layers == 2 else 2


def depth_anchor(selected: RunConfig) -> RunConfig:
    depth = alternate_depth(selected)
    return replace(selected, name=f"V anchor depth {depth}", n_layers=depth)


def depth_probe(selected: RunConfig) -> list[RunConfig]:
    anchor = depth_anchor(selected)
    probes = [
        replace(anchor, name=f"V depth {anchor.n_layers} d_model {width}", d_model=width)
        for width in WIDTHS
        if width != anchor.d_model
    ]
    probes += [
        replace(anchor, name=f"V depth {anchor.n_layers} heads {heads}", n_heads=heads)
        for heads in HEADS
        if heads != anchor.n_heads
    ]
    return probes


def capacity_neighbourhood(selected: RunConfig) -> list[RunConfig]:
    """Every single-coordinate move on the three capacity axes, from the final point."""
    moves = [
        replace(selected, name=f"V neighbour n_layers {value}", n_layers=value)
        for value in DEPTHS
        if value != selected.n_layers
    ]
    moves += [
        replace(selected, name=f"V neighbour d_model {value}", d_model=value)
        for value in WIDTHS
        if value != selected.d_model
    ]
    moves += [
        replace(selected, name=f"V neighbour n_heads {value}", n_heads=value)
        for value in HEADS
        if value != selected.n_heads
    ]
    return moves


def verdict(selected_ap: np.ndarray, probes: list[tuple[RunConfig, np.ndarray]]) -> list[dict]:
    better = []
    for run, ap in probes:
        mean, low, high = paired_margin(ap - selected_ap)
        if compare(selected_ap, ap) == "improves":
            better.append({"name": run.name, "delta": mean, "low": low, "high": high})
    return better


def all_deltas(selected_ap: np.ndarray, probes: list[tuple[RunConfig, np.ndarray]]) -> list[dict]:
    """Every probe's delta against the selected configuration, win or not -- for
    reporting the full neighbourhood (chart 4), not just the (usually empty) winners
    ``verdict`` returns."""
    rows = []
    for run, ap in probes:
        mean, low, high = paired_margin(ap - selected_ap)
        rows.append({"name": run.name, "delta": mean, "low": low, "high": high})
    return rows


def _ap(config: RunConfig, frame, partitions, directory: str, cache: dict) -> tuple[np.ndarray, bool]:
    """Score ``config``, and say whether it came from the cache (by digest)."""
    was_cached = config.digest in cache
    result, note = run_one(config, frame, partitions, directory=directory)
    ap = np.array([fold.average_precision for fold in result.folds], dtype=float)
    cache[config.digest] = ap
    return ap, note == "recorded"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument(
        "--output", type=str, default="results/eda-contract/architecture/greedy-validation.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    selected = declared[M_NAME]

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    cache: dict[str, np.ndarray] = {}
    selected_ap, _ = _ap(selected, frame, partitions, args.results, cache)

    print("\n=== LAYER 1: DEPTH PROBE (reopens the discarded depth) ===")
    probes_1 = depth_probe(selected)
    trained_1, cached_1 = 0, 0
    results_1: list[tuple[RunConfig, np.ndarray]] = []
    for probe in probes_1:
        ap, was_cached = _ap(probe, frame, partitions, args.results, cache)
        trained_1 += 0 if was_cached else 1
        cached_1 += 1 if was_cached else 0
        results_1.append((probe, ap))
    better_1 = verdict(selected_ap, results_1)

    print("\n=== LAYER 2: SINGLE-COORDINATE NEIGHBOURHOOD ===")
    moves = capacity_neighbourhood(selected)
    trained_2, cached_2 = 0, 0
    results_2: list[tuple[RunConfig, np.ndarray]] = []
    for move in moves:
        ap, was_cached = _ap(move, frame, partitions, args.results, cache)
        trained_2 += 0 if was_cached else 1
        cached_2 += 1 if was_cached else 0
        results_2.append((move, ap))
    better_2 = verdict(selected_ap, results_2)

    stable = not better_1 and not better_2
    conclusion = (
        "no probe at the unexplored depth and no single-coordinate move beat the "
        "selected configuration by the declared paired margin"
        if stable
        else "at least one probe beat the selected configuration by the declared paired margin"
    )

    document = {
        "selected": selected.name,
        "layer_1_depth_probe": {
            "anchor_depth": alternate_depth(selected),
            "trained": trained_1,
            "cached": cached_1,
            "better_than_selected": better_1,
            "all_probes": all_deltas(selected_ap, results_1),
        },
        "layer_2_neighbourhood": {
            "axes": ["n_layers", "d_model", "n_heads"],
            "trained": trained_2,
            "cached": cached_2,
            "better_than_selected": better_2,
            "all_probes": all_deltas(selected_ap, results_2),
        },
        "stable": stable,
        "conclusion": conclusion,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\n{document}")
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
