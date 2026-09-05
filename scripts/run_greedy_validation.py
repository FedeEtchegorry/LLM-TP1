"""Whether the directed path's own order decided where it stopped.

    .venv/Scripts/python -m scripts.run_greedy_validation \
        --parameters parameters-eda.txt --results results/v1-una-torre/eda-contract

Task 5 walks depth, then width, then heads, resolving each axis against the winner
of the previous one.  That visits seven of twenty-seven capacity points and admits
two biases the write-up cannot claim to have avoided without measuring them:

    conditioning  width and heads were never tried at the depth the path dropped
    ordering      the end point may describe the route rather than the model

Layer 1 reopens the discarded depth and crosses it with every width and head count.
Layer 2 takes one single-coordinate step on each capacity axis from the selected
point.  Both read only cross-validation; neither touches the holdout, and neither
promotes anything by itself -- the plan's decision rule does that.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import EDA_PARAMETERS, RunConfig, ladder_runs, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one
from src.model.representation_selection import compare, paired_margin
from src.model.results import RESULTS_DIR
from scripts.run_architecture import (
    ARCHITECTURE_DIR,
    FINAL_NAME,
    HEADS,
    SELECTION_FILE,
    WIDTHS,
    ap_scores,
    depth_candidates,
    head_candidates,
    numeric_candidates,
    width_candidates,
)

VALIDATION_FILE = "greedy-validation.json"
CAPACITY_AXES = ("n_layers", "d_model", "n_heads")


def architecture_base(declared: dict[str, RunConfig], selection: dict) -> RunConfig:
    """The point Task 5 walked the capacity axes from, named exactly as it named it.

    The path resolves the numeric embedding before touching depth, so the base is
    L2 when that stage kept its base and the winning ``A numeric ...`` run when it
    moved.  Rebuilding it through ``numeric_candidates`` rather than by hand is what
    makes every cached digest line up: a configuration under a different name is a
    different digest and would retrain.
    """
    l2 = ladder_runs(declared)
    base = next(run for name, run in l2.items() if name.startswith("L2"))
    chosen = selection["numeric_embedding"]["selected"]
    if chosen == base.numeric_embedding:
        return base
    return next(
        run for run in numeric_candidates(base) if run.numeric_embedding == chosen
    )


def alternate_depth(selected: RunConfig) -> int:
    """The depth the greedy never combined with any other width or head count."""
    return 1 if selected.n_layers == 2 else 2


def depth_anchor(selected: RunConfig, base: RunConfig) -> RunConfig:
    """The alternate-depth point, named exactly as Task 5 would have named it."""
    depth = alternate_depth(selected)
    if depth == base.n_layers:
        return base
    return replace(base, name=f"B depth {depth}", n_layers=depth)


def depth_probe(selected: RunConfig, base: RunConfig) -> list[RunConfig]:
    """Every width and head count at the depth the path abandoned: four new runs."""
    anchor = depth_anchor(selected, base)
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
    """Every single-coordinate move on the three capacity axes, from the final point.

    No generators of its own on purpose: reusing Task 5's means that when the greedy
    never moved, every point here is name-for-name what Task 5 already trained, so
    the digests match and the whole layer resolves from cache.
    """
    return (
        depth_candidates(selected)
        + width_candidates(selected)
        + head_candidates(selected)
    )


def verdict(
    selected_ap: np.ndarray, probes: list[tuple[RunConfig, np.ndarray]]
) -> list[dict]:
    """Probes that beat the frozen candidate by the declared paired margin.

    Deliberately asymmetric: ``improves`` counts and ``tie-break`` does not.  While
    the path was still searching, a tie-break was a cheap, reversible move; here M
    is already frozen, and overturning it on a difference the margin cannot resolve
    would be changing finalists on noise.
    """
    better = []
    for run, ap in probes:
        mean, low, high = paired_margin(np.asarray(ap, dtype=float) - selected_ap)
        if compare(selected_ap, ap) == "improves":
            better.append({"name": run.name, "delta": mean, "low": low, "high": high})
    return better


def validation_path(results: Path | str) -> Path:
    return Path(results) / ARCHITECTURE_DIR / VALIDATION_FILE


def read_selection(results: Path | str) -> dict:
    path = Path(results) / ARCHITECTURE_DIR / SELECTION_FILE
    if not path.exists():
        raise SystemExit(
            f"{path} is missing -- run scripts.run_architecture before validating it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_all(
    configs: list[RunConfig], frame, partitions, results: Path | str, force: bool
) -> tuple[list[tuple[RunConfig, np.ndarray]], int, int]:
    """Score every configuration, reporting how many were trained and how many cached."""
    scored: list[tuple[RunConfig, np.ndarray]] = []
    trained = cached = 0
    for config in configs:
        result, note = run_one(
            config, frame, partitions, directory=results, force=force
        )
        if note == "recorded":
            cached += 1
        else:
            trained += 1
        print(f"  {config.name}: {result.summary_row()} [{note}]", flush=True)
        scored.append((config, ap_scores(result)))
    return scored, trained, cached


def _layer(
    title: str,
    selected_ap: np.ndarray,
    configs: list[RunConfig],
    frame,
    partitions,
    results: Path | str,
    force: bool,
) -> tuple[dict, list[tuple[RunConfig, np.ndarray]]]:
    print(f"\n=== {title} ===")
    scored, trained, cached = _run_all(configs, frame, partitions, results, force)
    return (
        {
            "trained": trained,
            "cached": cached,
            "better_than_selected": verdict(selected_ap, scored),
        },
        scored,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    selection = read_selection(args.results)

    selected = declared[FINAL_NAME]
    base = architecture_base(declared, selection)

    frame = load_dataset()
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\nvalidating [{selected.name}] against the path that produced it")
    result, note = run_one(selected, frame, partitions, directory=args.results)
    selected_ap = ap_scores(result)
    print(f"  {selected.name}: {result.summary_row()} [{note}]")

    anchor = depth_anchor(selected, base)
    layer_1, _ = _layer(
        f"LAYER 1: THE DEPTH THE PATH DROPPED ({anchor.n_layers} blocks)",
        selected_ap,
        [anchor] + depth_probe(selected, base),
        frame,
        partitions,
        args.results,
        args.force,
    )
    layer_1["anchor_depth"] = anchor.n_layers

    layer_2, _ = _layer(
        "LAYER 2: ONE COORDINATE AWAY FROM THE SELECTED POINT",
        selected_ap,
        capacity_neighbourhood(selected),
        frame,
        partitions,
        args.results,
        args.force,
    )
    layer_2["axes"] = list(CAPACITY_AXES)

    better = layer_1["better_than_selected"] + layer_2["better_than_selected"]
    stable = not better
    document = {
        "selected": selected.name,
        "layer_1_depth_probe": layer_1,
        "layer_2_neighbourhood": layer_2,
        "stable": stable,
        "conclusion": (
            "no probe at the unexplored depth and no single-coordinate move beat the "
            "selected configuration by the declared paired margin"
            if stable
            else "a probe beat the selected configuration by the declared paired "
            "margin: " + ", ".join(entry["name"] for entry in better)
        ),
    }
    path = Path(args.output) if args.output else validation_path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"\ntrained {layer_1['trained'] + layer_2['trained']}, "
        f"cached {layer_1['cached'] + layer_2['cached']}, holdout untouched"
    )
    print(f"stable: {stable} -- {document['conclusion']}")
    print(f"verdict written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
