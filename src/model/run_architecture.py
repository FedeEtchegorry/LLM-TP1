"""Task 5: walk depth -> width -> heads from L2, one axis at a time, never a grid.

    .venv/bin/python -m src.model.run_architecture --parameters parameters-eda.txt \
        --results results/eda-contract

Each stage changes exactly one field from the currently selected configuration and is
resolved with the same paired-margin rule Task 3 uses (``representation_selection``).
Depth and width are genuinely ordinal -- more layers, more width is "more complex" --
so they ratchet through :func:`advance_complexity`. Heads are not ordinal: 8 is not
"more complex" than 2 in any useful sense, so that stage is resolved by
:func:`resolve_heads` instead (the correction Task 11 Step 1 requires from the start,
so the greedy-order validation in Task 11 checks the same walk this file performs).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import PARAMETERS_PATH, PROTOCOL, RunConfig, changed_fields, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one
from src.model.representation_selection import MOVES, compare
from src.model.results import RESULTS_DIR

BASE_NAME = "L2 learned embeddings with attention"

NUMERIC_MODES = ("affine", "buckets", "piecewise", "periodic")
DEPTHS = (1, 2, 3)
WIDTHS = (32, 64, 96)
HEADS = (2, 4, 8)
POST_ARCHITECTURE = (
    ("E learned positional", "positional", "learned"),
    ("F attention pooling", "pooling", "attention"),
    ("H dropout 0.3", "dropout", 0.3),
)
"""L2 arrives with ``affine+buckets``, one layer, ``d_model=64``, four heads, no
position, mean pooling and dropout 0.1. Neither four layers nor a width above 96 is
allowed in this story."""

ARCHITECTURE_DIR = "architecture"
SELECTION_FILE = "selection.json"


def selection_path(results: Path | str) -> Path:
    return Path(results) / ARCHITECTURE_DIR / SELECTION_FILE


def numeric_candidates(base: RunConfig) -> list[RunConfig]:
    return [
        replace(base, name=f"A numeric {mode}", numeric_embedding=mode)
        for mode in NUMERIC_MODES
        if mode != base.numeric_embedding
    ]


def depth_candidates(base: RunConfig) -> list[RunConfig]:
    return [
        replace(base, name=f"B depth {layers}", n_layers=layers)
        for layers in DEPTHS
        if layers != base.n_layers
    ]


def width_candidates(base: RunConfig) -> list[RunConfig]:
    return [
        replace(base, name=f"D d_model {width}", d_model=width)
        for width in WIDTHS
        if width != base.d_model
    ]


def head_candidates(base: RunConfig) -> list[RunConfig]:
    return [
        replace(base, name=f"C {heads} heads", n_heads=heads)
        for heads in HEADS
        if heads != base.n_heads
    ]


def advance_complexity(
    base: RunConfig,
    base_ap: np.ndarray,
    candidates: list[tuple[RunConfig, np.ndarray]],
) -> tuple[RunConfig, str]:
    """Ratchet through ordinal candidates, low to high; each must beat the winner so
    far, not just the original base. Returns the winner and **how** it won, because
    ``improves`` and ``tie-break`` are not reported the same way."""
    selected, selected_ap, how = base, base_ap, "base kept"
    for candidate, candidate_ap in candidates:
        outcome = compare(selected_ap, candidate_ap)
        if outcome in MOVES:
            selected, selected_ap, how = candidate, candidate_ap, outcome
    return selected, how


def resolve_heads(
    base: RunConfig,
    base_ap: np.ndarray,
    candidates: list[tuple[RunConfig, np.ndarray]],
) -> RunConfig:
    """Heads are not ordinal: 8 is not 'more complex' than 2 in any useful sense."""
    winners = [
        (run, ap)
        for run, ap in candidates
        if compare(base_ap, ap) in MOVES
    ]
    if not winners:
        return base
    run, _ = max(winners, key=lambda pair: (float(pair[1].mean()), -pair[0].n_heads))
    return run


def ap_of(config: RunConfig, frame, partitions, directory: str) -> np.ndarray:
    result, note = run_one(config, frame, partitions, directory=directory)
    ap = np.array([fold.average_precision for fold in result.folds], dtype=float)
    print(f"    {config.digest}  {config.name:<40s} AP {ap.mean():.4f} +/- {ap.std(ddof=1):.4f}   [{note}]")
    return ap


def _point(config: RunConfig, ap: np.ndarray) -> dict:
    return {
        "name": config.name,
        "digest": config.digest,
        "ap_mean": float(ap.mean()),
        "ap_std": float(ap.std(ddof=1)),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    base = declared[BASE_NAME]

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    def ap(config: RunConfig) -> np.ndarray:
        return ap_of(config, frame, partitions, args.results)

    stages: list[dict] = []

    print(f"\n=== NUMERIC EMBEDDING (base {base.numeric_embedding!r}) ===")
    base_ap = ap(base)
    candidates = [(run, ap(run)) for run in numeric_candidates(base)]
    numeric_selected, numeric_how = advance_complexity(base, base_ap, candidates)
    print(f"  -> {numeric_selected.numeric_embedding}  ({numeric_how})")
    stages.append({
        "axis": "numeric_embedding",
        "base": _point(base, base_ap),
        "candidates": [
            {**_point(run, run_ap), "outcome": compare(base_ap, run_ap)}
            for run, run_ap in candidates
        ],
        "selected": numeric_selected.numeric_embedding,
    })

    print(f"\n=== DEPTH (from {numeric_selected.n_layers}) ===")
    selected_ap = ap(numeric_selected) if numeric_selected is not base else base_ap
    depth_selected = numeric_selected
    depth_ap = selected_ap
    depth_points = []
    for candidate in depth_candidates(numeric_selected):
        candidate_ap = ap(candidate)
        outcome = compare(depth_ap, candidate_ap)
        depth_points.append({**_point(candidate, candidate_ap), "outcome": outcome})
        if outcome in MOVES:
            depth_selected, depth_ap = candidate, candidate_ap
    print(f"  -> n_layers={depth_selected.n_layers}")
    stages.append({
        "axis": "depth",
        "base": _point(numeric_selected, selected_ap),
        "candidates": depth_points,
        "selected": depth_selected.n_layers,
    })

    print(f"\n=== WIDTH (from n_layers={depth_selected.n_layers}, d_model={depth_selected.d_model}) ===")
    width_selected, width_ap = depth_selected, depth_ap
    width_points = []
    for candidate in width_candidates(depth_selected):
        candidate_ap = ap(candidate)
        outcome = compare(width_ap, candidate_ap)
        width_points.append({**_point(candidate, candidate_ap), "outcome": outcome})
        if outcome in MOVES:
            width_selected, width_ap = candidate, candidate_ap
    print(f"  -> d_model={width_selected.d_model}")
    stages.append({
        "axis": "d_model",
        "base": _point(depth_selected, depth_ap),
        "candidates": width_points,
        "selected": width_selected.d_model,
    })

    print(f"\n=== HEADS (from d_model={width_selected.d_model}, n_heads={width_selected.n_heads}) ===")
    head_pairs = [(run, ap(run)) for run in head_candidates(width_selected)]
    heads_selected = resolve_heads(width_selected, width_ap, head_pairs)
    print(f"  -> n_heads={heads_selected.n_heads}")
    stages.append({
        "axis": "n_heads",
        "base": _point(width_selected, width_ap),
        "candidates": [
            {**_point(run, run_ap), "outcome": compare(width_ap, run_ap)}
            for run, run_ap in head_pairs
        ],
        "selected": heads_selected.n_heads,
    })

    print("\n=== POSITION / POOLING / DROPOUT ===")
    current, current_ap = heads_selected, ap(heads_selected) if heads_selected is not width_selected else width_ap
    module_log: list[tuple[str, str]] = []
    module_points = []
    for label, field_name, value in POST_ARCHITECTURE:
        if getattr(current, field_name) == value:
            continue
        candidate = replace(current, name=label, **{field_name: value})
        candidate_ap = ap(candidate)
        outcome = compare(current_ap, candidate_ap)
        module_points.append({**_point(candidate, candidate_ap), "outcome": outcome, "base_at_time": current.name})
        if outcome in MOVES:
            current, current_ap = candidate, candidate_ap
            module_log.append((label, outcome))
        else:
            module_log.append((label, outcome))
    for label, outcome in module_log:
        print(f"  {label}: {outcome}")
    stages.append({
        "axis": "modules",
        "base": _point(heads_selected, ap(heads_selected) if heads_selected is not width_selected else width_ap),
        "candidates": module_points,
        "selected": current.name,
    })

    final_config = replace(current, name="M selected from directed comparisons")

    selection = {
        "numeric_embedding": {
            "base": base.numeric_embedding,
            "selected": numeric_selected.numeric_embedding,
        },
        "depth": {"tried": list(DEPTHS), "selected": depth_selected.n_layers},
        "d_model": {"tried": list(WIDTHS), "selected": width_selected.d_model},
        "n_heads": {"tried": list(HEADS), "selected": heads_selected.n_heads},
        "positional": {"base": base.positional, "selected": current.positional},
        "pooling": {"base": base.pooling, "selected": current.pooling},
        "dropout": {"base": base.dropout, "selected": current.dropout},
        "final_config": asdict(final_config),
        "stages": stages,
    }
    path = selection_path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, default=list), encoding="utf-8")
    print(f"\nresolved architecture written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
