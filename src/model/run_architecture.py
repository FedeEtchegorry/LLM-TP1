"""Follow the small, directed Transformer architecture path.

The runner deliberately walks one coordinate at a time.  It is not an
architecture grid: depth and width are ordered paths, while heads are resolved
as competing alternatives to the four-head configuration.

    .venv/Scripts/python -m src.model.run_architecture \
        --parameters parameters-eda.txt --results results/eda-contract
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import PARAMETERS_PATH, RunConfig, ladder_runs, load_parameters
from src.model.eda_contract import changed_fields, find_prefix, require_valid
from src.model.experiment import describe, partition, run_one
from src.model.protocol import EvaluationResult
from src.model.representation_selection import SEEDS, seed_mean, seed_spread
from src.model.results import RESULTS_DIR


NUMERIC_MODES = ("affine", "buckets", "piecewise")
DEPTHS = (1, 2, 3)
WIDTHS = (32, 64, 96)
HEADS = (2, 4, 8)
POST_ARCHITECTURE = (
    ("E learned positional", "positional", "learned"),
    ("F attention pooling", "pooling", "attention"),
    ("H dropout 0.3", "dropout", 0.3),
)

ARCHITECTURE_DIR = "architecture"
SELECTION_FILE = "selection.json"
FINAL_NAME = "M selected from directed comparisons"


def depth_candidates(base: RunConfig) -> list[RunConfig]:
    """The only deeper models considered: two and then three blocks."""
    return [
        replace(base, name=f"B depth {layers}", n_layers=layers)
        for layers in DEPTHS
        if layers != base.n_layers
    ]


def width_candidates(base: RunConfig) -> list[RunConfig]:
    """The two widths around the current architecture's width."""
    return [
        replace(base, name=f"D d_model {width}", d_model=width)
        for width in WIDTHS
        if width != base.d_model
    ]


def head_candidates(base: RunConfig) -> list[RunConfig]:
    """The non-reference head counts, all valid for every declared width."""
    return [
        replace(base, name=f"C {heads} heads", n_heads=heads)
        for heads in HEADS
        if heads != base.n_heads
    ]


def numeric_candidates(base: RunConfig) -> list[RunConfig]:
    """The three Transformer readings compared to L2's affine+buckets input."""
    return [
        replace(base, name=f"A numeric {mode}", numeric_embedding=mode)
        for mode in NUMERIC_MODES
        if mode != base.numeric_embedding
    ]


def resolve_stage(
    base: RunConfig,
    base_runs: list[np.ndarray],
    candidates: list[tuple[RunConfig, list[np.ndarray]]],
) -> tuple[RunConfig, str]:
    """Gana la media más alta sobre las tres semillas. Un empate conserva la base.

    Con la regla de mayor media, profundidad, ancho, heads y embedding numérico se
    resuelven todos igual: ya no hay un eje ordinal que exija recorrer de menor a
    mayor, ni un desempate por dispersión que distinga a los heads del resto. Las
    tres funciones que antes hacían esto por separado son ahora ésta.
    """
    incumbent = seed_mean(base_runs, label=base.name)
    best_run, best_runs, best_mean = base, base_runs, incumbent
    for candidate, runs in candidates:
        mean = seed_mean(runs, label=candidate.name)
        if mean > best_mean:
            best_run, best_runs, best_mean = candidate, runs, mean
    if best_run is base:
        return base, "base kept"
    return best_run, "higher mean"


def ap_scores(result: EvaluationResult) -> np.ndarray:
    """Five AP values in fold order, suitable for the declared paired rule."""
    ordered = sorted(result.folds, key=lambda fold: fold.fold_index)
    return np.asarray([fold.average_precision for fold in ordered], dtype=float)


def dropout_is_indicated() -> bool:
    """Return whether Task 5 supplied a measurable dropout trigger.

    It did not quantify either “train AP high” or “validation AP clearly lower”.
    The runner therefore records H as omitted instead of silently inventing a
    threshold.  A later plan amendment can replace this function with its declared
    diagnostic criterion without changing the directed path above it.
    """
    return False


def selection_path(results: Path | str) -> Path:
    return Path(results) / ARCHITECTURE_DIR / SELECTION_FILE


def write_selection(document: dict, results: Path | str) -> Path:
    path = selection_path(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _run_seeds(
    config: RunConfig, frame, partitions, results: Path | str, force: bool
) -> list[np.ndarray]:
    """Los AP por fold de las tres semillas declaradas, entrenando lo que falte.

    El nombre de la sección no cambia entre semillas; lo que cambia es ``seed``, que
    entra al digest, así que cada semilla es su propio registro y la 1337 suele
    resolverse por caché desde corridas anteriores.
    """
    runs: list[np.ndarray] = []
    for seed in SEEDS:
        result, note = run_one(
            replace(config, seed=seed), frame, partitions, directory=results, force=force
        )
        print(f"    seed {seed:<5d} AP {result.average_precision_mean:.4f} [{note}]", flush=True)
        runs.append(ap_scores(result))
    mean, spread = seed_mean(runs, label=config.name), seed_spread(runs, label=config.name)
    print(f"  {config.name:<32s} AP {mean:.4f} (3 semillas, dispersion {spread:.4f})", flush=True)
    return runs


def run_stage(
    title: str,
    base: RunConfig,
    base_runs: list[np.ndarray],
    candidates: list[RunConfig],
    evaluate,
    resolve=resolve_stage,
) -> tuple[RunConfig, list[np.ndarray], str]:
    """Una coordenada del recorrido: mide sus candidatos con tres semillas y resuelve."""
    print(f"\n=== {title} ===")
    scored = [(candidate, evaluate(candidate)) for candidate in candidates]
    selected, how = resolve(base, base_runs, scored)
    selected_runs = next((runs for run, runs in scored if run == selected), base_runs)
    print(f"  -> {title}: {selected.name}  ({how})")
    return selected, selected_runs, how


def _post_candidate(base: RunConfig, name: str, field: str, value: object) -> RunConfig:
    return replace(base, name=name, **{field: value})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    l2 = find_prefix(ladder_runs(declared), "L2")

    frame = load_dataset()
    partitions = partition(frame)
    describe(frame, partitions)

    def evaluate(config: RunConfig) -> np.ndarray:
        return _run(config, frame, partitions, args.results, args.force)

    print("\n=== L2, THE BASE OF THE PATH ===")
    current, current_ap = l2, evaluate(l2)

    current, current_ap, numeric_how = run_stage(
        "NUMERIC EMBEDDING", current, current_ap, numeric_candidates(current), evaluate
    )
    numeric_selected = current

    current, current_ap, depth_how = run_stage(
        "DEPTH", current, current_ap, depth_candidates(current), evaluate
    )
    depth_selected = current

    current, current_ap, width_how = run_stage(
        "WIDTH", current, current_ap, width_candidates(current), evaluate
    )
    width_selected = current

    current, current_ap, heads_how = run_stage(
        "HEADS", current, current_ap, head_candidates(current), evaluate
    )
    heads_selected = current

    print("\n=== POSITION, POOLING AND DROPOUT ===")
    post: dict[str, dict[str, object]] = {}
    for name, field, value in POST_ARCHITECTURE:
        base_value = getattr(current, field)
        if field == "dropout" and not dropout_is_indicated():
            post[field] = {
                "base": base_value,
                "selected": base_value,
                "outcome": "omitted: no quantitative dropout trigger declared",
            }
            print(f"  {name}: omitted, no declared dropout trigger")
            continue
        candidate = _post_candidate(current, name, field, value)
        candidate_runs = evaluate(candidate)
        selected, outcome = resolve_stage(current, current_ap, [(candidate, candidate_runs)])
        if selected is candidate:
            current, current_ap = candidate, candidate_runs
        post[field] = {
            "base": base_value,
            "selected": getattr(current, field),
            "outcome": outcome,
        }

    final_config = replace(current, name=FINAL_NAME)
    document = {
        "numeric_embedding": {
            "base": l2.numeric_embedding,
            "selected": numeric_selected.numeric_embedding,
            "outcome": numeric_how,
        },
        "depth": {"tried": list(DEPTHS), "selected": depth_selected.n_layers, "outcome": depth_how},
        "d_model": {"tried": list(WIDTHS), "selected": width_selected.d_model, "outcome": width_how},
        "n_heads": {"tried": list(HEADS), "selected": heads_selected.n_heads, "outcome": heads_how},
        "positional": post["positional"],
        "pooling": post["pooling"],
        "dropout": post["dropout"],
        "final_config": asdict(final_config),
    }
    path = write_selection(document, args.results)
    print(f"\narchitecture decision written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
