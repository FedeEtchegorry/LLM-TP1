"""Búsqueda por bracket adaptativo: tres variantes por eje, y se extiende si gana un borde.

    .venv/Scripts/python -m scripts.run_bracket_search \
        --parameters parameters-eda.txt --results results/eda-contract

Gana la media más alta sobre tres semillas, sin margen ni desempate por dispersión.
Los topes de cada eje están en ``LADDERS``.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import EDA_PARAMETERS, PROTOCOL, RunConfig, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one
from src.model.representation_selection import SEEDS, seed_mean, seed_spread
from src.model.results import RESULTS_DIR

LADDERS: dict[str, tuple[float, ...]] = {
    "n_layers": (1, 2, 3, 4),
    "d_model": (16, 32, 64, 96, 128),
    "n_heads": (1, 2, 4, 8, 16),
    "learning_rate": (2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3),
}
"""Los valores de cada eje, en orden; el último de cada tupla es un tope duro.

``learning_rate`` va al final, así que la arquitectura se eligió con ``lr=1e-4`` y el
paso se ajusta después: los tres ejes previos quedaron resueltos a otro protocolo.
"""

NUMERIC_MODES = ("affine", "buckets", "affine+buckets", "piecewise", "periodic")
"""El embedding numérico no es ordinal: se prueban los cinco modos que lee
``network.py`` y gana el mayor."""

MODULES = (
    ("positional", "learned"),
    ("pooling", "attention"),
    ("dropout", 0.3),
)
"""Alternativas de módulo, cada una un solo cambio contra la arquitectura elegida."""

ARCHITECTURE_DIR = "architecture"
SEARCH_FILE = "bracket-search.json"


def canonical_name(config: RunConfig) -> str:
    """Un nombre por configuración, derivado de sus campos."""
    return (
        f"arch d{config.d_model} L{config.n_layers} h{config.n_heads} "
        f"{config.numeric_embedding} {config.positional} {config.pooling} "
        f"do{config.dropout:g} lr{config.learning_rate:g}"
    )


def neighbours(axis: str, value: int) -> list[int]:
    """Los tres valores de la escalera alrededor del actual, recortados en los bordes."""
    ladder = LADDERS[axis]
    if value not in ladder:
        raise ValueError(f"{axis}={value} no está en la escalera declarada {ladder}")
    i = ladder.index(value)
    return list(ladder[max(0, i - 1) : i + 2])


def _valid(config: RunConfig) -> bool:
    """Los heads tienen que dividir al ancho; el resto de la validación la hace configs."""
    return config.d_model % config.n_heads == 0


def recorded_index(directory: str | Path) -> dict[str, tuple[str, list[float]]]:
    """Mapa configuración-sin-nombre -> (nombre registrado, AP por fold).

    Permite reutilizar cualquier corrida previa aunque se haya guardado con otro
    nombre, que es lo que el digest por sí solo no hace.
    """
    index: dict[str, tuple[str, list[float]]] = {}
    for path in glob.glob(str(Path(directory) / "*.json")):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        config = {k: v for k, v in document["config"].items() if k != "name"}
        folds = sorted(document["folds"], key=lambda f: f["fold_index"])
        index[json.dumps(config, sort_keys=True)] = (
            document["name"],
            [f["average_precision"] for f in folds],
        )
    return index


def _key(config: RunConfig) -> str:
    fields = {k: v for k, v in asdict(config).items() if k != "name"}
    for name in ("text_fields", "categorical_fields", "numeric_fields"):
        fields[name] = list(fields[name])
    return json.dumps(fields, sort_keys=True)


class Evaluator:
    """Mide una configuración con las tres semillas, reusando lo que ya exista."""

    def __init__(self, frame, partitions, directory: str, force: bool):
        self.frame, self.partitions = frame, partitions
        self.directory, self.force = directory, force
        self.index = recorded_index(directory)
        self.trained = 0
        self.reused = 0
        self.cache: dict[str, list[np.ndarray]] = {}

    def __call__(self, config: RunConfig) -> list[np.ndarray]:
        named = replace(config, name=canonical_name(config))
        key = _key(named)
        if key in self.cache:
            return self.cache[key]
        runs: list[np.ndarray] = []
        for seed in SEEDS:
            candidate = replace(named, seed=seed)
            found = None if self.force else self.index.get(_key(candidate))
            if found is not None:
                runs.append(np.asarray(found[1], dtype=float))
                self.reused += 1
                print(f"    seed {seed:<5d} AP {runs[-1].mean():.4f}  [reusa «{found[0]}»]", flush=True)
                continue
            result, _ = run_one(
                candidate, self.frame, self.partitions,
                directory=self.directory, force=self.force,
            )
            folds = sorted(result.folds, key=lambda f: f.fold_index)
            runs.append(np.asarray([f.average_precision for f in folds], dtype=float))
            self.trained += 1
            print(f"    seed {seed:<5d} AP {runs[-1].mean():.4f}  [entrenada]", flush=True)
        self.cache[key] = runs
        print(
            f"  {canonical_name(named):<52s} AP {seed_mean(runs):.4f}"
            f"  (disp entre semillas {seed_spread(runs):.4f})",
            flush=True,
        )
        return runs


def search_axis(axis: str, base: RunConfig, evaluate) -> tuple[RunConfig, dict]:
    """Prueba los tres valores alrededor del actual; si gana un borde corre la ventana.

    Termina cuando el ganador queda en el medio o al topar un extremo de ``LADDERS``.
    """
    ladder = LADDERS[axis]
    current = getattr(base, axis)
    scores: dict[int, float] = {}
    visited: set[int] = set()
    steps: list[dict] = []

    def value_of(v: int) -> float:
        if v in scores:
            return scores[v]
        candidate = replace(base, **{axis: v})
        if not _valid(candidate):
            scores[v] = float("-inf")
            print(f"  {axis}={v}: descartado, {candidate.d_model} no es divisible por {candidate.n_heads}")
            return scores[v]
        scores[v] = seed_mean(evaluate(candidate), label=f"{axis}={v}")
        return scores[v]

    while True:
        window = [v for v in neighbours(axis, current) if v not in visited or v in scores]
        for v in neighbours(axis, current):
            value_of(v)
        visited.update(neighbours(axis, current))
        winner = max(visited, key=lambda v: (scores[v], -v))
        steps.append({"window": neighbours(axis, current), "winner": winner,
                      "scores": {str(v): scores[v] for v in sorted(visited)}})
        i = ladder.index(winner)
        at_edge = winner in (min(neighbours(axis, current)), max(neighbours(axis, current)))
        more = (i > 0 and winner < current) or (i < len(ladder) - 1 and winner > current)
        if not at_edge or not more or winner == current:
            break
        print(f"  -> gano el borde {axis}={winner}, se extiende la ventana")
        current = winner

    best = max(visited, key=lambda v: (scores[v], -v))
    selected = replace(base, **{axis: best})
    return selected, {
        "axis": axis,
        "ladder": list(ladder),
        "evaluated": {str(v): scores[v] for v in sorted(visited)},
        "selected": best,
        "moved": best != getattr(base, axis),
        "steps": steps,
    }


def search_set(name: str, field: str, options, base: RunConfig, evaluate) -> tuple[RunConfig, dict]:
    """Un eje sin orden: se prueban todas las opciones y gana la mayor media."""
    scores: dict[str, float] = {}
    for option in options:
        candidate = replace(base, **{field: option})
        if not _valid(candidate):
            continue
        scores[str(option)] = seed_mean(evaluate(candidate), label=f"{field}={option}")
    current = str(getattr(base, field))
    best = max(scores, key=lambda k: scores[k])
    if current in scores and scores[current] >= scores[best]:
        best = current
    value = type(getattr(base, field))(best) if not isinstance(getattr(base, field), str) else best
    selected = replace(base, **{field: value})
    return selected, {
        "axis": field, "evaluated": scores, "selected": best,
        "moved": str(getattr(base, field)) != best,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    base = next(run for name, run in declared.items() if name.startswith("L2"))

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)
    evaluate = Evaluator(frame, partitions, args.results, args.force)

    print(f"\n=== BASE: {canonical_name(base)} (semillas {SEEDS}) ===")
    evaluate(base)

    document: dict[str, object] = {"seeds": list(SEEDS), "ladders": {k: list(v) for k, v in LADDERS.items()}}
    current = base

    print("\n=== EMBEDDING NUMERICO (sin orden: se prueban todos) ===")
    current, document["numeric_embedding"] = search_set(
        "numerico", "numeric_embedding", NUMERIC_MODES, current, evaluate)
    print(f"  -> {current.numeric_embedding}")

    for axis, title in (
        ("n_layers", "PROFUNDIDAD"),
        ("d_model", "ANCHO"),
        ("n_heads", "HEADS"),
        ("learning_rate", "LEARNING RATE"),
    ):
        print(f"\n=== {title} (bracket adaptativo sobre {LADDERS[axis]}) ===")
        current, document[axis] = search_axis(axis, current, evaluate)
        print(f"  -> {axis} = {getattr(current, axis)}")

    print("\n=== MODULOS (un cambio por vez sobre la arquitectura elegida) ===")
    modules: dict[str, object] = {}
    for field, value in MODULES:
        candidate = replace(current, **{field: value})
        before = seed_mean(evaluate(current), label="actual")
        after = seed_mean(evaluate(candidate), label=f"{field}={value}")
        moved = after > before
        if moved:
            current = candidate
        modules[field] = {"probado": value, "base": before, "alternativa": after, "adoptado": moved}
        print(f"  {field}={value}: {after:.4f} contra {before:.4f} -> {'adoptado' if moved else 'descartado'}")
    document["modules"] = modules

    final = replace(current, name=canonical_name(current))
    document["final"] = {"name": final.name, "config": asdict(final),
                         "ap": seed_mean(evaluate(final), label="final")}
    document["cost"] = {"entrenadas": evaluate.trained, "reusadas": evaluate.reused}

    out = Path(args.results) / ARCHITECTURE_DIR / SEARCH_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n=== ELEGIDA: {final.name}  AP {document['final']['ap']:.4f} ===")
    print(f"    entrenadas {evaluate.trained}, reusadas {evaluate.reused}")
    print(f"    escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
