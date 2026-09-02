"""La ablación que aísla la autoatención: la arquitectura elegida con cero bloques.

    .venv/Scripts/python -m src.model.run_attention_ablation \
        --results results/eda-contract

Compara la arquitectura elegida contra esa misma arquitectura con ``n_layers = 0``: un
solo campo de diferencia. ``L1`` no sirve para esto porque difiere en cuatro.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import EDA_PARAMETERS, PROTOCOL, RunConfig, load_parameters
from src.model.console import utf8_console
from src.model.experiment import describe, partition
from src.model.representation_selection import SEEDS, seed_mean, seed_spread
from src.model.results import RESULTS_DIR
from src.model.run_bracket_search import (
    ARCHITECTURE_DIR,
    SEARCH_FILE,
    Evaluator,
    canonical_name,
)

ABLATION_FILE = "attention-ablation.json"


def chosen(results: str | Path) -> RunConfig:
    """La arquitectura que escribió el recorrido, tal cual."""
    path = Path(results) / ARCHITECTURE_DIR / SEARCH_FILE
    if not path.exists():
        raise SystemExit(f"falta {path}: correr src.model.run_bracket_search primero")
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = dict(document["final"]["config"])
    for name in ("text_fields", "categorical_fields", "numeric_fields"):
        fields[name] = tuple(fields[name])
    return RunConfig(**fields)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    load_parameters(args.parameters)

    with_attention = chosen(args.results)
    without = replace(with_attention, n_layers=0)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)
    evaluate = Evaluator(frame, partitions, args.results, args.force)

    print(f"\n=== LA AUTOATENCION, AISLADA (semillas {SEEDS}) ===")
    print(f"    unico campo que cambia: n_layers {with_attention.n_layers} -> 0\n")

    runs_with = evaluate(with_attention)
    runs_without = evaluate(without)

    mean_with = seed_mean(runs_with, label="con atencion")
    mean_without = seed_mean(runs_without, label="sin atencion")
    delta = mean_with - mean_without

    document = {
        "seeds": list(SEEDS),
        "con_atencion": {
            "name": canonical_name(with_attention),
            "n_layers": with_attention.n_layers,
            "ap": mean_with,
            "sd_semillas": seed_spread(runs_with),
            "por_semilla": [float(np.mean(r)) for r in runs_with],
        },
        "sin_atencion": {
            "name": canonical_name(without),
            "n_layers": 0,
            "ap": mean_without,
            "sd_semillas": seed_spread(runs_without),
            "por_semilla": [float(np.mean(r)) for r in runs_without],
        },
        "delta": delta,
        "config": asdict(with_attention),
    }

    out = Path(args.results) / ARCHITECTURE_DIR / ABLATION_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"  con atencion ({with_attention.n_layers} bloques)  AP {mean_with:.4f}")
    print(f"  sin atencion (0 bloques)              AP {mean_without:.4f}")
    print(f"  aporte de la autoatencion             {delta:+.4f}")
    print(f"\n  escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
