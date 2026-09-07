"""TOKENIZER y INTERACTION: qué aportan los paréntesis, y por qué lo aportan.

    .venv/bin/python -m scripts.run_tokenizer_ablation --dry-run
    .venv/bin/python -m scripts.run_tokenizer_ablation --analyses tokenizer
    .venv/bin/python -m scripts.run_tokenizer_ablation

Los dos analyses comparten dos celdas de la grilla, así que pedirlos juntos cuesta cinco
configuraciones y no siete: el caché por digest reconoce las repetidas. ``--dry-run``
muestra la grilla, el digest de cada celda y cuáles ya están grabadas, sin entrenar nada.

La grilla y las dos lecturas viven en :mod:`src.model.ablation`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.ablation import (
    Measured,
    cells_for,
    interaction_contrasts,
    tokenizer_contrasts,
    expected_runs,
    interaction_series,
    markdown_table,
    plotted,
    read_tokenizer,
    read_interaction,
)
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import partition, run_one
from src.model.figures import FIGURES_DIR, interaction_lines, paired_contrasts
from src.model.representation_selection import SEEDS
from src.model.results import RESULTS_DIR, load

ANALYSES = {
    "tokenizer": ("tokenizer",),
    "interaction": ("interaction",),
    "both": ("tokenizer", "interaction"),
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "ablation"))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument("--analyses", choices=sorted(ANALYSES), default="both")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mostrar la grilla y el estado del cache sin entrenar",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="clave=valor",
        help="cambiar un campo sin declarar una seccion; repetible",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    analyses = ANALYSES[args.analyses]

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if args.config not in declared:
        raise SystemExit(
            f"{args.config!r} no está declarado en {args.parameters}; "
            f"hay {sorted(declared)}"
        )
    base = apply_overrides(declared[args.config], args.overrides)
    cells = cells_for(analyses)

    print(f"=== GRILLA ({args.analyses}) ===")
    print(
        f"{len(cells)} configuraciones × {len(SEEDS)} semillas = "
        f"{expected_runs(analyses)} entrenamientos"
    )
    cached = 0
    for cell in cells:
        for seed in SEEDS:
            config = cell.config(base, seed)
            recorded = load(config, args.results) is not None
            cached += recorded
            print(
                f"  {cell.key}  seed {seed:<5d} {config.digest}  "
                f"{cell.tokenizer:<11s} paréntesis={'sí' if cell.keep_brackets else 'no':<3s} "
                f"positional={cell.positional:<8s} "
                f"{'[grabado]' if recorded else ''}"
            )
    print(f"\n{cached} grabados, {expected_runs(analyses) - cached} por entrenar")

    if args.dry_run:
        print("\n--dry-run: no se entrenó nada")
        return 0

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)

    measured: list[Measured] = []
    for cell in cells:
        print(f"\n=== {cell.key}: {cell.label} ===")
        runs = []
        for seed in SEEDS:
            result, note = run_one(
                cell.config(base, seed),
                frame,
                partitions,
                directory=args.results,
                force=args.force,
            )
            ordered = sorted(result.folds, key=lambda fold: fold.fold_index)
            runs.append(tuple(fold.average_precision for fold in ordered))
            print(
                f"  seed {seed:<5d} AP {result.average_precision_mean:.4f} [{note}]",
                flush=True,
            )
        measured.append(Measured(cell=cell, runs=tuple(runs)))

    print("\n=== CELDAS ===")
    print(markdown_table(measured))

    figures = Path(args.figures)
    written = []
    if "tokenizer" in analyses:
        print("\n=== LECTURA DE TOKENIZER ===")
        print(read_tokenizer(measured))
        written.append(
            paired_contrasts(
                plotted(tokenizer_contrasts(measured)),
                title="TOKENIZER — qué mueve el AP: los paréntesis, no el tokenizador",
                xlabel="Diferencia de AP, pareada por semilla (barra = ±1 error)",
                path=figures / "tokenizer-contrasts.png",
            )
        )
    if "interaction" in analyses:
        print("\n=== LECTURA DE INTERACTION ===")
        print(read_interaction(measured))
        written.append(
            interaction_lines(
                interaction_series(measured),
                xlabels=("sin paréntesis", "con paréntesis"),
                title="INTERACTION — los paréntesis sirven sólo si hay posiciones",
                path=figures / "interaction-lines.png",
            )
        )
        written.append(
            paired_contrasts(
                plotted(interaction_contrasts(measured)),
                title="INTERACTION — el efecto de los paréntesis en cada nivel, y su interacción",
                xlabel="Diferencia de AP, pareada por semilla (barra = ±1 error)",
                path=figures / "interaction-contrasts.png",
            )
        )

    summary = Path(args.results) / "ablation" / f"tokenizer-{args.analyses}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "analyses": list(analyses),
                "base_digest": base.digest,
                "cells": [
                    {
                        "key": item.cell.key,
                        "label": item.cell.label,
                        "tokenizer": item.cell.tokenizer,
                        "keep_brackets": item.cell.keep_brackets,
                        "positional": item.cell.positional,
                        "mean": item.mean,
                        "spread": item.spread,
                        "runs": [list(run) for run in item.runs],
                    }
                    for item in measured
                ],
                "reading_tokenizer": read_tokenizer(measured) if "tokenizer" in analyses else None,
                "reading_interaction": read_interaction(measured) if "interaction" in analyses else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    for path in written:
        print(f"figura: {path}")
    print(f"resumen: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
