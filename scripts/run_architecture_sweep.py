"""Barrido de arquitectura: un factor por vez, con el resto congelado en la base.

    .venv/bin/python -m scripts.run_architecture_sweep --dry-run
    .venv/bin/python -m scripts.run_architecture_sweep --axes heads ffn
    .venv/bin/python -m scripts.run_architecture_sweep

El valor base se comparte entre todos los ejes y se mide una sola vez, así que el barrido
completo cuesta 19 configuraciones por semilla y no 28.

La grilla se valida antes de entrenar: si algún ``d_model`` no es divisible por su
``n_heads``, falla en el primer segundo en lugar de a mitad de la tercera hora.

La grilla, la lectura y la columna «¿distinguible del ruido?» viven en
:mod:`src.model.architecture_sweep`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.ablation import plotted
from src.model.architecture_sweep import (
    AXES,
    Measured,
    axes_for,
    contrasts,
    expected_runs,
    markdown_table,
    reading,
    unique_configs,
    validate,
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
from src.model.figures import FIGURES_DIR, paired_contrasts
from src.model.representation_selection import SEEDS
from src.model.results import RESULTS_DIR, load


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "architecture"))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument(
        "--axes",
        nargs="+",
        metavar="EJE",
        help=f"subconjunto de ejes; hay {' '.join(axis.key for axis in AXES)}",
    )
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

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if args.config not in declared:
        raise SystemExit(
            f"{args.config!r} no está declarado en {args.parameters}; "
            f"hay {sorted(declared)}"
        )
    base = apply_overrides(declared[args.config], args.overrides)
    axes = axes_for(tuple(args.axes) if args.axes else None)
    validate(base, axes)

    unique = unique_configs(base, axes)
    print("=== GRILLA ===")
    print(
        f"{len(axes)} ejes, {len(unique)} configuraciones distintas × "
        f"{len(SEEDS)} semillas = {expected_runs(base, axes)} entrenamientos"
    )
    cached = 0
    for axis in axes:
        base_value = getattr(base, axis.field)
        print(f"  {axis.label} (base: {base_value})")
        for value in axis.values:
            marks = []
            for seed in SEEDS:
                config = axis.config(base, value, seed)
                if load(config, args.results) is not None:
                    cached += 1
                    marks.append(str(seed))
            note = f"  [grabado: {', '.join(marks)}]" if marks else ""
            flag = " ← base" if value == base_value else ""
            print(f"    {axis.field}={value}{flag}{note}")
    print(f"\n{cached} grabados, {expected_runs(base, axes) - cached} por entrenar")

    if args.dry_run:
        print("\n--dry-run: no se entrenó nada")
        return 0

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)

    measured: list[Measured] = []
    for axis in axes:
        print(f"\n=== {axis.label} ===")
        for value in axis.values:
            runs = []
            for seed in SEEDS:
                result, note = run_one(
                    axis.config(base, value, seed),
                    frame,
                    partitions,
                    directory=args.results,
                    force=args.force,
                )
                ordered = sorted(result.folds, key=lambda fold: fold.fold_index)
                runs.append(tuple(fold.average_precision for fold in ordered))
                print(
                    f"  {axis.field}={value!s:<6s} seed {seed:<5d} "
                    f"AP {result.average_precision_mean:.4f} [{note}]",
                    flush=True,
                )
            measured.append(Measured(axis=axis, value=value, runs=tuple(runs)))

    print("\n=== TABLA ===")
    print(markdown_table(measured, base))

    print("\n=== LECTURA ===")
    conclusion = reading(measured, base)
    print(conclusion)

    figure = paired_contrasts(
        plotted(contrasts(measured, base)),
        title="Cada eje contra su base, pareado por semilla",
        xlabel="Diferencia de AP contra la base (barra = ±1 error)",
        path=Path(args.figures) / "architecture-sweep.png",
    )

    summary = Path(args.results) / "architecture" / "sweep.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "base_digest": base.digest,
                "axes": [axis.key for axis in axes],
                "points": [
                    {
                        "axis": item.axis.key,
                        "field": item.axis.field,
                        "value": item.value,
                        "mean": item.mean,
                        "spread": item.spread,
                        "runs": [list(run) for run in item.runs],
                    }
                    for item in measured
                ],
                "reading": conclusion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"figura: {figure}")
    print(f"resumen: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
