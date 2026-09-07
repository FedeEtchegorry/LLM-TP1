"""Curva de aprendizaje del Transformer contra la baseline lineal.

    .venv/bin/python -m scripts.run_learning_curve --dry-run
    .venv/bin/python -m scripts.run_learning_curve

Entrena los dos modelos con 500, 1.000, 2.000, 4.000 y 6.400 filas de entrenamiento por
fold, recortando por query entera y dejando las filas puntuadas intactas. Lo que se lee
es la pendiente, no quién está más arriba: la lectura la escribe
:func:`src.model.learning_curve.read_slopes`.

**Cada tamaño escribe en su propio directorio.** ``RunConfig.digest`` no incluye cuántas
filas entraron, así que compartir directorio haría que el segundo tamaño leyera el
resultado cacheado del primero y la curva saliera plana sin que nada avisara.

La medición es sobre validación cruzada y no sobre el holdout: la conclusión sobre la
pendiente tiene que quedar escrita antes de gastarlo.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.configs import (
    LOGISTIC,
    PARAMETERS_PATH,
    PROTOCOL,
    TRANSFORMER,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import partition, run_one
from src.model.figures import FIGURES_DIR, learning_curves
from src.model.learning_curve import (
    SIZES,
    Point,
    curve,
    markdown_table,
    read_slopes,
    subsample,
)
from src.model.representation_selection import SEEDS
from src.model.results import RESULTS_DIR, load

MODELS = (TRANSFORMER, LOGISTIC)
LABELS = {TRANSFORMER: "Transformer (dos torres)", LOGISTIC: "Baseline lineal"}


def size_directory(root: str, size: int) -> Path:
    return Path(root) / "learning-curve" / f"n{size}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "learning-curve"))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        metavar="N",
        help=f"filas de entrenamiento por punto (default: {' '.join(map(str, SIZES))})",
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
    sizes = sorted(args.sizes)

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if args.config not in declared:
        raise SystemExit(
            f"{args.config!r} no está declarado en {args.parameters}; "
            f"hay {sorted(declared)}"
        )
    base = apply_overrides(declared[args.config], args.overrides)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    query_ids = frame["query_id"].tolist()

    total = len(sizes) * len(MODELS) * len(SEEDS)
    print(f"=== GRILLA ===")
    print(
        f"{len(sizes)} tamaños × {len(MODELS)} modelos × {len(SEEDS)} semillas "
        f"= {total} entrenamientos"
    )
    cached = 0
    for size in sizes:
        shrunk = subsample(partitions, query_ids, size)
        rows = sum(len(fold.train_indices) for fold in shrunk.folds) // len(shrunk.folds)
        queries = len({query_ids[i] for i in shrunk.folds[0].train_indices})
        print(f"  n={size:<5d} {rows} filas/fold, {queries} queries en el fold 0")
        for model in MODELS:
            for seed in SEEDS:
                config = replace(base, model=model, seed=seed)
                recorded = load(config, size_directory(args.results, size)) is not None
                cached += recorded
    print(f"\n{cached} grabados, {total - cached} por entrenar")

    if args.dry_run:
        print("\n--dry-run: no se entrenó nada")
        return 0

    points: list[Point] = []
    for size in sizes:
        shrunk = subsample(partitions, query_ids, size)
        directory = size_directory(args.results, size)
        for model in MODELS:
            print(f"\n=== n={size} · {LABELS[model]} ===")
            runs = []
            for seed in SEEDS:
                result, note = run_one(
                    replace(base, model=model, seed=seed),
                    frame,
                    shrunk,
                    directory=directory,
                    force=args.force,
                )
                ordered = sorted(result.folds, key=lambda fold: fold.fold_index)
                runs.append(tuple(fold.average_precision for fold in ordered))
                print(
                    f"  seed {seed:<5d} AP {result.average_precision_mean:.4f} [{note}]",
                    flush=True,
                )
            points.append(Point(model=model, size=size, runs=tuple(runs)))

    print("\n=== CURVA ===")
    print(markdown_table(points))

    print("\n=== LECTURA DE LA PENDIENTE ===")
    reading = read_slopes(points)
    print(reading)

    figure = learning_curves(
        {
            LABELS[model]: [
                (point.size, point.mean, point.spread) for point in curve(points, model)
            ]
            for model in MODELS
        },
        title="Curva de aprendizaje contra la baseline",
        path=Path(args.figures) / "learning-curve.png",
    )

    summary = Path(args.results) / "learning-curve" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "sizes": sizes,
                "base_digest": base.digest,
                "points": [
                    {
                        "model": point.model,
                        "size": point.size,
                        "mean": point.mean,
                        "spread": point.spread,
                        "runs": [list(run) for run in point.runs],
                    }
                    for point in points
                ],
                "reading": reading,
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
