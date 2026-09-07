"""E5: los cinco ejes de ablación consolidados en un solo forest plot.

    .venv/bin/python -m scripts.run_ablation_summary

No entrena: lee los JSON que ya escribieron ``run_tokenizer_ablation`` y
``run_architecture_sweep``, reconstruye los contrastes pareados por semilla y dibuja
todo junto. Se puede volver a correr cada vez que llega una corrida nueva, y cuesta
segundos.

Los ejes que todavía no tengan JSON simplemente no aparecen, y el resumen dice cuáles
faltan en vez de dibujar una figura incompleta sin avisar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model.ablation_summary import (
    groups,
    markdown_table,
    plotted,
    reading,
)
from src.model.configs import (
    PARAMETERS_PATH,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.figures import FIGURES_DIR, grouped_forest
from src.model.results import RESULTS_DIR

SWEEP = "architecture/sweep.json"
TOKENIZER = "ablation/tokenizer-both.json"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "ablation"))
    parser.add_argument("--config", type=str, default="RUN")
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
    if args.config not in declared:
        raise SystemExit(
            f"{args.config!r} no está declarado en {args.parameters}; "
            f"hay {sorted(declared)}"
        )
    base = apply_overrides(declared[args.config], args.overrides)

    results = Path(args.results)
    sweep = read_json(results / SWEEP)
    tokenizer = read_json(results / TOKENIZER)

    print("=== FUENTES ===")
    for label, source, document in (
        ("tokenizador (D1)", TOKENIZER, tokenizer),
        ("barrido de ejes (D4, D8, D13, D6, D5)", SWEEP, sweep),
    ):
        state = "leído" if document else "FALTA — correr su runner"
        print(f"  {label:<40s} {results / source}  [{state}]")

    built = groups(sweep, tokenizer, base)
    if not built:
        print()
        print(reading(built))
        return 1

    print("\n=== ABLACIONES ===")
    print(markdown_table(built))

    print("\n=== LECTURA ===")
    conclusion = reading(built)
    print(conclusion)

    figure = grouped_forest(
        plotted(built),
        title="Estudio de ablación — cada módulo contra la configuración base",
        xlabel="Diferencia de PR-AUC contra la base (barra = ±1 error pareado)",
        path=Path(args.figures) / "ablation-forest.png",
    )

    summary = results / "ablation" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "base_digest": base.digest,
                "groups": [
                    {
                        "label": group.label,
                        "contrasts": [
                            {
                                "label": item.label.strip(),
                                "mean": item.mean,
                                "error": item.error,
                                "agree": item.agree,
                                "distinguishable": item.distinguishable,
                            }
                            for item in group.contrasts
                        ],
                    }
                    for group in built
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
