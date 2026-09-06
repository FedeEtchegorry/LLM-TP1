"""Los dos diagramas de arquitectura, el nuevo y el viejo, para el antes y después.

    .venv/bin/python -m scripts.run_architecture_diagram

No entrena ni lee resultados: dibuja lo que ``ARCHITECTURE.md`` y ``OLD_ARCHITECTURE.md``
declaran. Se corre cada vez que una de las dos descripciones cambia, que es la razón de
que el diagrama viva en el repo y no en un export sin fuente.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.console import utf8_console
from src.model.diagram import one_tower, personalised, render, two_towers
from src.model.figures import FIGURES_DIR


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=str, default=str(FIGURES_DIR / "architecture")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    directory = Path(args.figures)

    written = [
        render(
            two_towers(),
            title="Arquitectura de dos torres — fusión tardía",
            path=directory / "two-towers.png",
        ),
        render(
            one_tower(),
            title="Arquitectura de una sola torre — primera entrega",
            path=directory / "one-tower.png",
        ),
        render(
            personalised(),
            title="Cómo entraría la personalización de usuario",
            path=directory / "personalised.png",
        ),
    ]
    for path in written:
        print(f"figura: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
