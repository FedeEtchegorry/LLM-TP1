"""Runs the seven aspects in order, prints their tables and writes every figure.

    py -3.13 -m src.eda.run_eda
"""

from __future__ import annotations

from pathlib import Path

from src.eda import ranking
from src.eda.aspects import composition, package, price, product, search, target, text
from src.eda.loading import load_dataset
from src.eda.plots import FIGURES_DIR

ASPECTS = (target, text, price, product, package, composition, search)


def main(figures: Path = FIGURES_DIR) -> None:
    frame = load_dataset()
    for aspect in ASPECTS:
        aspect.analyse(frame, figures)
    ranking.analyse(frame, figures)
    print(f"\nFiguras escritas en {figures}/")


if __name__ == "__main__":
    main()
