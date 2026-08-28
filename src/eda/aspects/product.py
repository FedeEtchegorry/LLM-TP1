"""Aspect 3: what the product is, who makes it, where it comes from."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_level
from src.eda.rates import rate_by_level

COLUMNS: tuple[str, ...] = ("category", "brand", "storage_type", "country_of_origin")

TITLES: dict[str, str] = {
    "category": "BTR por categoria del producto",
    "brand": "BTR por marca",
    "storage_type": "BTR por tipo de almacenamiento",
    "country_of_origin": "BTR por pais de origen",
}


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """One table and one chart per catalogue attribute."""
    report.heading("Aspecto 3 - Que es el producto")

    tables: dict[str, pd.DataFrame] = {}
    for column in COLUMNS:
        table = rate_by_level(frame, column)
        report.show(table, caption=f"Tasa de compra por {column}:")
        bar_by_level(
            table,
            title=TITLES[column],
            path=figures / f"03-producto-{column}.png",
            target=frame["bought"].to_numpy(),
        )
        tables[column] = table

    return tables
