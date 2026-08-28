"""Aspect 2: the price, and the two ways of placing it inside its search."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.contribution import absorption, holding, separation
from src.eda.plots import bar_by_bucket
from src.eda.rates import rate_by_bucket

BUCKETS = 10

PLACEMENTS: tuple[tuple[str, str], ...] = (
    ("price_position", "posicion en la ventana del filtro"),
    ("price_rank", "puesto entre los precios que mostro la busqueda"),
)


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Rate the price by decile, then each placement, then each one holding the other."""
    report.heading("Aspecto 2 - El precio y la busqueda en la que aparece")

    inside = (frame["price"] >= frame["filter_price_min"]) & (frame["price"] <= frame["filter_price_max"])
    windows = frame.groupby("query_id").agg(
        low=("filter_price_min", "first"),
        high=("filter_price_max", "first"),
        cheapest=("price", "min"),
        dearest=("price", "max"),
    )
    coverage = (windows["dearest"] - windows["cheapest"]) / (windows["high"] - windows["low"])
    print()
    report.value("Filas con el precio dentro de la ventana del filtro", f"{inside.mean() * 100:.1f}%")
    report.value("Ancho mediano de la ventana del filtro", f"${(windows['high'] - windows['low']).median():.2f}")
    report.value("Ancho mediano de los precios mostrados", f"${(windows['dearest'] - windows['cheapest']).median():.2f}")
    report.value("Cobertura mediana de la ventana", f"{coverage.median():.2f}")

    absolute = rate_by_bucket(frame, "price", buckets=BUCKETS)
    report.show(absolute, caption="Tasa de compra por decil de precio (dolares):")
    bar_by_bucket(
        absolute,
        title="BTR por decil de precio absoluto",
        xlabel="Precio (USD)",
        path=figures / "02-precio-absoluto.png",
        target=frame["bought"].to_numpy(),
    )

    tables: dict[str, pd.DataFrame] = {"price": absolute}
    print("\nValores distintos que toma cada forma de ubicar el precio:")
    for column, label in PLACEMENTS:
        report.value(f"  {column} ({label})", frame[column].nunique())

    for column, label in PLACEMENTS:
        table = rate_by_bucket(frame, column, buckets=BUCKETS)
        report.show(table, caption=f"Tasa de compra por decil de {column} - {label}:")
        bar_by_bucket(
            table,
            title=f"BTR por decil de {column}",
            xlabel=label,
            path=figures / f"02-precio-{column.replace('_', '-')}.png",
            target=frame["bought"].to_numpy(),
        )
        spread, floor = separation(frame, column, buckets=BUCKETS)
        report.value(f"Separacion de {column}", f"{spread:.1f} pp contra un piso de {floor.mean:.2f} ± {floor.deviation:.2f}")
        tables[column] = table

    for other in ("price_rank", "price"):
        for held, read in (("price_position", other), (other, "price_position")):
            control = holding(frame, read, held)
            print(f"\nSeparacion de {read} dentro de cada tramo de {held}:")
            print(control.to_string())
            tables[f"{read}_dado_{held}"] = control
        print(f"\nResumen - price_position contra {other}:")
        print(absorption(frame, "price_position", other).to_string())

    return tables
