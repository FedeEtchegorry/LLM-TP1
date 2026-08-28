"""Aspect 6: the search the product appeared in, and when."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_level
from src.eda.rates import rate_by_level

MATCHED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("category", "filter_category"),
    ("storage_type", "filter_storage_type"),
)


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Compare each filter with its product column, then rate the query and the date."""
    report.heading("Aspecto 6 - El contexto de la busqueda")

    print()
    for column, filter_column in MATCHED_COLUMNS:
        agreement = (frame[column] == frame[filter_column]).mean()
        report.value(f"Filas con {column} == {filter_column}", f"{agreement * 100:.1f}%")

    sizes = frame.groupby("query_id").size()
    with_size = frame.assign(products_in_query=frame["query_id"].map(sizes))
    products_per_query = rate_by_level(with_size, "products_in_query").sort_index()
    report.show(products_per_query, caption="Tasa de compra segun cuantos productos trajo la busqueda:")
    bar_by_level(
        products_per_query,
        title="BTR segun cuantos productos devolvio la busqueda",
        path=figures / "06-busqueda-productos-por-query.png",
        target=frame["bought"].to_numpy(),
    )

    purchases = frame.groupby("query_id")["bought"].sum()
    purchases_per_query = purchases.value_counts().sort_index().rename("busquedas").to_frame()
    purchases_per_query.index.name = "compras"
    print()
    report.value("Busquedas", len(sizes))
    report.value(
        "Productos por busqueda",
        f"{sizes.min()} a {sizes.max()}, mediana {int(sizes.median())}",
    )
    print("\nBusquedas por cantidad de compras:")
    print(purchases_per_query.to_string())

    stamps = pd.to_datetime(frame["timestamp"])
    month = rate_by_level(frame.assign(month=stamps.dt.strftime("%Y-%m")), "month").sort_index()
    report.show(month, caption="Tasa de compra por mes:")
    bar_by_level(
        month,
        title="BTR por mes",
        path=figures / "06-busqueda-mes.png",
        target=frame["bought"].to_numpy(),
    )

    pooled = rate_by_level(frame, "month_of_year").sort_index()
    pooled["anios"] = frame.groupby("month_of_year")["timestamp"].apply(
        lambda s: pd.to_datetime(s).dt.year.nunique()
    )
    report.show(
        pooled[["rows", "rate"]],
        caption="Tasa de compra por mes del anio (los tres anios juntos):",
    )
    print("\nAnios que aporta cada mes:")
    print(pooled["anios"].to_string())
    bar_by_level(
        pooled[["rows", "rate"]],
        title="BTR por mes del anio (2024, 2025 y 2026 juntos)",
        path=figures / "06-busqueda-mes-del-anio.png",
        target=frame["bought"].to_numpy(),
    )

    weekday = rate_by_level(frame, "day_of_week").sort_index()
    report.show(weekday, caption="Tasa de compra por dia de la semana:")
    bar_by_level(
        weekday,
        title="BTR por dia de la semana",
        path=figures / "06-busqueda-dia-de-semana.png",
        target=frame["bought"].to_numpy(),
    )

    hour = rate_by_level(frame, "hour").sort_index()
    report.show(hour, caption="Tasa de compra por hora del dia:")
    bar_by_level(
        hour,
        title="BTR por hora del dia",
        path=figures / "06-busqueda-hora.png",
        target=frame["bought"].to_numpy(),
    )

    day = rate_by_level(frame, "day_of_month").sort_index()
    report.show(day, caption="Tasa de compra por dia del mes:")

    year = rate_by_level(frame.assign(year=stamps.dt.year.astype(str)), "year").sort_index()
    report.show(year, caption="Tasa de compra por anio:")
    print()
    report.value("Rango de fechas", f"{stamps.min():%Y-%m-%d} a {stamps.max():%Y-%m-%d}")

    return {
        "products_per_query": products_per_query,
        "purchases_per_query": purchases_per_query,
        "month": month,
        "month_of_year": pooled,
        "day_of_week": weekday,
        "day_of_month": day,
        "hour": hour,
        "year": year,
    }
