"""Aspect 6: the search the product appeared in, and when."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.aspects.text import TOP_TIER, tiers
from src.eda.plots import bar_by_level
from src.eda.rates import rate_by_level

MATCHED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("category", "filter_category"),
    ("storage_type", "filter_storage_type"),
)

FILTER_COLUMNS: tuple[str, ...] = (
    "filter_category",
    "filter_storage_type",
    "filter_price_min",
    "filter_price_max",
)
"""What the search asked for, as opposed to what the product is."""

CATEGORICAL_FILTERS: tuple[str, ...] = ("filter_category", "filter_storage_type")
"""The two of the four that name a level rather than a range."""

SECONDS_PER_DAY = 24 * 60 * 60


def query_spans(stamps: pd.Series, query_ids: pd.Series) -> pd.Series:
    """Days between the first and the last impression of each query."""
    moments = stamps.groupby(query_ids)
    return (moments.max() - moments.min()).dt.total_seconds() / SECONDS_PER_DAY


def query_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """The filter each query ran, one row per query."""
    return frame.groupby("query_id")[list(FILTER_COLUMNS)].first()


def filters_hold(frame: pd.DataFrame) -> bool:
    """Whether every filter column takes a single value inside each query."""
    return all(
        frame.groupby("query_id")[column].nunique().eq(1).all()
        for column in FILTER_COLUMNS
    )


def filter_reuse(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct filter combination, with the queries that ran it."""
    per_query = query_filters(frame)
    grouped = per_query.assign(filas=frame.groupby("query_id").size()).groupby(
        list(FILTER_COLUMNS), dropna=False
    )
    table = grouped.agg(busquedas=("filas", "size"), filas=("filas", "sum"))
    table["query_id"] = grouped.apply(
        lambda part: ", ".join(sorted(part.index)), include_groups=False
    )
    return table


def queries_per_filter(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """How many searches ran each distinct combination of ``columns``."""
    return query_filters(frame).groupby(list(columns), dropna=False).size()


def reuse_counts(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Combinations of ``columns`` run by exactly one search, by two, by three ..."""
    counts = queries_per_filter(frame, columns)
    table = counts.value_counts().sort_index().rename("combinaciones").to_frame()
    table["busquedas"] = table.index * table["combinaciones"]
    table.index.name = "busquedas que la corren"
    return table


def rivals(frame: pd.DataFrame, blocks: pd.Series) -> pd.Series:
    """How many *other* top-block products each row shares its query with."""
    top = (blocks == TOP_TIER).astype(int)
    return top.groupby(frame["query_id"]).transform("sum") - top


def competition(frame: pd.DataFrame, blocks: pd.Series) -> pd.DataFrame:
    """Buy rate of a top-block row against the top-block rivals it was shown with."""
    top = blocks == TOP_TIER
    grouped = frame[top].groupby(rivals(frame, blocks)[top])
    table = grouped["bought"].agg(rows="size", rate="mean")
    table.index.name = "rivales del bloque alto"
    return table


def additivity(frame: pd.DataFrame, blocks: pd.Series) -> pd.DataFrame:
    """Purchases a query produces against how many top-block products it holds."""
    per_query = frame.assign(top=(blocks == TOP_TIER).astype(int)).groupby("query_id").agg(
        purchases=("bought", "sum"), top=("top", "sum")
    )
    table = per_query.groupby("top")["purchases"].agg(
        busquedas="size", compras="sum", media="mean"
    )
    table.index.name = "productos del bloque alto"
    return table


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Compare each filter with its product column, then rate the query and the date."""
    report.heading("Aspecto 6 - El contexto de la busqueda")

    print()
    for column, filter_column in MATCHED_COLUMNS:
        agreement = (frame[column] == frame[filter_column]).mean()
        report.value(f"Filas con {column} == {filter_column}", f"{agreement * 100:.1f}%")

    reuse = filter_reuse(frame)
    report.value(
        "Filtros constantes dentro de cada busqueda",
        "si" if filters_hold(frame) else "no",
    )
    shared = reuse[reuse["busquedas"] > 1]
    report.value(
        "Combinaciones de filtro distintas",
        f"{len(reuse)} para {int(reuse['busquedas'].sum())} busquedas",
    )
    report.value(
        "Combinaciones que corrio mas de una busqueda",
        f"{len(shared)}, con {int(shared['busquedas'].sum())} busquedas"
        f" y {int(shared['filas'].sum())} filas",
    )
    counts = reuse_counts(frame, FILTER_COLUMNS)
    print("\nBusquedas que corren la misma combinacion de los cuatro campos:")
    print(counts.to_string())

    per_pair = queries_per_filter(frame, CATEGORICAL_FILTERS)
    report.value(
        "Busquedas por combinacion de los dos filtros categoricos",
        f"{len(per_pair)} combinaciones, de {per_pair.min()} a {per_pair.max()}"
        f" busquedas cada una (mediana {int(per_pair.median())})",
    )

    print("\nCombinaciones de filtro que corrio mas de una busqueda:")
    print(shared.to_string())

    stamps = pd.to_datetime(frame["timestamp"])

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

    blocks = tiers(frame)
    rate_by_rivals = competition(frame, blocks)
    report.show(
        rate_by_rivals,
        caption="Tasa de compra de un producto del bloque alto, segun cuantos otros"
        " del mismo bloque trajo su busqueda:",
    )
    bar_by_level(
        rate_by_rivals,
        title="BTR del bloque alto segun cuantos rivales del mismo bloque hubo",
        path=figures / "06-busqueda-competencia.png",
        target=frame.loc[blocks == TOP_TIER, "bought"].to_numpy(),
    )

    purchases_by_top = additivity(frame, blocks)
    print("\nCompras de una busqueda segun cuantos productos del bloque alto contiene:")
    print(purchases_by_top.round(2).to_string())

    spans = query_spans(stamps, frame["query_id"])
    print()
    report.value(
        "Dias entre la primera y la ultima impresion de una busqueda",
        f"minimo {spans.min():.0f}, mediana {spans.median():.0f},"
        f" maximo {spans.max():.0f} (cuartiles {spans.quantile(0.25):.0f}"
        f" y {spans.quantile(0.75):.0f})",
    )
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
        "query_filters": reuse,
        "filter_reuse_counts": counts,
        "products_per_query": products_per_query,
        "purchases_per_query": purchases_per_query,
        "competition": rate_by_rivals,
        "additivity": purchases_by_top,
        "query_spans": spans.rename("dias").to_frame(),
        "month": month,
        "month_of_year": pooled,
        "day_of_week": weekday,
        "day_of_month": day,
        "hour": hour,
        "year": year,
    }
