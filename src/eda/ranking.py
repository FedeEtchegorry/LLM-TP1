"""Every column on one scale: how far apart its buy rates sit.

Each aspect reads its own columns and stops there. This module measures all of
them the same way and against the same yardstick — the separation their own
shape reaches when ``bought`` is shuffled — so they can be compared in one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda import noise, report
from src.eda.aspects.composition import NO_ALLERGENS, NUTRITION_SENTINEL
from src.eda.plots import bar_separation_with_pvalues
from src.eda.rates import rate_by_bucket, rate_by_level

BUCKETS = 10
"""Quantile buckets used for a numeric column, unless its aspect uses fewer."""

NUTRITION_BUCKETS = 8
"""``nutrition_score`` keeps the bucket count aspect 5 reads it with."""

SOURCES: dict[str, str] = {
    "popularity_phrase": "title",
    "description_closing": "description",
    "price_position": "price + filter_price_*",
    "volume_in3": "dimensions_in",
    "year": "timestamp",
    "month_of_year": "timestamp",
    "day_of_week": "timestamp",
    "day_of_month": "timestamp",
    "hour": "timestamp",
    "products_in_query": "query_id",
}

CONSTRUCTED: frozenset[str] = frozenset(
    {
        "price_position",
        "products_in_query",
        "year",
        "month_of_year",
        "day_of_week",
        "day_of_month",
        "hour",
    }
)
"""Columns the analysis built: a combination of fields, or a scale it chose.

Everything else is a field read as it comes, including the ones that need
parsing — a title's parenthetical is what that field says, not a new variable.
"""

LEVEL_COLUMNS: tuple[str, ...] = (
    "popularity_phrase",
    "description_closing",
    "category",
    "allergens",
    "country_of_origin",
    "brand",
    "unit_of_measure",
    "products_in_query",
    "storage_type",
    "month_of_year",
    "day_of_week",
    "day_of_month",
    "hour",
    "year",
)

BUCKET_COLUMNS: tuple[str, ...] = (
    "price_position",
    "price",
    "net_weight_oz",
    "volume_in3",
)


@dataclass(frozen=True)
class Measurement:
    """One column measured against row-wise and query-wise nulls."""

    column: str
    grouping: str
    groups: int
    eligible_groups: int
    excluded_rows: int
    smallest: int
    separation: float
    row_result: noise.PermutationResult
    query_result: noise.PermutationResult


def _measure(
    table: pd.DataFrame,
    column: str,
    grouping: str,
    target: np.ndarray,
    groups: list[np.ndarray],
    query_ids: np.ndarray,
    query_permutations: np.ndarray,
    *,
    reps: int,
    seed: int,
    query_result: noise.PermutationResult | None = None,
) -> Measurement:
    separation = float(table["rate"].max() - table["rate"].min()) * 100
    sizes = table["rows"].to_numpy()
    return Measurement(
        column=column,
        grouping=grouping,
        groups=len(table),
        eligible_groups=len(table),
        excluded_rows=0,
        smallest=int(table["rows"].min()),
        separation=separation,
        row_result=noise.row_test(sizes, target, separation, reps=reps, seed=seed),
        query_result=query_result
        or noise.query_test(
            groups,
            target,
            query_ids,
            separation,
            reps=reps,
            seed=seed,
            permutations=query_permutations,
        ),
    )


def _level_groups(frame: pd.DataFrame, column: str, table: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of every observed level."""
    values = frame[column]
    groups = []
    for level in table.index:
        mask = values.isna() if pd.isna(level) else values.eq(level)
        groups.append(np.flatnonzero(mask.to_numpy()))
    return groups


def _bucket_groups(
    frame: pd.DataFrame, column: str, table: pd.DataFrame, *, buckets: int
) -> list[np.ndarray]:
    """Row positions of every observed quantile bucket."""
    codes = pd.qcut(frame[column], buckets, labels=False, duplicates="drop").to_numpy()
    return [np.flatnonzero(codes == code) for code in range(len(table))]


def _products_in_query_result(
    frame: pd.DataFrame, observed: float, *, reps: int, seed: int
) -> noise.PermutationResult:
    """Query-level test for a feature that is the query size itself."""
    per_query = frame.groupby("query_id")["bought"].agg(size="size", rate="mean")
    rows_per_level = per_query.groupby("size")["size"].sum()
    levels = rows_per_level.index
    sizes = per_query["size"].to_numpy()
    groups = [np.flatnonzero(sizes == level) for level in levels]
    return noise.query_rate_test(
        groups,
        per_query["rate"].to_numpy(),
        observed,
        reps=reps,
        seed=seed,
    )


def _label(column: str) -> str:
    source = SOURCES.get(column)
    return f"`{column}`" if source is None else f"`{column}` (de `{source}`)"


def derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the columns an aspect builds for itself, so all of them rank together."""
    stamps = pd.to_datetime(frame["timestamp"])
    return frame.assign(
        allergens=frame["allergens"].fillna(NO_ALLERGENS),
        products_in_query=frame["query_id"].map(frame.groupby("query_id").size()),
        year=stamps.dt.year.astype(str),
    )


def measurements(
    frame: pd.DataFrame, *, reps: int = noise.REPS, seed: int = 0
) -> list[Measurement]:
    """Measure every column under row and query-block permutations."""
    enriched = derived(frame)
    target = enriched["bought"].to_numpy()
    query_ids = enriched["query_id"].to_numpy()
    query_permutations = noise.query_permutations(target, query_ids, reps=reps, seed=seed)

    results = []
    for position, column in enumerate(LEVEL_COLUMNS):
        table = rate_by_level(enriched, column)
        observed = float(table["rate"].max() - table["rate"].min()) * 100
        query_result = (
            _products_in_query_result(
                enriched, observed, reps=reps, seed=seed + position + 1
            )
            if column == "products_in_query"
            else None
        )
        results.append(
            _measure(
                table,
                column,
                "nivel",
                target,
                _level_groups(enriched, column, table),
                query_ids,
                query_permutations,
                reps=reps,
                seed=seed + position + 1,
                query_result=query_result,
            )
        )

    for position, column in enumerate(BUCKET_COLUMNS, start=len(LEVEL_COLUMNS)):
        table = rate_by_bucket(enriched, column, buckets=BUCKETS)
        results.append(
            _measure(
                table,
                column,
                "decil",
                target,
                _bucket_groups(enriched, column, table, buckets=BUCKETS),
                query_ids,
                query_permutations,
                reps=reps,
                seed=seed + position + 1,
            )
        )

    del query_permutations

    # Aspect 5 keeps the sentinel out of the buckets; the ranking reads the column
    # the same way it is read there, so the two tables cannot disagree.
    scored = enriched[enriched["nutrition_score"] != NUTRITION_SENTINEL]
    scored_target = scored["bought"].to_numpy()
    scored_queries = scored["query_id"].to_numpy()
    scored_permutations = noise.query_permutations(
        scored_target, scored_queries, reps=reps, seed=seed + 10_000
    )
    nutrition = rate_by_bucket(scored, "nutrition_score", buckets=NUTRITION_BUCKETS)
    results.append(
        _measure(
            nutrition,
            "nutrition_score",
            "tramo",
            scored_target,
            _bucket_groups(
                scored, "nutrition_score", nutrition, buckets=NUTRITION_BUCKETS
            ),
            scored_queries,
            scored_permutations,
            reps=reps,
            seed=seed + len(LEVEL_COLUMNS) + len(BUCKET_COLUMNS) + 1,
        )
    )

    return sorted(results, key=lambda m: m.separation, reverse=True)


def table(
    frame: pd.DataFrame, *, reps: int = noise.REPS, seed: int = 0
) -> pd.DataFrame:
    measured = measurements(frame, reps=reps, seed=seed)
    rows = [
        {
            "columna": m.column,
            "origen": "construida" if m.column in CONSTRUCTED else "del dataset",
            "agrupada por": m.grouping,
            "grupos": m.groups,
            "grupos evaluados": m.eligible_groups,
            "filas excluidas": m.excluded_rows,
            "grupo mas chico": m.smallest,
            "separacion": round(m.separation, 1),
            "p95_filas": round(m.row_result.percentile, 2),
            "p_filas": m.row_result.p_value,
            "p95_query": round(m.query_result.percentile, 2),
            "p_query": m.query_result.p_value,
            "significativa_filas": m.row_result.significant,
            "significativa_query": m.query_result.significant,
        }
        for m in measured
    ]
    return pd.DataFrame(rows).set_index("columna")


SHOWN = [
    "origen",
    "agrupada por",
    "grupos",
    "grupos evaluados",
    "filas excluidas",
    "grupo mas chico",
    "separacion",
    "p95_filas",
    "p_filas",
    "p95_query",
    "p_query",
    "significativa_query",
]
"""Columns printed to the console."""


def markdown(
    frame: pd.DataFrame, *, reps: int = noise.REPS, seed: int = 0
) -> str:
    """The same ranking as a Markdown table, for pasting into the write-up."""
    ranking = table(frame, reps=reps, seed=seed)
    header = (
        "| Columna | Grupos | Separación | p95 filas | p filas |"
        " p95 query | p query |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [
        f"| {_label(column)} | {int(row['grupos evaluados'])} |"
        f" {row['separacion']:.1f} pp | {row['p95_filas']:.2f} |"
        f" {row['p_filas']:.4f} | {row['p95_query']:.2f} |"
        f" {row['p_query']:.4f} |".replace(",", ".")
        for column, row in ranking.iterrows()
    ]
    return "\n".join([header, *lines])


def analyse(
    frame: pd.DataFrame,
    figures: Path,
    *,
    reps: int = noise.REPS,
    seed: int = 0,
) -> pd.DataFrame:
    report.heading("Ranking - p-values por filas y por bloques de query_id")

    ranking = table(frame, reps=reps, seed=seed)
    print(
        "\np = (1 + permutaciones con separacion >= observada) / (1 + total)."
        f"\nSe usan {reps} permutaciones. p95 es una referencia visual;"
        f" la decision simple usa p <= {noise.ALPHA:.2f}."
        "\nFilas baraja compras entre impresiones. Query reasigna bloques completos"
        "\nentre query_id del mismo tamano y desordena posiciones dentro del bloque."
    )
    print(f"\n{ranking[SHOWN].to_string()}")
    bar_separation_with_pvalues(
        ranking,
        title="Separacion observada y percentil 95: filas contra query_id",
        path=figures / "07-ranking-pvalues.png",
    )
    return ranking


if __name__ == "__main__":
    from src.eda.loading import load_dataset

    print(markdown(load_dataset()))
