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
from src.eda.plots import bar_separation_vs_floor
from src.eda.rates import SMALL_GROUP, rate_by_bucket, rate_by_level

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
    """One column, how it is grouped, and how it stands against its own floor."""

    column: str
    grouping: str
    groups: int
    smallest: int
    separation: float
    floor: noise.Floor
    exceedance: float
    """Percent of shuffles separating at least as much as this column does."""

    @property
    def verdict(self) -> str:
        return self.floor.verdict(self.separation)


def _measure(
    table: pd.DataFrame, column: str, grouping: str, target: np.ndarray
) -> Measurement:
    large = table[table["rows"] >= SMALL_GROUP]
    separation = float(large["rate"].max() - large["rate"].min()) * 100
    sizes = large["rows"].to_numpy()
    return Measurement(
        column=column,
        grouping=grouping,
        groups=len(table),
        smallest=int(table["rows"].min()),
        separation=separation,
        floor=noise.floor(sizes, target),
        exceedance=noise.exceedance(separation, sizes, target),
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


def measurements(frame: pd.DataFrame) -> list[Measurement]:
    """Measure every column, widest separation first."""
    enriched = derived(frame)
    target = enriched["bought"].to_numpy()

    results = [
        _measure(rate_by_level(enriched, column), column, "nivel", target)
        for column in LEVEL_COLUMNS
    ]
    results += [
        _measure(rate_by_bucket(enriched, column, buckets=BUCKETS), column, "decil", target)
        for column in BUCKET_COLUMNS
    ]

    # Aspect 5 keeps the sentinel out of the buckets; the ranking reads the column
    # the same way it is read there, so the two tables cannot disagree.
    scored = enriched[enriched["nutrition_score"] != NUTRITION_SENTINEL]
    results.append(
        _measure(
            rate_by_bucket(scored, "nutrition_score", buckets=NUTRITION_BUCKETS),
            "nutrition_score",
            "tramo",
            scored["bought"].to_numpy(),
        )
    )

    return sorted(results, key=lambda m: m.separation, reverse=True)


def table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "columna": m.column,
            "origen": "construida" if m.column in CONSTRUCTED else "del dataset",
            "agrupada por": m.grouping,
            "grupos": m.groups,
            "grupo mas chico": m.smallest,
            "separacion": round(m.separation, 1),
            "piso": round(m.floor.mean, 2),
            "desvio": round(m.floor.deviation, 2),
            "piso_lo": m.floor.low,
            "piso_hi": m.floor.high,
            "% del azar que la supera": round(m.exceedance, 1),
            "supera": m.verdict,
        }
        for m in measurements(frame)
    ]
    return pd.DataFrame(rows).set_index("columna")


SHOWN = [
    "origen",
    "agrupada por",
    "grupos",
    "grupo mas chico",
    "separacion",
    "piso",
    "desvio",
    "% del azar que la supera",
    "supera",
]
"""Columns printed to the console; ``piso_lo``/``piso_hi`` only feed the chart."""


def markdown(frame: pd.DataFrame) -> str:
    """The same ranking as a Markdown table, for pasting into the write-up."""
    header = (
        "| Columna | Agrupada por | Grupos | Grupo más chico | Separación |"
        " Piso de ruido | % del azar que la supera | ¿Supera? |\n"
        "|---|---|---:|---:|---:|---:|---:|---|"
    )
    lines = [
        f"| {_label(m.column)} | {m.grouping} | {m.groups} | {m.smallest:,} |"
        f" {m.separation:.1f} pp | {m.floor.mean:.2f} ± {m.floor.deviation:.2f} pp |"
        f" {m.exceedance:.0f}% |"
        f" {'**sí**' if m.verdict == 'si' else m.verdict} |".replace(",", ".")
        for m in measurements(frame)
    ]
    return "\n".join([header, *lines])


def analyse(frame: pd.DataFrame, figures: Path) -> pd.DataFrame:
    report.heading("Ranking - Separacion de cada columna contra su piso de ruido")

    ranking = table(frame)
    print(
        "\nPiso = separacion que alcanza una columna de la misma forma (mismos"
        "\ngrupos, mismos tamanos) al barajar 'bought'. Cada una de las"
        f"\n{noise.SEEDS} semillas da su percentil {noise.PERCENTILE} sobre"
        f" {noise.REPS} barajadas;"
        "\nel piso es la media de esas 100 y el desvio, cuanto se mueven entre si."
        "\nSupera: 'si' arriba de piso+desvio, 'no' abajo de piso-desvio,"
        "\n'?' adentro del rango, donde la simulacion no alcanza para decidir."
    )
    print(f"\n{ranking[SHOWN].to_string()}")
    bar_separation_vs_floor(
        ranking,
        title="Separacion de cada columna contra el piso que alcanza por azar",
        path=figures / "07-ranking-piso-de-ruido.png",
    )
    return ranking


if __name__ == "__main__":
    from src.eda.loading import load_dataset

    print(markdown(load_dataset()))
