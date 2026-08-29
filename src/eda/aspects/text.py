"""Aspect 1: the title parenthetical and the closing line of the description."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_level
from src.eda.rates import rate_by_level

HIGH_RATE = 0.5
"""Rate above which a phrase joins the top block; between it and 0 is the middle."""

TOP_TIER = "alta"
"""Name of the block whose phrases buy more than half the time."""

WORD = r"[a-z0-9]+"
"""A word of the listing: lowercase letters and digits, everything else a separator."""

TEXT_FIELDS: tuple[str, ...] = ("title", "description")
"""The two text columns, each measured on its own."""


def _tier(rate: float) -> str:
    if rate > HIGH_RATE:
        return TOP_TIER
    if rate > 0.0:
        return "baja"
    return "nula"


def tiers(frame: pd.DataFrame) -> pd.Series:
    """The block of each row, read from the buy rate of its own phrase."""
    rates = rate_by_level(frame, "popularity_phrase")["rate"]
    return frame["popularity_phrase"].map(rates).map(_tier)


def words(frame: pd.DataFrame, column: str) -> pd.Series:
    """The words of one text column, one list per row."""
    return frame[column].str.lower().str.findall(WORD)


def vocabulary(frame: pd.DataFrame, column: str) -> pd.Series:
    """How many rows each word of ``column`` appears in, most widespread first."""
    return words(frame, column).map(set).explode().value_counts()


def shape(frame: pd.DataFrame) -> pd.DataFrame:
    """Each text column as a sequence: words to embed, and how long a row runs."""
    rows = []
    for column in TEXT_FIELDS:
        frequency = vocabulary(frame, column)
        lengths = words(frame, column).str.len()
        rows.append(
            {
                "columna": column,
                "palabras distintas": len(frequency),
                "palabras por fila": f"{int(lengths.median())} (max {int(lengths.max())})",
                "en todas las filas": int((frequency == len(frame)).sum()),
                "en una sola fila": int((frequency == 1).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("columna")


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Rate the two text columns by level, then by the blocks the phrases form."""
    report.heading("Aspecto 1 - El texto del listing (title + description)")

    phrase = rate_by_level(frame, "popularity_phrase")
    report.show(phrase, caption="Tasa de compra por frase final del titulo:")
    bar_by_level(
        phrase,
        title="BTR por la frase entre parentesis del titulo",
        path=figures / "01-texto-frase-titulo.png",
        target=frame["bought"].to_numpy(),
    )

    closing = rate_by_level(frame, "description_closing")
    report.show(closing, caption="Tasa de compra por oracion final de la descripcion:")
    bar_by_level(
        closing,
        title="BTR por la oracion final de la descripcion",
        path=figures / "01-texto-cierre-descripcion.png",
        target=frame["bought"].to_numpy(),
    )

    crossed = pd.crosstab(frame["description_closing"], frame["popularity_phrase"])
    distinct_phrases = (crossed > 0).sum(axis=1)

    blocks = tiers(frame)
    tier = rate_by_level(frame.assign(tier=blocks), "tier")
    distinct_tiers = (pd.crosstab(frame["description_closing"], blocks) > 0).sum(axis=1)

    print()
    report.value("Titulos con parentesis", f"{(frame['popularity_phrase'] != '(sin frase)').mean() * 100:.1f}%")
    report.value("Frases distintas por cierre de descripcion", f"{distinct_phrases.min()} a {distinct_phrases.max()}")
    report.value("Bloques distintos por cierre de descripcion", f"{distinct_tiers.min()} a {distinct_tiers.max()}")

    report.show(tier, caption="Tasa de compra por bloque de popularidad:")
    bar_by_level(
        tier,
        title="BTR por bloque de popularidad",
        path=figures / "01-texto-bloques.png",
        target=frame["bought"].to_numpy(),
    )

    sequences = shape(frame)
    print("\nForma de cada columna de texto como secuencia:")
    print(sequences.to_string())

    vocabularies = [set(vocabulary(frame, column).index) for column in TEXT_FIELDS]
    shared = vocabularies[0] & vocabularies[1]
    print()
    report.value("Palabras compartidas por title y description", len(shared))
    report.value(
        "Union de los dos vocabularios",
        len(vocabularies[0] | vocabularies[1]),
    )

    return {"phrase": phrase, "closing": closing, "tier": tier, "sequences": sequences}
