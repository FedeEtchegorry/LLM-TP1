"""Aspect 1: the title parenthetical and the closing line of the description."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_level
from src.eda.rates import rate_by_level

HIGH_RATE = 0.5
"""Rate above which a phrase joins the top block; between it and 0 is the middle."""


def _tier(rate: float) -> str:
    if rate > HIGH_RATE:
        return "alta"
    if rate > 0.0:
        return "baja"
    return "nula"


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

    tiers = frame["popularity_phrase"].map(phrase["rate"]).map(_tier)
    tier = rate_by_level(frame.assign(tier=tiers), "tier")
    distinct_tiers = (pd.crosstab(frame["description_closing"], tiers) > 0).sum(axis=1)

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

    return {"phrase": phrase, "closing": closing, "tier": tier}
