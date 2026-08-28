"""Aspect 0: the target column, and ``cart`` beside it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_level
from src.eda.rates import overall_rate, rate_by_level


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Measure the global rate, cross ``cart`` against ``bought``, chart both."""
    report.heading("Aspecto 0 - El target y la fuga")

    print()
    report.value("Filas (impresiones)", len(frame))
    report.value("Compras", int(frame["bought"].sum()))
    report.value("BTR global", f"{overall_rate(frame) * 100:.1f}%")

    crossed = pd.crosstab(frame["cart"], frame["bought"])
    print("\nTabla cruzada cart x bought:")
    print(crossed.to_string())

    cart = rate_by_level(frame, "cart")
    report.show(cart, caption="Tasa de compra segun cart:")
    bar_by_level(
        cart,
        title="BTR segun si el producto llego al carrito",
        path=figures / "00-target-cart.png",
        target=frame["bought"].to_numpy(),
    )

    return {"cart": cart, "crosstab": crossed}
