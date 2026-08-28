"""Aspect 4: the packaging — its unit, its weight and its box."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import report
from src.eda.plots import bar_by_bucket, bar_by_level
from src.eda.rates import rate_by_bucket, rate_by_level

BUCKETS = 10


def analyse(frame: pd.DataFrame, figures: Path) -> dict[str, pd.DataFrame]:
    """Measure ``package_size`` against ``net_weight_oz``, then rate each size column."""
    report.heading("Aspecto 4 - El envase")

    countable = frame["unit_of_measure"] == "ct"
    ratio = (frame.loc[~countable, "net_weight_oz"] / frame.loc[~countable, "package_oz"]).describe()
    print()
    report.value("Filas contables ('ct'), sin peso convertible", int(countable.sum()))
    report.value(
        "Cociente net_weight_oz / package_size en onzas",
        f"min {ratio['min']:.2f}, mediana {ratio['50%']:.2f}, max {ratio['max']:.2f}",
    )
    report.value("Valores distintos de dimensions_in", frame["dimensions_in"].nunique())

    unit = rate_by_level(frame, "unit_of_measure")
    report.show(unit, caption="Tasa de compra por unidad de medida:")
    bar_by_level(
        unit,
        title="BTR por unidad de medida del envase",
        path=figures / "04-envase-unidad.png",
        target=frame["bought"].to_numpy(),
    )

    weight = rate_by_bucket(frame, "net_weight_oz", buckets=BUCKETS)
    report.show(weight, caption="Tasa de compra por decil de peso neto:")
    bar_by_bucket(
        weight,
        title="BTR por decil de peso neto",
        xlabel="Peso neto (oz)",
        path=figures / "04-envase-peso.png",
        target=frame["bought"].to_numpy(),
    )

    volume = rate_by_bucket(frame, "volume_in3", buckets=BUCKETS)
    report.show(volume, caption="Tasa de compra por decil de volumen del envase:")
    bar_by_bucket(
        volume,
        title="BTR por decil de volumen del envase",
        xlabel="Volumen (pulgadas cubicas)",
        path=figures / "04-envase-volumen.png",
        target=frame["bought"].to_numpy(),
    )

    return {"unit_of_measure": unit, "net_weight_oz": weight, "volume_in3": volume}
