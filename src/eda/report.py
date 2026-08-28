"""Printing what the aspects compute, in a shape fit for the slides."""

from __future__ import annotations

import pandas as pd


def heading(text: str) -> None:
    print(f"\n{text}")
    print("=" * len(text))


def show(table: pd.DataFrame, *, caption: str) -> None:
    display = table.copy()
    display["rate"] = (display["rate"] * 100).round(1).astype(str) + "%"
    display = display.rename(columns={"rows": "filas", "rate": "% comprado"})
    print(f"\n{caption}")
    print(display.to_string())


def value(label: str, measurement: object) -> None:
    print(f"{label}: {measurement}")
