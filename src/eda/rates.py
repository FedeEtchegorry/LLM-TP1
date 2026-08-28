"""The one place a buy rate is computed, so no two numbers can disagree."""

from __future__ import annotations

import pandas as pd

TARGET = "bought"
"""One row is one impression; its rate over a group is that group's BTR."""

SMALL_GROUP = 100
"""Groups under this many rows are set aside: their rate moves too much to read."""


def overall_rate(frame: pd.DataFrame) -> float:
    return float(frame[TARGET].mean())


def rate_by_level(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows and buy rate per level of ``column``, most bought first.

    Nulls stay as their own level instead of being dropped.
    """
    grouped = frame.groupby(column, dropna=False)[TARGET].agg(rows="size", rate="mean")
    return grouped.sort_values("rate", ascending=False)


MAX_DECIMALS = 3
"""Cap for a derived column whose ratio has no natural precision."""


def decimals(values: pd.Series) -> int:
    """How many decimal places the column actually carries, capped."""
    text = values.dropna().round(6).map(lambda value: f"{value:.6f}".rstrip("0"))
    places = text.str.split(".").str[1].str.len().max()
    return min(int(places or 0), MAX_DECIMALS)


def _label(interval: pd.Interval, places: int, lowest: float | None = None) -> str:
    """A bucket's edges, written with the precision of the column they cut.

    ``pd.qcut`` pushes the first left edge below the minimum; ``lowest`` puts the
    real minimum back, as a closed bracket.
    """
    if lowest is not None:
        return f"[{lowest:.{places}f}, {interval.right:.{places}f}]"
    return f"({interval.left:.{places}f}, {interval.right:.{places}f}]"


def rate_by_bucket(frame: pd.DataFrame, column: str, *, buckets: int = 10) -> pd.DataFrame:
    """Rows and buy rate per quantile bucket, kept in bucket order.

    Nulls are dropped, unlike ``rate_by_level``: they fall outside every interval
    ``pd.qcut`` produces. And ``buckets`` is a request, not a guarantee — under
    ``duplicates="drop"`` repeated values collapse edges, so fewer and uneven
    buckets can come back.
    """
    edges = pd.qcut(frame[column], buckets, duplicates="drop")
    grouped = frame.groupby(edges, observed=True)[TARGET].agg(rows="size", rate="mean")
    places = decimals(frame[column])
    lowest = float(frame[column].min())
    grouped.index = [
        _label(interval, places, lowest if position == 0 else None)
        for position, interval in enumerate(grouped.index)
    ]
    return grouped
