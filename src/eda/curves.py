"""Binned response curves shared by the tables and the figures."""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.eda.dataset import BtrData

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class BinnedResponse:
    """Purchase rate against a numeric column, in equal-count quantile bins."""
    edges: np.ndarray
    """Bin boundaries, length ``n_bins + 1``."""
    centers: np.ndarray
    """Mean ``price_pct`` inside each bin -- where the marker is drawn."""
    rates: np.ndarray
    """Share of rows in the bin that were bought."""
    counts: np.ndarray
    """Rows per bin."""
    baseline: float
    """Purchase rate over all selected rows, drawn as the reference line."""
    def confidence_band(self, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        """Normal-approximation interval around each bin's rate."""
        half = z * np.sqrt(self.rates * (1.0 - self.rates) / self.counts)
        return self.rates - half, self.rates + half


PriceCurve = BinnedResponse
"""Historical name for :class:`BinnedResponse`, kept for existing imports."""


def binned_response(
    values: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> BinnedResponse:
    """Split ``values`` into equal-count bins and take the buy rate in each."""
    finite = np.isfinite(values)
    values, target = values[finite], target[finite].astype(np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9

    centers, rates, counts = [], [], []
    for index in range(n_bins):
        selected = (values >= edges[index]) & (values < edges[index + 1])
        rows = int(selected.sum())
        if rows == 0:
            continue
        centers.append(float(values[selected].mean()))
        rates.append(float(target[selected].mean()))
        counts.append(rows)

    return BinnedResponse(
        edges=edges,
        centers=np.array(centers),
        rates=np.array(rates),
        counts=np.array(counts),
        baseline=float(target.mean()),
    )


def price_pct_curve(data: BtrData, tier: str = "A", n_bins: int = 10) -> BinnedResponse:
    """Bin ``price_pct`` within one popularity tier and take the buy rate per bin."""
    mask = np.array([value == tier for value in data.oracle_tier])
    return binned_response(data.numeric["price_pct"][mask], data.target[mask], n_bins)


def numeric_curve(
    data: BtrData, field: str, *, tier: str | None = None, n_bins: int = 10
) -> BinnedResponse:
    """The same treatment for any numeric column, optionally within one tier."""
    if tier is None:
        return binned_response(data.numeric[field], data.target, n_bins)
    mask = np.array([value == tier for value in data.oracle_tier])
    return binned_response(data.numeric[field][mask], data.target[mask], n_bins)


@dataclass(frozen=True)
class LevelRates:
    """Purchase rate for every level of a categorical column, with intervals."""
    labels: tuple[str, ...]
    counts: np.ndarray
    bought: np.ndarray
    rates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    baseline: float
    """Purchase rate over all rows -- the line each level is compared against."""
    @property
    def spans_baseline(self) -> np.ndarray:
        """True where the level's interval contains the overall rate."""
        return (self.lower <= self.baseline) & (self.baseline <= self.upper)


def level_rates(
    labels: Sequence[str],
    target: np.ndarray,
    *,
    sort_by_rate: bool = True,
    min_rows: int = 1,
) -> LevelRates:
    """Buy rate per level with a Wilson interval."""
    grouped: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for label, value in zip(labels, target, strict=True):
        grouped[label][0] += int(value)
        grouped[label][1] += 1

    items = [(label, pair) for label, pair in grouped.items() if pair[1] >= min_rows]
    if sort_by_rate:
        items.sort(key=lambda item: -item[1][0] / item[1][1])
    else:
        items.sort(key=lambda item: item[0])

    bought = np.array([pair[0] for _, pair in items], dtype=np.float64)
    counts = np.array([pair[1] for _, pair in items], dtype=np.float64)
    lower, upper = wilson_interval(bought, counts)
    return LevelRates(
        labels=tuple(label for label, _ in items),
        counts=counts,
        bought=bought,
        rates=bought / counts,
        lower=lower,
        upper=upper,
        baseline=float(target.mean()),
    )


def wilson_interval(
    successes: np.ndarray, total: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion."""
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half = (
        z
        * np.sqrt(rate * (1.0 - rate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def quarter_rates(data: BtrData) -> LevelRates:
    """Buy rate per calendar quarter, in chronological order."""
    quarters = tuple(
        f"{moment.year}Q{(moment.month - 1) // 3 + 1}"
        for moment in (
            datetime.strptime(stamp, TIMESTAMP_FORMAT) for stamp in data.timestamps
        )
    )
    return level_rates(quarters, data.target, sort_by_rate=False)


def query_spans(data: BtrData) -> np.ndarray:
    """Days between the earliest and latest event inside each query."""
    by_query: dict[str, list[datetime]] = collections.defaultdict(list)
    for query, stamp in zip(data.query_ids, data.timestamps, strict=True):
        by_query[query].append(datetime.strptime(stamp, TIMESTAMP_FORMAT))
    return np.array(
        [
            (max(moments) - min(moments)).days
            for moments in by_query.values()
            if len(moments) > 1
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class AdditivitySeries:
    """Purchases per query against the number of tier-A products it shows."""
    tier_a_counts: np.ndarray
    queries: np.ndarray
    purchases: np.ndarray
    slope: float
    """Purchase rate of a tier-A row: the per-product prediction if nothing competes."""
    @property
    def means(self) -> np.ndarray:
        return self.purchases / self.queries

    @property
    def total_queries(self) -> int:
        return int(self.queries.sum())


def additivity_series(data: BtrData) -> AdditivitySeries:
    """Group queries by their tier-A count and total the purchases in each group."""
    by_query: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for query, tier, target in zip(
        data.query_ids, data.oracle_tier, data.target, strict=True
    ):
        by_query[query][0] += int(tier == "A")
        by_query[query][1] += int(target)

    grouped: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for tier_a_count, purchases in by_query.values():
        grouped[tier_a_count][0] += purchases
        grouped[tier_a_count][1] += 1

    keys = sorted(grouped)
    tier_a_rows = np.array([tier == "A" for tier in data.oracle_tier])
    return AdditivitySeries(
        tier_a_counts=np.array(keys),
        queries=np.array([grouped[key][1] for key in keys]),
        purchases=np.array([grouped[key][0] for key in keys]),
        slope=float(data.target[tier_a_rows].mean()),
    )
