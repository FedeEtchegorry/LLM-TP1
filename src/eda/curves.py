"""Binned response curves shared by the tables and the figures.

:mod:`src.eda.run_structure` prints these as Markdown and :mod:`src.eda.figures`
draws them as SVG. Both read the functions here, so a table and the chart beside
it cannot drift apart -- which is what went wrong when the figures were generated
from numbers pasted into a throwaway script.

These are descriptive summaries over the whole dataset, not model inputs. They
are computed once, for the write-up. Nothing here is fitted, and nothing here
feeds a model; anything estimated for a model lives in :mod:`src.eda.features`
and is fitted per fold.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from src.eda.dataset import BtrData


@dataclass(frozen=True)
class PriceCurve:
    """Purchase rate against ``price_pct``, in equal-count quantile bins."""

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


def price_pct_curve(data: BtrData, tier: str = "A", n_bins: int = 10) -> PriceCurve:
    """Bin ``price_pct`` within one popularity tier and take the buy rate per bin.

    Holding the tier fixed is the point: the tier dominates the purchase decision,
    so a curve over all rows would mostly plot the tier mix. Within tier A the
    remaining shape is the inverted U.
    """

    mask = np.array([value == tier for value in data.oracle_tier])
    values = data.numeric["price_pct"][mask]
    target = data.target[mask].astype(np.float64)

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

    return PriceCurve(
        edges=edges,
        centers=np.array(centers),
        rates=np.array(rates),
        counts=np.array(counts),
        baseline=float(target.mean()),
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
    """Group queries by their tier-A count and total the purchases in each group.

    Every query is represented, including the single query holding five tier-A
    products; dropping it would make the counts fail to sum to 2,012.
    """

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
