"""How much a column separates when it has no relationship with the target.

Shuffling ``bought`` across the same groups keeps their sizes and the overall
rate but breaks any link to the column, so what separation is left is chance.
Repeating that under many seeds gives the floor a mean and a deviation of its
own, and a column is only decided when it falls outside ``media +/- desvio``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

REPS = 2000
"""Shuffles per seed."""

SEEDS = 100

PERCENTILE = 95
"""Percentile of one seed's shuffled separations taken as that seed's floor."""


@dataclass(frozen=True)
class Floor:
    """The separation chance reaches, and how much that number itself moves."""

    mean: float
    deviation: float

    @property
    def low(self) -> float:
        return self.mean - self.deviation

    @property
    def high(self) -> float:
        return self.mean + self.deviation

    def verdict(self, separation: float) -> str:
        """``si`` above the range, ``no`` below it, ``?`` inside it."""
        if separation > self.high:
            return "si"
        if separation < self.low:
            return "no"
        return "?"


def _rates(sizes: tuple[int, ...], positives: int, total: int, reps: int, seed: int) -> np.ndarray:
    """Buy rate of every group under ``reps`` shuffles, as ``reps`` by groups.

    Dealing shuffled rows into fixed-size groups draws each group's purchases
    without replacement from what the previous groups left, so a chain of
    hypergeometric draws gives the same distribution without ever building a
    permutation.
    """
    rng = np.random.default_rng(seed)
    good = np.full(reps, positives)
    bad = np.full(reps, total - positives)

    out = np.empty((reps, len(sizes)))
    for column, size in enumerate(sizes):
        drawn = rng.hypergeometric(good, bad, size)
        out[:, column] = drawn / size
        good -= drawn
        bad -= size - drawn
    return out


@lru_cache(maxsize=None)
def _separations(sizes: tuple[int, ...], positives: int, total: int, reps: int, seed: int) -> np.ndarray:
    rates = _rates(sizes, positives, total, reps, seed)
    return (rates.max(axis=1) - rates.min(axis=1)) * 100


def _shape(sizes: Sequence[int], target: np.ndarray) -> tuple[tuple[int, ...], int, int]:
    """The group sizes and the pool they are drawn from."""
    values = np.asarray(target)
    counts = tuple(int(size) for size in sizes)
    if sum(counts) > values.shape[0]:
        raise ValueError(
            f"los grupos suman {sum(counts)} filas y el target tiene {values.shape[0]}: "
            "una fila contada en varios grupos no se puede barajar"
        )
    return counts, int(values.sum()), int(values.shape[0])


def separations(
    sizes: Sequence[int],
    target: np.ndarray,
    *,
    reps: int = REPS,
    seed: int = 0,
) -> np.ndarray:
    """Separation in percentage points produced by each shuffle of one seed."""
    counts, positives, total = _shape(sizes, target)
    return _separations(counts, positives, total, reps, seed)


def floor(
    sizes: Sequence[int],
    target: np.ndarray,
    *,
    seeds: int = SEEDS,
    reps: int = REPS,
    percentile: int = PERCENTILE,
) -> Floor:
    """Mean and deviation of the per-seed floors."""
    counts, positives, total = _shape(sizes, target)
    per_seed = np.array(
        [
            np.percentile(_separations(counts, positives, total, reps, seed), percentile)
            for seed in range(seeds)
        ]
    )
    return Floor(float(per_seed.mean()), float(per_seed.std()))


def exceedance(
    observed: float,
    sizes: Sequence[int],
    target: np.ndarray,
    *,
    seeds: int = SEEDS,
    reps: int = REPS,
) -> float:
    """Percent of all shuffles, across every seed, separating at least as much."""
    counts, positives, total = _shape(sizes, target)
    shares = [
        (_separations(counts, positives, total, reps, seed) >= observed).mean()
        for seed in range(seeds)
    ]
    return float(np.mean(shares)) * 100


def level_bands(
    sizes: Sequence[int],
    target: np.ndarray,
    *,
    percentile: int = PERCENTILE,
    reps: int = REPS,
    seed: int = 0,
) -> np.ndarray:
    """Low and high buy rate each group reaches by chance, one ``[lo, hi]`` per group."""
    counts, positives, total = _shape(sizes, target)
    rates = _rates(counts, positives, total, reps, seed) * 100
    tail = (100 - percentile) / 2
    return np.percentile(rates, [tail, 100 - tail], axis=0).T
