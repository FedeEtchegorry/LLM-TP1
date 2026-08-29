"""Empirical p-values for the separation of buy rates between groups.

The statistic is always ``max(BTR) - min(BTR)``.  Two null distributions are
available: shuffling purchases between rows, and reassigning complete purchase
blocks between queries of the same size.  The second keeps the number of
purchases and the dependence structure of every query while breaking its link
to the feature being measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

REPS = 10_000
"""Permutations used by a reported empirical p-value."""

ALPHA = 0.05
"""Decision threshold for an empirical p-value."""

PERCENTILE = 95
"""Null percentile shown as a visual reference next to the p-value."""


@dataclass(frozen=True)
class PermutationResult:
    """The null reference and empirical p-value for one observed separation."""

    percentile: float
    p_value: float
    reps: int

    @property
    def significant(self) -> bool:
        return self.p_value <= ALPHA


def empirical_p(null: np.ndarray, observed: float) -> float:
    """Upper-tail empirical p-value, with the standard plus-one correction."""
    values = np.asarray(null, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("la distribucion nula debe ser un vector no vacio")
    return float((np.count_nonzero(values >= observed) + 1) / (values.size + 1))


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


def row_test(
    sizes: Sequence[int],
    target: np.ndarray,
    observed: float,
    *,
    reps: int = REPS,
    seed: int = 0,
    percentile: int = PERCENTILE,
) -> PermutationResult:
    """Test ``observed`` against purchases shuffled freely between rows."""
    null = separations(sizes, target, reps=reps, seed=seed)
    return PermutationResult(
        percentile=float(np.percentile(null, percentile)),
        p_value=empirical_p(null, observed),
        reps=reps,
    )


def _query_positions(query_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Row positions of queries, stacked by query size."""
    grouped: dict[object, list[int]] = {}
    for position, query_id in enumerate(np.asarray(query_ids).tolist()):
        grouped.setdefault(query_id, []).append(position)

    by_size: dict[int, list[np.ndarray]] = {}
    for positions in grouped.values():
        block = np.asarray(positions, dtype=np.int32)
        by_size.setdefault(len(block), []).append(block)
    return {size: np.stack(blocks) for size, blocks in by_size.items()}


def query_permutations(
    target: np.ndarray,
    query_ids: np.ndarray,
    *,
    reps: int = REPS,
    seed: int = 0,
) -> np.ndarray:
    """Purchase vectors reassigned between queries of the same size.

    Every source query keeps its complete purchase vector.  Its vector is given
    to another query with the same row count, and positions inside the assigned
    vector are randomized because row order is not part of this test.
    """
    values = np.asarray(target, dtype=np.int8)
    queries = np.asarray(query_ids)
    if values.ndim != 1 or queries.ndim != 1 or len(values) != len(queries):
        raise ValueError("target y query_ids deben ser vectores del mismo largo")
    if reps < 1:
        raise ValueError("reps debe ser positivo")

    rng = np.random.default_rng(seed)
    permuted = np.empty((reps, len(values)), dtype=np.int8)
    for positions in _query_positions(queries).values():
        n_queries, size = positions.shape
        blocks = values[positions]
        for rep in range(reps):
            assigned = blocks[rng.permutation(n_queries)].copy()
            order = np.argsort(rng.random((n_queries, size)), axis=1)
            assigned = np.take_along_axis(assigned, order, axis=1)
            permuted[rep, positions.ravel()] = assigned.ravel()
    return permuted


def group_separations(
    permutations: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    batch_size: int = 200,
) -> np.ndarray:
    """Separation of explicit row groups over precomputed target permutations."""
    values = np.asarray(permutations)
    indices = [np.asarray(group, dtype=np.int32) for group in groups]
    if values.ndim != 2:
        raise ValueError("permutations debe tener forma reps x filas")
    if len(indices) < 2 or any(group.size == 0 for group in indices):
        raise ValueError("se necesitan al menos dos grupos no vacios")

    out = np.empty(values.shape[0], dtype=float)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = values[start:stop]
        rates = np.stack([batch[:, group].mean(axis=1) for group in indices], axis=1)
        out[start:stop] = (rates.max(axis=1) - rates.min(axis=1)) * 100
    return out


def query_test(
    groups: Sequence[np.ndarray],
    target: np.ndarray,
    query_ids: np.ndarray,
    observed: float,
    *,
    reps: int = REPS,
    seed: int = 0,
    percentile: int = PERCENTILE,
    permutations: np.ndarray | None = None,
) -> PermutationResult:
    """Test ``observed`` against a null that preserves complete query blocks."""
    shuffled = (
        query_permutations(target, query_ids, reps=reps, seed=seed)
        if permutations is None
        else np.asarray(permutations)
    )
    if shuffled.shape != (reps, len(target)):
        raise ValueError("las permutaciones no coinciden con reps y el largo del target")
    null = group_separations(shuffled, groups)
    return PermutationResult(
        percentile=float(np.percentile(null, percentile)),
        p_value=empirical_p(null, observed),
        reps=reps,
    )


def query_rate_test(
    groups: Sequence[np.ndarray],
    rates: np.ndarray,
    observed: float,
    *,
    reps: int = REPS,
    seed: int = 0,
    percentile: int = PERCENTILE,
) -> PermutationResult:
    """Permute one observed rate per query between query-level groups."""
    values = np.asarray(rates, dtype=float)
    indices = [np.asarray(group, dtype=np.int32) for group in groups]
    if values.ndim != 1 or len(indices) < 2:
        raise ValueError("se necesitan tasas por query y al menos dos grupos")

    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for rep in range(reps):
        shuffled = rng.permutation(values)
        means = [shuffled[group].mean() for group in indices]
        null[rep] = (max(means) - min(means)) * 100
    return PermutationResult(
        percentile=float(np.percentile(null, percentile)),
        p_value=empirical_p(null, observed),
        reps=reps,
    )


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
