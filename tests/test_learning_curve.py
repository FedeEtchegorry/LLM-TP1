"""El submuestreo no puede tocar la validación ni partir una query, y la lectura de la
pendiente tiene que salir en el sentido que corresponda."""

from __future__ import annotations

import pytest

from src.model.learning_curve import (
    LOGISTIC,
    TRANSFORMER,
    Point,
    curve,
    markdown_table,
    read_slopes,
    slopes_by_seed,
    subsample,
)
from src.partitions import DataPartitions, FoldIndices

ROWS_PER_QUERY = 5
QUERIES = 200


def synthetic() -> tuple[DataPartitions, list]:
    """Doscientas queries de cinco filas: 160 de entrenamiento y 40 de validación."""
    query_ids = [index // ROWS_PER_QUERY for index in range(QUERIES * ROWS_PER_QUERY)]
    cut = 160 * ROWS_PER_QUERY
    folds = (
        FoldIndices(
            fold_index=0,
            train_indices=tuple(range(cut)),
            validation_indices=tuple(range(cut, QUERIES * ROWS_PER_QUERY)),
        ),
    )
    return DataPartitions(test_indices=(), folds=folds), query_ids


def test_validation_rows_are_identical_at_every_size():
    partitions, query_ids = synthetic()
    original = partitions.folds[0].validation_indices
    for size in (50, 200, 800):
        shrunk = subsample(partitions, query_ids, size)
        assert shrunk.folds[0].validation_indices == original


def test_a_query_is_never_split_between_kept_and_dropped():
    partitions, query_ids = synthetic()
    kept = set(subsample(partitions, query_ids, 200).folds[0].train_indices)
    seen: dict[int, set[bool]] = {}
    for index in partitions.folds[0].train_indices:
        seen.setdefault(query_ids[index], set()).add(index in kept)
    assert all(len(memberships) == 1 for memberships in seen.values())


def test_the_subsample_lands_near_the_budget():
    partitions, query_ids = synthetic()
    for size in (100, 300, 700):
        rows = len(subsample(partitions, query_ids, size).folds[0].train_indices)
        assert size <= rows < size + ROWS_PER_QUERY


def test_the_subsample_is_deterministic_so_both_models_see_the_same_rows():
    partitions, query_ids = synthetic()
    first = subsample(partitions, query_ids, 300).folds[0].train_indices
    second = subsample(partitions, query_ids, 300).folds[0].train_indices
    assert first == second


def test_a_bigger_budget_never_drops_rows_below_the_total():
    partitions, query_ids = synthetic()
    everything = subsample(partitions, query_ids, 10_000).folds[0].train_indices
    assert set(everything) == set(partitions.folds[0].train_indices)


def point(model: str, size: int, mean: float, spread: float = 0.002) -> Point:
    return Point(
        model=model,
        size=size,
        runs=tuple(tuple([mean + offset] * 5) for offset in (-spread, 0.0, spread)),
    )


def rising(model: str, start: float, per_decade: float) -> list[Point]:
    """Una curva cuyo AP crece linealmente con log10(filas)."""
    import numpy as np

    return [
        point(model, size, start + per_decade * float(np.log10(size / 500)))
        for size in (500, 1000, 2000, 4000, 6400)
    ]


def test_slope_recovers_the_growth_per_decade():
    fitted = slopes_by_seed(rising(TRANSFORMER, 0.50, 0.10), TRANSFORMER)
    assert fitted == pytest.approx([0.10, 0.10, 0.10], abs=1e-6)


def test_reads_a_data_limit_when_the_transformer_climbs_faster():
    points = rising(TRANSFORMER, 0.50, 0.14) + rising(LOGISTIC, 0.70, 0.02)
    reading = read_slopes(points)
    assert "el límite no es la arquitectura" in reading


def test_reads_a_negative_result_when_the_slopes_match():
    points = rising(TRANSFORMER, 0.50, 0.06) + rising(LOGISTIC, 0.70, 0.06)
    reading = read_slopes(points)
    assert "indistinguibles" in reading
    assert "resultado negativo" in reading


def test_reads_the_baseline_pulling_away():
    points = rising(TRANSFORMER, 0.50, 0.02) + rising(LOGISTIC, 0.70, 0.16)
    reading = read_slopes(points)
    assert "la alejarían" in reading


def test_every_reading_states_it_avoided_the_holdout():
    points = rising(TRANSFORMER, 0.50, 0.06) + rising(LOGISTIC, 0.70, 0.06)
    assert "antes de gastarlo" in read_slopes(points)


def test_an_incomplete_curve_says_so():
    assert "incompleta" in read_slopes([point(TRANSFORMER, 500, 0.5)])


def test_curve_returns_one_model_ordered_by_size():
    points = rising(TRANSFORMER, 0.5, 0.1) + rising(LOGISTIC, 0.7, 0.1)
    sizes = [item.size for item in curve(points, LOGISTIC)]
    assert sizes == sorted(sizes)
    assert all(item.model == LOGISTIC for item in curve(points, LOGISTIC))


def test_markdown_table_has_one_row_per_size():
    points = rising(TRANSFORMER, 0.5, 0.1) + rising(LOGISTIC, 0.7, 0.1)
    rendered = markdown_table(points).splitlines()
    assert len(rendered) == 2 + 5
