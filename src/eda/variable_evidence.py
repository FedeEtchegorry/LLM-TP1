"""Measured evidence for keeping or dropping each column."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eda.dataset import BtrData
from src.eda.evaluation import EvaluationResult, evaluate_across_folds, logistic_scorer
from src.eda.features import (
    BagOfWords,
    CategoricalOneHot,
    FeatureSpec,
    Indices,
    MissingIndicators,
    NumericBuckets,
    NumericScaled,
)
from src.partitions import DataPartitions

PRICE_BINS = 10

TEXT_COLUMN = "title + description + ingredients"
"""The three text fields are scored together: they share one bag-of-words block."""


@dataclass(frozen=True)
class Column:
    """A dataset column, how it is encoded, and what the write-up does with it."""
    name: str
    kind: str
    """``text``, ``categorical`` or ``numeric``."""
    keep: bool
    reason: str
    """Why the write-up keeps or drops it -- the claim this module tests."""
    in_full_set: bool = True
    """Whether it joins the leave-one-out feature set (the text block does not)."""


COLUMNS: tuple[Column, ...] = (
    Column(
        "popularity_phrase",
        "categorical",
        True,
        "the trailing parenthetical of the title; the dominant signal",
    ),
    Column(
        TEXT_COLUMN,
        "text",
        True,
        "carries the popularity phrase as words",
        in_full_set=False,
    ),
    Column("price_pct", "numeric", True, "position of the price inside the filter"),
    Column("price", "numeric", True, "absolute price"),
    Column("category", "categorical", True, "product category"),
    Column("allergens", "categorical", True, "declared allergens"),
    Column("storage_type", "categorical", True, "storage the product needs"),
    Column("unit_of_measure", "categorical", True, "unit behind package_size"),
    Column("net_weight_oz", "numeric", True, "net weight"),
    Column("nutrition_score", "numeric", True, "nutrition score, 0 read as missing"),
    Column("brand", "categorical", False, "15 brands; the write-up says no gain"),
    Column("country_of_origin", "categorical", False, "10 countries; no gain claimed"),
    Column("package_value", "numeric", False, "number in package_size, unit stripped"),
    Column("volume_in3", "numeric", False, "cubic inches from dimensions_in"),
    Column("month_index", "numeric", False, "months since the first event"),
)


def column_blocks(column: Column) -> tuple[object, ...]:
    """The feature blocks that represent one column, freshly constructed."""
    if column.kind == "text":
        return (BagOfWords(),)
    if column.kind == "categorical":
        return (CategoricalOneHot((column.name,)),)
    blocks: list[object] = [
        NumericScaled((column.name,)),
        NumericBuckets((column.name,), n_bins=PRICE_BINS),
    ]
    if column.name == "nutrition_score":
        blocks.append(MissingIndicators())
    return tuple(blocks)


def univariate_specs() -> tuple[tuple[Column, FeatureSpec], ...]:
    """One feature set per column, each containing only that column."""
    return tuple(
        (column, FeatureSpec(column.name, column_blocks(column)))
        for column in COLUMNS
    )


def full_spec(exclude: str | None = None) -> FeatureSpec:
    """Every column in :data:`COLUMNS` that joins the full set, minus one."""
    blocks: list[object] = []
    for column in COLUMNS:
        if not column.in_full_set or column.name == exclude:
            continue
        blocks.extend(column_blocks(column))
    name = "full tabular set" if exclude is None else f"without {exclude}"
    return FeatureSpec(name, tuple(blocks))


def evaluate_spec(
    data: BtrData, partitions: DataPartitions, spec: FeatureSpec
) -> EvaluationResult:
    return evaluate_across_folds(
        spec.name, data.target, partitions, logistic_scorer(data, spec)
    )


def univariate_results(
    data: BtrData, partitions: DataPartitions
) -> list[tuple[Column, EvaluationResult]]:
    """Score every column on its own."""
    return [
        (column, evaluate_spec(data, partitions, spec))
        for column, spec in univariate_specs()
    ]


def leave_one_out_results(
    data: BtrData, partitions: DataPartitions
) -> tuple[EvaluationResult, list[tuple[Column, EvaluationResult]]]:
    """Score the full tabular set, then the same set with each column removed."""
    full = evaluate_spec(data, partitions, full_spec())
    removed = [
        (column, evaluate_spec(data, partitions, full_spec(exclude=column.name)))
        for column in COLUMNS
        if column.in_full_set
    ]
    return full, removed


@dataclass
class ExternalColumn:
    """A feature block over an array supplied from outside the dataset loader."""
    values: np.ndarray
    name: str = "external"

    def fit(self, data: BtrData, train_indices: Indices) -> "ExternalColumn":
        del data, train_indices  # nothing to estimate
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        del data
        return self.values[list(indices)].astype(np.float64)[:, None]


def read_cart_column(path: Path) -> np.ndarray:
    """Read ``cart`` straight from the CSV, for the leakage demonstration only."""
    with path.open(encoding="utf-8", newline="") as handle:
        return np.array(
            [row["cart"].strip().lower() == "true" for row in csv.DictReader(handle)],
            dtype=np.int8,
        )


def leakage_results(
    data: BtrData, partitions: DataPartitions, dataset_path: Path
) -> list[tuple[str, EvaluationResult]]:
    """What a model scores with ``cart``, and what it scores honestly without it."""
    cart = read_cart_column(dataset_path)
    honest = FeatureSpec(
        "impression-time features (no cart)",
        (
            CategoricalOneHot(("popularity_phrase", "category", "allergens")),
            NumericBuckets(("price_pct",), n_bins=PRICE_BINS),
        ),
    )
    leaking = FeatureSpec(
        "the same features plus cart",
        (
            CategoricalOneHot(("popularity_phrase", "category", "allergens")),
            NumericBuckets(("price_pct",), n_bins=PRICE_BINS),
            ExternalColumn(cart, name="cart"),
        ),
    )
    cart_only = FeatureSpec("cart alone", (ExternalColumn(cart, name="cart"),))
    return [
        (spec.name, evaluate_spec(data, partitions, spec))
        for spec in (honest, leaking, cart_only)
    ]


def cart_crosstab(path: Path) -> dict[tuple[str, str], int]:
    """Counts of ``cart`` against ``bought``; the empty cell is the whole story."""
    counts: dict[tuple[str, str], int] = {
        (cart, bought): 0 for cart in ("false", "true") for bought in ("false", "true")
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[(row["cart"].strip().lower(), row["bought"].strip().lower())] += 1
    return counts


def numeric_correlations(
    data: BtrData, fields: Sequence[str]
) -> np.ndarray:
    """Pearson correlation between numeric columns, missing values dropped pairwise."""
    size = len(fields)
    matrix = np.eye(size)
    for i in range(size):
        for j in range(i + 1, size):
            left = data.numeric[fields[i]]
            right = data.numeric[fields[j]]
            usable = np.isfinite(left) & np.isfinite(right)
            value = float(np.corrcoef(left[usable], right[usable])[0, 1])
            matrix[i, j] = matrix[j, i] = value
    return matrix


@dataclass(frozen=True)
class FoldComposition:
    """Row counts and purchase rates per fold, plus the fixed test holdout."""
    labels: tuple[str, ...]
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    positive_rates: tuple[tuple[float, float], ...]
    test_positive_rate: float
    shared_queries: int
    """Queries appearing in both the training and validation half of any fold."""


def fold_composition(data: BtrData, partitions: DataPartitions) -> FoldComposition:
    """Summarise the partition, including the check that no query is split."""
    labels, train, validation, test, rates = [], [], [], [], []
    shared = 0
    for fold in partitions.folds:
        train_queries = {data.query_ids[index] for index in fold.train_indices}
        validation_queries = {data.query_ids[index] for index in fold.validation_indices}
        shared += len(train_queries & validation_queries)
        labels.append(f"fold {fold.fold_index}")
        train.append(len(fold.train_indices))
        validation.append(len(fold.validation_indices))
        test.append(len(partitions.test_indices))
        rates.append(
            (
                float(data.target[list(fold.train_indices)].mean()),
                float(data.target[list(fold.validation_indices)].mean()),
            )
        )
    return FoldComposition(
        labels=tuple(labels),
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        positive_rates=tuple(rates),
        test_positive_rate=float(data.target[list(partitions.test_indices)].mean()),
        shared_queries=shared,
    )
