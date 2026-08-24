"""Feature blocks that are fitted on training rows and applied to any rows.

Every block follows the same two-step contract:

``fit(data, train_indices)``
    Estimate whatever the block needs from the training rows *only* --
    vocabulary, category set, mean and scale, bucket edges.

``transform(data, indices)``
    Produce a dense matrix for the requested rows using the fitted state.

Keeping estimation behind ``fit`` is what makes the reported scores honest: a
vocabulary built over all 10,000 rows would let validation rows influence the
feature space even though their labels are never seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.eda.dataset import BtrData

Indices = Sequence[int]


class NotFittedError(RuntimeError):
    """Raised when a block is transformed before it is fitted."""


@dataclass
class BagOfWords:
    """Binary word-occurrence features over ``title + description + ingredients``."""

    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 1
    name: str = "bag_of_words"
    _vectorizer: CountVectorizer | None = field(default=None, init=False, repr=False)

    def fit(self, data: BtrData, train_indices: Indices) -> "BagOfWords":
        self._vectorizer = CountVectorizer(
            binary=True,
            lowercase=True,
            token_pattern=r"[a-z0-9]+",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        self._vectorizer.fit([data.text[index] for index in train_indices])
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        vectorizer = _require(self._vectorizer, self.name)
        matrix = vectorizer.transform([data.text[index] for index in indices])
        return np.asarray(matrix.todense(), dtype=np.float64)

    @property
    def vocabulary_size(self) -> int:
        return len(_require(self._vectorizer, self.name).vocabulary_)


@dataclass
class CategoricalOneHot:
    """One-hot columns for categorical fields; unseen values become all-zero."""

    fields: tuple[str, ...]
    name: str = "categorical"
    _levels: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)

    def fit(self, data: BtrData, train_indices: Indices) -> "CategoricalOneHot":
        self._levels = {}
        for name in self.fields:
            column = self._column(data, name)
            self._levels[name] = tuple(
                sorted({column[index] for index in train_indices})
            )
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        if not self._levels:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        blocks = []
        for name in self.fields:
            column = self._column(data, name)
            lookup = {level: position for position, level in enumerate(self._levels[name])}
            block = np.zeros((len(indices), len(lookup)), dtype=np.float64)
            for row, index in enumerate(indices):
                position = lookup.get(column[index])
                if position is not None:
                    block[row, position] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((len(indices), 0))

    @staticmethod
    def _column(data: BtrData, name: str) -> tuple[str, ...]:
        if name == "popularity_phrase":
            return data.popularity_phrase
        if name == "oracle_tier":
            return data.oracle_tier
        return data.categorical[name]


@dataclass
class NumericScaled:
    """Numeric columns, median-imputed and standardised on the training rows."""

    fields: tuple[str, ...]
    name: str = "numeric"
    _medians: dict[str, float] = field(default_factory=dict, init=False)
    _means: dict[str, float] = field(default_factory=dict, init=False)
    _scales: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, data: BtrData, train_indices: Indices) -> "NumericScaled":
        self._medians, self._means, self._scales = {}, {}, {}
        for name in self.fields:
            train_values = data.numeric[name][list(train_indices)]
            median = float(np.nanmedian(train_values))
            filled = np.where(np.isnan(train_values), median, train_values)
            self._medians[name] = median
            self._means[name] = float(filled.mean())
            self._scales[name] = float(filled.std()) or 1.0
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        if not self._medians:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        columns = []
        for name in self.fields:
            values = data.numeric[name][list(indices)]
            filled = np.where(np.isnan(values), self._medians[name], values)
            columns.append((filled - self._means[name]) / self._scales[name])
        return np.column_stack(columns)


@dataclass
class NumericBuckets:
    """Quantile buckets per numeric column, with edges taken from training rows.

    This is the block that matters most: the purchase-rate curve against
    ``price_pct`` is an inverted U, and a single linear coefficient cannot
    represent a hump. One-hot buckets can.
    """

    fields: tuple[str, ...]
    n_bins: int = 10
    name: str = "numeric_buckets"
    _edges: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _medians: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, data: BtrData, train_indices: Indices) -> "NumericBuckets":
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self._edges, self._medians = {}, {}
        for name in self.fields:
            train_values = data.numeric[name][list(train_indices)]
            median = float(np.nanmedian(train_values))
            filled = np.where(np.isnan(train_values), median, train_values)
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            self._medians[name] = median
            self._edges[name] = np.unique(np.quantile(filled, quantiles))
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        if not self._edges:
            raise NotFittedError(f"{self.name} must be fitted before transform")
        blocks = []
        for name in self.fields:
            values = data.numeric[name][list(indices)]
            filled = np.where(np.isnan(values), self._medians[name], values)
            edges = self._edges[name]
            assignment = np.searchsorted(edges, filled, side="right")
            block = np.zeros((len(filled), len(edges) + 1), dtype=np.float64)
            block[np.arange(len(filled)), assignment] = 1.0
            blocks.append(block)
        return np.hstack(blocks)


@dataclass
class Crossed:
    """The outer product of two blocks: one column per pair of their columns.

    An additive model gives every product the same price response, differing only
    by a constant offset per popularity level. Crossing the two blocks lets the
    whole price curve change shape from one level to the next.

    This exists to *measure* interaction, not to ship it. A Transformer models
    feature interaction natively -- attention across the field tokens, then the
    feed-forward layers -- so the gap between the additive row and the crossed row
    in :mod:`src.eda.run_interactions` is the evidence for using one. Explicit
    crosses do not scale: they multiply the column count and need to be chosen by
    hand, one pair at a time.
    """

    left: object
    right: object
    name: str = "crossed"

    def fit(self, data: BtrData, train_indices: Indices) -> "Crossed":
        self.left.fit(data, train_indices)
        self.right.fit(data, train_indices)
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        left = self.left.transform(data, indices)
        right = self.right.transform(data, indices)
        product = left[:, :, None] * right[:, None, :]
        return product.reshape(left.shape[0], -1)


@dataclass
class MissingIndicators:
    """Binary flags for sentinel-coded missing values."""

    name: str = "missing_indicators"

    def fit(self, data: BtrData, train_indices: Indices) -> "MissingIndicators":
        del data, train_indices  # nothing to estimate
        return self

    def transform(self, data: BtrData, indices: Indices) -> np.ndarray:
        return data.nutrition_missing[list(indices)].astype(np.float64)[:, None]


@dataclass(frozen=True)
class FeatureSpec:
    """A named set of blocks, fitted and applied together."""

    name: str
    blocks: tuple[object, ...]
    uses_oracle: bool = False
    """True when a block reads a field derived from whole-dataset label rates."""

    def fit_transform(
        self, data: BtrData, train_indices: Indices, other_indices: Indices
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit on ``train_indices`` and transform both index sets."""

        for block in self.blocks:
            block.fit(data, train_indices)
        return (
            self._stack(data, train_indices),
            self._stack(data, other_indices),
        )

    def _stack(self, data: BtrData, indices: Indices) -> np.ndarray:
        return np.hstack([block.transform(data, indices) for block in self.blocks])


def _require(value, name: str):
    if value is None:
        raise NotFittedError(f"{name} must be fitted before transform")
    return value
