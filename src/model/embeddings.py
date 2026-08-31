from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd

from src.model.encoding import categorical_column, numeric_column, tokenize


@dataclass
class Continuous:

    name: str
    _centre: float = field(default=0.0, init=False)
    _scale: float = field(default=1.0, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        values = numeric_column(frame.iloc[list(train_indices)], self.name)
        present = values[~np.isnan(values)]
        self._centre = float(np.median(present)) if present.size else 0.0
        spread = float(present.std()) if present.size else 0.0
        self._scale = spread or 1.0
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        values = numeric_column(frame.iloc[list(indices)], self.name)
        filled = np.where(np.isnan(values), self._centre, values)
        return ((filled - self._centre) / self._scale).reshape(-1, 1)


@dataclass
class QuantileBucketsBlock:

    name: str
    n_bins: int = 10
    _edges: np.ndarray | None = field(default=None, init=False)
    _centre: float = field(default=0.0, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        values = numeric_column(frame.iloc[list(train_indices)], self.name)
        present = values[~np.isnan(values)]
        self._centre = float(np.median(present)) if present.size else 0.0
        filled = np.where(np.isnan(values), self._centre, values)
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self._edges = np.unique(np.quantile(filled, quantiles))
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError(f"{self.name} buckets were never fitted")
        values = numeric_column(frame.iloc[list(indices)], self.name)
        filled = np.where(np.isnan(values), self._centre, values)
        assigned = np.digitize(filled, self._edges)
        block = np.zeros((len(filled), len(self._edges) + 1), dtype=np.float64)
        block[np.arange(len(filled)), assigned] = 1.0
        return block


@dataclass
class ContinuousAndBuckets:

    name: str
    n_bins: int = 10
    _continuous: Continuous = field(init=False)
    _buckets: QuantileBucketsBlock = field(init=False)

    def __post_init__(self) -> None:
        self._continuous = Continuous(self.name)
        self._buckets = QuantileBucketsBlock(self.name, self.n_bins)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        self._continuous.fit(frame, train_indices)
        self._buckets.fit(frame, train_indices)
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        return np.hstack(
            [
                self._continuous.transform(frame, indices),
                self._buckets.transform(frame, indices),
            ]
        )


@dataclass
class PiecewiseLinear:

    name: str
    n_bins: int = 10
    _edges: np.ndarray | None = field(default=None, init=False)
    _centre: float = field(default=0.0, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        values = numeric_column(frame.iloc[list(train_indices)], self.name)
        present = values[~np.isnan(values)]
        self._centre = float(np.median(present)) if present.size else 0.0
        filled = np.where(np.isnan(values), self._centre, values)
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self._edges = np.unique(np.quantile(filled, quantiles))
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError(f"{self.name} was never fitted")
        values = numeric_column(frame.iloc[list(indices)], self.name)
        filled = np.where(np.isnan(values), self._centre, values)
        lower, upper = self._edges[:-1], self._edges[1:]
        width = np.where(upper > lower, upper - lower, 1.0)
        ratio = (filled[:, None] - lower[None, :]) / width[None, :]
        return np.clip(ratio, 0.0, 1.0)


@dataclass
class Periodic:

    name: str
    n_frequencies: int = 4
    _low: float = field(default=0.0, init=False)
    _span: float = field(default=1.0, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        values = numeric_column(frame.iloc[list(train_indices)], self.name)
        present = values[~np.isnan(values)]
        if present.size:
            self._low = float(present.min())
            span = float(present.max() - present.min())
            self._span = span or 1.0
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        values = numeric_column(frame.iloc[list(indices)], self.name)
        filled = np.where(np.isnan(values), self._low + self._span / 2.0, values)
        scaled = np.clip((filled - self._low) / self._span, 0.0, 1.0)
        angles = [(2.0**k) * np.pi * scaled for k in range(self.n_frequencies)]
        return np.hstack(
            [np.sin(angle).reshape(-1, 1) for angle in angles]
            + [np.cos(angle).reshape(-1, 1) for angle in angles]
        )


@dataclass
class TargetEncoded:

    name: str
    smoothing: float = 20.0
    _rates: dict[str, float] = field(default_factory=dict, init=False)
    _prior: float = field(default=0.0, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        target = training["bought"].to_numpy().astype(float)
        levels = categorical_column(training, self.name)
        self._prior = float(target.mean())
        grouped = pd.DataFrame(
            {"level": levels.to_numpy(), "y": target}
        ).groupby("level")["y"]
        counts, means = grouped.count(), grouped.mean()
        shrunk = (counts * means + self.smoothing * self._prior) / (
            counts + self.smoothing
        )
        self._rates = shrunk.to_dict()
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        levels = categorical_column(frame.iloc[list(indices)], self.name)
        return (
            levels.map(self._rates).fillna(self._prior).to_numpy(dtype=np.float64)
        ).reshape(-1, 1)


def stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "big"
    )


@dataclass
class HashedLevels:

    fields: tuple[str, ...]
    n_columns: int = 32

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        rows = frame.iloc[list(indices)]
        block = np.zeros((len(rows), self.n_columns), dtype=np.float64)
        for name in self.fields:
            for position, value in enumerate(categorical_column(rows, name)):
                column = stable_hash(f"{name}={value}") % self.n_columns
                block[position, column] = 1.0
        return block


@dataclass
class OrdinalLevels:

    fields: tuple[str, ...]
    _codes: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        self._codes = {
            name: {
                level: index
                for index, level in enumerate(
                    sorted(categorical_column(training, name).unique())
                )
            }
            for name in self.fields
        }
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        rows = frame.iloc[list(indices)]
        return np.hstack(
            [
                categorical_column(rows, name)
                .map(self._codes[name])
                .fillna(-1)
                .to_numpy(dtype=np.float64)
                .reshape(-1, 1)
                for name in self.fields
            ]
        )


@dataclass
class TfidfWords:

    fields: tuple[str, ...]
    _vocabulary: dict[str, int] = field(default_factory=dict, init=False)
    _idf: np.ndarray | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        documents = [self._words(row) for _, row in training.iterrows()]
        seen = sorted({word for document in documents for word in document})
        self._vocabulary = {word: index for index, word in enumerate(seen)}
        counts = np.zeros(len(seen), dtype=np.float64)
        for document in documents:
            for word in document:
                counts[self._vocabulary[word]] += 1.0
        self._idf = np.log((1.0 + len(documents)) / (1.0 + counts)) + 1.0
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("the tf-idf block was never fitted")
        rows = frame.iloc[list(indices)]
        block = np.zeros((len(rows), len(self._vocabulary)), dtype=np.float64)
        for position, (_, row) in enumerate(rows.iterrows()):
            for word in self._words(row):
                column = self._vocabulary.get(word)
                if column is not None:
                    block[position, column] = self._idf[column]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        return block / np.where(norms > 0.0, norms, 1.0)

    def _words(self, row: pd.Series) -> set[str]:
        return {word for name in self.fields for word in tokenize(row[name])}
