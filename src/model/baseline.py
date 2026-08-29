"""
The feature set is the write-up's best honest baseline -- the popularity phrase, the
price percentile in quantile buckets, the category and the allergens. Levels and
bucket edges are fitted on training rows only, so a validation row can never
introduce a column the model did not see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model.encoding import categorical_column, tokenize
from src.model.protocol import ScoreFold
from src.model.records import TrainedFold

CATEGORICAL_FIELDS: tuple[str, ...] = ("popularity_phrase", "category", "allergens")
"""Every column the bar reads. ``cart`` is absent here and in every other feature set."""

NUMERIC_FIELD = "price_position"
"""``docs/EDA.md`` calls this ``price_pct``; ``loading.py`` derives it with the same formula."""

N_BUCKETS = 10
TARGET = "bought"


@dataclass
class OneHotLevels:
    """Categorical levels seen in training. An unseen level becomes a row of zeros."""

    fields: tuple[str, ...]
    _levels: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        self._levels = {
            name: tuple(sorted(categorical_column(training, name).unique()))
            for name in self.fields
        }
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        rows = frame.iloc[list(indices)]
        blocks = []
        for name in self.fields:
            levels = self._levels[name]
            position = {level: index for index, level in enumerate(levels)}
            block = np.zeros((len(rows), len(levels)), dtype=np.float64)
            for row, value in enumerate(categorical_column(rows, name)):
                column = position.get(value)
                if column is not None:
                    block[row, column] = 1.0
            blocks.append(block)
        return np.hstack(blocks)


@dataclass
class QuantileBuckets:
    """Quantile edges from the training rows, one indicator column per bucket."""

    name: str
    n_bins: int = N_BUCKETS
    _edges: np.ndarray | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        values = frame.iloc[list(train_indices)][self.name].to_numpy(dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self._edges = np.unique(np.quantile(values, quantiles))
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError(f"{self.name} buckets were never fitted")
        values = frame.iloc[list(indices)][self.name].to_numpy(dtype=np.float64)
        assigned = np.digitize(values, self._edges)
        block = np.zeros((len(values), len(self._edges) + 1), dtype=np.float64)
        block[np.arange(len(values)), assigned] = 1.0
        return block


@dataclass
class WordIndicators:
    """A binary bag of words over the declared text fields, fitted on training rows.

    This is the counterweight to the frozen sentence encoder of ``pretrained.py``.
    It has no idea that *customer* and *shopper* mean nearly the same thing, which is
    exactly why it separates ``(Customer Favorite)`` from ``(Shopper Favorite)`` that
    a semantic embedding folds together. A word absent from the training rows gets no
    column, so an unseen word contributes nothing rather than shifting the row.
    """

    fields: tuple[str, ...]
    _vocabulary: dict[str, int] = field(default_factory=dict, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        seen = {word for _, row in training.iterrows() for word in self._words(row)}
        self._vocabulary = {word: index for index, word in enumerate(sorted(seen))}
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        rows = frame.iloc[list(indices)]
        block = np.zeros((len(rows), len(self._vocabulary)), dtype=np.float64)
        for position, (_, row) in enumerate(rows.iterrows()):
            for word in self._words(row):
                column = self._vocabulary.get(word)
                if column is not None:
                    block[position, column] = 1.0
        return block

    def _words(self, row: pd.Series) -> set[str]:
        return {word for name in self.fields for word in tokenize(row[name])}


def feature_blocks(
    *,
    text_fields: tuple[str, ...] = (),
    categorical_fields: tuple[str, ...] = CATEGORICAL_FIELDS,
    numeric_fields: tuple[str, ...] = (NUMERIC_FIELD,),
    n_buckets: int = N_BUCKETS,
) -> list:
    """The declared columns as fit/transform blocks, in a fixed order.

    ``pretrained.py`` builds the same list to hang its embedding block beside, so the
    tabular half of a transfer run is identical to the tabular half of the bar.
    """
    blocks: list = []
    if text_fields:
        blocks.append(WordIndicators(text_fields))
    if categorical_fields:
        blocks.append(OneHotLevels(categorical_fields))
    blocks += [QuantileBuckets(name, n_buckets) for name in numeric_fields]
    if not blocks:
        raise ValueError("a feature set must declare at least one field")
    return blocks


def fit_blocks(blocks: list, frame: pd.DataFrame, train_indices) -> list:
    for block in blocks:
        block.fit(frame, train_indices)
    return blocks


def design_matrix(blocks: list, frame: pd.DataFrame, indices) -> np.ndarray:
    return np.hstack([block.transform(frame, indices) for block in blocks])


def target_of(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET].to_numpy().astype(np.int8)


def logistic_scorer(
    frame: pd.DataFrame,
    *,
    text_fields: tuple[str, ...] = (),
    categorical_fields: tuple[str, ...] = CATEGORICAL_FIELDS,
    numeric_fields: tuple[str, ...] = (NUMERIC_FIELD,),
    n_buckets: int = N_BUCKETS,
    c: float = 1.0,
    folds: list | None = None,
) -> ScoreFold:
    """The bar as a ``ScoreFold``: refits its own encoders on each fold's train rows."""
    target = target_of(frame)

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        blocks = fit_blocks(
            feature_blocks(
                text_fields=text_fields,
                categorical_fields=categorical_fields,
                numeric_fields=numeric_fields,
                n_buckets=n_buckets,
            ),
            frame,
            train_indices,
        )
        train_matrix = design_matrix(blocks, frame, train_indices)
        scored_matrix = design_matrix(blocks, frame, scored_indices)
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        model.fit(train_matrix, target[list(train_indices)])
        if folds is not None:
            folds.append(TrainedFold(parameters=train_matrix.shape[1] + 1))
        return model.predict_proba(scored_matrix)[:, 1]

    return score_fold
