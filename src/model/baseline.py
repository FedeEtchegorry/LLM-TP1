"""The linear bar the Transformer has to beat: AP 0.813 in ``docs/EDA.md`` §8.

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

from src.eda.aspects.composition import NO_ALLERGENS
from src.model.protocol import ScoreFold

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
            name: tuple(sorted(_column(training, name).unique()))
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
            for row, value in enumerate(_column(rows, name)):
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


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Categorical values as strings, with missing allergens kept as their own level."""
    return frame[name].fillna(NO_ALLERGENS).astype(str)


def target_of(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET].to_numpy().astype(np.int8)


def logistic_scorer(
    frame: pd.DataFrame,
    *,
    categorical_fields: tuple[str, ...] = CATEGORICAL_FIELDS,
    numeric_fields: tuple[str, ...] = (NUMERIC_FIELD,),
    n_buckets: int = N_BUCKETS,
    c: float = 1.0,
) -> ScoreFold:
    """The bar as a ``ScoreFold``: refits its own encoders on each fold's train rows."""
    target = target_of(frame)

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        blocks = [OneHotLevels(categorical_fields)] if categorical_fields else []
        blocks += [QuantileBuckets(name, n_buckets) for name in numeric_fields]
        for block in blocks:
            block.fit(frame, train_indices)

        train_matrix = np.hstack(
            [block.transform(frame, train_indices) for block in blocks]
        )
        scored_matrix = np.hstack(
            [block.transform(frame, scored_indices) for block in blocks]
        )
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        model.fit(train_matrix, target[list(train_indices)])
        return model.predict_proba(scored_matrix)[:, 1]

    return score_fold
