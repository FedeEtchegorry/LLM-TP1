"""One row becomes one heterogeneous sequence.

    [CLS] · text tokens · one token per categorical field · one per numeric field

Which columns enter is a knob rather than a constant: ``title`` and ``description``
are separate fields, so they can be measured apart and then together. Vocabulary,
categorical levels, imputation medians, scales and bucket edges are all fitted on the
training rows of a fold and never on the rows being scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
import torch

from src.eda.aspects.composition import NO_ALLERGENS, NUTRITION_SENTINEL

TEXT_FIELDS: tuple[str, ...] = ("title", "description", "ingredients")
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category",
    "storage_type",
    "allergens",
    "unit_of_measure",
)
NUMERIC_FIELDS: tuple[str, ...] = (
    "price",
    "price_position",
    "net_weight_oz",
    "nutrition_score",
)
SENTINEL_FIELDS: frozenset[str] = frozenset({"nutrition_score"})
"""Columns where a literal zero means "not applicable" rather than a low score."""

PAD, WORD_UNK, CLS = 0, 1, 2
N_SPECIAL = 3

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: object) -> list[str]:
    return _TOKEN.findall(str(text).lower())


@dataclass(frozen=True)
class EncodingSpec:
    """Which columns enter the sequence, and how finely numbers are bucketed."""

    text_fields: tuple[str, ...] = TEXT_FIELDS
    categorical_fields: tuple[str, ...] = CATEGORICAL_FIELDS
    numeric_fields: tuple[str, ...] = NUMERIC_FIELDS
    n_buckets: int = 10
    max_text_tokens: int = 64

    def __post_init__(self) -> None:
        if not (self.text_fields or self.categorical_fields or self.numeric_fields):
            raise ValueError("a spec must carry at least one field")


@dataclass(frozen=True)
class EncodedRows:
    """The tensors the network consumes, already padded to a common width."""

    token_ids: torch.Tensor
    field_ids: torch.Tensor
    padding_mask: torch.Tensor
    numeric_values: torch.Tensor
    numeric_buckets: torch.Tensor
    numeric_missing: torch.Tensor

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def to(self, device) -> "EncodedRows":
        """The same rows on another device. Cheap and idempotent when already there."""
        return EncodedRows(
            token_ids=self.token_ids.to(device),
            field_ids=self.field_ids.to(device),
            padding_mask=self.padding_mask.to(device),
            numeric_values=self.numeric_values.to(device),
            numeric_buckets=self.numeric_buckets.to(device),
            numeric_missing=self.numeric_missing.to(device),
        )

    def select(self, rows: torch.Tensor) -> "EncodedRows":
        """The same columns for a subset of rows, in the order given: one batch."""
        return EncodedRows(
            token_ids=self.token_ids[rows],
            field_ids=self.field_ids[rows],
            padding_mask=self.padding_mask[rows],
            numeric_values=self.numeric_values[rows],
            numeric_buckets=self.numeric_buckets[rows],
            numeric_missing=self.numeric_missing[rows],
        )


@dataclass
class RowEncoder:
    """Fits its vocabulary and statistics on training rows, then encodes any rows."""

    spec: EncodingSpec = field(default_factory=EncodingSpec)
    _words: dict[str, int] = field(default_factory=dict, init=False)
    _levels: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    _level_unknown: dict[str, int] = field(default_factory=dict, init=False)
    _centres: dict[str, float] = field(default_factory=dict, init=False)
    _scales: dict[str, float] = field(default_factory=dict, init=False)
    _edges: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _text_width: int = field(default=0, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        self._fit_words(training)
        self._fit_levels(training)
        self._fit_numbers(training)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame, indices) -> EncodedRows:
        if not self._fitted:
            raise RuntimeError("the encoder was never fitted")
        rows = frame.iloc[list(indices)]
        token_ids, field_ids, mask = self._discrete(rows)
        values, buckets, missing = self._numeric(rows)
        return EncodedRows(
            token_ids=torch.from_numpy(token_ids),
            field_ids=torch.from_numpy(field_ids),
            padding_mask=torch.from_numpy(mask),
            numeric_values=torch.from_numpy(values),
            numeric_buckets=torch.from_numpy(buckets),
            numeric_missing=torch.from_numpy(missing),
        )

    @property
    def vocabulary_size(self) -> int:
        """Words plus categorical levels plus the three special tokens."""
        levels = sum(len(codes) + 1 for codes in self._levels.values())
        return N_SPECIAL + len(self._words) + levels

    @property
    def n_fields(self) -> int:
        """One field identifier for ``[CLS]`` and one per discrete column."""
        return 1 + len(self.spec.text_fields) + len(self.spec.categorical_fields)

    @property
    def sequence_length(self) -> int:
        return 1 + self._text_width + len(self.spec.categorical_fields)

    @property
    def n_numeric(self) -> int:
        return len(self.spec.numeric_fields)

    @property
    def text_width(self) -> int:
        """How many positions the text fields occupy, after ``[CLS]``."""
        return self._text_width

    def bucket_edges(self, name: str) -> np.ndarray:
        """The training quantile cuts for one numeric column, for interpretability."""
        return self._edges[name]

    def standardise(self, name: str, values: np.ndarray) -> np.ndarray:
        """Put raw values on the scale the network was trained to read them on."""
        return (values - self._centres[name]) / self._scales[name]

    def _fit_words(self, training: pd.DataFrame) -> None:
        seen: set[str] = set()
        longest = 0
        for _, row in training[list(self.spec.text_fields)].iterrows():
            tokens = [token for name in self.spec.text_fields for token in tokenize(row[name])]
            seen.update(tokens)
            longest = max(longest, len(tokens))
        self._words = {
            word: N_SPECIAL + position for position, word in enumerate(sorted(seen))
        }
        self._text_width = min(longest, self.spec.max_text_tokens)

    def _fit_levels(self, training: pd.DataFrame) -> None:
        self._levels = {}
        self._level_unknown = {}
        offset = N_SPECIAL + len(self._words)
        for name in self.spec.categorical_fields:
            levels = sorted(categorical_column(training, name).unique())
            self._levels[name] = {
                level: offset + position for position, level in enumerate(levels)
            }
            self._level_unknown[name] = offset + len(levels)
            offset += len(levels) + 1

    def _fit_numbers(self, training: pd.DataFrame) -> None:
        self._centres, self._scales, self._edges = {}, {}, {}
        for name in self.spec.numeric_fields:
            values = numeric_column(training, name)
            present = values[~np.isnan(values)]
            centre = float(np.median(present)) if present.size else 0.0
            spread = float(present.std()) if present.size else 0.0
            self._centres[name] = centre
            self._scales[name] = spread or 1.0
            filled = np.where(np.isnan(values), centre, values)
            quantiles = np.linspace(0.0, 1.0, self.spec.n_buckets + 1)[1:-1]
            self._edges[name] = np.unique(np.quantile(filled, quantiles))

    def _discrete(
        self, rows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``[CLS]``, the text tokens padded to a common width, then one token each."""
        width = self.sequence_length
        token_ids = np.full((len(rows), width), PAD, dtype=np.int64)
        field_ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=bool)

        token_ids[:, 0] = CLS
        mask[:, 0] = True

        for position, row in enumerate(rows.itertuples(index=False)):
            cursor = 1
            for field_index, name in enumerate(self.spec.text_fields, start=1):
                for word in tokenize(getattr(row, name)):
                    if cursor > self._text_width:
                        break
                    token_ids[position, cursor] = self._words.get(word, WORD_UNK)
                    field_ids[position, cursor] = field_index
                    mask[position, cursor] = True
                    cursor += 1

        start = 1 + self._text_width
        field_index = 1 + len(self.spec.text_fields)
        for offset, name in enumerate(self.spec.categorical_fields):
            column = start + offset
            codes = self._levels[name]
            unknown = self._level_unknown[name]
            token_ids[:, column] = [
                codes.get(value, unknown) for value in categorical_column(rows, name)
            ]
            field_ids[:, column] = field_index + offset
            mask[:, column] = True

        return token_ids, field_ids, mask

    def _numeric(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardised value, bucket index and the flag, one column per field."""
        n_rows, n_fields = len(rows), self.n_numeric
        values = np.zeros((n_rows, n_fields), dtype=np.float32)
        buckets = np.zeros((n_rows, n_fields), dtype=np.int64)
        missing = np.zeros((n_rows, n_fields), dtype=np.float32)

        for column, name in enumerate(self.spec.numeric_fields):
            raw = numeric_column(rows, name)
            absent = np.isnan(raw)
            filled = np.where(absent, self._centres[name], raw)
            values[:, column] = (filled - self._centres[name]) / self._scales[name]
            buckets[:, column] = np.digitize(filled, self._edges[name])
            missing[:, column] = absent.astype(np.float32)

        return values, buckets, missing


def categorical_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Values as strings; a missing allergen list is a level, not a dropped row."""
    return frame[name].fillna(NO_ALLERGENS).astype(str)


def numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Values as floats, with a sentinel zero turned into an honest missing value."""
    values = frame[name].to_numpy(dtype=np.float64, copy=True)
    if name in SENTINEL_FIELDS:
        values[values == NUTRITION_SENTINEL] = np.nan
    return values
