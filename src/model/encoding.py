"""Fold-local preprocessing for the text and tabular model towers.

``RowEncoder.transform`` deliberately returns two independent batches. Text keeps
only ``[CLS]`` and word-token positions, while the tabular branch has a stable
30-column contract. Anything learned from data (the word vocabulary and the price
interval bounds) is still fitted only on the training rows of a fold.

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
CATEGORICAL_FIELDS: tuple[str, ...] = ("category", "allergens")
NUMERIC_FIELDS: tuple[str, ...] = ("price_position",)
SENTINEL_FIELDS: frozenset[str] = frozenset({"nutrition_score"})
"""Columns where a literal zero means "not applicable" rather than a low score."""

PAD, WORD_UNK, CLS = 0, 1, 2
N_SPECIAL = 3

CATEGORY_LEVELS: tuple[str, ...] = (
    "Baby",
    "Bakery",
    "Beverages",
    "Dairy",
    "Frozen",
    "Household",
    "Meat",
    "Pantry",
    "Personal Care",
    "Produce",
    "Seafood",
    "Snacks",
)
ALLERGEN_LEVELS: tuple[str, ...] = (
    NO_ALLERGENS,
    "Fish",
    "Milk",
    "Peanuts",
    "Shellfish",
    "Soy",
    "Tree nuts",
    "Wheat",
)
PRICE_PIECES = 10
TABULAR_WIDTH = len(CATEGORY_LEVELS) + len(ALLERGEN_LEVELS) + PRICE_PIECES

_TOKEN_TYPE = {name: position for position, name in enumerate(TEXT_FIELDS)}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: object) -> list[str]:
    if text is None or pd.isna(text):
        return []
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
class TextBatch:
    """Text-only inputs, each with shape ``(rows, sequence_length)``."""

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device) -> "TextBatch":
        return TextBatch(
            input_ids=self.input_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )

    def select(self, rows: torch.Tensor) -> "TextBatch":
        return TextBatch(
            input_ids=self.input_ids[rows],
            token_type_ids=self.token_type_ids[rows],
            attention_mask=self.attention_mask[rows],
        )


@dataclass(frozen=True)
class TabBatch:
    """The independent 30-dimensional tabular input."""

    x_tab: torch.Tensor

    def __len__(self) -> int:
        return int(self.x_tab.shape[0])

    def to(self, device) -> "TabBatch":
        return TabBatch(x_tab=self.x_tab.to(device))

    def select(self, rows: torch.Tensor) -> "TabBatch":
        return TabBatch(x_tab=self.x_tab[rows])


@dataclass(frozen=True)
class EncodedRows:
    """The tensors the network consumes, already padded to a common width."""

    token_ids: torch.Tensor
    field_ids: torch.Tensor
    padding_mask: torch.Tensor
    numeric_values: torch.Tensor
    numeric_buckets: torch.Tensor
    numeric_missing: torch.Tensor
    numeric_ratios: torch.Tensor

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
            numeric_ratios=self.numeric_ratios.to(device),
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
            numeric_ratios=self.numeric_ratios[rows],
        )


@dataclass
class RowEncoder:
    """Fits its vocabulary and statistics on training rows, then encodes any rows."""

    spec: EncodingSpec = field(default_factory=EncodingSpec)
    _words: dict[str, int] = field(default_factory=dict, init=False)
    _centres: dict[str, float] = field(default_factory=dict, init=False)
    _scales: dict[str, float] = field(default_factory=dict, init=False)
    _edges: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _bounds: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _tabular_price_centre: float = field(default=0.0, init=False)
    _tabular_price_bounds: np.ndarray = field(
        default_factory=lambda: np.empty(0), init=False
    )
    _text_width: int = field(default=0, init=False)
    _fitted: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        self._fit_words(training)
        self._fit_numbers(training)
        self._fit_tabular_price(training)
        self._fitted = True
        return self

    def transform(
        self, frame: pd.DataFrame, indices
    ) -> tuple[TextBatch, TabBatch]:
        """Encode rows into independent text and tabular tower inputs."""
        if not self._fitted:
            raise RuntimeError("the encoder was never fitted")
        rows = frame.iloc[list(indices)]
        return self._text_batch(rows), self._tab_batch(rows)

    @property
    def vocabulary_size(self) -> int:
        """Size of the text-only vocabulary, including the three special tokens."""
        return N_SPECIAL + len(self._words)

    @property
    def n_fields(self) -> int:
        """One field identifier for ``[CLS]`` and one per discrete column."""
        return 1 + len(self.spec.text_fields) + len(self.spec.categorical_fields)

    @property
    def sequence_length(self) -> int:
        """Padded width of ``[CLS]`` followed by word-token positions."""
        return 1 + self._text_width

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

    def _fit_numbers(self, training: pd.DataFrame) -> None:
        self._centres, self._scales, self._edges, self._bounds = {}, {}, {}, {}
        for name in self.spec.numeric_fields:
            values = numeric_column(training, name)
            present = values[~np.isnan(values)]
            centre = float(np.median(present)) if present.size else 0.0
            spread = float(present.std()) if present.size else 0.0
            self._centres[name] = centre
            self._scales[name] = spread or 1.0
            filled = np.where(np.isnan(values), centre, values)
            quantiles = np.linspace(0.0, 1.0, self.spec.n_buckets + 1)
            self._edges[name] = np.unique(np.quantile(filled, quantiles[1:-1]))
            self._bounds[name] = self._interval_bounds(filled, quantiles)

    def _interval_bounds(self, filled: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        cuts = np.quantile(filled, quantiles)
        wanted = self.spec.n_buckets + 1
        for position in range(1, wanted):
            if cuts[position] <= cuts[position - 1]:
                cuts[position] = np.nextafter(cuts[position - 1], np.inf)
        return cuts

    def _fit_tabular_price(self, training: pd.DataFrame) -> None:
        """Fit the tabular tower's ten price intervals from training rows only."""
        if "price_position" not in self.spec.numeric_fields:
            self._tabular_price_centre = 0.0
            self._tabular_price_bounds = np.empty(0)
            return
        values = numeric_column(training, "price_position")
        present = values[~np.isnan(values)]
        self._tabular_price_centre = float(np.median(present)) if present.size else 0.0
        filled = np.where(np.isnan(values), self._tabular_price_centre, values)
        quantiles = np.linspace(0.0, 1.0, PRICE_PIECES + 1)
        self._tabular_price_bounds = self._interval_bounds(filled, quantiles)

    def _text_batch(self, rows: pd.DataFrame) -> TextBatch:
        """Build a text-only sequence with no categorical or numeric positions."""
        width = self.sequence_length
        input_ids = np.full((len(rows), width), PAD, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=bool)
        token_type_ids = np.zeros((len(rows), width), dtype=np.int64)

        input_ids[:, 0] = CLS
        attention_mask[:, 0] = True

        for row_position, row in enumerate(rows.itertuples(index=False)):
            cursor = 1
            for fallback_type, name in enumerate(self.spec.text_fields):
                token_type = _TOKEN_TYPE.get(name, fallback_type)
                for word in tokenize(getattr(row, name)):
                    if cursor >= width:
                        break
                    input_ids[row_position, cursor] = self._words.get(word, WORD_UNK)
                    token_type_ids[row_position, cursor] = token_type
                    attention_mask[row_position, cursor] = True
                    cursor += 1

        return TextBatch(
            input_ids=torch.from_numpy(input_ids),
            token_type_ids=torch.from_numpy(token_type_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )

    def _tab_batch(self, rows: pd.DataFrame) -> TabBatch:
        """Build category, allergen and price-piece blocks."""
        x_tab = np.zeros((len(rows), TABULAR_WIDTH), dtype=np.float32)

        if "category" in self.spec.categorical_fields:
            category_index = {value: i for i, value in enumerate(CATEGORY_LEVELS)}
            for row, value in enumerate(categorical_column(rows, "category")):
                column = category_index.get(value)
                if column is not None:
                    x_tab[row, column] = 1.0

        allergen_start = len(CATEGORY_LEVELS)
        if "allergens" in self.spec.categorical_fields:
            allergen_index = {value: i for i, value in enumerate(ALLERGEN_LEVELS)}
            for row, value in enumerate(categorical_column(rows, "allergens")):
                column = allergen_index.get(value)
                if column is not None:
                    x_tab[row, allergen_start + column] = 1.0

        price_start = allergen_start + len(ALLERGEN_LEVELS)
        if "price_position" in self.spec.numeric_fields:
            raw = numeric_column(rows, "price_position")
            missing = np.isnan(raw)
            filled = np.where(missing, self._tabular_price_centre, raw)
            x_tab[:, price_start : price_start + PRICE_PIECES] = (
                self._ratios_for_bounds(self._tabular_price_bounds, filled)
            )

        return TabBatch(x_tab=torch.from_numpy(x_tab))

    def _numeric(
        self, rows: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Standardised value, bucket index and the flag, one column per field."""
        n_rows, n_fields = len(rows), self.n_numeric
        n_buckets = self.spec.n_buckets
        values = np.zeros((n_rows, n_fields), dtype=np.float32)
        buckets = np.zeros((n_rows, n_fields), dtype=np.int64)
        missing = np.zeros((n_rows, n_fields), dtype=np.float32)
        ratios = np.zeros((n_rows, n_fields, n_buckets), dtype=np.float32)

        for column, name in enumerate(self.spec.numeric_fields):
            raw = numeric_column(rows, name)
            absent = np.isnan(raw)
            filled = np.where(absent, self._centres[name], raw)
            values[:, column] = (filled - self._centres[name]) / self._scales[name]
            buckets[:, column] = np.digitize(filled, self._edges[name])
            missing[:, column] = absent.astype(np.float32)
            ratios[:, column, :] = self.piecewise_ratios(name, filled)

        return values, buckets, missing, ratios

    def piecewise_ratios(self, name: str, values: np.ndarray) -> np.ndarray:
        """``(rows, n_buckets)``: how far the value travelled through each bucket."""
        return self._ratios_for_bounds(self._bounds[name], values)

    @staticmethod
    def _ratios_for_bounds(bounds: np.ndarray, values: np.ndarray) -> np.ndarray:
        lower, upper = bounds[:-1], bounds[1:]
        width = np.where(upper > lower, upper - lower, 1.0)
        travelled = (np.asarray(values, dtype=np.float64)[:, None] - lower[None, :])
        return np.clip(travelled / width[None, :], 0.0, 1.0).astype(np.float32)


def categorical_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Values as strings; a missing allergen list is a level, not a dropped row."""
    return frame[name].fillna(NO_ALLERGENS).astype(str)


def numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Values as floats, with a sentinel zero turned into an honest missing value."""
    values = frame[name].to_numpy(dtype=np.float64, copy=True)
    if name in SENTINEL_FIELDS:
        values[values == NUTRITION_SENTINEL] = np.nan
    return values
