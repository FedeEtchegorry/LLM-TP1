"""Fold-local preprocessing for the text and tabular model towers.

``RowEncoder.transform`` deliberately returns two independent batches. Text keeps
only ``[CLS]`` and word-token positions, while the tabular branch has a stable
30-column contract. Anything learned from data (the tokenizer's vocabulary and the
price interval bounds) is still fitted only on the training rows of a fold.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
import torch

from src.eda.aspects.composition import NO_ALLERGENS, NUTRITION_SENTINEL
from src.model.tokenization import (
    CLS,
    PAD,
    SEP,
    WORDPIECE,
    Tokenizer,
    tokenizer_for,
)
from src.model.tokenization import UNK as WORD_UNK

TEXT_FIELDS: tuple[str, ...] = ("title", "description", "ingredients")
CATEGORICAL_FIELDS: tuple[str, ...] = ("category", "allergens")
NUMERIC_FIELDS: tuple[str, ...] = ("price_position",)
SENTINEL_FIELDS: frozenset[str] = frozenset({"nutrition_score"})
"""Columns where a literal zero means "not applicable" rather than a low score."""

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
PRICE_START = len(CATEGORY_LEVELS) + len(ALLERGEN_LEVELS)
PRICE_SLICE = slice(PRICE_START, PRICE_START + PRICE_PIECES)
"""Where the ten price pieces sit inside ``x_tab``, for whoever has to read or
rewrite them: the counterfactual sweep and the price-axis diagnostic."""

_TOKEN_TYPE = {name: position for position, name in enumerate(TEXT_FIELDS)}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: object) -> list[str]:
    """The plain word regex, kept for the linear baseline and the bag of embeddings.

    The Transformer reads the tokenizer its run declares instead.
    """
    return _TOKEN.findall(text_or_empty(text).lower())


def text_or_empty(value: object) -> str:
    """Missing text is no text. Without this, ``NaN`` becomes the word ``nan``."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


@dataclass(frozen=True)
class EncodingSpec:
    """Which columns enter the sequence, and how finely numbers are bucketed."""

    text_fields: tuple[str, ...] = TEXT_FIELDS
    categorical_fields: tuple[str, ...] = CATEGORICAL_FIELDS
    numeric_fields: tuple[str, ...] = NUMERIC_FIELDS
    n_buckets: int = 10
    tokenizer: str = WORDPIECE
    keep_brackets: bool = True
    """WordPiece with the parentheses is the architecture; the word regex stays
    reachable as the ablation's control. Both reach the run's digest."""
    max_text_tokens: int = 64
    """Token budget the three text fields share, specials excluded. The sequence adds
    ``[CLS]`` and one ``[SEP]`` per field, and is the same width for every fold."""

    def __post_init__(self) -> None:
        if not (self.text_fields or self.categorical_fields or self.numeric_fields):
            raise ValueError("a spec must carry at least one field")


@dataclass(frozen=True)
class FieldSpan:
    """Where one text field sits in a row, and how many of its tokens did not fit."""

    name: str
    start: int
    kept: int
    total: int

    @property
    def dropped(self) -> int:
        return self.total - self.kept


def longest_first(totals: list[int], budget: int) -> list[int]:
    """Share the budget by trimming the longest field, not by consuming left to right.

    No field is emptied while another one still has tokens to give up, and a tie is
    resolved against the later field: the end of the title is where the phrase lives.
    """
    if sum(totals) <= budget:
        return list(totals)
    cap = 0
    while sum(min(total, cap + 1) for total in totals) <= budget:
        cap += 1
    kept = [min(total, cap) for total in totals]
    for position, total in enumerate(totals):
        if sum(kept) >= budget:
            break
        if total > kept[position]:
            kept[position] += 1
    return kept


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


@dataclass
class RowEncoder:
    """Fits its vocabulary and statistics on training rows, then encodes any rows."""

    spec: EncodingSpec = field(default_factory=EncodingSpec)
    _tokenizer: Tokenizer | None = field(default=None, init=False)
    _centres: dict[str, float] = field(default_factory=dict, init=False)
    _scales: dict[str, float] = field(default_factory=dict, init=False)
    _edges: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _tabular_price_centre: float = field(default=0.0, init=False)
    _tabular_price_bounds: np.ndarray = field(
        default_factory=lambda: np.empty(0), init=False
    )
    _fitted: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = frame.iloc[list(train_indices)]
        self._fit_tokenizer(training)
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
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise RuntimeError("the encoder was never fitted")
        return self._tokenizer

    @property
    def vocabulary_size(self) -> int:
        """Rows the token embedding table needs, specials included."""
        return self.tokenizer.vocabulary_size

    def tokens(self, text: object) -> list[str]:
        return self.tokenizer.tokens(text_or_empty(text))

    def encode(self, text: object) -> list[int]:
        """The same tokens as ids, with no special token added."""
        return self.tokenizer.encode(text_or_empty(text))

    @property
    def sequence_length(self) -> int:
        """Fixed width: ``[CLS]``, the shared budget, and one ``[SEP]`` per field."""
        return 1 + self.spec.max_text_tokens + len(self.spec.text_fields)

    def layout(self, rows: pd.DataFrame) -> list[tuple[FieldSpan, ...]]:
        """Where every field lands in every row, once the budget has been shared out."""
        return [
            self._spans(self._row_tokens(row))
            for row in rows.itertuples(index=False)
        ]

    def bucket_edges(self, name: str) -> np.ndarray:
        """The training quantile cuts for one numeric column, for interpretability."""
        return self._edges[name]

    def standardise(self, name: str, values: np.ndarray) -> np.ndarray:
        """Put raw values on the scale the network was trained to read them on."""
        return (values - self._centres[name]) / self._scales[name]

    def _fit_tokenizer(self, training: pd.DataFrame) -> None:
        texts = [
            text_or_empty(value)
            for name in self.spec.text_fields
            for value in training[name]
        ]
        self._tokenizer = tokenizer_for(
            self.spec.tokenizer, self.spec.keep_brackets
        ).fit(texts)

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
            quantiles = np.linspace(0.0, 1.0, self.spec.n_buckets + 1)
            self._edges[name] = np.unique(np.quantile(filled, quantiles[1:-1]))

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

    def _row_tokens(self, row) -> list[list[int]]:
        return [self.encode(getattr(row, name)) for name in self.spec.text_fields]

    def _spans(self, tokens: list[list[int]]) -> tuple[FieldSpan, ...]:
        kept = longest_first(
            [len(field) for field in tokens], self.spec.max_text_tokens
        )
        spans, cursor = [], 1
        for name, field, keep in zip(self.spec.text_fields, tokens, kept):
            spans.append(FieldSpan(name, cursor, keep, len(field)))
            cursor += keep + 1
        return tuple(spans)

    def _text_batch(self, rows: pd.DataFrame) -> TextBatch:
        """``[CLS] title [SEP] description [SEP] ingredients [SEP]``, then padding."""
        width = self.sequence_length
        input_ids = np.full((len(rows), width), PAD, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=bool)
        token_type_ids = np.zeros((len(rows), width), dtype=np.int64)

        input_ids[:, 0] = CLS
        attention_mask[:, 0] = True

        for row_position, row in enumerate(rows.itertuples(index=False)):
            tokens = self._row_tokens(row)
            for position, (span, field) in enumerate(
                zip(self._spans(tokens), tokens)
            ):
                token_type = _TOKEN_TYPE.get(span.name, position)
                end = span.start + span.kept
                input_ids[row_position, span.start : end] = field[: span.kept]
                input_ids[row_position, end] = SEP
                token_type_ids[row_position, span.start : end + 1] = token_type
                attention_mask[row_position, span.start : end + 1] = True

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

        if "price_position" in self.spec.numeric_fields:
            raw = numeric_column(rows, "price_position")
            filled = np.where(np.isnan(raw), self._tabular_price_centre, raw)
            x_tab[:, PRICE_SLICE] = self.tabular_price_ratios(filled)

        return TabBatch(x_tab=torch.from_numpy(x_tab))

    def tabular_price_ratios(self, values: np.ndarray) -> np.ndarray:
        """``(len(values), PRICE_PIECES)``: the price block ``x_tab`` would carry."""
        return self._ratios_for_bounds(self._tabular_price_bounds, values)

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
