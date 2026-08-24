"""Column-oriented access to the supermarket BTR dataset with derived fields.

Loading is deliberately separate from feature construction: this module reads the
CSV and derives quantities that are properties of a row alone, never of the
dataset as a whole. Anything that must be estimated from a sample -- vocabularies,
category sets, scalers, bucket edges -- belongs in :mod:`src.eda.features`, where
it is fitted on training indices only.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATASET_PATH = Path("data/supermarket_products.csv")

TEXT_FIELDS: tuple[str, ...] = ("title", "description", "ingredients")
"""Fields concatenated into the text used for bag-of-words features."""

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category",
    "storage_type",
    "allergens",
    "unit_of_measure",
    "brand",
    "country_of_origin",
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "price",
    "price_pct",
    "net_weight_oz",
    "nutrition_score",
)

NUTRITION_SENTINEL = 0.0
"""``nutrition_score`` is 0 exactly for Household and Personal Care rows.

The column's genuine range is 18-99, so the sentinel is recorded as missing
rather than passed through as an extreme low score.
"""

_TRAILING_PARENTHETICAL = re.compile(r"\(([^)]+)\)\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")

NO_PHRASE = "<none>"
"""Value used when a title carries no trailing parenthetical."""

_ORACLE_TIER_A = frozenset(
    {"Customer Favorite", "Best Seller", "Top Rated", "#1 Pick"}
)
_ORACLE_TIER_B = frozenset(
    {"Well Reviewed", "Shopper Favorite", "Highly Rated", "Popular Choice"}
)


@dataclass(frozen=True)
class BtrData:
    """One row per product impression, with row-local derived fields."""

    target: np.ndarray
    query_ids: tuple[str, ...]
    text: tuple[str, ...]
    popularity_phrase: tuple[str, ...]
    oracle_tier: tuple[str, ...]
    categorical: dict[str, tuple[str, ...]]
    numeric: dict[str, np.ndarray]
    nutrition_missing: np.ndarray
    timestamps: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.target.shape[0])

    @property
    def positive_rate(self) -> float:
        """Fraction of impressions that were bought."""

        return float(self.target.mean())


def load_btr_data(path: Path | str = DEFAULT_DATASET_PATH) -> BtrData:
    """Read the dataset and derive row-local fields.

    ``cart`` is never read. It is a later step of the same funnel as ``bought``
    (no row has ``bought`` true with ``cart`` false), so using it as a feature
    would leak the target.

    ``filter_category`` and ``filter_storage_type`` are not read either: they
    duplicate ``category`` and ``storage_type`` in every row. The price filter
    bounds are read only to derive ``price_pct``.
    """

    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{source} contains no rows")

    target = np.fromiter(
        (_parse_bool(row["bought"], "bought") for row in rows),
        dtype=np.int8,
        count=len(rows),
    )
    numeric = {
        "price": _column(rows, "price"),
        "net_weight_oz": _column(rows, "net_weight_oz"),
        "nutrition_score": _column(rows, "nutrition_score"),
        "price_pct": _price_pct(rows),
    }
    nutrition_missing = (numeric["nutrition_score"] == NUTRITION_SENTINEL).astype(
        np.int8
    )
    numeric["nutrition_score"] = np.where(
        nutrition_missing == 1, np.nan, numeric["nutrition_score"]
    )

    phrases = tuple(_popularity_phrase(row["title"]) for row in rows)
    return BtrData(
        target=target,
        query_ids=tuple(row["query_id"] for row in rows),
        text=tuple(" ".join(row[field] for field in TEXT_FIELDS) for row in rows),
        popularity_phrase=phrases,
        oracle_tier=tuple(_oracle_tier(phrase) for phrase in phrases),
        categorical={
            field: tuple(row[field] for row in rows) for field in CATEGORICAL_FIELDS
        },
        numeric=numeric,
        nutrition_missing=nutrition_missing,
        timestamps=tuple(row["timestamp"] for row in rows),
    )


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    This is the tokenization every reported vocabulary count refers to.
    """

    return _TOKEN.findall(text.lower())


def _popularity_phrase(title: str) -> str:
    """Return the trailing parenthetical of a title, or :data:`NO_PHRASE`.

    The phrase is observable at impression time, so it is a legitimate feature.
    """

    match = _TRAILING_PARENTHETICAL.search(title)
    return match.group(1) if match else NO_PHRASE


def _oracle_tier(phrase: str) -> str:
    """Map a popularity phrase to the hand-assigned tier A, B or C.

    The grouping was chosen after inspecting purchase rates over the *whole*
    dataset, so any model using it is an oracle and its score is an upper bound,
    not an achievable baseline. Use :attr:`BtrData.popularity_phrase` for a
    feature that can be learned honestly from a training fold.
    """

    if phrase in _ORACLE_TIER_A:
        return "A"
    if phrase in _ORACLE_TIER_B:
        return "B"
    return "C"


def _column(rows: Sequence[dict[str, str]], field: str) -> np.ndarray:
    return np.fromiter(
        (float(row[field]) for row in rows), dtype=np.float64, count=len(rows)
    )


def _price_pct(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Position of the price inside the shopper's requested price window.

    Every row satisfies its own filter, so this is always in [0, 1]; the width is
    guarded anyway so a degenerate window yields 0.5 rather than a division error.
    """

    price = _column(rows, "price")
    low = _column(rows, "filter_price_min")
    high = _column(rows, "filter_price_max")
    width = high - low
    return np.where(width > 0, (price - low) / np.where(width > 0, width, 1.0), 0.5)


def _parse_bool(value: str, field: str) -> int:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1
    if normalized == "false":
        return 0
    raise ValueError(f"{field} must be 'true' or 'false', got {value!r}")
