"""Column-oriented access to the supermarket BTR dataset with derived fields."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
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

AUDITED_NUMERIC_FIELDS: tuple[str, ...] = (
    "package_value",
    "volume_in3",
    "month_index",
)
"""Numeric columns derived only so that discarding them is a measurement."""

NUTRITION_SENTINEL = 0.0
"""``nutrition_score`` is 0 exactly for Household and Personal Care rows."""

_TRAILING_PARENTHETICAL = re.compile(r"\(([^)]+)\)\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")
_LEADING_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
_DIMENSIONS = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*x\s*([0-9]+(?:\.[0-9]+)?)\s*x\s*([0-9]+(?:\.[0-9]+)?)"
)
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

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
    """Read the dataset and derive row-local fields."""
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
        "package_value": _package_value(rows),
        "volume_in3": _package_volume(rows),
        "month_index": _month_index(rows),
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
    """Split text into lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def _popularity_phrase(title: str) -> str:
    """Return the trailing parenthetical of a title, or :data:`NO_PHRASE`."""
    match = _TRAILING_PARENTHETICAL.search(title)
    return match.group(1) if match else NO_PHRASE


def _oracle_tier(phrase: str) -> str:
    """Map a popularity phrase to the hand-assigned tier A, B or C."""
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
    """Position of the price inside the shopper's requested price window."""
    price = _column(rows, "price")
    low = _column(rows, "filter_price_min")
    high = _column(rows, "filter_price_max")
    width = high - low
    return np.where(width > 0, (price - low) / np.where(width > 0, width, 1.0), 0.5)


def _package_value(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """The number in ``package_size``, with its unit left in ``unit_of_measure``."""
    return np.fromiter(
        (_leading_number(row["package_size"]) for row in rows),
        dtype=np.float64,
        count=len(rows),
    )


def _package_volume(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Cubic inches from ``dimensions_in``, parsed as ``length x width x height``."""
    values = []
    for row in rows:
        match = _DIMENSIONS.search(row["dimensions_in"])
        if match is None:
            values.append(float("nan"))
            continue
        length, width, height = (float(group) for group in match.groups())
        values.append(length * width * height)
    return np.asarray(values, dtype=np.float64)


def _month_index(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Months elapsed since the earliest event, as a continuous time coordinate."""
    months = np.fromiter(
        (
            datetime.strptime(row["timestamp"], _TIMESTAMP_FORMAT).year * 12
            + datetime.strptime(row["timestamp"], _TIMESTAMP_FORMAT).month
            for row in rows
        ),
        dtype=np.float64,
        count=len(rows),
    )
    return months - months.min()


def _leading_number(value: str) -> float:
    match = _LEADING_NUMBER.match(value)
    return float(match.group(1)) if match else float("nan")


def _parse_bool(value: str, field: str) -> int:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1
    if normalized == "false":
        return 0
    raise ValueError(f"{field} must be 'true' or 'false', got {value!r}")
