"""Reads the BTR dataset and adds the derived columns the aspects need."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/supermarket_products.csv")

MONTH_NAMES: tuple[str, ...] = (
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
)

WEEKDAY_NAMES: tuple[str, ...] = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")

NO_PHRASE = "(sin frase)"

OUNCES_PER_UNIT: dict[str, float] = {
    "oz": 1.0,
    "fl oz": 1.0,
    "lb": 16.0,
    "gal": 128.0,
}
"""``ct`` is a count, not a weight, so it converts to nothing and stays missing."""

_PARENTHETICAL = r"\(([^)]+)\)\s*$"
_SENTENCE_BREAK = r"\.\s+"
_LEADING_AMOUNT = r"^([\d.]+)\s"
_DIMENSIONS = r"([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)"


def popularity_phrase(titles: pd.Series) -> pd.Series:
    return titles.str.extract(_PARENTHETICAL, expand=False).fillna(NO_PHRASE)


def description_closing(descriptions: pd.Series) -> pd.Series:
    trimmed = descriptions.str.strip().str.rstrip(".")
    return trimmed.str.split(_SENTENCE_BREAK, regex=True).str[-1]


def price_position(frame: pd.DataFrame) -> pd.Series:
    """Where the price sits inside its query's filter window, from 0 to 1."""
    width = frame["filter_price_max"] - frame["filter_price_min"]
    return (frame["price"] - frame["filter_price_min"]) / width


def price_rank(frame: pd.DataFrame) -> pd.Series:
    """Where the price ranks among the products the same query actually showed."""
    return frame.groupby("query_id")["price"].rank(pct=True)


def package_ounces(frame: pd.DataFrame) -> pd.Series:
    """``package_size`` converted to ounces; missing for countable units."""
    amounts = frame["package_size"].str.extract(_LEADING_AMOUNT, expand=False).astype(float)
    return amounts * frame["unit_of_measure"].map(OUNCES_PER_UNIT)


def package_volume(frame: pd.DataFrame) -> pd.Series:
    """Cubic inches from the ``length x width x height`` string in ``dimensions_in``."""
    sides = frame["dimensions_in"].str.extract(_DIMENSIONS).astype(float)
    return sides[0] * sides[1] * sides[2]


def month_of_year(timestamps: pd.Series) -> pd.Series:
    """The calendar month, pooling every year: enero de 2024, 2025 y 2026 juntos."""
    months = pd.to_datetime(timestamps).dt.month
    return months.map(lambda m: f"{m:02d} {MONTH_NAMES[m - 1]}")


def day_of_week(timestamps: pd.Series) -> pd.Series:
    """The weekday, pooling every week of the dataset."""
    days = pd.to_datetime(timestamps).dt.dayofweek
    return days.map(lambda i: f"{i + 1} {WEEKDAY_NAMES[i]}")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["popularity_phrase"] = popularity_phrase(frame["title"])
    frame["description_closing"] = description_closing(frame["description"])
    frame["price_position"] = price_position(frame)
    frame["price_rank"] = price_rank(frame)
    frame["package_oz"] = package_ounces(frame)
    frame["volume_in3"] = package_volume(frame)
    stamps = pd.to_datetime(frame["timestamp"])
    frame["month_of_year"] = month_of_year(frame["timestamp"])
    frame["day_of_week"] = day_of_week(frame["timestamp"])
    frame["day_of_month"] = stamps.dt.day
    frame["hour"] = stamps.dt.hour
    return frame
