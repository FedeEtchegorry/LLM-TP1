"""Structural and data-quality facts about the dataset.

Run with ``python -m src.eda.run_structure``.

Everything printed here is a direct count over the CSV -- no model, no split.
These are the claims the EDA write-up cites, emitted so they can be re-checked.
"""

from __future__ import annotations

import argparse
import collections
import statistics
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from src.eda.curves import additivity_series, price_pct_curve
from src.eda.dataset import (
    DEFAULT_DATASET_PATH,
    NO_PHRASE,
    TEXT_FIELDS,
    BtrData,
    load_btr_data,
    tokenize,
)

TIER_ORDER = ("A", "B", "C")


def tier_table(data: BtrData) -> str:
    counts: dict[str, list[int]] = {tier: [0, 0] for tier in TIER_ORDER}
    for tier, target in zip(data.oracle_tier, data.target, strict=True):
        counts[tier][0] += int(target)
        counts[tier][1] += 1
    lines = ["| Tier | Rows | Bought | Rate |", "|---|---:|---:|---:|"]
    for tier in TIER_ORDER:
        bought, rows = counts[tier]
        lines.append(f"| {tier} | {rows:,} | {bought:,} | {bought / rows:.3f} |")
    return "\n".join(lines)


def phrase_table(data: BtrData) -> str:
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    tiers: dict[str, str] = {}
    for phrase, tier, target in zip(
        data.popularity_phrase, data.oracle_tier, data.target, strict=True
    ):
        counts[phrase][0] += int(target)
        counts[phrase][1] += 1
        tiers[phrase] = tier
    ordered = sorted(counts.items(), key=lambda item: -item[1][0] / item[1][1])
    lines = ["| Phrase | Bought / rows | Rate | Tier |", "|---|---:|---:|:--:|"]
    for phrase, (bought, rows) in ordered:
        label = "*(no parenthetical)*" if phrase == NO_PHRASE else phrase
        lines.append(
            f"| {label} | {bought} / {rows} | {bought / rows:.3f} | {tiers[phrase]} |"
        )
    return "\n".join(lines)


def query_additivity_table(data: BtrData) -> str:
    """Purchases per query against the number of tier-A products it contains."""

    series = additivity_series(data)
    lines = [
        "| Tier-A products in query | Queries | Total purchases | Mean |",
        "|---:|---:|---:|---:|",
    ]
    for count, queries, purchases, mean in zip(
        series.tier_a_counts, series.queries, series.purchases, series.means, strict=True
    ):
        lines.append(f"| {count} | {queries:,} | {purchases:,} | {mean:.2f} |")
    lines.append(f"| **total** | **{series.total_queries:,}** | | |")
    return "\n".join(lines)


def competition_table(data: BtrData) -> str:
    """Tier-A buy rate against the number of tier-A rivals in the same query."""

    by_query: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for query, tier in zip(data.query_ids, data.oracle_tier, strict=True):
        by_query[query][0] += int(tier == "A")
    grouped: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for query, tier, target in zip(
        data.query_ids, data.oracle_tier, data.target, strict=True
    ):
        if tier != "A":
            continue
        rivals = by_query[query][0] - 1
        grouped[rivals][0] += int(target)
        grouped[rivals][1] += 1
    lines = ["| Other tier-A products | Bought / rows | Rate |", "|---:|---:|---:|"]
    for rivals in sorted(grouped):
        bought, rows = grouped[rivals]
        if rows < 30:
            continue
        lines.append(f"| {rivals} | {bought} / {rows} | {bought / rows:.3f} |")
    return "\n".join(lines)


def price_pct_deciles(data: BtrData, tier: str = "A", n_bins: int = 10) -> str:
    """The inverted U as a table; :mod:`src.eda.figures` draws the same numbers."""

    curve = price_pct_curve(data, tier=tier, n_bins=n_bins)
    lines = ["| Decile | Range | Bought / rows | Rate |", "|---:|---|---:|---:|"]
    for index, (rate, rows) in enumerate(zip(curve.rates, curve.counts, strict=True)):
        lines.append(
            f"| {index} | {curve.edges[index]:.2f} - {curve.edges[index + 1]:.2f} "
            f"| {round(rate * rows)} / {rows} | {rate:.3f} |"
        )
    return "\n".join(lines)


def quarterly_table(data: BtrData) -> str:
    grouped: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for stamp, target in zip(data.timestamps, data.target, strict=True):
        moment = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
        key = f"{moment.year}Q{(moment.month - 1) // 3 + 1}"
        grouped[key][0] += int(target)
        grouped[key][1] += 1
    lines = ["| Quarter | Rows | Buy rate |", "|---|---:|---:|"]
    for key in sorted(grouped):
        bought, rows = grouped[key]
        lines.append(f"| {key} | {rows:,} | {bought / rows:.3f} |")
    return "\n".join(lines)


def timestamp_dispersion(data: BtrData) -> tuple[float, float]:
    """Median within-query span and median within-query pairwise difference.

    The two differ substantially, so the write-up must say which it quotes.
    """

    by_query: dict[str, list[datetime]] = collections.defaultdict(list)
    for query, stamp in zip(data.query_ids, data.timestamps, strict=True):
        by_query[query].append(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    spans, pairwise = [], []
    for moments in by_query.values():
        if len(moments) < 2:
            continue
        ordered = sorted(moments)
        spans.append((ordered[-1] - ordered[0]).days)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairwise.append((ordered[j] - ordered[i]).days)
    return statistics.median(spans), statistics.median(pairwise)


def vocabulary_size(data: BtrData) -> int:
    """Distinct tokens over ``title + description + ingredients``."""

    words: set[str] = set()
    for text in data.text:
        words.update(tokenize(text))
    return len(words)


def universal_words(data: BtrData) -> list[str]:
    """Tokens present in every row -- template scaffolding with zero variance."""

    counts: collections.Counter[str] = collections.Counter()
    for text in data.text:
        counts.update(set(tokenize(text)))
    return sorted(word for word, count in counts.items() if count == len(data))


def leakage_crosstab(path: Path) -> str:
    """Cross-tabulate ``cart`` against ``bought`` straight from the CSV.

    ``cart`` is excluded from :class:`~src.eda.dataset.BtrData` on purpose, so
    this reads the file directly to show why.
    """

    import csv

    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[(row["cart"], row["bought"])] += 1
    lines = ["| cart | bought | rows |", "|---|---|---:|"]
    for cart in ("false", "true"):
        for bought in ("false", "true"):
            lines.append(f"| {cart} | {bought} | {counts[(cart, bought)]:,} |")
    return "\n".join(lines)


def filter_redundancy(path: Path) -> str:
    import csv

    same_category = same_storage = inside_window = total = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            same_category += row["category"] == row["filter_category"]
            same_storage += row["storage_type"] == row["filter_storage_type"]
            inside_window += (
                float(row["filter_price_min"])
                <= float(row["price"])
                <= float(row["filter_price_max"])
            )
    return "\n".join(
        [
            "| Check | Rows satisfying |",
            "|---|---:|",
            f"| `category == filter_category` | {same_category:,} / {total:,} |",
            f"| `storage_type == filter_storage_type` | {same_storage:,} / {total:,} |",
            f"| `filter_price_min <= price <= filter_price_max` | {inside_window:,} / {total:,} |",
        ]
    )


def nutrition_sentinel(data: BtrData) -> str:
    grouped: collections.Counter[str] = collections.Counter()
    categories = data.categorical["category"]
    for index, missing in enumerate(data.nutrition_missing):
        if missing:
            grouped[categories[index]] += 1
    lines = ["| Category | Rows with nutrition_score = 0 |", "|---|---:|"]
    for category, count in grouped.most_common():
        lines.append(f"| {category} | {count:,} |")
    lines.append(f"| **total** | **{sum(grouped.values()):,}** |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args(argv)

    data = load_btr_data(args.dataset)
    span, pairwise = timestamp_dispersion(data)
    words = universal_words(data)

    sections = [
        (f"Rows {len(data):,} · queries {len(set(data.query_ids)):,} · "
         f"positive rate {data.positive_rate:.4f}", ""),
        ("Buy rate by hand-assigned tier", tier_table(data)),
        ("Every popularity phrase", phrase_table(data)),
        ("`cart` against `bought`", leakage_crosstab(args.dataset)),
        ("Filter columns are redundant", filter_redundancy(args.dataset)),
        ("`nutrition_score = 0` by category", nutrition_sentinel(data)),
        ("Competition between tier-A products", competition_table(data)),
        ("Purchases per query", query_additivity_table(data)),
        ("price_pct deciles within tier A", price_pct_deciles(data)),
        ("Buy rate by quarter", quarterly_table(data)),
        (
            "Timestamp dispersion within a query",
            f"- median within-query span (max − min): **{span:.0f} days**\n"
            f"- median within-query pairwise difference: **{pairwise:.0f} days**",
        ),
        (
            f"Vocabulary over {' + '.join(TEXT_FIELDS)}",
            f"- distinct lowercase alphanumeric tokens: **{vocabulary_size(data)}**\n"
            f"- tokens present in every row ({len(words)}): "
            + ", ".join(f"`{word}`" for word in words),
        ),
    ]
    for heading, body in sections:
        print(f"\n### {heading}\n")
        if body:
            print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
