"""Evidence, figure by figure, for what each column is worth."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.charts import (
    bar_chart,
    error_arms,
    interval_arms,
    matrix_chart,
    response_bars,
    response_grid,
    save_figure,
    stacked_bar_chart,
)
from src.eda.curves import (
    binned_response,
    level_rates,
    numeric_curve,
    quarter_rates,
    query_spans,
)
from src.eda.dataset import DEFAULT_DATASET_PATH, NO_PHRASE, BtrData, load_btr_data
from src.eda.evaluation import EvaluationResult
from src.eda.variable_evidence import (
    Column,
    cart_crosstab,
    fold_composition,
    leakage_results,
    leave_one_out_results,
    numeric_correlations,
    univariate_results,
)
from src.partitions import DataPartitions, build_query_partitions

DEFAULT_FIGURE_DIRECTORY = Path("docs/figures")

NUMERIC_PANEL_FIELDS = (
    ("price_pct", "position in filter window"),
    ("price", "dollars"),
    ("net_weight_oz", "ounces"),
    ("nutrition_score", "score, 18-99"),
    ("package_value", "number in package_size"),
    ("volume_in3", "cubic inches"),
    ("month_index", "months since first event"),
)

CORRELATION_FIELDS = (
    "price",
    "price_pct",
    "net_weight_oz",
    "nutrition_score",
    "package_value",
    "volume_in3",
    "month_index",
)


def _curve_frame(curve) -> pd.DataFrame:
    """A binned response curve as a frame, with its confidence band."""
    lower, upper = curve.confidence_band()
    return pd.DataFrame(
        {
            "bin": [f"{center:.3g}" for center in curve.centers],
            "rate": curve.rates,
            "lower": np.clip(lower, 0.0, 1.0),
            "upper": np.clip(upper, 0.0, 1.0),
            "rows": curve.counts,
        }
    )


def phrase_figure(data: BtrData, out: Path) -> Path:
    """Every popularity phrase, its buy rate and how certain that rate is."""
    rates = level_rates(data.popularity_phrase, data.target)
    frame = pd.DataFrame(
        {
            "phrase": [
                "(no parenthetical)" if label == NO_PHRASE else label
                for label in rates.labels
            ],
            "rate": rates.rates,
            "lower": rates.lower,
            "upper": rates.upper,
            "rows": rates.counts.astype(int),
        }
    )
    highest_zero = float(frame.loc[frame["rate"] == 0, "upper"].max())
    bar_chart(
        frame,
        label="phrase",
        value="rate",
        yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
        color_per_bar=True,
        title="One phrase in the title carries most of the label",
        subtitle=(
            f"buy rate per trailing parenthetical, with a 95% Wilson interval. The "
            f"four leftmost phrases are tier A, the next four tier B, and the "
            f"twelve at zero are tier C -- bounded above by {highest_zero:.3f}, so "
            f"the sample rules out any real rate above that"
        ),
        ylabel="P(bought)",
        value_fmt="%.3f",
        reference=data.positive_rate,
        reference_label=f"dataset average {data.positive_rate:.4f}",
        figsize=(12, 6),
    )
    return save_figure(out / "phrase-buy-rates.png")


def univariate_figure(
    results: Iterable[tuple[Column, EvaluationResult]], data: BtrData, out: Path
) -> Path:
    """Average precision of each column on its own, best encoding, five folds."""
    ordered = sorted(results, key=lambda item: -item[1].average_precision_mean)
    frame = pd.DataFrame(
        {
            "column": [column.name for column, _ in ordered],
            "ap": [result.average_precision_mean for _, result in ordered],
            "sd": [result.average_precision_std for _, result in ordered],
            "disposition": [
                "kept" if column.keep else "dropped" for column, _ in ordered
            ],
        }
    )
    bar_chart(
        frame,
        label="column",
        value="ap",
        series="disposition",
        legend_title="disposition",
        yerr=error_arms(frame["ap"], frame["sd"]),
        title="What each column is worth on its own",
        subtitle=(
            "average precision, 5 query-grouped folds, mean ± sd. Numeric columns "
            "get a linear term and quantile buckets. Every dropped column lands on "
            "the random line; nutrition_score and storage_type do too, and are kept "
            "for reasons this chart does not measure"
        ),
        ylabel="average precision (PR-AUC)",
        value_fmt="%.3f",
        reference=data.positive_rate,
        reference_label=f"random {data.positive_rate:.3f}",
        figsize=(12, 6),
    )
    return save_figure(out / "variable-univariate-ap.png")


def leave_one_out_figure(
    full: EvaluationResult,
    removed: Iterable[tuple[Column, EvaluationResult]],
    out: Path,
) -> Path:
    """Change in average precision when one column leaves the full feature set."""
    deltas = sorted(
        (
            (column, result.average_precision_mean - full.average_precision_mean, result)
            for column, result in removed
        ),
        key=lambda item: item[1],
    )
    frame = pd.DataFrame(
        {
            "column": [column.name for column, _, _ in deltas],
            "delta": [delta for _, delta, _ in deltas],
            "sd": [result.average_precision_std for _, _, result in deltas],
            "verdict": [
                "costs AP to remove" if delta < -0.002 else "free to remove"
                for _, delta, _ in deltas
            ],
        }
    )
    bar_chart(
        frame,
        label="column",
        value="delta",
        series="verdict",
        legend_title="verdict",
        yerr=error_arms(frame["delta"], frame["sd"], floor=None),
        title="What each column adds that the others do not",
        subtitle=(
            f"change in average precision when the column is removed from the full "
            f"tabular set (AP {full.average_precision_mean:.3f} ± "
            f"{full.average_precision_std:.3f}). Bars below zero cost something to "
            f"remove; bars at or above zero are free, and all five dropped columns "
            f"are in that group"
        ),
        ylabel="Δ average precision when removed",
        value_fmt="%+.3f",
        figsize=(12, 6),
    )
    return save_figure(out / "variable-leave-one-out.png")


def leakage_figures(
    data: BtrData,
    partitions: DataPartitions,
    dataset_path: Path,
    out: Path,
) -> tuple[Path, Path, list[tuple[str, EvaluationResult]]]:
    """The empty cell in the ``cart`` cross-tabulation, and what it is worth."""
    counts = cart_crosstab(dataset_path)
    frame = pd.DataFrame(
        [
            [counts[("false", "false")], counts[("false", "true")]],
            [counts[("true", "false")], counts[("true", "true")]],
        ],
        index=["cart = false", "cart = true"],
        columns=["bought = false", "bought = true"],
    )
    matrix_chart(
        frame,
        title="`cart` is a later step of the same funnel",
        subtitle=(
            "no impression was ever bought without being carted first, so a model "
            "given this column learns 'not carted, therefore not bought' -- a rule "
            "that cannot be evaluated when a search page renders"
        ),
        value_fmt="{:,.0f}",
        colorbar_label="impressions",
        highlight=(0, 1),
        highlight_note="never happens",
        figsize=(8, 5),
    )
    grid_path = save_figure(out / "cart-leakage.png")

    scores = leakage_results(data, partitions, dataset_path)
    panel = pd.DataFrame(
        {
            "feature_set": [name for name, _ in scores],
            "ap": [result.average_precision_mean for _, result in scores],
            "sd": [result.average_precision_std for _, result in scores],
            "available": [
                "impression-time" if index == 0 else "not available at render time"
                for index, _ in enumerate(scores)
            ],
        }
    )
    bar_chart(
        panel,
        label="feature_set",
        value="ap",
        series="available",
        legend_title="available when the page renders?",
        yerr=error_arms(panel["ap"], panel["sd"]),
        title="What the leak buys, and why it is worthless",
        subtitle=(
            "average precision over the same folds. Adding cart is worth +0.133 AP "
            "and none of it is available to a deployed model: at the moment a "
            "search page renders, nothing has been carted yet"
        ),
        ylabel="average precision (PR-AUC)",
        value_fmt="%.3f",
        reference=data.positive_rate,
        reference_label=f"random {data.positive_rate:.3f}",
    )
    return grid_path, save_figure(out / "cart-leakage-ap.png"), scores


def numeric_shapes_figure(data: BtrData, out: Path, tier: str = "A") -> Path:
    """Response curves for every numeric column, on one shared scale."""
    panels = [
        (f"{field}\n({units})", _curve_frame(numeric_curve(data, field, tier=tier)))
        for field, units in NUMERIC_PANEL_FIELDS
    ]
    tier_rows = sum(1 for value in data.oracle_tier if value == tier)
    tier_rate = float(
        data.target[[value == tier for value in data.oracle_tier]].mean()
    )
    response_grid(
        panels,
        title="Price bends. Nothing else does.",
        subtitle=(
            f"P(bought) by decile of each numeric column, within tier {tier} "
            f"(n = {tier_rows:,}); the dashed line is the tier average and the "
            f"whiskers are 95% intervals. price_pct and price carry the same hump "
            f"-- one is derived from the other -- and the remaining five, including "
            f"the three parsed only so that dropping them is a measurement, are "
            f"flat inside their own noise"
        ),
        reference=tier_rate,
    )
    return save_figure(out / "numeric-response-grid.png")


def categorical_levels_figures(data: BtrData, out: Path) -> tuple[Path, Path]:
    """Buy rate per level for the categorical columns kept, then those dropped."""
    kept = ("category", "allergens", "storage_type", "unit_of_measure")
    dropped = ("brand", "country_of_origin")
    return (
        _levels_figure(
            data,
            kept,
            out / "levels-kept.png",
            title="The categorical columns that move the rate",
            subtitle=(
                "buy rate per level with a 95% Wilson interval. Eight levels clear "
                "the dataset average by more than their interval -- three in "
                "category and five in allergens -- and those are what the +0.028 "
                "and +0.012 AP rows in the baseline table are made of"
            ),
        ),
        _levels_figure(
            data,
            dropped,
            out / "levels-dropped.png",
            title="`brand` and `country_of_origin` say nothing",
            subtitle=(
                "24 of these 25 intervals contain the dataset average. One marginal "
                "escape out of 25 independent 95% intervals is what chance alone "
                "produces, and removing brand costs no average precision"
            ),
        ),
    )


def _levels_figure(
    data: BtrData,
    fields: tuple[str, ...],
    path: Path,
    *,
    title: str,
    subtitle: str,
) -> Path:
    rows = []
    for field in fields:
        rates = level_rates(data.categorical[field], data.target)
        for label, rate, low, high, count in zip(
            rates.labels, rates.rates, rates.lower, rates.upper, rates.counts,
            strict=True,
        ):
            spans = low <= data.positive_rate <= high
            rows.append(
                {
                    "level": f"{field}: {label}",
                    "rate": float(rate),
                    "lower": float(low),
                    "upper": float(high),
                    "rows": int(count),
                    "separates": "interval spans the average"
                    if spans
                    else "interval clears the average",
                }
            )
    frame = pd.DataFrame(rows)
    bar_chart(
        frame,
        label="level",
        value="rate",
        series="separates",
        legend_title="vs the dataset average",
        yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
        title=title,
        subtitle=subtitle,
        ylabel="P(bought)",
        value_fmt="%.3f",
        reference=data.positive_rate,
        reference_label=f"dataset average {data.positive_rate:.4f}",
        figsize=(14, 6),
    )
    return save_figure(path)


def nutrition_sentinel_figure(data: BtrData, out: Path) -> Path:
    """The spike at zero, and which categories it comes from."""
    raw = np.where(
        data.nutrition_missing == 1, 0.0, np.nan_to_num(data.numeric["nutrition_score"])
    )
    categories = np.array(data.categorical["category"])
    non_food = np.isin(categories, ("Household", "Personal Care"))
    edges = np.linspace(0, 100, 21)
    frame = pd.DataFrame(
        {
            "Household and Personal Care": np.histogram(raw[non_food], bins=edges)[0],
            "every other category": np.histogram(raw[~non_food], bins=edges)[0],
        },
        index=[f"{int(edge)}" for edge in edges[:-1]],
    )
    stacked_bar_chart(
        frame,
        title="`nutrition_score = 0` means 'not applicable'",
        subtitle=(
            f"{int(data.nutrition_missing.sum()):,} rows sit at zero and every one "
            f"is Household or Personal Care; nothing at all falls between 0 and "
            f"{int(np.nanmin(data.numeric['nutrition_score'])):d}, where the genuine "
            f"range starts. A literal zero would tell the model these are the "
            f"worst-scoring products in the catalogue"
        ),
        xlabel="nutrition_score (bin lower edge)",
        ylabel="rows",
        legend_title="category",
        rotate=False,
    )
    return save_figure(out / "nutrition-sentinel.png")


def timestamp_figures(data: BtrData, out: Path) -> tuple[Path, Path]:
    """Buy rate over time, and how far apart one query's events are."""
    quarters = quarter_rates(data)
    frame = pd.DataFrame(
        {
            "quarter": list(quarters.labels),
            "rate": quarters.rates,
            "lower": quarters.lower,
            "upper": quarters.upper,
            "rows": quarters.counts.astype(int),
        }
    )
    bar_chart(
        frame,
        label="quarter",
        value="rate",
        yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
        title="No drift worth modelling",
        subtitle=(
            "buy rate per quarter with a 95% Wilson interval. Every complete "
            "quarter's interval overlaps the dataset average; the last bar is a "
            "partial quarter of 113 rows and its interval is wide enough to "
            "swallow the apparent rise"
        ),
        xlabel="quarter",
        ylabel="P(bought)",
        value_fmt="%.3f",
        reference=data.positive_rate,
        reference_label=f"dataset average {data.positive_rate:.4f}",
    )
    curve_path = save_figure(out / "buy-rate-by-quarter.png")

    spans = query_spans(data)
    edges = np.linspace(0, spans.max(), 21)
    histogram = pd.DataFrame(
        {"queries": np.histogram(spans, bins=edges)[0]},
        index=[f"{int(edge)}" for edge in edges[:-1]],
    )
    stacked_bar_chart(
        histogram,
        title="A `query_id` is not a browsing session",
        subtitle=(
            f"days between the first and last event inside one query; median "
            f"{np.median(spans):.0f} days across {len(spans):,} queries. Sorting by "
            f"time and cutting would split almost every query down the middle -- "
            f"the exact leak the grouping exists to prevent"
        ),
        xlabel="span of a query, in days (bin lower edge)",
        ylabel="queries",
        legend_title="",
        rotate=False,
    )
    return curve_path, save_figure(out / "query-time-span.png")


def correlation_figure(data: BtrData, out: Path) -> Path:
    """Correlation between the numeric columns: only the derived pair is related."""
    frame = pd.DataFrame(
        numeric_correlations(data, CORRELATION_FIELDS),
        index=list(CORRELATION_FIELDS),
        columns=list(CORRELATION_FIELDS),
    )
    matrix_chart(
        frame,
        title="The numeric columns are unrelated to each other",
        subtitle=(
            "Pearson correlation. The only strong pair is price with price_pct, "
            "which is expected: one is derived from the other. There is no "
            "structure here for a rotation to compress, so PCA over these columns "
            "is a change of basis with no compression to offer"
        ),
        colorbar_label="Pearson r",
        figsize=(9, 7),
    )
    return save_figure(out / "numeric-correlation.png")


def partition_figure(data: BtrData, partitions: DataPartitions, out: Path) -> Path:
    """The split itself: rows per part, purchase rate per part, queries never shared."""
    composition = fold_composition(data, partitions)
    frame = pd.DataFrame(
        {
            "train": composition.train,
            "validation": composition.validation,
            "test (never scored)": composition.test,
        },
        index=list(composition.labels),
    )
    stacked_bar_chart(
        frame,
        title="Query-grouped, stratified, with a test set nobody touches",
        subtitle=(
            f"impressions per fold, and the buy rate in each part. "
            f"{composition.shared_queries} queries appear on both sides of any "
            f"fold, and the same test holdout -- buy rate "
            f"{composition.test_positive_rate:.4f} -- sits outside all five"
        ),
        xlabel="fold",
        ylabel="impressions",
        legend_title="split",
        value_fmt="%.0f",
        inside_labels=True,
        annotations=[
            f"train {train:.4f} / valid {validation:.4f}"
            for train, validation in composition.positive_rates
        ],
        rotate=False,
    )
    return save_figure(out / "partition-folds.png")


def filter_redundancy_figure(data: BtrData, out: Path) -> Path:
    """Where the price falls inside the window the shopper asked for."""
    frame = _curve_frame(binned_response(data.numeric["price_pct"], data.target, 12))
    response_bars(
        frame,
        label="bin",
        value="rate",
        yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
        title="The filter is always satisfied; the position inside it is not",
        subtitle=(
            "every row obeys its own price filter in 10,000 rows out of 10,000, so "
            "the three filter columns carry nothing as columns -- but where the "
            "price sits between them separates buyers from non-buyers even before "
            "the popularity tier is held fixed"
        ),
        xlabel=(
            "price_pct  —  (price − filter_price_min) / "
            "(filter_price_max − filter_price_min)"
        ),
        reference=data.positive_rate,
        reference_label=f"dataset average {data.positive_rate:.4f}",
    )
    return save_figure(out / "price-pct-derived.png")


def markdown_table(
    rows: Iterable[tuple[str, EvaluationResult]], header: str, delta_from: float | None
) -> str:
    lines = [
        f"| {header} | ROC-AUC | PR-AUC (AP) |" + (" Δ AP |" if delta_from else ""),
        "|---|---:|---:|" + ("---:|" if delta_from else ""),
    ]
    for name, result in rows:
        delta = (
            f" {result.average_precision_mean - delta_from:+.3f} |"
            if delta_from
            else ""
        )
        lines.append(
            f"| {name} | {result.roc_auc_mean:.3f} ± {result.roc_auc_std:.3f} "
            f"| {result.average_precision_mean:.3f} ± "
            f"{result.average_precision_std:.3f} |{delta}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    data = load_btr_data(args.dataset)
    partitions = build_query_partitions(
        data.target.tolist(),
        data.query_ids,
        n_folds=args.folds,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )
    written: list[Path] = []

    print("### Descriptive figures\n")
    written.append(phrase_figure(data, args.out))
    written.append(filter_redundancy_figure(data, args.out))
    written.append(numeric_shapes_figure(data, args.out))
    written.extend(categorical_levels_figures(data, args.out))
    written.append(nutrition_sentinel_figure(data, args.out))
    written.extend(timestamp_figures(data, args.out))
    written.append(correlation_figure(data, args.out))
    written.append(partition_figure(data, partitions, args.out))

    print("Scoring each column on its own ...")
    univariate = univariate_results(data, partitions)
    written.append(univariate_figure(univariate, data, args.out))

    print("Scoring the full set with one column removed at a time ...")
    full, removed = leave_one_out_results(data, partitions)
    written.append(leave_one_out_figure(full, removed, args.out))

    print("Measuring the cart leak ...")
    grid_path, panel_path, leak_scores = leakage_figures(
        data, partitions, args.dataset, args.out
    )
    written.extend([grid_path, panel_path])

    print("\n--- Markdown: what each column is worth alone ---\n")
    print(
        markdown_table(
            (
                (f"{column.name} ({'keep' if column.keep else 'drop'})", result)
                for column, result in sorted(
                    univariate, key=lambda item: -item[1].average_precision_mean
                )
            ),
            "Column, alone",
            None,
        )
    )

    print("\n--- Markdown: leave-one-out over the full tabular set ---\n")
    print(
        markdown_table(
            [("full tabular set", full)]
            + [
                (f"without {column.name}", result)
                for column, result in sorted(
                    removed, key=lambda item: item[1].average_precision_mean
                )
            ],
            "Feature set",
            full.average_precision_mean,
        )
    )

    print("\n--- Markdown: the cart leak ---\n")
    print(markdown_table(leak_scores, "Feature set", None))

    print()
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
