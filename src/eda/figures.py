"""The two headline figures for the write-up, drawn from computed data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.eda.charts import (
    FIGSIZE,
    caption,
    interval_arms,
    plt,
    response_bars,
    save_figure,
)
from src.eda.curves import AdditivitySeries, BinnedResponse

__all__ = ["draw_price_curve", "draw_additivity", "save_figure"]


def draw_price_curve(curve: BinnedResponse, tier: str = "A") -> Figure:
    """Purchase rate against ``price_pct`` within one tier: the inverted U."""
    lower, upper = curve.confidence_band()
    frame = pd.DataFrame(
        {
            "bin": [f"{center:.2f}" for center in curve.centers],
            "rate": curve.rates,
            "lower": np.clip(lower, 0.0, 1.0),
            "upper": np.clip(upper, 0.0, 1.0),
        }
    )
    return response_bars(
        frame,
        label="bin",
        value="rate",
        yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
        title="Price sits in an inverted U, not on a slope",
        subtitle=(
            f"P(bought) by decile of price_pct, within tier {tier} only "
            f"(n = {int(curve.counts.sum()):,}); whiskers are a 95% interval. The "
            f"rate rises from {curve.rates[0]:.2f} to {curve.rates.max():.2f} and "
            f"falls back to {curve.rates[-1]:.2f} -- a single linear coefficient "
            f"cannot represent that shape, a bucket table can"
        ),
        xlabel="price_pct  —  where the price sits inside the shopper's filter window",
        reference=curve.baseline,
        reference_label=f"tier-{tier} average {curve.baseline:.3f}",
        value_fmt="%.2f",
    )


def draw_additivity(series: AdditivitySeries, min_queries: int = 15) -> Figure:
    """Mean purchases per query against its tier-A count, with the additive line."""
    keep = [
        index
        for index, queries in enumerate(series.queries)
        if queries >= min_queries
    ]
    frame = pd.DataFrame(
        {
            "tier_a": [int(series.tier_a_counts[index]) for index in keep],
            "queries": [int(series.queries[index]) for index in keep],
            "mean": [float(series.means[index]) for index in keep],
        }
    )
    frame["predicted"] = series.slope * frame["tier_a"]

    plt.figure(figsize=FIGSIZE)
    positions = np.arange(len(frame))
    bars = plt.bar(positions, frame["mean"], label="observed mean purchases")
    plt.bar_label(bars, fmt="%.2f", label_type="edge", fontsize=8, color="black",
                  padding=2)
    plt.plot(
        positions,
        frame["predicted"],
        color="C1",
        linestyle="--",
        marker="o",
        label=(
            f"{series.slope:.2f} × (tier-A count), the prediction "
            f"if products do not compete"
        ),
    )
    plt.xticks(
        positions,
        [
            f"{count}\nn={n:,}"
            for count, n in zip(frame["tier_a"], frame["queries"], strict=True)
        ],
    )
    plt.xlabel("number of tier-A products shown in the query")
    plt.ylabel("mean purchases")
    plt.title("Purchases per query are additive")
    plt.legend(title="purchases per query")
    plt.margins(y=0.15)
    plt.tight_layout()
    caption(
        "each tier-A product contributes its own purchases whatever sits beside "
        "it, so there is no competition effect to model: the write-up scores each "
        "impression on its own rather than ranking a query's candidates against "
        "each other"
    )
    return plt.gcf()
