"""Regenerate the two headline figures used by ``docs/EDA.md``."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eda.curves import additivity_series, price_pct_curve
from src.eda.dataset import DEFAULT_DATASET_PATH, load_btr_data
from src.eda.figures import draw_additivity, draw_price_curve, save_figure

DEFAULT_FIGURE_DIRECTORY = Path("docs/figures")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    parser.add_argument("--tier", default="A")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)

    data = load_btr_data(args.dataset)

    curve = price_pct_curve(data, tier=args.tier, n_bins=args.bins)
    series = additivity_series(data)

    written = [
        save_figure(
            args.out / "price-inverted-u.png", draw_price_curve(curve, tier=args.tier)
        ),
        save_figure(args.out / "purchases-additive.png", draw_additivity(series)),
    ]
    for path in written:
        print(f"wrote {path}")

    peak = int(curve.rates.argmax())
    print(
        f"\nprice_pct curve: peak {curve.rates[peak]:.3f} at "
        f"{curve.centers[peak]:.2f}, ends {curve.rates[0]:.3f} / "
        f"{curve.rates[-1]:.3f}, tier-{args.tier} average {curve.baseline:.3f}"
    )
    print(
        f"additivity: {series.total_queries:,} queries, "
        f"per-product slope {series.slope:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
