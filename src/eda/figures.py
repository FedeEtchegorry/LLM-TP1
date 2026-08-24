"""SVG figures for the write-up, drawn from computed data.

The two charts in ``docs/EDA.md`` were originally emitted by a throwaway script
holding a pasted copy of the numbers, so the repository could not regenerate
them and nothing stopped them from disagreeing with the tables. Here the numbers
come from :mod:`src.eda.curves` -- the same functions
:mod:`src.eda.run_structure` prints -- so figure and table always match.

Each file is standalone: literal colours and an opaque background rectangle, so
it renders correctly wherever it is embedded, including a Markdown preview that
supplies no stylesheet of its own.
"""

from __future__ import annotations

import math
from pathlib import Path

from src.eda.curves import AdditivitySeries, PriceCurve

INK = "#0e1116"
MUTED = "#7f8a95"
LABEL = "#4b545e"
GRID = "#e9edf1"
AXIS = "#c8d1da"
SURFACE = "#ffffff"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

_STYLE = f"""<style>
  .gr{{stroke:{GRID};stroke-width:1}}
  .ax{{stroke:{AXIS};stroke-width:1.25}}
  .tk{{fill:{MUTED};font-size:12px;font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace}}
  .tk.sm{{font-size:10px}}
  .tk.rl{{font-size:11px;letter-spacing:.04em;font-family:Archivo,Helvetica,Arial,sans-serif}}
  .axl{{fill:{LABEL};font-size:12.5px;font-family:Archivo,Helvetica,Arial,sans-serif}}
  .ln{{fill:none;stroke:{SERIES_1};stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
  .band{{fill:rgba(42,120,214,.15);stroke:none}}
  .dot{{fill:{SERIES_1};stroke:{SURFACE};stroke-width:2}}
  .halo{{fill:transparent}}
  .ref{{stroke:{MUTED};stroke-width:1.25;stroke-dasharray:5 4}}
  .pk{{fill:{INK};font-size:12.5px;font-weight:600;font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace}}
  .bar{{fill:{SERIES_1}}}
  .pred{{fill:none;stroke:{SERIES_2};stroke-width:2;stroke-dasharray:6 4;stroke-linecap:round}}
  .pdot{{fill:{SERIES_2};stroke:{SURFACE};stroke-width:1.5}}
  .lg{{font-size:12.5px;font-family:Archivo,Helvetica,Arial,sans-serif;fill:{LABEL}}}
</style>"""


def _ticks(low: float, high: float, step: float) -> list[float]:
    """Tick values at ``step`` intervals covering ``[low, high]``."""

    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    return [index * step for index in range(first, last + 1)]


def _bounds(values: list[float], step: float, pad: float = 0.5) -> tuple[float, float]:
    """Round the data range outward to a multiple of ``step``, plus padding."""

    low = (math.floor(min(values) / step) - pad) * step
    high = (math.ceil(max(values) / step) + pad) * step
    return low, high


def render_price_curve(curve: PriceCurve) -> str:
    """Purchase rate against ``price_pct`` within one tier: the inverted U."""

    lower, upper = curve.confidence_band()
    width, height = 760, 360
    left, right, top, bottom = 64, 26, 26, 58

    x_low, x_high = 0.0, float(curve.centers.max()) + 0.06
    y_low, y_high = _bounds(list(lower) + list(upper) + [curve.baseline], 0.1)

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (width - left - right)

    def sy(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * (height - top - bottom)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Probability of '
        f'purchase against price_pct within tier A, an inverted U peaking near 0.5" '
        f'class="chart">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>',
        _STYLE,
        "<title>P(bought) vs price_pct within tier A</title>",
    ]
    for value in _ticks(y_low, y_high, 0.1):
        out.append(
            f'<line class="gr" x1="{left}" y1="{sy(value):.1f}" '
            f'x2="{width - right}" y2="{sy(value):.1f}"/>'
        )
        out.append(
            f'<text class="tk" x="{left - 10}" y="{sy(value) + 4:.1f}" '
            f'text-anchor="end">{value:.1f}</text>'
        )
    for value in _ticks(x_low, x_high, 0.2):
        out.append(
            f'<text class="tk" x="{sx(value):.1f}" y="{height - bottom + 22}" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    out.append(
        f'<line class="ax" x1="{left}" y1="{height - bottom}" '
        f'x2="{width - right}" y2="{height - bottom}"/>'
    )

    upper_edge = " ".join(
        f"{sx(x):.1f},{sy(min(y_high, value)):.1f}"
        for x, value in zip(curve.centers, upper, strict=True)
    )
    lower_edge = " ".join(
        f"{sx(x):.1f},{sy(max(y_low, value)):.1f}"
        for x, value in reversed(list(zip(curve.centers, lower, strict=True)))
    )
    out.append(f'<polygon class="band" points="{upper_edge} {lower_edge}"/>')
    out.append(
        f'<line class="ref" x1="{left}" y1="{sy(curve.baseline):.1f}" '
        f'x2="{width - right}" y2="{sy(curve.baseline):.1f}"/>'
    )
    out.append(
        f'<text class="tk rl" x="{width - right}" y="{sy(curve.baseline) - 9:.1f}" '
        f'text-anchor="end">tier-A average  {curve.baseline:.3f}</text>'
    )
    out.append(
        '<polyline class="ln" points="'
        + " ".join(
            f"{sx(x):.1f},{sy(rate):.1f}"
            for x, rate in zip(curve.centers, curve.rates, strict=True)
        )
        + '"/>'
    )
    for x, rate, count in zip(curve.centers, curve.rates, curve.counts, strict=True):
        out.append(
            f'<g class="hit"><title>price_pct = {x:.2f} - P(bought) {rate:.3f} '
            f"over {count} products</title>"
            f'<circle class="halo" cx="{sx(x):.1f}" cy="{sy(rate):.1f}" r="13"/>'
            f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(rate):.1f}" r="4.5"/></g>'
        )

    # Label the peak and the two ends only; a number on every point is noise.
    peak = int(curve.rates.argmax())
    for index, anchor, offset in (
        (peak, "middle", -16),
        (0, "start", 22),
        (len(curve.rates) - 1, "end", 24),
    ):
        nudge = 6 if anchor == "start" else (-6 if anchor == "end" else 0)
        out.append(
            f'<text class="pk" x="{sx(curve.centers[index]) + nudge:.1f}" '
            f'y="{sy(curve.rates[index]) + offset:.1f}" '
            f'text-anchor="{anchor}">{curve.rates[index]:.2f}</text>'
        )
    out.append(
        f'<text class="axl" x="{(left + width - right) / 2:.1f}" y="{height - 10}" '
        f'text-anchor="middle">price_pct  -  where the price sits inside the '
        f"shopper’s filter window</text>"
    )
    out.append(
        f'<text class="axl" transform="translate(16,{(top + height - bottom) / 2:.1f}) '
        f'rotate(-90)" text-anchor="middle">P(bought)</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def render_additivity(series: AdditivitySeries, min_queries: int = 15) -> str:
    """Mean purchases per query against its tier-A count, with the additive line."""

    keep = [
        index
        for index, queries in enumerate(series.queries)
        if queries >= min_queries
    ]
    counts = [int(series.tier_a_counts[index]) for index in keep]
    means = [float(series.means[index]) for index in keep]
    queries = [int(series.queries[index]) for index in keep]
    predicted = [series.slope * count for count in counts]

    width, height = 760, 372
    left, right, top, bottom = 64, 26, 30, 76
    y_high = max(max(means), max(predicted)) * 1.12

    def sy(value: float) -> float:
        return top + (y_high - value) / y_high * (height - top - bottom)

    def cx(position: int) -> float:
        return left + (position + 0.5) * (width - left - right) / len(counts)

    bar_width = (width - left - right) / len(counts) * 0.52
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Mean purchases '
        f'per query rises linearly with the number of tier A products" class="chart">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>',
        _STYLE,
        "<title>Purchases per query vs tier-A count</title>",
    ]
    for value in _ticks(0.0, y_high, 0.5):
        out.append(
            f'<line class="gr" x1="{left}" y1="{sy(value):.1f}" '
            f'x2="{width - right}" y2="{sy(value):.1f}"/>'
        )
        out.append(
            f'<text class="tk" x="{left - 10}" y="{sy(value) + 4:.1f}" '
            f'text-anchor="end">{value:g}</text>'
        )
    for position, (count, mean, n_queries) in enumerate(
        zip(counts, means, queries, strict=True)
    ):
        bar_height = sy(0) - sy(mean)
        out.append(
            f'<g class="hit"><title>{n_queries} queries showed {count} tier-A '
            f"product(s) - {mean:.2f} purchases on average</title>"
            f'<rect class="bar" x="{cx(position) - bar_width / 2:.1f}" '
            f'y="{sy(mean):.1f}" width="{bar_width:.1f}" '
            f'height="{max(bar_height, 1.5):.1f}" rx="3"/></g>'
        )
        out.append(
            f'<text class="pk" x="{cx(position):.1f}" y="{sy(mean) - 10:.1f}" '
            f'text-anchor="middle">{mean:.2f}</text>'
        )
        out.append(
            f'<text class="tk" x="{cx(position):.1f}" y="{height - bottom + 22}" '
            f'text-anchor="middle">{count}</text>'
        )
        out.append(
            f'<text class="tk sm" x="{cx(position):.1f}" y="{height - bottom + 38}" '
            f'text-anchor="middle">n={n_queries}</text>'
        )
    out.append(
        '<polyline class="pred" points="'
        + " ".join(
            f"{cx(position):.1f},{sy(value):.1f}"
            for position, value in enumerate(predicted)
        )
        + '"/>'
    )
    for position, (count, value) in enumerate(zip(counts, predicted, strict=True)):
        out.append(
            f'<g class="hit"><title>{series.slope:.2f} x {count} = {value:.2f} '
            f"predicted if products do not compete</title>"
            f'<circle class="pdot" cx="{cx(position):.1f}" cy="{sy(value):.1f}" '
            f'r="4"/></g>'
        )
    out.append(
        f'<line class="ax" x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" '
        f'y2="{sy(0):.1f}"/>'
    )
    out.append(
        f'<line class="pred" x1="{left + 8}" y1="{top + 4}" x2="{left + 34}" '
        f'y2="{top + 4}"/>'
        f'<text class="lg" x="{left + 42}" y="{top + 8}">'
        f"{series.slope:.2f} x (tier-A count), the prediction if products do not "
        f"compete</text>"
    )
    out.append(
        f'<text class="axl" x="{(left + width - right) / 2:.1f}" '
        f'y="{height - bottom + 60}" text-anchor="middle">number of tier-A products '
        f"shown in the query</text>"
    )
    out.append(
        f'<text class="axl" transform="translate(16,{(top + height - bottom) / 2:.1f}) '
        f'rotate(-90)" text-anchor="middle">mean purchases</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def write_figure(path: Path, markup: str) -> Path:
    """Write ``markup`` to ``path``, creating the directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markup + "\n", encoding="utf-8")
    return path
