"""Charts for the write-up: stock Matplotlib, no theme, pandas frames in."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGSIZE = (10, 6)
"""The one figure size, so the charts sit together on a slide."""

DPI = 120


def save_figure(path: Path, figure: plt.Figure | None = None) -> Path:
    """Write the current figure to ``path`` as a PNG and close it."""
    figure = figure if figure is not None else plt.gcf()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def distinct_colors(count: int) -> list:
    """``count`` visually distinct colours for a chart with one bar per level."""
    palette = plt.cm.tab20.colors
    return [palette[index % len(palette)] for index in range(count)]


def error_arms(
    values: Sequence[float], spread: Sequence[float], *, floor: float | None = 0.0
) -> np.ndarray:
    """Symmetric error bars, clipped so they never cross ``floor``."""
    values = np.asarray(values, dtype=float)
    spread = np.asarray(spread, dtype=float)
    upper = spread
    lower = spread if floor is None else np.minimum(spread, values - floor)
    return np.vstack([np.maximum(lower, 0.0), upper])


def interval_arms(
    values: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> np.ndarray:
    """Error bars from absolute bounds, for an asymmetric interval like Wilson's."""
    values = np.asarray(values, dtype=float)
    return np.vstack(
        [
            np.maximum(values - np.asarray(lower, dtype=float), 0.0),
            np.maximum(np.asarray(upper, dtype=float) - values, 0.0),
        ]
    )


def bar_chart(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    ylabel: str,
    xlabel: str = "",
    subtitle: str = "",
    yerr: np.ndarray | None = None,
    series: str | None = None,
    legend_title: str = "",
    color_per_bar: bool = False,
    value_fmt: str | None = "%.3f",
    label_type: str = "edge",
    reference: float | None = None,
    reference_label: str = "",
    log: bool = False,
    figsize: tuple[float, float] = FIGSIZE,
) -> plt.Figure:
    """Vertical bars, one per row of ``frame``."""
    if frame.empty:
        raise ValueError("bar_chart needs at least one row")

    plt.figure(figsize=figsize)
    positions = np.arange(len(frame))
    heights = frame[value].to_numpy(dtype=float)

    if series is not None:
        for name, group in frame.groupby(series, sort=False):
            index = frame.index.get_indexer(group.index)
            bars = plt.bar(
                positions[index],
                group[value].to_numpy(dtype=float),
                yerr=None if yerr is None else yerr[:, index],
                capsize=5,
                label=str(name),
            )
            _label_bars(bars, value_fmt, label_type)
    else:
        bars = plt.bar(
            positions,
            heights,
            yerr=yerr,
            capsize=5,
            color=distinct_colors(len(frame)) if color_per_bar else None,
        )
        _label_bars(bars, value_fmt, label_type)

    if reference is not None:
        plt.axhline(
            reference, linestyle="--", linewidth=1, color="black",
            label=reference_label or None,
        )
    if series is not None or reference_label:
        plt.legend(title=legend_title or series or "")
    if log:
        plt.yscale("log")
    if heights.min() < 0:
        plt.axhline(0, linewidth=1, color="black")

    plt.xticks(positions, frame[label].astype(str), rotation=45, ha="right")
    _finish(title, subtitle, xlabel, ylabel)
    return plt.gcf()


def stacked_bar_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "",
    subtitle: str = "",
    legend_title: str = "",
    value_fmt: str | None = None,
    inside_labels: bool = False,
    annotations: Sequence[str] = (),
    rotate: bool = True,
    figsize: tuple[float, float] = FIGSIZE,
) -> plt.Figure:
    """One bar per row, split into the columns of ``frame`` and stacked."""
    plt.figure(figsize=figsize)
    positions = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for column in frame.columns:
        counts = frame[column].to_numpy(dtype=float)
        bars = plt.bar(positions, counts, bottom=bottom, label=str(column))
        if inside_labels and value_fmt is not None:
            plt.bar_label(
                bars, fmt=value_fmt, label_type="center", fontsize=8, color="white"
            )
        bottom = bottom + counts

    for position, note in zip(positions, annotations, strict=False):
        plt.text(position, bottom[position], f"\n{note}", ha="center", va="bottom",
                 fontsize=8)

    labels = frame.index.astype(str)
    if rotate:
        plt.xticks(positions, labels, rotation=45, ha="right")
    else:
        plt.xticks(positions, labels)
    if len(frame.columns) > 1 or legend_title:
        plt.legend(title=legend_title)
    plt.margins(y=0.15)
    _finish(title, subtitle, xlabel, ylabel)
    return plt.gcf()


def response_bars(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    title: str,
    xlabel: str,
    subtitle: str = "",
    ylabel: str = "P(bought)",
    yerr: np.ndarray | None = None,
    reference: float | None = None,
    reference_label: str = "",
    value_fmt: str | None = "%.3f",
    figsize: tuple[float, float] = FIGSIZE,
) -> plt.Figure:
    """Purchase rate per bin of a numeric column, as bars with an interval."""
    return bar_chart(
        frame,
        label=label,
        value=value,
        title=title,
        subtitle=subtitle,
        xlabel=xlabel,
        ylabel=ylabel,
        yerr=yerr,
        reference=reference,
        reference_label=reference_label,
        value_fmt=value_fmt,
        figsize=figsize,
    )


def response_grid(
    panels: Sequence[tuple[str, pd.DataFrame]],
    *,
    title: str,
    subtitle: str = "",
    columns: int = 4,
    ylabel: str = "P(bought)",
    reference: float | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """The same response chart for several columns, on one shared vertical scale."""
    rows = -(-len(panels) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    flat = [axis for row in axes for axis in row]
    highest = max(float(frame["rate"].max()) for _, frame in panels)

    for axis, (name, frame) in zip(flat, panels, strict=False):
        positions = np.arange(len(frame))
        axis.bar(
            positions,
            frame["rate"].to_numpy(dtype=float),
            yerr=interval_arms(frame["rate"], frame["lower"], frame["upper"]),
            capsize=3,
        )
        if reference is not None:
            axis.axhline(reference, linestyle="--", linewidth=1, color="black")
        axis.set_xticks(positions)
        axis.set_xticklabels(frame["bin"].astype(str), rotation=45, ha="right",
                             fontsize=7)
        axis.set_title(name, fontsize=10)
        axis.set_ylim(0, min(1.0, highest * 1.25))
        axis.tick_params(labelsize=8)
    for axis in flat[len(panels):]:
        axis.set_visible(False)
    for index, axis in enumerate(flat[: len(panels)]):
        if index % columns == 0:
            axis.set_ylabel(ylabel, fontsize=9)

    figure.suptitle(title)
    figure.tight_layout()
    if subtitle:
        caption(subtitle)
    return figure


def matrix_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    subtitle: str = "",
    value_fmt: str = "{:.2f}",
    colorbar_label: str = "",
    highlight: tuple[int, int] | None = None,
    highlight_note: str = "",
    figsize: tuple[float, float] = (9, 7),
) -> plt.Figure:
    """A labelled grid: a correlation matrix or a cross-tabulation."""
    plt.figure(figsize=figsize)
    values = frame.to_numpy(dtype=float)
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(image, label=colorbar_label)

    threshold = values.min() + 0.5 * (values.max() - values.min())
    for (row, column), value in np.ndenumerate(values):
        plt.text(
            column,
            row,
            value_fmt.format(value),
            ha="center",
            va="center",
            fontsize=9,
            color="black" if value > threshold else "white",
        )
    if highlight is not None:
        row, column = highlight
        plt.gca().add_patch(
            plt.Rectangle(
                (column - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2.5,
            )
        )
        if highlight_note:
            plt.text(column, row + 0.34, highlight_note, ha="center", va="top",
                     fontsize=9, color="red")

    plt.xticks(range(values.shape[1]), frame.columns.astype(str), rotation=45,
               ha="right")
    plt.yticks(range(values.shape[0]), frame.index.astype(str))
    _finish(title, subtitle, "", "")
    return plt.gcf()


def scatter_panels(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    groupings: Sequence[tuple[str, str]],
    title: str,
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[float, float] = (12, 5.5),
) -> plt.Figure:
    """The same points twice, coloured by two different columns."""
    figure, axes = plt.subplots(1, len(groupings), figsize=figsize, squeeze=False)
    for axis, (column, panel_title) in zip(axes[0], groupings, strict=True):
        order = frame[column].value_counts().index
        for name in order:
            group = frame[frame[column] == name]
            axis.scatter(group[x], group[y], s=6, alpha=0.55, label=str(name))
        axis.set_title(panel_title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.legend(title=column, fontsize=8, markerscale=2)
    axes[0][0].set_ylabel(ylabel)

    figure.suptitle(title)
    figure.tight_layout()
    if subtitle:
        caption(subtitle)
    return figure


def _label_bars(bars, value_fmt: str | None, label_type: str) -> None:
    """Write each bar's value on it, inside in white or above it in black."""
    if value_fmt is None:
        return
    if label_type == "center":
        plt.bar_label(bars, fmt=value_fmt, label_type="center", fontsize=8,
                      color="white")
    else:
        plt.bar_label(bars, fmt=value_fmt, label_type="edge", fontsize=8,
                      color="black", padding=2)


def _finish(title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    if subtitle:
        caption(subtitle)


def caption(subtitle: str, width: int = 130) -> None:
    """Put the explanatory line under the chart, wrapped to the figure width."""
    text = textwrap.fill(subtitle, width=width)
    plt.gcf().text(0.5, -0.01, text, ha="center", va="top", fontsize=8)
