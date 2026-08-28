"""Three bar charts: one for named levels, one for ordered buckets, one for the ranking."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eda import noise  # noqa: E402

FIGURES_DIR = Path("figures")
FIGSIZE = (10, 6)
DPI = 120

BAR_COLOR = "#4C72B0"
NOISE_COLOR = "#B0B0B0"
REFERENCE_COLOR = "#C44E52"

BUILT_COLOR = "#55A868"
"""Bar fill for a column the analysis built, in the ranking chart."""

# The bar is drawn thinner than the chance range behind it, so the range shows on both sides.
BAND_THICKNESS = 0.82
BAR_THICKNESS = 0.40


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def _bands(table: pd.DataFrame, target: np.ndarray | None) -> np.ndarray | None:
    if target is None:
        return None
    return noise.level_bands(table["rows"].to_numpy(), target)


def bar_by_level(
    table: pd.DataFrame,
    *,
    title: str,
    path: Path,
    target: np.ndarray | None = None,
) -> Path:
    """Horizontal bars per level; ``target`` adds the range each reaches by chance."""
    figure, axes = plt.subplots(figsize=FIGSIZE)
    labels = [str(level) for level in table.index]
    percentages = (table["rate"] * 100).to_numpy()
    bands = _bands(table, target)

    if bands is not None:
        axes.axvline(
            float(np.asarray(target).mean()) * 100,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1,
            zorder=0,
            label="BTR global",
        )
        axes.barh(
            labels,
            bands[:, 1] - bands[:, 0],
            left=bands[:, 0],
            height=BAND_THICKNESS,
            color=NOISE_COLOR,
            alpha=0.5,
            zorder=1,
            label=f"rango por azar ({noise.PERCENTILE}%)",
        )

    thickness = BAR_THICKNESS if bands is not None else BAND_THICKNESS
    axes.barh(labels, percentages, height=thickness, color=BAR_COLOR, zorder=2)
    axes.invert_yaxis()
    span = max(percentages.max(), 1.0)
    for y, (percentage, rows) in enumerate(zip(percentages, table["rows"])):
        axes.text(percentage + span * 0.01, y, f"{percentage:.1f}%  (n={rows})", va="center", fontsize=8, zorder=3)
    axes.set_xlim(0, span * 1.28)
    axes.set_xlabel("% comprado")
    axes.set_title(title)
    if bands is not None:
        axes.legend(loc="lower right", fontsize=8)
    return _save(figure, path)


def bar_by_bucket(
    table: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    path: Path,
    target: np.ndarray | None = None,
) -> Path:
    """Vertical bars in bucket order, so a non-monotonic shape stays visible."""
    figure, axes = plt.subplots(figsize=FIGSIZE)
    labels = [str(bucket) for bucket in table.index]
    percentages = (table["rate"] * 100).to_numpy()
    positions = range(len(labels))
    bands = _bands(table, target)

    if bands is not None:
        axes.axhline(
            float(np.asarray(target).mean()) * 100,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1,
            zorder=0,
            label="BTR global",
        )
        axes.bar(
            list(positions),
            bands[:, 1] - bands[:, 0],
            bottom=bands[:, 0],
            width=BAND_THICKNESS,
            color=NOISE_COLOR,
            alpha=0.5,
            zorder=1,
            label=f"rango por azar ({noise.PERCENTILE}%)",
        )

    thickness = BAR_THICKNESS if bands is not None else BAND_THICKNESS
    axes.bar(positions, percentages, width=thickness, color=BAR_COLOR, zorder=2)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    span = max(percentages.max(), 1.0)
    for x, (percentage, rows) in enumerate(zip(percentages, table["rows"])):
        axes.text(x, percentage + span * 0.01, f"n={rows}", ha="center", va="bottom", fontsize=7, zorder=3)
    axes.set_ylim(0, span * 1.15)
    axes.set_ylabel("% comprado")
    axes.set_xlabel(xlabel)
    axes.set_title(title)
    if bands is not None:
        axes.legend(loc="upper right", fontsize=8)
    return _save(figure, path)


def bar_separation_vs_floor(
    table: pd.DataFrame,
    *,
    title: str,
    path: Path,
) -> Path:
    """One bar per column: its separation, against the floor its shape reaches.

    Hue says where the column comes from, the red mark says what it has to beat,
    and a faded bar is one that does not beat it.
    """
    figure, axes = plt.subplots(figsize=(10, 7))
    labels = [str(column) for column in table.index]
    separation = table["separacion"].to_numpy()
    low = table["piso_lo"].to_numpy()
    high = table["piso_hi"].to_numpy()
    built = (table["origen"] == "construida").to_numpy()
    positions = np.arange(len(labels))

    for y, (value, lo, hi, is_built) in enumerate(zip(separation, low, high, built)):
        clears = value > hi
        axes.barh(
            y,
            value,
            color=BUILT_COLOR if is_built else BAR_COLOR,
            alpha=1.0 if clears else 0.38,
            zorder=2,
        )
        axes.add_patch(
            plt.Rectangle(
                (lo, y - 0.42), max(hi - lo, 0.05), 0.84,
                color=REFERENCE_COLOR, alpha=0.9, zorder=4,
            )
        )
        axes.text(max(value, hi) + 0.6, y, f"{value:.1f} vs {(lo + hi) / 2:.1f}", va="center", fontsize=8, zorder=5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR, label="campo del dataset - supera"),
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR, alpha=0.38, label="campo del dataset - no supera"),
        plt.Rectangle((0, 0), 1, 1, color=BUILT_COLOR, label="construida por el analisis - supera"),
        plt.Rectangle((0, 0), 1, 1, color=BUILT_COLOR, alpha=0.38, label="construida por el analisis - no supera"),
        plt.Rectangle((0, 0), 1, 1, color=REFERENCE_COLOR, alpha=0.9, label="piso de ruido (media +/- desvio)"),
    ]
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlim(0, max(separation.max(), high.max()) * 1.24)
    axes.set_xlabel("Separacion (puntos porcentuales)")
    axes.set_title(title)
    axes.legend(handles=handles, loc="lower right", fontsize=8)
    return _save(figure, path)
