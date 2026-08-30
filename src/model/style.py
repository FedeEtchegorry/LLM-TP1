"""Shared visual style for every model-side figure.

Split from ``figures.py`` so a slide's look and the data it draws are two different
concerns: the palette or the resolution can change here without touching a single
chart. ``figures.py`` imports its constants and its saving helper from this module
instead of defining its own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update(
    {
        # Presentation projection, not a laptop screen.
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # A default for the charts that opt into a grid; it is never turned on here,
        # so an axes with no ``.grid()`` call stays exactly as bare as it was.
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)

MODEL_COLOR = "#4C72B0"
BAR_COLOR = "#C44E52"
OBSERVED_COLOR = "#55A868"
NEUTRAL = "#B0B0B0"
HIGHLIGHT = "#8172B2"

PALETTE = (MODEL_COLOR, BAR_COLOR, OBSERVED_COLOR, HIGHLIGHT, "#CCB974", "#64B5CD")


def save(figure: plt.Figure, path: Path) -> Path:
    """Write one figure to ``path``, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path
