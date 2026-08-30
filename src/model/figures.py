"""The model side's charts, kept apart from the numbers that produce them.

Same division as ``src/eda/plots.py``, and the same conventions: the Agg backend so a
run needs no display, one file per figure under ``figures/``, and Spanish labels
because these end up on the slides while the code and the docs stay in English.
"""

from __future__ import annotations

from pathlib import Path

from src.model.style import (  # sets the Agg backend before pyplot is imported below
    BAR_COLOR,
    HIGHLIGHT,
    MODEL_COLOR,
    NEUTRAL,
    OBSERVED_COLOR,
    PALETTE,
)
from src.model.style import save as _save

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURES_DIR = Path("figures")
FIGSIZE = (10, 6)
WIDE = (12, 5)


# ---------------------------------------------------------------------------
# Stage 4: what the frozen encoder thinks the popularity phrases mean.
# ---------------------------------------------------------------------------


def similarity_against_gap(
    pairs: pd.DataFrame,
    *,
    contrast: pd.Series | None,
    title: str,
    path: Path,
) -> Path:
    """Every pair of phrases: how close the encoder puts them against how differently
    they actually behave.

    If semantics carried the signal the cloud would slope down -- similar wording,
    similar buy rate. The pair that matters is marked, and it sits where the claim
    fails: nearly identical text, a 62-point gap in BTR.
    """
    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.scatter(
        pairs["cosine"],
        pairs["rate_gap"] * 100,
        s=26,
        color=NEUTRAL,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
        label="pares de frases",
    )
    if contrast is not None:
        axes.scatter(
            [contrast["cosine"]],
            [contrast["rate_gap"] * 100],
            s=170,
            color=BAR_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
            label=f"{contrast['left']} vs {contrast['right']}",
        )
        axes.annotate(
            f"coseno {contrast['cosine']:.3f}\nbrecha {contrast['rate_gap'] * 100:.1f} pp",
            xy=(contrast["cosine"], contrast["rate_gap"] * 100),
            xytext=(-14, -46),
            textcoords="offset points",
            fontsize=9,
            color=BAR_COLOR,
            ha="right",
        )
    axes.set_xlabel("Similitud coseno entre las dos frases (MiniLM congelado)")
    axes.set_ylabel("Diferencia de BTR observada (puntos porcentuales)")
    axes.set_title(title)
    axes.grid(alpha=0.25, zorder=0)
    axes.legend(loc="upper left", fontsize=9)
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Stage 6: is it right, is it honest, is it useful, and why.
# ---------------------------------------------------------------------------


def roc_and_pr(
    curves: list[tuple[str, pd.DataFrame, pd.DataFrame, float, float]],
    *,
    positive_rate: float,
    title: str,
    path: Path,
) -> Path:
    """ROC beside PR, one line per model, with the two chance references drawn in.

    Both panels, not one: ROC looks generous at a 13% positive rate and PR does not,
    and the write-up quotes PR-AUC for exactly that reason.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE)

    left.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1, label="azar")
    right.axhline(
        positive_rate,
        color=NEUTRAL,
        linestyle="--",
        linewidth=1,
        label=f"azar ({positive_rate:.3f})",
    )

    for position, (name, roc, pr, roc_auc, average_precision) in enumerate(curves):
        colour = PALETTE[position % len(PALETTE)]
        left.plot(roc["fpr"], roc["tpr"], color=colour, linewidth=1.8,
                  label=f"{name} (AUC {roc_auc:.3f})")
        right.plot(pr["recall"], pr["precision"], color=colour, linewidth=1.8,
                   label=f"{name} (AP {average_precision:.3f})")

    left.set_xlabel("Falsos positivos")
    left.set_ylabel("Verdaderos positivos")
    left.set_title("ROC")
    left.grid(alpha=0.25)
    left.legend(loc="lower right", fontsize=8)

    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_title("Precision-Recall")
    right.grid(alpha=0.25)
    right.legend(loc="upper right", fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def calibration(table: pd.DataFrame, *, title: str, path: Path, error: float) -> Path:
    """Predicted BTR against observed BTR, with the perfect diagonal for reference.

    ``BTR`` is the mean of the predicted probability, so this is the business metric
    itself and not a stand-in for it. A point above the diagonal is a bin the model
    over-promises on.
    """
    figure, axes = plt.subplots(figsize=FIGSIZE)
    limit = float(max(table["predicted"].max(), table["observed"].max(), 0.05)) * 1.1

    axes.plot([0, limit], [0, limit], color=NEUTRAL, linestyle="--", linewidth=1,
              label="calibracion perfecta")
    axes.errorbar(
        table["predicted"],
        table["observed"],
        # Wilson's interval is not centred on the raw rate, so a bin at 0% or 100%
        # lands on its own bound and rounding can push the arm a hair below zero.
        yerr=[
            (table["observed"] - table["low"]).clip(lower=0.0),
            (table["high"] - table["observed"]).clip(lower=0.0),
        ],
        fmt="o",
        color=MODEL_COLOR,
        ecolor=MODEL_COLOR,
        elinewidth=1.2,
        capsize=3,
        markersize=7,
        label="decil de score (IC Wilson 95%)",
    )
    for _, row in table.iterrows():
        axes.annotate(
            f"{int(row['bin'])}",
            xy=(row["predicted"], row["observed"]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
            color=MODEL_COLOR,
        )

    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("BTR predicho (media de p en el decil)")
    axes.set_ylabel("BTR observado en el decil")
    axes.set_title(f"{title}   —   error de calibracion {error:.4f}")
    axes.grid(alpha=0.25)
    axes.legend(loc="upper left", fontsize=9)
    return _save(figure, path)


def ranking_gains(
    tables: list[tuple[str, pd.DataFrame]], *, title: str, path: Path
) -> Path:
    """Precision@k on the left, lift@k on the right: the promotion budget curve."""
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE)

    for position, (name, table) in enumerate(tables):
        colour = PALETTE[position % len(PALETTE)]
        percentages = table["fraction"] * 100
        left.plot(percentages, table["precision"] * 100, "o-", color=colour, label=name)
        right.plot(percentages, table["lift"], "o-", color=colour, label=name)

    right.axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=1, label="azar")
    for axes, ylabel, panel in (
        (left, "Precision@k (%)", "Precision en el tope del ranking"),
        (right, "Lift@k (veces sobre el azar)", "Lift en el tope del ranking"),
    ):
        axes.set_xlabel("k: % del catalogo promocionado")
        axes.set_ylabel(ylabel)
        axes.set_title(panel)
        axes.set_xscale("log")
        axes.grid(alpha=0.25, which="both")
        axes.legend(loc="best", fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def attention_by_group(table: pd.DataFrame, *, title: str, path: Path) -> Path:
    """Where ``[CLS]`` looks, by group of positions, in mass and per token.

    Two panels because they answer different questions: mass is what a big group wins
    by being big, per-token is what two decisive words win by being decisive.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 6))
    layers = sorted(table["layer"].unique())
    order = (
        table[table["layer"] == layers[-1]]
        .sort_values("per_token", ascending=True)["group"]
        .tolist()
    )
    positions = np.arange(len(order))
    height = 0.8 / max(len(layers), 1)

    for index, layer in enumerate(layers):
        rows = table[table["layer"] == layer].set_index("group").reindex(order)
        offset = (index - (len(layers) - 1) / 2) * height
        colour = PALETTE[index % len(PALETTE)]
        left.barh(positions + offset, rows["mass"], height=height,
                  color=colour, label=f"capa {layer + 1}")
        right.barh(positions + offset, rows["per_token"], height=height,
                   color=colour, label=f"capa {layer + 1}")

    for axes, xlabel, panel in (
        (left, "Masa de atencion del [CLS]", "Cuanta atencion recibe el grupo"),
        (right, "Atencion por token", "Cuanta atencion recibe cada token"),
    ):
        axes.set_yticks(positions)
        axes.set_yticklabels(order, fontsize=8)
        axes.set_xlabel(xlabel)
        axes.set_title(panel)
        axes.grid(alpha=0.25, axis="x")
        axes.legend(loc="lower right", fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def price_recovery(
    sweep: pd.DataFrame, axis: pd.DataFrame, *, title: str, path: Path
) -> Path:
    """The observed hump against the model's own price response, and the bucket axis."""
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE)

    left.plot(sweep["bucket"] + 1, sweep["observed"] * 100, "o-",
              color=OBSERVED_COLOR, linewidth=2, label="BTR observado")
    left.plot(sweep["bucket"] + 1, sweep["counterfactual"] * 100, "s--",
              color=MODEL_COLOR, linewidth=2, label="respuesta del modelo (contrafactico)")
    left.set_xlabel("Decil de price_position")
    left.set_ylabel("% comprado")
    left.set_title("La U invertida, observada y aprendida")
    left.grid(alpha=0.25)
    left.legend(loc="best", fontsize=8)

    right.plot(axis["bucket"] + 1, axis["component"], "o-",
               color=HIGHLIGHT, linewidth=2)
    right.axhline(0.0, color=NEUTRAL, linewidth=1)
    right.set_xlabel("Decil de price_position")
    right.set_ylabel("Primera componente del embedding de bucket")
    right.set_title("Los 10 vectores aprendidos, sobre su eje principal")
    right.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def errors_by_level(
    table: pd.DataFrame, *, title: str, path: Path, top: int = 24
) -> Path:
    """Observed against predicted BTR for each level, so the failures are visible."""
    rows = table.head(top)
    figure, axes = plt.subplots(figsize=(10, max(6, 0.34 * len(rows))))
    positions = np.arange(len(rows))

    axes.barh(positions + 0.19, rows["observed"] * 100, height=0.38,
              color=OBSERVED_COLOR, label="BTR observado")
    axes.barh(positions - 0.19, rows["predicted"] * 100, height=0.38,
              color=MODEL_COLOR, label="BTR predicho (media)")
    for y, (_, row) in enumerate(rows.iterrows()):
        axes.text(
            max(row["observed"], row["predicted"]) * 100 + 1.0,
            y,
            f"n={int(row['rows'])}   brecha {row['gap'] * 100:+.1f} pp",
            va="center",
            fontsize=7,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels([str(level) for level in rows["level"]], fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("% comprado")
    axes.set_xlim(0, max(rows["observed"].max(), rows["predicted"].max()) * 100 * 1.35)
    axes.set_title(title)
    axes.grid(alpha=0.25, axis="x")
    axes.legend(loc="lower right", fontsize=9)
    return _save(figure, path)


def training_curves(curves: pd.DataFrame, *, title: str, path: Path) -> Path:
    """Train against held-out loss and AP, epoch by epoch: the over/underfitting slide."""
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE)
    for fold, rows in curves.groupby("fold_index"):
        colour = PALETTE[int(fold) % len(PALETTE)]
        left.plot(rows["epoch"], rows["train_loss"], color=colour, linewidth=1.5,
                  label=f"fold {fold} — train")
        left.plot(rows["epoch"], rows["validation_loss"], color=colour, linewidth=1.5,
                  linestyle="--", label=f"fold {fold} — corte")
        right.plot(rows["epoch"], rows["train_ap"], color=colour, linewidth=1.5)
        right.plot(rows["epoch"], rows["validation_ap"], color=colour, linewidth=1.5,
                   linestyle="--")

    left.set_xlabel("Epoca")
    left.set_ylabel("BCE loss")
    left.set_title("Loss: continua es train, punteada es el corte")
    left.grid(alpha=0.25)
    left.legend(loc="best", fontsize=7, ncol=2)

    right.set_xlabel("Epoca")
    right.set_ylabel("Average precision")
    right.set_title("AP: se registra, no decide la epoca")
    right.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)
