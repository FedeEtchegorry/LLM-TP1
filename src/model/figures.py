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


def encoding_ladder(
    folds: pd.DataFrame,
    *,
    title: str,
    path: Path,
    reference: float | None = None,
    reference_label: str = "barra lineal",
    noise: float | None = None,
) -> Path:
    """Mean AP per encoding, with the spread across folds drawn on top."""
    grouped = folds.groupby("encoding")["average_precision"]
    table = grouped.agg(["mean", "std", "count"]).sort_values("mean")
    deviations = table["std"].fillna(0.0)

    figure, axes = plt.subplots(figsize=(10, 0.55 * len(table) + 2.2))
    positions = np.arange(len(table))
    axes.errorbar(
        table["mean"], positions, xerr=deviations, fmt="o",
        color=MODEL_COLOR, ecolor="#888888", capsize=4, markersize=9,
        lw=1.6, zorder=3,
    )
    for position, (mean, deviation) in enumerate(zip(table["mean"], deviations)):
        axes.text(
            mean + deviation + 0.006, position, f"{mean:.3f}",
            va="center", fontsize=9, color="#333333",
        )

    if noise is not None:
        best = table["mean"].iloc[-1]
        axes.axvspan(best - noise, best + noise, color=NEUTRAL, alpha=0.28, zorder=0)
        axes.text(
            best - noise, -0.62,
            f"± ruido de reentrenar ({noise:.3f}): todo lo que cae aca es indistinguible",
            fontsize=8.5, color="#555555", va="center",
        )
    if reference is not None:
        axes.axvline(reference, color=BAR_COLOR, ls="--", lw=1.3, zorder=2)
        axes.text(
            reference, len(table) - 0.4, f" {reference_label} {reference:.3f}",
            color=BAR_COLOR, fontsize=9, va="top",
        )

    low = min(table["mean"] - deviations)
    high = max([*(table["mean"] + deviations), reference or -np.inf])
    margin = 0.12 * (high - low)
    axes.set_xlim(low - margin, high + 1.6 * margin)
    axes.set_ylim(-0.9, len(table) - 0.3)
    axes.set_yticks(positions)
    axes.set_yticklabels(table.index, fontsize=9)
    axes.set_xlabel("PR-AUC (AP), media entre folds ± desvio")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.25)
    axes.set_axisbelow(True)
    return _save(figure, path)


def paired_differences(
    pairs: pd.DataFrame,
    *,
    baseline: str,
    title: str,
    path: Path,
    value: str = "average_precision",
    group: str = "encoding",
) -> Path:
    """Every paired difference against one baseline, with its 95% interval."""
    wide = pairs.pivot_table(index=["seed", "fold"], columns=group, values=value)
    others = [name for name in wide.columns if name != baseline]

    figure, axes = plt.subplots(figsize=(10, 0.9 * len(others) + 2.6))
    for position, name in enumerate(others):
        difference = (wide[name] - wide[baseline]).dropna().to_numpy()
        jitter = (np.arange(len(difference)) % 5 - 2) * 0.035
        axes.scatter(
            difference, np.full(len(difference), position) + jitter,
            color=NEUTRAL, s=26, zorder=2, edgecolor="white", linewidth=0.5,
        )
        mean = difference.mean()
        half = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
        crosses_zero = (mean - half) <= 0.0 <= (mean + half)
        colour = BAR_COLOR if crosses_zero else OBSERVED_COLOR
        axes.errorbar(
            mean, position, xerr=half, fmt="D", color=colour,
            capsize=5, markersize=7, zorder=4, lw=2,
        )
        axes.text(
            mean, position + 0.30,
            f"{mean:+.4f}  ({(difference > 0).sum()}/{len(difference)} a favor)",
            ha="center", fontsize=9, color=colour,
        )
    axes.axvline(0.0, color="#333333", lw=1.4, zorder=1)
    axes.set_yticks(np.arange(len(others)))
    axes.set_yticklabels(others, fontsize=9)
    axes.set_ylim(-0.7, len(others) - 0.25)
    axes.set_xlabel(f"Δ AP contra {baseline}, un punto por par (semilla x fold)")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.25)
    axes.set_axisbelow(True)
    return _save(figure, path)


def noise_against_effect(
    repeats: pd.DataFrame,
    effects: dict[str, float],
    *,
    title: str,
    path: Path,
    value: str = "ap",
) -> Path:
    """Retraining spread on the left, the measured differences on the same scale."""
    values = repeats[value].to_numpy()
    spread = values.std(ddof=1)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    left.hist(values, bins=10, color=MODEL_COLOR, alpha=0.85, edgecolor="white")
    left.axvline(values.mean(), color="#333333", lw=1.5)
    left.axvspan(
        values.mean() - spread, values.mean() + spread, color=NEUTRAL, alpha=0.30, zorder=0
    )
    left.set_xlabel("AP de la MISMA configuracion, reentrenada")
    left.set_ylabel("corridas")
    left.set_title(f"Ruido de reentrenar\ndesvio = {spread:.4f}", fontsize=10)
    left.grid(alpha=0.2)
    left.set_axisbelow(True)

    names = list(effects)
    sizes = [effects[name] for name in names]
    colours = [OBSERVED_COLOR if abs(v) > spread else BAR_COLOR for v in sizes]
    positions = np.arange(len(names))
    right.barh(positions, sizes, color=colours, height=0.6)
    right.axvline(0.0, color="#333333", lw=1.2)
    for edge in (-spread, spread):
        right.axvline(edge, color=NEUTRAL, ls="--", lw=1.6)
    right.axvspan(-spread, spread, color=NEUTRAL, alpha=0.22, zorder=0)
    right.text(
        spread, len(names) - 0.4, "  ±1 desvio de reentrenar",
        fontsize=9, color="#555555", va="top",
    )
    for position, size in zip(positions, sizes):
        right.text(
            size + (0.0012 if size >= 0 else -0.0012), position, f"{size:+.4f}",
            va="center", ha="left" if size >= 0 else "right", fontsize=9,
        )
    right.set_yticks(positions)
    right.set_yticklabels(names, fontsize=9)
    right.set_xlabel("Δ AP medido")
    right.set_title("Lo que medimos, a la misma escala", fontsize=10)
    right.grid(axis="x", alpha=0.2)
    right.set_axisbelow(True)

    figure.suptitle(title, y=1.02)
    return _save(figure, path)


def seed_collapse(
    pairs: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    title: str,
    path: Path,
    value: str = "ap",
    group: str = "mode",
) -> Path:
    """The effect and its p-value as paired observations accumulate seed by seed."""
    from scipy import stats

    wide = pairs.pivot_table(index=["offset", "fold"], columns=group, values=value)
    seeds = sorted(pairs["offset"].unique())

    means, halves, pvalues, counts = [], [], [], []
    for depth in range(1, len(seeds) + 1):
        subset = wide.loc[wide.index.get_level_values("offset").isin(seeds[:depth])]
        difference = (subset[candidate] - subset[baseline]).dropna().to_numpy()
        means.append(difference.mean())
        halves.append(1.96 * difference.std(ddof=1) / np.sqrt(len(difference)))
        pvalues.append(stats.ttest_1samp(difference, 0.0).pvalue)
        counts.append(len(difference))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(1, len(seeds) + 1)

    left.errorbar(
        x, means, yerr=halves, fmt="o-", color=MODEL_COLOR,
        capsize=5, lw=1.8, markersize=7,
    )
    left.axhline(0.0, color="#333333", lw=1.3)
    left.set_xticks(x)
    left.set_xticklabels([f"{n} pares" for n in counts])
    left.set_ylabel(f"Δ AP: {candidate} - {baseline}")
    left.set_title("El efecto se encoge al sumar semillas", fontsize=10)
    left.grid(alpha=0.25)
    left.set_axisbelow(True)

    right.plot(x, pvalues, "o-", color=HIGHLIGHT, lw=1.8, markersize=7)
    right.axhline(0.05, color=BAR_COLOR, ls="--", lw=1.5)
    right.text(x[0], 0.05, " p = 0,05", color=BAR_COLOR, fontsize=9, va="bottom")
    for position, value_p in zip(x, pvalues):
        right.text(position, value_p + 0.02, f"{value_p:.3f}", ha="center", fontsize=9)
    right.set_xticks(x)
    right.set_xticklabels([f"{n} pares" for n in counts])
    right.set_ylim(0, max(max(pvalues) * 1.35, 0.28))
    right.set_ylabel("p (t pareado)")
    right.set_title("Y la significancia se evapora", fontsize=10)
    right.grid(alpha=0.25)
    right.set_axisbelow(True)

    figure.suptitle(title, y=1.02)
    return _save(figure, path)


def encoding_families(
    folds: pd.DataFrame,
    *,
    title: str,
    path: Path,
    labels: dict[str, str] | None = None,
) -> Path:
    """Every encoding measured, one panel per family of column."""
    families = list(dict.fromkeys(folds["block"]))
    labels = labels or {}
    heights = [max(folds[folds["block"] == f]["encoding"].nunique(), 1) for f in families]
    figure, axes = plt.subplots(
        len(families), 1,
        figsize=(11, 0.5 * sum(heights) + 1.6 * len(families)),
        gridspec_kw={"height_ratios": heights},
    )
    axes = np.atleast_1d(axes)

    for panel, family in zip(axes, families):
        subset = folds[folds["block"] == family]
        table = (
            subset.groupby("encoding")["average_precision"]
            .agg(["mean", "std"])
            .sort_values("mean")
        )
        positions = np.arange(len(table))
        deviations = table["std"].fillna(0.0)
        panel.errorbar(
            table["mean"], positions, xerr=deviations, fmt="o",
            color=MODEL_COLOR, ecolor="#888888", capsize=3.5, markersize=7.5,
            lw=1.4, zorder=3,
        )
        for position, (mean, deviation) in enumerate(zip(table["mean"], deviations)):
            panel.text(
                mean + deviation + 0.004, position, f"{mean:.3f}",
                va="center", fontsize=8.5, color="#333333",
            )
        panel.set_yticks(positions)
        panel.set_yticklabels(table.index, fontsize=8.5)
        panel.set_title(labels.get(family, family), fontsize=10, loc="left")
        panel.grid(axis="x", alpha=0.22)
        panel.set_axisbelow(True)
        panel.set_ylim(-0.6, len(table) - 0.4)
        low = float((table["mean"] - deviations).min())
        high = float((table["mean"] + deviations).max())
        margin = 0.10 * max(high - low, 1e-3)
        panel.set_xlim(low - margin, high + 4.0 * margin)

    axes[-1].set_xlabel("PR-AUC (AP), media entre folds ± desvio")
    figure.suptitle(title, y=1.005)
    figure.tight_layout()
    return _save(figure, path)


def one_versus_two(
    comparisons: pd.DataFrame,
    *,
    title: str,
    path: Path,
) -> Path:
    """Each contrast where a second encoding was added to a first, against zero."""
    table = comparisons.sort_values("delta")
    positions = np.arange(len(table))
    colours = []
    for delta, half in zip(table["delta"], table["half"]):
        if delta - half <= 0.0 <= delta + half:
            colours.append(NEUTRAL)
        elif delta > 0:
            colours.append(OBSERVED_COLOR)
        else:
            colours.append(BAR_COLOR)

    figure, axes = plt.subplots(figsize=(10, 0.85 * len(table) + 2.4))
    axes.errorbar(
        table["delta"], positions, xerr=table["half"], fmt="D",
        ecolor="#666666", capsize=6, markersize=0, lw=1.8, zorder=2,
    )
    axes.scatter(table["delta"], positions, color=colours, s=90, zorder=3, edgecolor="white")
    axes.axvline(0.0, color="#333333", lw=1.5, zorder=1)
    for position, (delta, half) in enumerate(zip(table["delta"], table["half"])):
        axes.text(
            delta, position + 0.26,
            f"{delta:+.4f}  IC95 [{delta - half:+.4f}, {delta + half:+.4f}]",
            ha="center", fontsize=8.5, color="#333333",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels(table["contrast"], fontsize=9)
    axes.set_ylim(-0.7, len(table) - 0.2)
    axes.set_xlabel("Δ AP de agregar la segunda codificacion")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.25)
    axes.set_axisbelow(True)
    return _save(figure, path)
