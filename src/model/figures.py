"""The model side's charts, kept apart from the numbers that produce them.

Same division as ``src/eda/plots.py``, and the same conventions: the Agg backend so a
run needs no display, one file per figure under ``figures/``, and Spanish labels
because these end up on the slides while the code and the docs stay in English.
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
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

T95_DF4 = 2.776
"""Two-sided 95% critical value of Student's t at 4 degrees of freedom: this protocol
always produces 5 paired folds, too few to lean on the normal approximation instead."""


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

    The x-axis label already says which encoder; a caption under the axes spells out,
    in words a slide-reader cannot skim past, that this is MiniLM's own frozen
    embedding space and nothing our Transformer learned -- and not a claim about the
    task in general, since a supervised model reading the same phrases through its own
    trained weights would not be bound to reproduce this geometry.
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
    figure.text(
        0.5, 0.01,
        "Mide el espacio de embeddings de MiniLM congelado -- no el modelo "
        "entrenado en este trabajo, ni una afirmacion sobre la tarea en general.",
        ha="center", fontsize=8.5, color=NEUTRAL,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
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


def ladder_waterfall(ladder: pd.DataFrame, *, title: str, path: Path) -> Path:
    """Horizontal bars for the ladder, L0 through L4, sorted by rung regardless of the
    frame's own row order. Each bar carries its between-fold error bar and, beside it,
    the delta against the rung before it; a dashed line at L0's own AP runs through the
    whole plot, so a rung that never clears the linear bar is visible without reading
    the axis. This is also where L1 and L2 read as indistinguishable despite L2 costing
    100k more parameters -- their bars and error bars overlap almost exactly.

    ``ladder`` is ``results.summary_frame()`` filtered to the five ``[L0 ...]``..
    ``[L4 ...]`` rows -- one per rung, carrying ``name``, ``average_precision_mean`` and
    ``average_precision_std``.
    """
    rows = ladder.assign(rung=ladder["name"].str.extract(r"^L(\d)")[0].astype(int))
    rows = rows.sort_values("rung").reset_index(drop=True)
    l0_ap = float(rows.loc[rows["rung"] == 0, "average_precision_mean"].iloc[0])

    figure, axes = plt.subplots(figsize=FIGSIZE)
    positions = np.arange(len(rows))

    axes.barh(
        positions,
        rows["average_precision_mean"],
        xerr=rows["average_precision_std"],
        color=BAR_COLOR,
        ecolor="black",
        capsize=4,
        height=0.6,
        zorder=2,
    )
    axes.axvline(
        l0_ap,
        color=NEUTRAL,
        linewidth=2,
        linestyle="--",
        zorder=1,
        label=f"L0 (AP {l0_ap:.3f})",
    )

    previous = None
    for y, (_, row) in zip(positions, rows.iterrows()):
        label = f"AP {row['average_precision_mean']:.3f}"
        if previous is not None:
            label += f"   Δ {row['average_precision_mean'] - previous:+.3f}"
        previous = row["average_precision_mean"]
        axes.text(
            row["average_precision_mean"] + row["average_precision_std"] + 0.012,
            y,
            label,
            va="center",
            fontsize=9,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(rows["name"], fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel("Average precision (media entre folds)")
    axes.set_title(title)
    axes.grid(alpha=0.25, axis="x")
    axes.legend(loc="lower right", fontsize=9)
    return _save(figure, path)


def ablation_forest(
    folds: pd.DataFrame,
    base_name: str,
    variant_names: list[str],
    *,
    title: str,
    path: Path,
) -> Path:
    """Forest plot of paired per-fold AP deltas against ``base_name``.

    ``folds`` is ``results.fold_frame()`` (or a slice of it), carrying ``name``,
    ``fold_index`` and ``average_precision``. The pairing is the whole point: for each
    variant, ``delta_k = AP_variant(fold k) - AP_base(fold k)`` is computed fold by fold
    and only then averaged, so subtracting cancels the fold's own difficulty instead of
    comparing two loose means. The interval is Student's t at 4 degrees of freedom
    (``T95_DF4``), not the normal approximation -- there are only 5 paired folds. A
    variant missing one of the base's folds is left out of the plot, with a warning
    printed, rather than imputed.
    """
    base = folds[folds["name"] == base_name].set_index("fold_index")["average_precision"]
    if base.empty:
        raise ValueError(f"no hay folds registrados para la base {base_name!r}")

    labels: list[str] = []
    deltas_by_variant: list[np.ndarray] = []
    skipped: list[str] = []
    for variant in variant_names:
        scored = folds[folds["name"] == variant].set_index("fold_index")["average_precision"]
        missing = sorted(set(base.index) - set(scored.index))
        if missing:
            print(f"[ablation_forest] {variant!r}: faltan los folds {missing}, se excluye")
            skipped.append(variant)
            continue
        labels.append(variant)
        deltas_by_variant.append((scored.loc[base.index] - base).to_numpy())

    if not labels:
        raise ValueError("ninguna variante tiene todos los folds de la base pareados")

    figure, axes = plt.subplots(figsize=(10, max(4, 0.6 * len(labels) + 1)))
    positions = np.arange(len(labels))

    for y, deltas in zip(positions, deltas_by_variant):
        mean = float(deltas.mean())
        halfwidth = T95_DF4 * float(deltas.std(ddof=1)) / sqrt(len(deltas))
        crosses_zero = mean - halfwidth <= 0 <= mean + halfwidth
        colour = NEUTRAL if crosses_zero else HIGHLIGHT
        axes.errorbar(
            [mean],
            [y],
            xerr=[[halfwidth], [halfwidth]],
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=1.8,
            capsize=4,
            markersize=8,
            zorder=3,
        )
        axes.text(
            mean + halfwidth + 0.006,
            y,
            f"{mean:+.3f}  [{mean - halfwidth:+.3f}, {mean + halfwidth:+.3f}]",
            va="center",
            fontsize=8,
            color=colour,
        )

    axes.axvline(0.0, color="#333333", linewidth=2.5, zorder=1)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel(f"Δ AP pareado vs [{base_name}] (IC 95%, t de Student, 4 g.l.)")
    axes.set_title(title)
    axes.grid(alpha=0.25, axis="x")
    if skipped:
        axes.text(
            0.0,
            -0.14,
            "excluidas por folds faltantes: " + ", ".join(skipped),
            transform=axes.transAxes,
            fontsize=8,
            color=NEUTRAL,
        )
    figure.tight_layout()
    return _save(figure, path)


def learning_rate_sweep(
    folds: pd.DataFrame,
    names: list[str],
    *,
    epoch_ceiling: int,
    title: str,
    path: Path,
) -> Path:
    """Two panels sharing a log x-axis of learning rate: validation AP on top, mean
    ``best_epoch`` on the bottom, with a dashed line at ``epoch_ceiling``.

    ``folds`` is ``results.fold_frame()`` (or a slice of it); ``names`` are the runs
    that make up the sweep, one learning rate each, read from each run's own
    ``config.learning_rate`` rather than assumed from ``names`` order. A point where
    at least one fold's ``best_epoch`` reaches ``epoch_ceiling`` is annotated on the
    bottom panel: early stopping never cut that fold off, so its plateau is a
    training-budget artefact, not evidence the rate itself is fine. Checked per fold
    rather than on the mean, since a couple of folds pinned at the ceiling can still
    average below it.
    """
    points = []
    for name in names:
        rows = folds[folds["name"] == name]
        if rows.empty:
            print(f"[learning_rate_sweep] {name!r}: sin folds registrados, se excluye")
            continue
        average_precision = rows["average_precision"].astype(float)
        best_epoch = rows["best_epoch"].dropna().astype(float)
        points.append(
            {
                "name": name,
                "learning_rate": float(rows["config.learning_rate"].iloc[0]),
                "ap_mean": float(average_precision.mean()),
                "ap_std": float(average_precision.std(ddof=1)) if len(average_precision) > 1 else 0.0,
                "best_epoch_mean": float(best_epoch.mean()) if not best_epoch.empty else float("nan"),
                "hit_ceiling": bool((best_epoch >= epoch_ceiling).any()),
            }
        )
    if not points:
        raise ValueError("ninguno de los puntos del barrido tiene folds registrados")

    table = pd.DataFrame(points).sort_values("learning_rate").reset_index(drop=True)

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    top.plot(table["learning_rate"], table["ap_mean"], "o-", color=MODEL_COLOR, linewidth=2, zorder=3)
    top.fill_between(
        table["learning_rate"],
        table["ap_mean"] - table["ap_std"],
        table["ap_mean"] + table["ap_std"],
        color=MODEL_COLOR,
        alpha=0.2,
        zorder=2,
        label="± 1 desvio entre folds",
    )
    for _, row in table.iterrows():
        top.annotate(
            f"{row['ap_mean']:.3f} ± {row['ap_std']:.3f}",
            xy=(row["learning_rate"], row["ap_mean"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    top.set_ylabel("AP de validacion (media entre folds)")
    top.set_xscale("log")
    top.grid(alpha=0.25)
    top.legend(loc="best", fontsize=8)
    plt.setp(top.get_xticklabels(), visible=False)

    bottom.plot(
        table["learning_rate"], table["best_epoch_mean"], "o-",
        color=BAR_COLOR, linewidth=2, zorder=3,
    )
    bottom.axhline(
        epoch_ceiling,
        color=NEUTRAL,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
        label=f"techo de epocas ({epoch_ceiling})",
    )
    for _, row in table[table["hit_ceiling"]].iterrows():
        bottom.annotate(
            "toca el techo:\nel early stopping\nnunca cortó en algun fold",
            xy=(row["learning_rate"], row["best_epoch_mean"]),
            xytext=(60, 48),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=BAR_COLOR,
            arrowprops=dict(arrowstyle="->", color=BAR_COLOR, linewidth=1.2),
        )
    bottom.set_xscale("log")
    bottom.set_xticks(table["learning_rate"])
    bottom.set_xticklabels([f"{rate:g}" for rate in table["learning_rate"]])
    bottom.set_xlabel("Learning rate (escala log)")
    bottom.set_ylabel("best_epoch (media entre folds)")
    bottom.set_ylim(0, epoch_ceiling * 1.28)
    bottom.grid(alpha=0.25)
    bottom.legend(loc="lower left", fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, path)


def seed_variance(
    summary: pd.DataFrame,
    seed_runs: list[str],
    config_runs: list[str],
    *,
    seed_row_label: str,
    config_row_label: str,
    title: str,
    path: Path,
) -> Path:
    """Two rows of AP points and the range across them: one configuration under
    several seeds on top, several different configurations (each at its own default
    seed) on the bottom. The point is the comparison between the two ranges, not
    either row alone -- when the between-seed range is as wide as, or wider than, the
    between-config range, picking among the configurations has no empirical support.

    ``summary`` is ``results.summary_frame()``; ``seed_runs`` and ``config_runs`` are
    the run names for each row, read by ``average_precision_mean``. A row whose points
    are numerically identical (``max == min``, e.g. a deterministic model like the
    logistic bar scored under different seeds) gets no range bracket -- a zero-width
    bar would read as a measured spread of zero rather than the absence of one -- and
    is labelled "(determinístico)" instead.
    """

    def _values(names: list[str]) -> list[float]:
        values, missing = [], []
        for name in names:
            match = summary[summary["name"] == name]
            if match.empty:
                missing.append(name)
                continue
            values.append(float(match["average_precision_mean"].iloc[0]))
        if missing:
            print(f"[seed_variance] sin registrar: {missing}")
        return values

    seed_values = _values(seed_runs)
    config_values = _values(config_runs)
    if not seed_values or not config_values:
        raise ValueError("faltan corridas registradas para armar seed_variance")

    figure, axes = plt.subplots(figsize=(9, 4.2))
    rows = ((1, seed_row_label, seed_values), (0, config_row_label, config_values))
    spans: dict[str, float] = {}

    for y, label, values in rows:
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
        axes.scatter(
            values,
            [y + offset for offset in jitter],
            s=90,
            color=MODEL_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        low, high = min(values), max(values)
        span = high - low
        spans[label] = span
        if span < 1e-6:
            axes.text(
                low, y - 0.24, "(determinístico: mismo AP en las 3 semillas)",
                ha="center", fontsize=8, color=NEUTRAL,
            )
        else:
            axes.plot([low, high], [y - 0.16, y - 0.16], color=BAR_COLOR, linewidth=2.4, zorder=2)
            for edge in (low, high):
                axes.plot([edge, edge], [y - 0.20, y - 0.12], color=BAR_COLOR, linewidth=2.4, zorder=2)
            axes.text(
                (low + high) / 2, y - 0.30, f"rango {span:.3f}",
                ha="center", fontsize=9, color=BAR_COLOR,
            )

    axes.set_yticks([0, 1])
    axes.set_yticklabels([config_row_label, seed_row_label], fontsize=9)
    axes.set_ylim(-0.65, 1.6)
    axes.set_xlabel("Average precision")
    axes.set_title(title)
    axes.grid(alpha=0.25, axis="x")

    seed_span, config_span = spans[seed_row_label], spans[config_row_label]
    if seed_span >= 1e-6 and config_span >= 1e-6:
        comparison = "mayor que" if seed_span > config_span else "menor o igual que"
        figure.text(
            0.5, 0.015,
            f"rango entre semillas ({seed_span:.3f}) {comparison} rango entre "
            f"configuraciones ({config_span:.3f})",
            ha="center", fontsize=9, color="#333333",
        )

    figure.tight_layout(rect=(0, 0.06, 1, 1))
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


ENCODING_LABELS = {
    "bolsa binaria": "Bolsa de palabras",
    "tf-idf": "TF-IDF",
    "one-hot": "One-hot",
    "target encoding suavizado": "Target encoding",
    "continuo estandarizado": "Affine",
    "buckets por cuantiles": "Buckets",
    "continuo + buckets": "Affine + buckets",
    "piecewise-linear": "Piecewise",
}
"""Tidy display names, aligned with the vocabulary chart 3 uses for the Transformer's
numeric embedding (affine/buckets/piecewise) -- these are still a different
mechanism (see ``model_alias`` module docstring): here they are scalar features
handed to a plain logistic regression, there they are learned embedding vectors."""


def encoding_families(
    folds: pd.DataFrame,
    *,
    title: str,
    path: Path,
    labels: dict[str, str] | None = None,
    encoding_labels: dict[str, str] | None = None,
) -> Path:
    """Every encoding measured, one panel per family of column -- always scored with
    the same plain logistic regression (Modelo A's model family); only the column's
    representation changes between bars."""
    families = list(dict.fromkeys(folds["block"]))
    labels = labels or {}
    encoding_text = {**ENCODING_LABELS, **(encoding_labels or {})}
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
            color=MODEL_COLOR, ecolor="#3a3a38", capsize=4, markersize=7.5,
            lw=1.5, elinewidth=1.3, capthick=1.3, zorder=3,
        )
        for position, (mean, deviation) in enumerate(zip(table["mean"], deviations)):
            panel.text(
                mean + deviation + 0.004, position, f"{mean:.3f} ± {deviation:.3f}",
                va="center", fontsize=8.5, color="#333333",
            )
        panel.set_yticks(positions)
        panel.set_yticklabels([encoding_text.get(name, name) for name in table.index], fontsize=8.5)
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


# ---------------------------------------------------------------------------
# Ejercicio 2 (EDA contract): chart 2 of the flow -- the L0a/L0/L1/L2/L0b ladder
# read between its two diagnostic bounds, rather than sorted by rung digit like
# ``ladder_waterfall`` (which would put L0a, L0 and L0b all on the same "rung 0").
# ---------------------------------------------------------------------------


CATEGORICAL_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
"""One distinct hue per rung, in narrative (``order``) sequence -- floor and ceiling
included, no bar singled out as neutral."""


def eda_ladder_waterfall(
    ladder: pd.DataFrame,
    *,
    order: list[str],
    floor: str,
    ceiling: str,
    recover_for: str | None = None,
    label_fn: Callable[[str], str] | None = None,
    title: str,
    path: Path,
) -> Path:
    """The text ladder read against its floor (no text) and ceiling (hand-extracted
    key), in the fixed narrative order -- never sorted by name.

    ``ladder`` is ``results.summary_frame()`` (or an equivalent frame) carrying
    ``name``, ``average_precision_mean`` and ``average_precision_std``, filtered to
    the rows named in ``order``. ``floor`` and ``ceiling`` must both be in ``order``.

    Bare, presentation-only chart: no title, no recovered-fraction annotation.
    ``label_fn`` maps each row's full declared name to the text drawn on the axis
    (default: ``src.model.model_alias.alias_label``, which reads L0/L1/L2 as Modelo
    A/B/C). The recovered-fraction number the write-up quotes is computed
    separately, from the same ``ladder`` frame, not read off this figure.
    """
    if label_fn is None:
        from src.model.model_alias import alias_label as label_fn
    rows = ladder.set_index("name").loc[order].reset_index()
    floor_ap = float(rows.loc[rows["name"] == floor, "average_precision_mean"].iloc[0])
    ceiling_ap = float(rows.loc[rows["name"] == ceiling, "average_precision_mean"].iloc[0])

    figure, axes = plt.subplots(figsize=FIGSIZE)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    positions = np.arange(len(rows))[::-1]  # first in ``order`` drawn at the top

    colours = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(rows))]
    axes.barh(
        positions,
        rows["average_precision_mean"],
        xerr=rows["average_precision_std"],
        color=colours,
        ecolor="#52514e",
        capsize=3,
        error_kw={"linewidth": 1.3},
        height=0.62,
        zorder=2,
    )

    for y, (_, row) in zip(positions, rows.iterrows()):
        axes.text(
            row["average_precision_mean"] + row["average_precision_std"] + 0.014,
            y,
            f"{row['average_precision_mean']:.3f} ± {row['average_precision_std']:.3f}",
            va="center",
            fontsize=9,
            color="#52514e",
        )

    axes.set_yticks(positions)
    axes.set_yticklabels([label_fn(name) for name in rows["name"]], fontsize=10.5)
    axes.set_xlabel("Average precision")
    axes.grid(alpha=0.15, axis="x")
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("#52514e")
        spine.set_linewidth(1.0)
    axes.tick_params(axis="y", length=0)
    axes.margins(x=0.14)

    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Ejercicio 2 (EDA contract): charts 3-5 of the flow. Implemented and unit-tested
# against synthetic fixtures; not yet exercised against real data, which needs the
# full architecture search (Task 5), its greedy-order validation (Task 11) and the
# holdout (Task 7/9) to have actually run. See ``run_eda_contract_figures.py``.
# ---------------------------------------------------------------------------


def architecture_path(stages: list[dict], *, title: str, path: Path) -> Path:
    """One row per search stage, the base and every candidate plotted by AP, the
    selected point highlighted and connected across stages into a single path.

    Each item of ``stages`` is
    ``{"stage": str, "points": [{"label": str, "ap": float, "outcome": str}, ...],
    "selected": str}``, where ``outcome`` is one of ``"base"``, ``"improves"``,
    ``"tie-break"``, ``"inconclusive"`` or ``"loses"`` and ``selected`` names the
    point (by ``label``) that stage's comparison kept -- exactly what
    ``advance_complexity``/``resolve_heads`` return, made drawable.
    """
    outcome_colour = {
        "base": NEUTRAL,
        "improves": OBSERVED_COLOR,
        "tie-break": HIGHLIGHT,
        "inconclusive": NEUTRAL,
        "loses": BAR_COLOR,
    }
    figure, axes = plt.subplots(figsize=(WIDE[0], 0.9 * len(stages) + 1.6))
    positions = np.arange(len(stages))[::-1]

    path_x: list[float] = []
    path_y: list[float] = []
    for y, stage in zip(positions, stages):
        for point in stage["points"]:
            colour = outcome_colour.get(point["outcome"], NEUTRAL)
            is_selected = point["label"] == stage["selected"]
            axes.scatter(
                point["ap"], y,
                s=140 if is_selected else 55,
                color=colour,
                edgecolor="black" if is_selected else "none",
                linewidth=1.4 if is_selected else 0,
                zorder=4 if is_selected else 3,
            )
            axes.text(
                point["ap"], y + 0.22,
                point["label"],
                ha="center", fontsize=7.5,
                fontweight="bold" if is_selected else "normal",
                color="#222222" if is_selected else "#666666",
            )
            if is_selected:
                path_x.append(point["ap"])
                path_y.append(y)

    axes.plot(path_x, path_y, color="#333333", linewidth=1.2, linestyle=":", zorder=1)
    axes.set_yticks(positions)
    axes.set_yticklabels([stage["stage"] for stage in stages], fontsize=9)
    axes.set_xlabel("Average precision (media entre folds)")
    axes.set_title(title)
    axes.grid(alpha=0.2, axis="x")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour, markersize=9, label=label)
        for label, colour in (
            ("base / sin evidencia suficiente", NEUTRAL),
            ("mejora clara", OBSERVED_COLOR),
            ("desempate (media y desvio)", HIGHLIGHT),
            ("pierde", BAR_COLOR),
        )
    ]
    axes.legend(handles=handles, loc="lower right", fontsize=8)
    figure.tight_layout()
    return _save(figure, path)


def architecture_grid(stages: list[dict], *, title: str, path: Path) -> Path:
    """One subplot per capacity axis (embedding numerico, profundidad, ancho,
    heads), all in a single figure: every candidate at that axis as a horizontal
    bar by AP, one distinct colour per bar, the axis's winner outlined, the exact
    AP value printed beside each bar.

    Each item of ``stages`` is the same shape ``architecture_path`` takes --
    ``{"stage": str, "points": [{"label", "ap", "ap_std"?, "outcome"}, ...],
    "selected": str}`` -- with ``ap_std`` optional (defaults to 0, e.g. for the
    bundled example fixture, which carries no per-point spread). ``outcome`` is
    still used to outline the winning bar, not to colour it.
    """
    columns = 2
    grid_rows = -(-len(stages) // columns)
    figure, grid = plt.subplots(grid_rows, columns, figsize=(WIDE[0], 3.4 * grid_rows))
    panels = np.atleast_1d(grid).ravel()

    for panel, stage in zip(panels, stages):
        points = stage["points"]
        positions = np.arange(len(points))[::-1]
        colours = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(points))]
        edges = ["black" if point["label"] == stage["selected"] else "none" for point in points]
        errors = [point.get("ap_std", 0.0) for point in points]
        panel.barh(
            positions,
            [point["ap"] for point in points],
            xerr=errors,
            color=colours,
            edgecolor=edges,
            linewidth=1.6,
            ecolor="#3a3a38",
            error_kw={"elinewidth": 1.3, "capthick": 1.3},
            capsize=4,
            height=0.6,
            zorder=2,
        )
        for y, point, error in zip(positions, points, errors):
            panel.text(
                point["ap"] + error + 0.006,
                y,
                f"{point['ap']:.3f} ± {error:.3f}",
                va="center", fontsize=8, color="#52514e",
            )
        panel.set_yticks(positions)
        panel.set_yticklabels([point["label"] for point in points], fontsize=8.5)
        panel.set_xlabel("Average precision", fontsize=9)
        panel.set_title(stage["stage"][:1].upper() + stage["stage"][1:], fontsize=10.5)
        panel.grid(alpha=0.15, axis="x")
        panel.set_axisbelow(True)
        panel.margins(x=0.22)
        for spine in panel.spines.values():
            spine.set_visible(True)
            spine.set_color("#52514e")
            spine.set_linewidth(0.8)

    for panel in panels[len(stages):]:
        panel.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=1.6, markersize=9,
                   label="elegida en ese eje"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    if title:
        figure.suptitle(title, fontsize=11.5)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95 if title else 1.0))
    return _save(figure, path)


def greedy_neighbourhood_forest(
    moves: pd.DataFrame,
    *,
    selected_name: str,
    title: str,
    path: Path,
) -> Path:
    """Every single-coordinate probe from the selected point, as a delta with its
    paired margin -- the same picture as ``ablation_forest``, but for Task 11's
    depth-reopening and single-move neighbourhood rather than the module axes.

    ``moves`` carries ``name``, ``delta``, ``low`` and ``high`` (the paired-margin
    bounds of ``delta``, from ``representation_selection.paired_margin``). A move
    whose margin clears zero (``low > 0``) is drawn in the "improves" colour; every
    other move, including a tie-break, is neutral, because Task 11 only ever accepts
    ``improves`` to replace the selected configuration.
    """
    table = moves.sort_values("delta")
    positions = np.arange(len(table))
    colours = [OBSERVED_COLOR if low > 0 else NEUTRAL for low in table["low"]]

    figure, axes = plt.subplots(figsize=(10, 0.6 * len(table) + 2.0))
    axes.errorbar(
        table["delta"], positions,
        xerr=[table["delta"] - table["low"], table["high"] - table["delta"]],
        fmt="o", ecolor="#888888", capsize=4, markersize=0, lw=1.6, zorder=2,
    )
    axes.scatter(table["delta"], positions, color=colours, s=80, zorder=3, edgecolor="white")
    axes.axvline(0.0, color="#333333", linewidth=2.0, zorder=1)
    for position, (delta, low, high) in enumerate(zip(table["delta"], table["low"], table["high"])):
        axes.text(
            delta, position + 0.24,
            f"{delta:+.4f}  [{low:+.4f}, {high:+.4f}]",
            ha="center", fontsize=8, color="#333333",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels(table["name"], fontsize=9)
    axes.set_ylim(-0.7, len(table) - 0.2)
    axes.set_xlabel(f"Δ AP pareado vs [{selected_name}]")
    axes.set_title(title)
    axes.grid(axis="x", alpha=0.25)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, path)


def final_candidates_bar(rows: list[dict], *, title: str, path: Path) -> Path:
    """The last chart of the flow: only the frozen finalists, on the holdout.

    ``rows`` is ``[{"name": str, "ap": float, "std": float}, ...]`` -- typically the
    linear reference and the chosen candidate, in that order. Unlike every earlier
    chart this one is not a decision aid: it exists to report the single number the
    whole search was aimed at, once, after the holdout has actually been opened.
    """
    figure, axes = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(rows))
    heights = [row["ap"] for row in rows]
    errors = [row["std"] for row in rows]
    colours = [NEUTRAL] + [MODEL_COLOR] * (len(rows) - 1)

    axes.bar(positions, heights, yerr=errors, color=colours[: len(rows)], capsize=6, zorder=2)
    for x, row in zip(positions, rows):
        axes.text(
            x, row["ap"] + row["std"] + 0.01,
            f"AP {row['ap']:.3f} ± {row['std']:.3f}",
            ha="center", fontsize=9,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([row["name"] for row in rows], fontsize=9)
    axes.set_ylabel("Average precision (holdout)")
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return _save(figure, path)
