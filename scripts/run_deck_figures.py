"""Todas las figuras de ablacion de la presentacion, en un solo lugar.

    python3 scripts/run_deck_figures.py

Lee los JSON que ya escribieron los runners (`sweep.json`) y los resultados crudos por
corrida; no entrena nada y no importa nada de `src/`, asi que corre con cualquier Python
con matplotlib. Escribe en `figures/deck/` para no pisar las figuras de los runners.

Criterio unico para las figuras, para que se puedan mirar juntas:
un factor por vez desde la configuracion declarada, AP absoluto (no diferencias
pareadas), media de 5 folds x 3 semillas y la dispersion entre semillas como barra de
error. La configuracion declarada va en naranja y su AP se repite como linea horizontal.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- paleta, validada contra el fondo claro de las diapositivas -------------------
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
BASE = "#eb6834"
BASE_SOFT = "#f7c3ab"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LEGEND = [Patch(facecolor=BASE, label="Configuración declarada"),
          Patch(facecolor=SERIES, label="Variación")]

# --- la configuracion declarada, campo por campo ----------------------------------
DECLARED = dict(n_heads=4, n_layers=1, d_model=64, ffn_multiplier=4, dropout=0.1,
                pooling="cls", positional="learned", embedding_norm=True,
                tab_tower="mlp", fusion_head="mlp", tokenizer="wordpiece",
                keep_brackets=True)
PROTOCOL = dict(pooler_projection=True, early_stopping="ap", epochs=150)

SWEEP_AXES = {
    "heads": ("n_heads", "Cantidad de heads"),
    "layers": ("n_layers", "Encoders apilados"),
    "width": ("d_model", "d_model"),
    "ffn": ("ffn_multiplier", "Dimensión del FFN (× d_model)"),
    "dropout": ("dropout", "Dropout"),
    "pooling": ("pooling", "Pooler"),
    "positional": ("positional", "Positional encoding"),
    "embnorm": ("embedding_norm", "LayerNorm + Dropout de cierre"),
    "tab": ("tab_tower", "Torre tabular"),
    "head": ("fusion_head", "MLP de salida"),
}


# --- lectura de resultados crudos --------------------------------------------------
def runs_by_config(results: Path) -> dict:
    """AP medio por (campos movidos, semilla), leyendo cada corrida grabada."""
    table: dict[tuple, dict] = {}
    for path in glob.glob(str(results / "*.json")):
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        config = run.get("config", {})
        if any(config.get(field) != value for field, value in PROTOCOL.items()):
            continue
        moved = tuple(sorted((field, config.get(field)) for field in DECLARED
                             if config.get(field) != DECLARED[field]))
        folds = [fold["average_precision"] for fold in run["folds"]]
        if folds:
            table.setdefault(moved, {})[config.get("seed")] = statistics.fmean(folds)
    return table


def summarise(by_seed: dict) -> tuple[float, float]:
    means = list(by_seed.values())
    return statistics.fmean(means), (statistics.stdev(means) if len(means) > 1 else 0.0)


def one_factor(table: dict, field: str, value) -> tuple[float, float] | None:
    key = () if value == DECLARED[field] else ((field, value),)
    found = table.get(key)
    return summarise(found) if found else None


def tick(value) -> str:
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)


# --- un panel ----------------------------------------------------------------------
def draw_panel(axes, title, points, base_value, floor, ceiling, *, width=0.62,
               label_size=10, title_size=12, value_size=9.5, labels=None):
    axes.set_facecolor(SURFACE)
    reference = next(mean for value, (mean, _) in points if value == base_value)
    axes.axhline(reference, color=BASE_SOFT, linewidth=1.4, zorder=1)
    for position, (value, (mean, spread)) in enumerate(points):
        axes.bar(position, mean - floor, bottom=floor, width=width,
                 color=BASE if value == base_value else SERIES, linewidth=0, zorder=3)
        axes.errorbar(position, mean, yerr=spread, fmt="none", ecolor=SECONDARY,
                      elinewidth=1.3, capsize=4, zorder=4)
        axes.text(position, mean + spread + 0.004, f"{mean:.3f}".replace(".", ","),
                  ha="center", va="bottom", fontsize=value_size, color=SECONDARY)
    shown = labels or [tick(value) for value, _ in points]
    axes.set_xticks(range(len(points)))
    axes.set_xticklabels([name + ("\n(Base)" if value == base_value else "")
                          for name, (value, _) in zip(shown, points)],
                         fontsize=label_size, color=INK)
    axes.set_ylim(floor, ceiling)
    axes.set_xlim(-0.62, len(points) - 0.38)
    axes.set_title(title, fontsize=title_size, color=INK, pad=10)
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.spines["left"].set_color(AXIS)
    axes.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)


def bounds(groups, pad_low=0.018, pad_high=0.020):
    values = [pair for points in groups for _, pair in points]
    return (min(m - s for m, s in values) - pad_low,
            max(m + s for m, s in values) + pad_high)


def finish(figure, out: Path, title, first, second, *, rect, legend_y=0.005,
           title_y=0.978, ncol=2):
    figure.legend(handles=LEGEND, loc="lower center", ncol=ncol, frameon=False,
                  fontsize=11, bbox_to_anchor=(0.5, legend_y), labelcolor=SECONDARY)
    figure.suptitle(title, fontsize=20, color=INK, y=title_y)
    figure.text(0.5, title_y - 0.052, first, ha="center", fontsize=11.5, color=SECONDARY)
    figure.text(0.5, title_y - 0.093, second, ha="center", fontsize=11.5, color=MUTED)
    figure.tight_layout(rect=rect)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    print(f"figura: {out}")


# --- las tres figuras ---------------------------------------------------------------
def architecture(sweep: dict, table: dict, out: Path) -> None:
    grouped = {key: [] for key in SWEEP_AXES}
    for point in sweep["points"]:
        grouped[point["axis"]].append((point["value"], (point["mean"], point["spread"])))
    for key, (field, _) in SWEEP_AXES.items():          # valores corridos fuera de la grilla
        known = {value for value, _ in grouped[key]}
        for (moved, by_seed) in table.items():
            if len(moved) == 1 and moved[0][0] == field and moved[0][1] not in known:
                grouped[key].append((moved[0][1], summarise(by_seed)))
        # los ejes de texto conservan el orden declarado en el barrido; los numericos
        # se reordenan para que un valor corrido aparte caiga en su lugar
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value, _ in grouped[key]):
            grouped[key].sort(key=lambda item: item[0])

    floor, ceiling = bounds(grouped.values())
    figure, grid = plt.subplots(2, 5, figsize=(19.5, 8.6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    for index, (key, (field, label)) in enumerate(SWEEP_AXES.items()):
        axes = grid[index // 5][index % 5]
        draw_panel(axes, label, grouped[key], DECLARED[field], floor, ceiling)
        if index % 5 == 0:
            axes.set_ylabel("Average precision", fontsize=10, color=SECONDARY)
        else:
            axes.set_yticklabels([])
    piso = f"{floor:.2f}".replace(".", ",")
    finish(figure, out, "Barrido de arquitectura — average precision por eje",
           "Media de 5 folds × 3 semillas  ·  La barra de error es la dispersión entre semillas",
           f"Un factor por vez desde la configuración declarada  ·  El eje vertical arranca en {piso}, no en cero",
           rect=(0, 0.045, 1, 0.875))


def tokenizer_grid(table: dict, out: Path) -> None:
    """El 2x2 entero: el unico lugar donde se ve la celda de control."""
    cells = {}
    for tokenizer in ("whole-word", "wordpiece"):
        for brackets in (False, True):
            moved = tuple(sorted([(f, v) for f, v in
                                  (("tokenizer", tokenizer), ("keep_brackets", brackets))
                                  if v != DECLARED[f]]))
            found = table.get(moved)
            if not found:
                raise SystemExit(f"falta la celda {tokenizer}/{brackets}")
            cells[(tokenizer, brackets)] = summarise(found)

    floor, ceiling = bounds([[(k, v) for k, v in cells.items()]])
    figure, axes = plt.subplots(figsize=(11.0, 6.6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    series = [(False, "Sin paréntesis", SERIES), (True, "Con paréntesis", BASE)]
    groups = [("whole-word", "Palabras enteras"), ("wordpiece", "WordPiece")]
    width = 0.30
    for offset, (brackets, label, colour) in zip((-0.5, 0.5), series):
        for index, (tokenizer, _) in enumerate(groups):
            mean, spread = cells[(tokenizer, brackets)]
            position = index + offset * (width + 0.03)
            axes.bar(position, mean - floor, bottom=floor, width=width, color=colour,
                     linewidth=0, zorder=3)
            axes.errorbar(position, mean, yerr=spread, fmt="none", ecolor=SECONDARY,
                          elinewidth=1.4, capsize=5, zorder=4)
            axes.text(position, mean + spread + 0.0022, f"{mean:.4f}".replace(".", ","),
                      ha="center", va="bottom", fontsize=12, color=INK)
    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels([label for _, label in groups], fontsize=14, color=INK)
    axes.set_ylim(floor, ceiling)
    axes.set_xlim(-0.62, len(groups) - 0.38)
    axes.set_ylabel("Average precision", fontsize=12, color=SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.spines["left"].set_color(AXIS)
    axes.tick_params(axis="both", length=0, colors=MUTED, labelsize=11)
    axes.legend(handles=[Patch(facecolor=colour, label=label)
                         for _, label, colour in series],
                loc="upper left", frameon=False, fontsize=12, labelcolor=SECONDARY)
    piso = f"{floor:.2f}".replace(".", ",")
    figure.suptitle("El delimitador, no el tokenizador", fontsize=20, color=INK, y=0.975)
    figure.text(0.5, 0.917, "AP por celda del 2×2  ·  Media de 5 folds × 3 semillas  ·  "
                "la barra de error es la dispersión entre semillas",
                ha="center", fontsize=11.5, color=SECONDARY)
    figure.text(0.5, 0.878, f"Todas con positional = learned  ·  El eje vertical arranca "
                f"en {piso}, no en cero", ha="center", fontsize=11.5, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.855))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    print(f"figura: {out}")


def interaction(table: dict, out: Path) -> None:
    """La interaccion parentesis x positional: la afirmacion ES que no son paralelas."""
    lines = [
        ("learned", "positional = learned", SERIES),
        ("none", "positional = none", BASE),
    ]
    x = [0.0, 1.0]
    names = ["Sin paréntesis", "Con paréntesis"]

    data = {}
    for positional, _, _ in lines:
        for brackets in (False, True):
            moved = tuple(sorted([(f, v) for f, v in
                                  (("keep_brackets", brackets), ("positional", positional))
                                  if v != DECLARED[f]]))
            found = table.get(moved)
            if not found:
                raise SystemExit(f"falta la celda positional={positional} brackets={brackets}")
            data[(positional, brackets)] = summarise(found)

    figure, axes = plt.subplots(figsize=(11.4, 6.8), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for dodge, (positional, label, colour) in zip((-0.012, 0.012), lines):
        means = [data[(positional, brackets)][0] for brackets in (False, True)]
        spreads = [data[(positional, brackets)][1] for brackets in (False, True)]
        shifted = [value + dodge for value in x]
        axes.errorbar(shifted, means, yerr=spreads, fmt="none", ecolor=colour,
                      elinewidth=1.6, capsize=5, alpha=0.75, zorder=3)
        axes.plot(shifted, means, color=colour, linewidth=2.4, zorder=4)
        axes.plot(shifted, means, "o", color=colour, markersize=11,
                  markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=5)
        delta = means[1] - means[0]
        axes.text(0.5 + dodge, (means[0] + means[1]) / 2 + (0.0028 if delta > 0 else -0.0042),
                  f"{delta:+.4f}".replace(".", ","), ha="center",
                  va="bottom" if delta > 0 else "top", fontsize=13, color=colour)
        axes.text(-0.07, means[0], label, ha="right", va="center",
                  fontsize=12.5, color=colour)

    both = [pair for pair in data.values()]
    floor = min(m - s for m, s in both) - 0.010
    ceiling = max(m + s for m, s in both) + 0.012

    interaction_value = ((data[("learned", True)][0] - data[("learned", False)][0])
                         - (data[("none", True)][0] - data[("none", False)][0]))
    axes.text(0.28, floor + 0.004,
              f"Interacción  {interaction_value:+.4f}".replace(".", ",") + "   ·   3/3 semillas",
              ha="center", va="bottom", fontsize=13, color=INK)

    axes.set_xticks(x)
    axes.set_xticklabels(names, fontsize=14, color=INK)
    axes.set_xlim(-0.62, 1.18)
    axes.set_ylim(floor, ceiling)
    axes.set_ylabel("Average precision", fontsize=12, color=SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.spines["left"].set_color(AXIS)
    axes.tick_params(axis="both", length=0, colors=MUTED, labelsize=11)
    axes.legend(handles=[Patch(facecolor=colour, label=label) for _, label, colour in lines],
                loc="lower right", frameon=False, fontsize=12, labelcolor=SECONDARY)

    figure.suptitle("Los paréntesis sirven sólo si hay posiciones", fontsize=20,
                    color=INK, y=0.975)
    figure.text(0.5, 0.917,
                "Si el efecto del delimitador no dependiera del orden, las dos líneas "
                "serían paralelas  ·  Todas con WordPiece",
                ha="center", fontsize=11.5, color=SECONDARY)
    figure.text(0.5, 0.878,
                "La interacción está tirada por una semilla: +0,071 · +0,015 · +0,012  ·  "
                "Lo que no depende de umbrales es el signo, seis de seis",
                ha="center", fontsize=11.5, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 0.855))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    print(f"figura: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/v2-dos-torres")
    parser.add_argument("--figures", default="figures/deck")
    args = parser.parse_args()

    results = Path(args.results)
    figures = Path(args.figures)
    sweep = json.loads((results / "architecture" / "sweep.json").read_text(encoding="utf-8"))
    table = runs_by_config(results)

    architecture(sweep, table, figures / "barrido-arquitectura.png")
    tokenizer_grid(table, figures / "tokenizador-2x2.png")
    interaction(table, figures / "interaccion-parentesis-posiciones.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
