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
THIRD = "#7d54c7"
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
                keep_brackets=True, learning_rate=1e-4)
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
    "lr": ("learning_rate", "Learning rate"),
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
        text = f"{value:g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            text = f"{mantissa}e{int(exponent)}"
        return text.replace(".", ",")
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
           title_y=0.978, ncol=2, gap1=0.052, gap2=0.093):
    figure.legend(handles=LEGEND, loc="lower center", ncol=ncol, frameon=False,
                  fontsize=11, bbox_to_anchor=(0.5, legend_y), labelcolor=SECONDARY)
    figure.suptitle(title, fontsize=20, color=INK, y=title_y)
    figure.text(0.5, title_y - gap1, first, ha="center", fontsize=11.5, color=SECONDARY)
    figure.text(0.5, title_y - gap2, second, ha="center", fontsize=11.5, color=MUTED)
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

    present = {key: (field, label) for key, (field, label) in SWEEP_AXES.items()
              if any(value == DECLARED[field] for value, _ in grouped[key])}
    missing = [key for key in SWEEP_AXES if key not in present]
    if missing:
        print(f"barrido: falta la base declarada para {missing}, se omiten del panorama "
              f"(quedan puntos sueltos sin el punto de referencia para compararlos)")

    floor, ceiling = bounds(grouped[key] for key in present)
    n_axes = len(present)
    n_cols = 6
    n_rows = -(-n_axes // n_cols)  # ceil
    figure, grid = plt.subplots(n_rows, n_cols, figsize=(3.9 * n_cols, 4.3 * n_rows), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    for index, (key, (field, label)) in enumerate(present.items()):
        axes = grid[index // n_cols][index % n_cols]
        draw_panel(axes, label, grouped[key], DECLARED[field], floor, ceiling)
        if index % n_cols == 0:
            axes.set_ylabel("Average precision", fontsize=10, color=SECONDARY)
        else:
            axes.set_yticklabels([])
    for leftover in range(n_axes, n_rows * n_cols):
        grid[leftover // n_cols][leftover % n_cols].set_visible(False)
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


def embedder_focus(sweep: dict, table: dict, out: Path) -> None:
    """Deep-dive de la capa de embedding: positional encoding y el LayerNorm + Dropout
    de cierre. Distinto del barrido general porque aca interesa contrastar los dos ejes
    que pidio la catedra al nombrar "el embedder", uno al lado del otro y mas grandes."""
    keys = ("positional", "embnorm")

    grouped = {key: [] for key in keys}
    for point in sweep["points"]:
        if point["axis"] in grouped:
            grouped[point["axis"]].append((point["value"], (point["mean"], point["spread"])))
    for key in keys:
        field, _ = SWEEP_AXES[key]
        known = {value for value, _ in grouped[key]}
        for (moved, by_seed) in table.items():
            if len(moved) == 1 and moved[0][0] == field and moved[0][1] not in known:
                grouped[key].append((moved[0][1], summarise(by_seed)))
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value, _ in grouped[key]):
            grouped[key].sort(key=lambda item: item[0])

    floor, ceiling = bounds(grouped.values())
    figure, row = plt.subplots(1, len(keys), figsize=(12.6, 6.6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    for index, key in enumerate(keys):
        field, label = SWEEP_AXES[key]
        axes = row[index]
        draw_panel(axes, label, grouped[key], DECLARED[field], floor, ceiling,
                  width=0.5, label_size=13, title_size=15, value_size=12)
        if index == 0:
            axes.set_ylabel("Average precision", fontsize=12, color=SECONDARY)
        else:
            axes.set_yticklabels([])
    piso = f"{floor:.2f}".replace(".", ",")
    finish(figure, out, "La capa de embedding — average precision por eje",
           "Media de 5 folds × 3 semillas  ·  La barra de error es la dispersión entre semillas",
           f"Un factor por vez desde la configuración declarada  ·  El eje vertical arranca en {piso}, no en cero",
           rect=(0, 0.08, 1, 0.78), legend_y=0.02, gap1=0.09, gap2=0.16)


def interaction(table: dict, out: Path) -> None:
    """La interaccion parentesis x positional: la afirmacion ES que no son paralelas.
    Tres lineas: learned y none son la interaccion declarada; sinusoidal responde una
    pregunta aparte -- si alcanza con tener posiciones, o hace falta que sean
    aprendidas."""
    lines = [
        ("learned", "positional = learned", SERIES),
        ("sinusoidal", "positional = sinusoidal", THIRD),
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

    def declutter(values, min_gap):
        """Empuja valores demasiado cercanos para que las etiquetas no se pisen,
        preservando el orden. No mueve el punto real, solo donde se escribe el texto."""
        order = sorted(range(len(values)), key=lambda i: values[i])
        pushed = [values[order[0]]]
        for i in order[1:]:
            pushed.append(max(values[i], pushed[-1] + min_gap))
        result = [0.0] * len(values)
        for slot, i in enumerate(order):
            result[i] = pushed[slot]
        return result

    dodges = (-0.024, 0.0, 0.024)
    per_line = []
    for dodge, (positional, label, colour) in zip(dodges, lines):
        means = [data[(positional, brackets)][0] for brackets in (False, True)]
        spreads = [data[(positional, brackets)][1] for brackets in (False, True)]
        delta = means[1] - means[0]
        per_line.append(dict(dodge=dodge, label=label, colour=colour,
                             means=means, spreads=spreads, delta=delta))

    label_ys = declutter([line["means"][0] for line in per_line], min_gap=0.011)
    mid_ys = declutter([(line["means"][0] + line["means"][1]) / 2 for line in per_line],
                       min_gap=0.009)

    for line, label_y, mid_y in zip(per_line, label_ys, mid_ys):
        dodge, label, colour = line["dodge"], line["label"], line["colour"]
        means, spreads, delta = line["means"], line["spreads"], line["delta"]
        shifted = [value + dodge for value in x]
        axes.errorbar(shifted, means, yerr=spreads, fmt="none", ecolor=colour,
                      elinewidth=1.6, capsize=5, alpha=0.75, zorder=3)
        axes.plot(shifted, means, color=colour, linewidth=2.4, zorder=4)
        axes.plot(shifted, means, "o", color=colour, markersize=11,
                  markeredgecolor=SURFACE, markeredgewidth=2.5, zorder=5)
        axes.text(0.5 + dodge, mid_y, f"{delta:+.4f}".replace(".", ","), ha="center",
                  va="center", fontsize=12.5, color=colour,
                  bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
        axes.plot([-0.07, -0.025], [label_y, means[0]], color=colour, linewidth=0.9,
                  alpha=0.55, zorder=2) if abs(label_y - means[0]) > 1e-6 else None
        axes.text(-0.08, label_y, label, ha="right", va="center",
                  fontsize=12.5, color=colour)

    both = [pair for pair in data.values()]
    floor = min(m - s for m, s in both) - 0.010
    ceiling = max(m + s for m, s in both) + 0.012

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
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    print(f"figura: {out}")


# --- resultados: de fit al holdout --------------------------------------------------
DISPLAY_NAME = {"fit": "fit", "corte": "corte", "validation": "validación", "test": "test"}
SEEN_COLOUR = {"gradientes": SERIES, "solo el encoder": THIRD, "nada": BASE}


SHORT_QUESTION = {
    ("fit", "corte"): "memorización",
    ("corte", "validation"): "generalización",
    ("validation", "test"): "sobrevive al holdout",
}


def resultados_puntos(overfitting: dict, out: Path) -> None:
    """La misma idea que el forest plot original, prolija: sin flechas cruzando el panel.

    Los cuatro niveles siguen siendo un punto con su desvío, pero las brechas se leen en
    un carril aparte a la izquierda -- en coordenadas de fila, no de AP -- así una brecha
    chica (corte→validación) y una grande (fit→corte) no compiten por el mismo espacio.
    Los márgenes son fijos (``add_axes``, no ``tight_layout``): el texto de los carriles
    vive fuera del área de datos y layout automático no lo ve al calcular espacio.
    """
    levels = overfitting["levels"]
    sizes = overfitting["rows_per_level"]
    gaps = overfitting["gaps"]
    positions = list(range(len(levels)))[::-1]

    figure = plt.figure(figsize=(11.8, 6.2), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes = figure.add_axes((0.44, 0.16, 0.52, 0.62))
    axes.set_facecolor(SURFACE)
    trans = axes.get_yaxis_transform()

    values = [level["average_precision"] for level in levels]
    floor, ceiling = min(values) - 0.028, max(values) + 0.028

    for position, level in zip(positions, levels):
        name = level["name"]
        colour = SEEN_COLOUR[level["seen"]]
        is_test = name == "test"
        marker = "D" if is_test else "o"
        size = 10 if is_test else 11
        if level["deviation"] > 0:
            axes.errorbar(level["average_precision"], position, xerr=level["deviation"],
                          fmt="none", ecolor=colour, elinewidth=1.5, capsize=4, zorder=3,
                          alpha=0.85)
        axes.plot(level["average_precision"], position, marker, markersize=size,
                  color=colour, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        axes.text(level["average_precision"], position + 0.30,
                  f"{level['average_precision']:.3f}".replace(".", ","),
                  ha="center", fontsize=12, color=colour, fontweight="bold")

        n = sizes[name]
        n_text = (f"n = {n:,}" if is_test else f"n ≈ {n:,}").replace(",", ".")
        axes.text(-0.36, position + 0.09, DISPLAY_NAME[name], transform=trans,
                  ha="left", fontsize=13, color=INK, fontweight="bold")
        axes.text(-0.36, position - 0.20, f"{level['seen']} · {n_text}", transform=trans,
                  ha="left", fontsize=9, color=MUTED)

    for (upper, lower), (high, low) in zip(zip(positions, positions[1:]), zip(levels, levels[1:])):
        gap = next(g for g in gaps if g["origin"] == high["name"] and g["destination"] == low["name"])
        delta = gap["delta"]
        colour = SERIES if delta > 0 else BASE
        middle = (upper + lower) / 2
        tag = SHORT_QUESTION[(high["name"], low["name"])]
        axes.annotate("", xy=(-0.62, lower + 0.30), xytext=(-0.62, upper - 0.30),
                      xycoords=trans, textcoords=trans,
                      arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.6))
        axes.text(-0.62, middle, f"{delta:+.3f}".replace(".", ",") + "\n" + tag,
                  transform=trans, ha="center", va="center", fontsize=9.5, color=colour,
                  fontweight="bold", linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE, edgecolor=colour,
                           linewidth=0.8))

    axes.set_yticks([])
    axes.set_ylim(-0.6, len(levels) - 0.4)
    axes.set_xlim(floor, ceiling)
    axes.set_xlabel("Average precision", fontsize=11, color=SECONDARY)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.tick_params(axis="x", length=0, colors=MUTED, labelsize=10)

    figure.suptitle("Resultados — fit, corte, validación, test", fontsize=19, color=INK, y=0.955)
    figure.text(0.5, 0.885,
                "Cuatro niveles según cuánto vio el modelo  ·  la brecha real es memorización, no el salto a test",
                ha="center", fontsize=10.5, color=SECONDARY)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    print(f"figura: {out}")


def resultados_cascada(overfitting: dict, out: Path) -> None:
    """Alternativa al forest plot: un puente entre los cuatro niveles.

    Cada nivel es una barra plana, no un punto con desvío -- la lectura no es "cuatro
    intervalos independientes" sino "una trayectoria con dos caídas y una subida", que es
    lo que la diapositiva efectivamente afirma.
    """
    levels = overfitting["levels"]
    sizes = overfitting["rows_per_level"]
    gaps = overfitting["gaps"]
    x = list(range(len(levels)))
    half = 0.30

    figure, axes = plt.subplots(figsize=(10.6, 6.0), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    values = [level["average_precision"] for level in levels]
    floor, ceiling = min(values) - 0.05, max(values) + 0.045

    for position, level in zip(x, levels):
        value, spread = level["average_precision"], level["deviation"]
        colour = SEEN_COLOUR[level["seen"]]
        if spread > 0:
            axes.add_patch(plt.Rectangle((position - half, value - spread), 2 * half,
                                         2 * spread, facecolor=colour, alpha=0.16,
                                         edgecolor="none", zorder=2))
        axes.plot([position - half, position + half], [value, value], color=colour,
                  linewidth=5, solid_capstyle="round", zorder=4)
        axes.text(position, value + spread + 0.010, f"{value:.3f}".replace(".", ","),
                  ha="center", fontsize=13, color=colour, fontweight="bold")

    for i in range(len(levels) - 1):
        gap = next(g for g in gaps if g["origin"] == levels[i]["name"]
                  and g["destination"] == levels[i + 1]["name"])
        y0, y1 = values[i], values[i + 1]
        colour = SERIES if gap["delta"] > 0 else BASE
        axes.fill_between([i + half, i + 1 - half], [y0, y0], [y1, y1],
                          color=colour, alpha=0.16, zorder=1)
        axes.plot([i + half, i + 1 - half], [y0, y1], color=colour, linewidth=1.6,
                  linestyle="--", zorder=3)
        mid_x, mid_y = i + 0.5, (y0 + y1) / 2
        tag = SHORT_QUESTION[(levels[i]["name"], levels[i + 1]["name"])]
        axes.text(mid_x, mid_y, f"{gap['delta']:+.3f}".replace(".", ",") + "\n" + tag,
                  ha="center", va="center", fontsize=10.5, color=colour, fontweight="bold",
                  linespacing=1.6,
                  bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE, edgecolor=colour,
                           linewidth=0.9), zorder=5)

    labels = []
    for level in levels:
        n = sizes[level["name"]]
        n_text = (f"n = {n:,}" if level["name"] == "test" else f"n ≈ {n:,}").replace(",", ".")
        labels.append(f"{DISPLAY_NAME[level['name']]}\n{level['seen']}\n{n_text}")
    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=10.5, color=INK)
    axes.set_xlim(-0.6, len(levels) - 0.4)
    axes.set_ylim(floor, ceiling)
    axes.set_ylabel("Average precision", fontsize=11, color=SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(AXIS)
    axes.spines["left"].set_color(AXIS)
    axes.tick_params(axis="both", length=0, colors=MUTED, labelsize=10)

    figure.suptitle("Resultados — fit, corte, validación, test", fontsize=20, color=INK, y=0.975)
    figure.text(0.5, 0.915,
                "La trayectoria del modelo entre los cuatro niveles, no cuatro puntos sueltos  ·  "
                "azul = sube, naranja = cae",
                ha="center", fontsize=11, color=SECONDARY)
    figure.tight_layout(rect=(0, 0.02, 1, 0.90))
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
    embedder_focus(sweep, table, figures / "embedder.png")
    tokenizer_grid(table, figures / "tokenizador-2x2.png")
    interaction(table, figures / "interaccion-parentesis-posiciones.png")

    overfitting_paths = glob.glob(str(results / "overfitting" / "*.json"))
    if overfitting_paths:
        overfitting = json.loads(Path(overfitting_paths[0]).read_text(encoding="utf-8"))
        resultados_puntos(overfitting, figures / "resultados-puntos.png")
        resultados_cascada(overfitting, figures / "resultados-cascada.png")
    else:
        print("aviso: no encontré results/overfitting/*.json, salteo las figuras de resultados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
