"""Block diagrams of the two architectures, drawn from a declaration rather than by hand.

Declaring the boxes and the arrows here makes the diagram regenerate from
``scripts.run_architecture_diagram`` like every other figure in ``figures/``, instead of
living as a raster export whose source has to be reopened to fix a tensor shape.

Two conventions the drawing enforces:

**An arrow's label is the tensor that travels along it, not the one the box below
produces.** The tokenizer receives raw text and emits ``(B, L)`` integers; ``d`` appears
only after the embedding layer looks those integers up, so labelling the arrow out of the
tokenizer ``(B, L, d)`` would conflate ids with embeddings. The shapes are attached to
edges and read in order.

**A dashed border means the box has a declared alternative.** ``ARCHITECTURE.md`` §9
lists the modules that vary, and drawing that on the diagram says at a glance which parts
of the design were measured against something else and which are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model.style import BAR_COLOR, HIGHLIGHT, NEUTRAL, OBSERVED_COLOR
from src.model.style import save as _save

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

TEXT_COLOR = OBSERVED_COLOR
TABULAR_COLOR = "#D98C00"
FUSION_COLOR = "#52514e"
OUTPUT_COLOR = BAR_COLOR
HIGHLIGHT_COLOR = HIGHLIGHT

BOX_HEIGHT = 9.0
EDGE_LABEL_SIZE = 8.5


@dataclass(frozen=True)
class Box:
    """One stage, positioned by the centre of its rectangle."""

    key: str
    title: str
    cx: float
    cy: float
    width: float
    lines: tuple[str, ...] = ()
    colour: str = FUSION_COLOR
    dashed: bool = False
    height: float = BOX_HEIGHT

    @property
    def top(self) -> float:
        return self.cy + self.height / 2

    @property
    def bottom(self) -> float:
        return self.cy - self.height / 2


@dataclass(frozen=True)
class Edge:
    """An arrow between two boxes, carrying the shape of what travels along it."""

    origin: str
    destination: str
    label: str = ""
    elbow: bool = False
    entry_dx: float = 0.0
    """Where on the destination's top edge the arrow lands, relative to its centre.
    Two elbows converging on the same box need different entry points, or the second
    one drawn covers the first one's head."""


@dataclass(frozen=True)
class Diagram:
    """Everything one figure draws, so the runner declares and does not lay out."""

    boxes: tuple[Box, ...]
    edges: tuple[Edge, ...]
    notes: tuple[str, ...] = ()
    headers: tuple[tuple[str, float, float, str], ...] = ()
    footnote: str = ""
    figsize: tuple[float, float] = (12.5, 14.0)
    annotations: tuple[tuple[str, float, float], ...] = ()


def _draw_box(axes, box: Box) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (box.cx - box.width / 2, box.bottom),
            box.width,
            box.height,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.6,
            edgecolor=box.colour,
            facecolor="white",
            linestyle=(0, (5, 3)) if box.dashed else "solid",
            zorder=2,
        )
    )
    title_gap, line_gap = 2.6, 2.2
    span = title_gap + (len(box.lines) - 1) * line_gap if box.lines else 0.0
    top = box.cy + span / 2
    axes.text(
        box.cx, top, box.title,
        ha="center", va="center", fontsize=12, color="#22211f", zorder=3,
    )
    for index, line in enumerate(box.lines):
        axes.text(
            box.cx, top - title_gap - index * line_gap, line,
            ha="center", va="center", fontsize=8.5, color="#52514e", zorder=3,
        )


def _draw_edge(axes, edge: Edge, boxes: dict[str, Box]) -> None:
    origin, destination = boxes[edge.origin], boxes[edge.destination]
    if edge.elbow:
        _draw_elbow(axes, origin, destination, edge.label, edge.entry_dx)
        return
    axes.add_patch(
        FancyArrowPatch(
            (origin.cx, origin.bottom),
            (destination.cx, destination.top),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=origin.colour, zorder=1,
        )
    )
    if edge.label:
        axes.text(
            origin.cx + 0.8, (origin.bottom + destination.top) / 2, edge.label,
            ha="left", va="center", fontsize=EDGE_LABEL_SIZE, color="#52514e", zorder=3,
        )


def _draw_elbow(axes, origin: Box, destination: Box, label: str, entry_dx: float) -> None:
    """A right-angled path for the two arrows that converge on the fusion box."""
    corner = destination.top + 6.0
    entry = destination.cx + entry_dx
    axes.plot(
        [origin.cx, origin.cx, entry],
        [origin.bottom, corner, corner],
        color=origin.colour, linewidth=1.4, solid_joinstyle="miter", zorder=1,
    )
    axes.add_patch(
        FancyArrowPatch(
            (entry, corner),
            (entry, destination.top),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=origin.colour, zorder=1,
        )
    )
    if label:
        axes.text(
            origin.cx + 0.8, (origin.bottom + corner) / 2, label,
            ha="left", va="center", fontsize=EDGE_LABEL_SIZE, color="#52514e", zorder=3,
        )


def render(diagram: Diagram, *, title: str, path):
    """Draw one declared diagram and write it where every other figure lands."""
    figure, axes = plt.subplots(figsize=diagram.figsize)
    boxes = {box.key: box for box in diagram.boxes}

    for edge in diagram.edges:
        _draw_edge(axes, edge, boxes)
    for box in diagram.boxes:
        _draw_box(axes, box)

    for label, x, y, colour in diagram.headers:
        axes.text(x, y, label, ha="center", va="center",
                  fontsize=13, color=colour, zorder=3)
    for label, x, y in diagram.annotations:
        axes.text(x, y, label, ha="left", va="center",
                  fontsize=9, color="#52514e", zorder=3)

    lowest = min(box.bottom for box in diagram.boxes)
    if diagram.footnote:
        axes.text(
            2, lowest - 6, diagram.footnote,
            ha="left", va="top", fontsize=8.5, color=NEUTRAL, zorder=3,
        )

    axes.set_xlim(0, 100)
    axes.set_ylim(lowest - (12 if diagram.footnote else 4), 104)
    axes.set_axis_off()
    axes.set_title(title, fontsize=15, pad=8)
    figure.tight_layout()
    return _save(figure, path)


def two_towers() -> Diagram:
    """The late-fusion architecture of ``ARCHITECTURE.md``.

    Dashed boxes are the modules §9 declares alternatives for.
    """
    text_x, tab_x, mid_x = 28.0, 74.0, 51.0
    boxes = (
        Box("text_in", "title · description · ingredients", text_x, 95, 40,
            ("texto libre",), TEXT_COLOR),
        Box("tokenizer", "Tokenizador", text_x, 80, 40,
            ("WordPiece entrenado sobre el corpus",
             "[CLS] A [SEP] B [SEP] C [SEP]",
             "conserva la puntuación"),
            TEXT_COLOR, dashed=True, height=11.5),
        Box("embedding", "Capa de embedding", text_x, 64, 40,
            ("LayerNorm(tok + seg + pos)", "→ Dropout(0.1)"),
            TEXT_COLOR, dashed=True, height=10.5),
        Box("encoder", "Encoder", text_x, 47, 40,
            ("N × [ MHA bidireccional + FFN ]",
             "residual + LayerNorm después de cada suma (post-LN)",
             "FFN: Linear(d→4d) · GELU · Linear(4d→d)"),
            TEXT_COLOR, dashed=True, height=12.0),
        Box("pooler", "Pooler", text_x, 31, 40,
            ("h[CLS] → Linear(d→d) + tanh",), TEXT_COLOR, dashed=True),

        Box("tab_in", "category · allergens · price_position", tab_x, 95, 40,
            ("categóricas + numérica",), TABULAR_COLOR),
        Box("encoding", "Codificación tabular", tab_x, 80, 40,
            ("one-hot(12) · one-hot(7)",
             "piecewise-linear(10) + faltante(1)"),
            TABULAR_COLOR, height=11.5),
        Box("tab_mlp", "MLP", tab_x, 64, 40,
            ("Linear(30→32) → ReLU → Dropout", "→ Linear(32→16)"),
            TABULAR_COLOR, dashed=True, height=10.5),

        Box("fusion", "Fusión", mid_x, 8, 30, ("concat",), FUSION_COLOR, dashed=True),
        Box("head", "MLP de salida", mid_x, -8, 36,
            ("Linear(80→32) → ReLU → Dropout", "→ Linear(32→1)"),
            FUSION_COLOR, dashed=True, height=10.5),
        Box("output", "sigmoid → p(bought) = BTR", mid_x, -24, 42,
            ("BCEWithLogitsLoss · PR-AUC primaria",), OUTPUT_COLOR),
    )
    edges = (
        Edge("text_in", "tokenizer", "texto crudo"),
        Edge("tokenizer", "embedding", "(B, L) enteros"),
        Edge("embedding", "encoder", "(B, L, d)"),
        Edge("encoder", "pooler", "(B, L, d)"),
        Edge("tab_in", "encoding", ""),
        Edge("encoding", "tab_mlp", "x_tab (B, 30)"),
        Edge("pooler", "fusion", "h_text (B, 64)", elbow=True, entry_dx=-7.0),
        Edge("tab_mlp", "fusion", "h_tab (B, 16)", elbow=True, entry_dx=7.0),
        Edge("fusion", "head", "(B, 80)"),
        Edge("head", "output", "logit (B, 1)"),
    )
    return Diagram(
        boxes=boxes,
        edges=edges,
        headers=(
            ("TORRE DE TEXTO", text_x, 101.5, TEXT_COLOR),
            ("TORRE TABULAR", tab_x, 101.5, TABULAR_COLOR),
        ),
        footnote=(
            "B = batch · L = longitud de la secuencia · d = d_model (64 en la "
            "configuración base)\n"
            "x_tab = 12 category + 7 allergens + 10 price_position + 1 faltante = 30. "
            "Las 4.455 filas sin alérgeno declarado son todo-ceros, no una columna.\n"
            "Borde punteado: el módulo tiene alternativas declaradas y medidas "
            "(tokenizador, positional, encoder, pooler, torre tabular y MLP de salida)."
        ),
        figsize=(13.0, 15.0),
    )


def personalised() -> Diagram:
    """Where a user factor would enter, and what it would cost elsewhere.

    The first constraint is data, not architecture. The dataset has 22 columns and none
    of them identifies a person -- ``query_id`` is a search, not a user -- so
    the current target is a per-product marginal ``P(bought | producto)`` and no
    architectural change makes it conditional on someone.

    The second claim is that the user tower should reuse the text encoder rather than
    learn an id embedding. An id table has nothing to say about a user seen once, and
    most users are seen once; a user described by the titles of what they bought is a
    sequence the existing tower already knows how to read.

    The third is that the fusion has to be multiplicative. Personalisation *is* an
    interaction -- the same product ranking differently for two people -- and a
    concatenation followed by a linear layer can only add the two contributions.
    """
    text_x, user_x, tab_x = 16.0, 48.0, 82.0
    gate_x, mid_x = 32.0, 57.0
    boxes = (
        Box("text_in", "Producto: texto", text_x, 94, 28,
            ("title · description", "ingredients"), TEXT_COLOR, height=12.0),
        Box("user_in", "Usuario: historial", user_x, 94, 28,
            ("títulos de lo que compró", "o vio antes"),
            HIGHLIGHT_COLOR, height=12.0),
        Box("tab_in", "Producto: tabular", tab_x, 94, 28,
            ("category · allergens", "price_position"), TABULAR_COLOR, height=12.0),

        Box("text_tower", "Torre de texto", text_x, 74, 28,
            ("encoder + pooler", "sin cambios"), TEXT_COLOR, height=12.0),
        Box("user_tower", "Torre de usuario", user_x, 74, 28,
            ("el mismo encoder,", "pesos compartidos", "→ media sobre el historial"),
            HIGHLIGHT_COLOR, dashed=True, height=13.5),
        Box("tab_tower", "Torre tabular", tab_x, 74, 28,
            ("MLP", "sin cambios"), TABULAR_COLOR, height=12.0),

        Box("gate", "Modulación (FiLM)", gate_x, 50, 34,
            ("h_text ⊙ σ(W·h_user) + b(h_user)",
             "el usuario cambia cómo se lee el texto"),
            HIGHLIGHT_COLOR, dashed=True, height=11.5),
        Box("fusion", "Fusión", mid_x, 26, 28, ("concat",), FUSION_COLOR, height=9.0),
        Box("head", "MLP de salida", mid_x, 8, 32,
            ("Linear → ReLU → Linear",), FUSION_COLOR, height=9.5),
        Box("output", "p(bought | producto, usuario)", mid_x, -9, 40,
            ("BTR personalizado",), OUTPUT_COLOR, height=9.5),
    )
    edges = (
        Edge("text_in", "text_tower", ""),
        Edge("user_in", "user_tower", ""),
        Edge("tab_in", "tab_tower", ""),
        Edge("text_tower", "gate", "h_text", elbow=True, entry_dx=-8.0),
        Edge("user_tower", "gate", "h_user", elbow=True, entry_dx=8.0),
        Edge("gate", "fusion", "", elbow=True, entry_dx=-6.0),
        Edge("tab_tower", "fusion", "h_tab", elbow=True, entry_dx=6.0),
        Edge("fusion", "head", ""),
        Edge("head", "output", "logit"),
    )
    return Diagram(
        boxes=boxes,
        edges=edges,
        footnote=(
            "El dataset no tiene columna de usuario: query_id es una búsqueda, no una "
            "persona. Sin ese dato\n"
            "el BTR es una marginal por producto y ninguna arquitectura lo vuelve "
            "condicional.\n\n"
            "Lo que también cambia: la partición pasa a ser por usuario y no por "
            "query · la métrica pasa a ser\n"
            "ranking por usuario (NDCG@k) y no PR-AUC global · el BTR deja de ser una "
            "propiedad del producto."
        ),
        figsize=(12.5, 12.0),
    )


def one_tower() -> Diagram:
    """The single-sequence architecture of ``OLD_ARCHITECTURE.md``.

    Carries the dilution arithmetic as an annotation: the three tabular columns took 7%
    of the pooled vector by construction, independently of how much signal they held.
    """
    mid_x = 50.0
    boxes = (
        Box("sequence", "Una sola secuencia por fila", mid_x, 92, 76,
            ("[CLS] · 49 posiciones de texto (≈39 reales + padding) · "
             "⟨category⟩ · ⟨allergens⟩ · price_position",
             "todas las posiciones son vectores de ℝ^d_model",
             "positional = none, porque las columnas no tienen orden"),
            NEUTRAL, height=13.0),
        Box("encoder", "Un único encoder", mid_x, 70, 50,
            ("N × [ MHA + FFN ]", "texto y tabulares comparten la atención"),
            NEUTRAL, height=10.5),
        Box("pooling", "mean pooling", mid_x, 52, 50,
            ("sobre todas las posiciones no-padding,", "tabulares incluidas"),
            NEUTRAL, height=10.5),
        Box("head", "Cabeza", mid_x, 36, 42,
            ("Linear(d→d) → ReLU → Linear(d→1)",), NEUTRAL),
        Box("output", "sigmoid → p(bought) = BTR", mid_x, 21, 42,
            ("mismo target, misma pérdida, misma partición",), OUTPUT_COLOR),
    )
    edges = (
        Edge("sequence", "encoder", "(B, 53, d)"),
        Edge("encoder", "pooling", "(B, 53, d)"),
        Edge("pooling", "head", "(B, d)"),
        Edge("head", "output", "logit (B, 1)"),
    )
    return Diagram(
        boxes=boxes,
        edges=edges,
        annotations=(
            ("posiciones promediadas ≈ 1 + 39,4 + 2 + 1 = 43,4", 4, 46),
            ("peso de lo tabular = 3 / 43,4 ≈ 7 %", 4, 42),
        ),
        footnote=(
            "Las 3 columnas tabulares se llevaban el 7 % del promedio por construcción,\n"
            "sin importar cuánta señal tuvieran. Eso es lo que la fusión tardía corrige."
        ),
        figsize=(11.0, 11.0),
    )
