"""Display-only relabeling for the story's figures: L0/L1/L2 become the bare letters
A/B/C (no "Modelo" prefix -- the charts already make clear these are models), and
every architecture-search variant becomes "C (campo=valor, ...)" instead of its
axis-letter codename (``B depth 2``, ``D d_model 96``, ``V neighbour n_heads 8``,
...). This only parses the predictable name patterns run_architecture.py and
run_greedy_validation.py generate -- it never touches the underlying run names,
digests or caching, purely how charts label things.

    A  = el lineal sobre el texto crudo (la referencia).
    A* = la cota: sin texto crudo, con la frase del título extraída a mano.
    B  = L1, el mismo contrato de columnas con embeddings aprendidos, sin
         autoatención.
    C  = L2, B con un bloque de autoatención agregado.
    C* = C con la arquitectura resuelta por la búsqueda.
"""

from __future__ import annotations

import re
from collections.abc import Callable

BASE_ALIAS = {
    # The two diagnostic bounds are A (same logistic model) with its input changed,
    # not a model of their own -- named that way rather than "piso"/"techo".
    "L0a linear, no text": "A (sin texto)",
    "L0 linear raw EDA": "A",
    "L1 learned embeddings, no attention": "B",
    "L2 learned embeddings with attention": "C",
    "L0b linear, extracted key only": "A*",
    "M selected from directed comparisons": "C*",
    "FINAL bracket d96 L2 h4 piecewise do0.3 lr2e-4 seed99": "C*",
    "L0* lineal (tf-idf, 453 params)": "A",
    "L0* swept-best linear": "A",
}

FIELD_LABEL = {
    "numeric_embedding": "embedding numérico",
    "n_layers": "profundidad",
    "d_model": "ancho",
    "n_heads": "heads",
    "positional": "posición",
    "pooling": "pooling",
    "dropout": "dropout",
}

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], list[tuple[str, str]]]]] = [
    (re.compile(r"^A numeric (\w+)$"), lambda m: [("numeric_embedding", m.group(1))]),
    (re.compile(r"^B depth (\d+)$"), lambda m: [("n_layers", m.group(1))]),
    (re.compile(r"^D d_model (\d+)$"), lambda m: [("d_model", m.group(1))]),
    (re.compile(r"^C (\d+) heads$"), lambda m: [("n_heads", m.group(1))]),
    (re.compile(r"^E learned positional$"), lambda m: [("positional", "learned")]),
    (re.compile(r"^F attention pooling$"), lambda m: [("pooling", "attention")]),
    (re.compile(r"^H dropout ([\d.]+)$"), lambda m: [("dropout", m.group(1))]),
    (re.compile(r"^V anchor depth (\d+)$"), lambda m: [("n_layers", m.group(1))]),
    (re.compile(r"^V depth (\d+) d_model (\d+)$"), lambda m: [("n_layers", m.group(1)), ("d_model", m.group(2))]),
    (re.compile(r"^V depth (\d+) heads (\d+)$"), lambda m: [("n_layers", m.group(1)), ("n_heads", m.group(2))]),
    (re.compile(r"^V neighbour n_layers (\d+)$"), lambda m: [("n_layers", m.group(1))]),
    (re.compile(r"^V neighbour d_model (\d+)$"), lambda m: [("d_model", m.group(1))]),
    (re.compile(r"^V neighbour n_heads (\d+)$"), lambda m: [("n_heads", m.group(1))]),
]


def alias_label(name: str) -> str:
    """Best-effort display label for a declared or generated run name.

    A known base name (the ladder, or M) maps directly to its Modelo A/B/C/C*
    alias. A recognised architecture-search codename becomes "Modelo C
    (campo=valor, ...)", read off the single field (or two, for a Layer-1 probe
    that reopens depth and moves a second axis at once) the pattern encodes.
    Anything else -- an unrecognised name -- is returned unchanged rather than
    guessed at.
    """
    if name in BASE_ALIAS:
        return BASE_ALIAS[name]
    for pattern, extractor in _PATTERNS:
        match = pattern.match(name)
        if match:
            changes = ", ".join(
                f"{FIELD_LABEL.get(field, field)}={value}" for field, value in extractor(match)
            )
            return f"C ({changes})"
    return name


def variant_label(name: str) -> str:
    """Same matching as :func:`alias_label`, but without the "Modelo C" prefix --
    for a chart where every point is already known to be a Modelo C variant (the
    architecture search and its greedy-order validation), so repeating the model
    name on each bar would be redundant. The plain base point reads as "Base"; a
    recognised codename reads as just its changed field(s); anything else falls
    back to :func:`alias_label`."""
    if name == "L2 learned embeddings with attention":
        return "Base"
    for pattern, extractor in _PATTERNS:
        match = pattern.match(name)
        if match:
            return ", ".join(
                f"{FIELD_LABEL.get(field, field)}={value}" for field, value in extractor(match)
            )
    return alias_label(name)
