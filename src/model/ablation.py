"""La grilla que comparten D1 y D10, y cómo se lee en cada uno de los dos sentidos.

D1 pregunta si la ganancia la dan los paréntesis o el tipo de tokenizador. D10 pregunta
*por qué* la dan, apagando el mecanismo: sin positional encoding la self-attention es
permutación-equivariante y ``[CLS]`` queda invariante a permutaciones del resto, así que
el modelo ve un multiconjunto de tokens y el paréntesis está presente pero es inutilizable
como delimitador.

Los dos tickets se miden sobre la misma grilla porque **comparten dos celdas**. El caché
va por digest, así que declararlas una sola vez hace que correr los dos cueste seis
configuraciones y no ocho:

======  ===========  ==============  ============  ==========
celda   tokenizer    keep_brackets   positional    tickets
======  ===========  ==============  ============  ==========
A       whole-word   False           learned       D1
F       whole-word   True            learned       D1
B       wordpiece    True            learned       D1, D10
C       wordpiece    False           learned       D1, D10
D       wordpiece    True            none          D10
E       wordpiece    False           none          D10
======  ===========  ==============  ============  ==========

D1 son las cuatro primeras: un 2x2 de ``tokenizer`` por ``keep_brackets``, no un eje de
tres brazos. Un eje de tres mueve las subpalabras y la puntuación a la vez entre v1 y la
propuesta, y ninguna diferencia queda atribuible a una sola de las dos.

Las dos filas no son simétricas y conviene decirlo antes de que lo pregunten. En
``wordpiece`` el factor es limpio: ``keep_brackets=False`` borra los paréntesis y deja el
resto de la puntuación intacta. En ``whole-word`` es más grueso, porque ``False`` tiene que
reproducir v1 exactamente y v1 borraba toda la puntuación; ``True`` devuelve la puntuación
entera, no sólo los paréntesis. La afirmación fuerte sobre los paréntesis vive en la fila
``wordpiece``; la fila ``whole-word`` aporta la línea base histórica y el contraste de
palabras enteras.

**D1 corre con ``positional = learned`` fijo, y eso no es un detalle.** Corrido con
``positional = none`` las cuatro celdas de texto miden lo mismo, D1 daría cero y la
conclusión sería que los paréntesis no sirven. Queda declarado en la grilla en vez de
confiado a que alguien se acuerde al lanzarlo.

Un matiz que conviene sostener al enunciar el resultado: los paréntesis **no** reducen la
colisión de embeddings. Bajo WordPiece ``(Customer Favorite)`` y ``(Shopper Favorite)``
--67,7 % contra 2,8 %-- siguen compartiendo ``favorite``, ``(`` y ``)``. Lo que aportan es
un delimitador para que la atención se ancle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.model.configs import RunConfig
from src.model.representation_selection import SEEDS, seed_mean, seed_spread

D1 = "D1"
D10 = "D10"

WHOLE_WORD = "whole-word"
WORDPIECE = "wordpiece"
LEARNED = "learned"
NONE = "none"


@dataclass(frozen=True)
class Cell:
    """Una configuración de la grilla, con los tickets que la usan."""

    key: str
    label: str
    tokenizer: str
    keep_brackets: bool
    positional: str
    tickets: tuple[str, ...]

    def config(self, base: RunConfig, seed: int) -> RunConfig:
        return replace(
            base,
            tokenizer=self.tokenizer,
            keep_brackets=self.keep_brackets,
            positional=self.positional,
            seed=seed,
        )


GRID = (
    Cell("A", "regex de v1 (sin puntuación)", WHOLE_WORD, False, LEARNED, (D1,)),
    Cell("F", "palabras enteras con puntuación", WHOLE_WORD, True, LEARNED, (D1,)),
    Cell("B", "WordPiece con paréntesis", WORDPIECE, True, LEARNED, (D1, D10)),
    Cell("C", "WordPiece sin paréntesis (control)", WORDPIECE, False, LEARNED, (D1, D10)),
    Cell("D", "WordPiece con paréntesis · positional=none", WORDPIECE, True, NONE, (D10,)),
    Cell("E", "WordPiece sin paréntesis · positional=none", WORDPIECE, False, NONE, (D10,)),
)

BY_KEY = {cell.key: cell for cell in GRID}


def cells_for(tickets: tuple[str, ...]) -> tuple[Cell, ...]:
    """Las celdas que hacen falta para los tickets pedidos, sin repetir las compartidas."""
    return tuple(cell for cell in GRID if set(cell.tickets) & set(tickets))


@dataclass(frozen=True)
class Measured:
    """Una celda ya corrida con las tres semillas."""

    cell: Cell
    runs: tuple[tuple[float, ...], ...]

    @property
    def mean(self) -> float:
        return seed_mean([list(run) for run in self.runs], label=self.cell.key)

    @property
    def spread(self) -> float:
        return seed_spread([list(run) for run in self.runs], label=self.cell.key)


FLOAT_TOLERANCE = 1e-12
"""Guarda contra ruido de coma flotante, no un umbral con significado. Sin ella, dos
mediciones de dispersión nula se separan por 1e-16 y la comparación se vuelve un
volado."""


def distinguishable(delta: float, spreads: tuple[float, ...]) -> bool:
    """Regla declarada, no un test: una diferencia cuenta si supera la suma de las
    dispersiones entre semillas de las dos celdas que compara.

    Es deliberadamente conservadora y deliberadamente simple. Los cinco folds comparten
    filas de entrenamiento, así que su dispersión no es un intervalo de confianza y
    fabricar un p-valor acá sería darle a un número descriptivo una autoridad que no
    tiene. Lo que se reporta es la regla y el resultado de aplicarla.
    """
    return abs(delta) > sum(spreads) + FLOAT_TOLERANCE


def _by_key(measured: list[Measured]) -> dict[str, Measured]:
    return {item.cell.key: item for item in measured}


def read_d1(measured: list[Measured]) -> str:
    """La conclusión de D1, escrita en cualquiera de los sentidos en que salga.

    Los cuatro contrastes del 2x2 se reportan siempre; la conclusión la deciden los dos
    de la fila ``wordpiece`` y la columna sin puntuación, que son los limpios.
    """
    found = _by_key(measured)
    missing = {"A", "F", "B", "C"} - set(found)
    if missing:
        return f"D1 incompleto: faltan las celdas {sorted(missing)}"

    a, f, b, c = found["A"], found["F"], found["B"], found["C"]
    brackets = b.mean - c.mean
    punctuation = f.mean - a.mean
    tokenizer = c.mean - a.mean
    tokenizer_marked = b.mean - f.mean
    brackets_real = distinguishable(brackets, (b.spread, c.spread))
    punctuation_real = distinguishable(punctuation, (f.spread, a.spread))
    tokenizer_real = distinguishable(tokenizer, (c.spread, a.spread))

    lines = [
        f"paréntesis en wordpiece  (B − C) = {brackets:+.4f}  "
        f"{'distinguible' if brackets_real else 'dentro del ruido'}",
        f"puntuación en whole-word (F − A) = {punctuation:+.4f}  "
        f"{'distinguible' if punctuation_real else 'dentro del ruido'}",
        f"tokenizador sin puntuación (C − A) = {tokenizer:+.4f}  "
        f"{'distinguible' if tokenizer_real else 'dentro del ruido'}",
        f"tokenizador con puntuación (B − F) = {tokenizer_marked:+.4f}",
    ]
    if brackets_real and not tokenizer_real:
        lines.append(
            "La mejora viene de conservar los paréntesis, no de WordPiece en general: "
            "el control queda al nivel del regex de v1."
        )
    elif brackets_real and tokenizer_real:
        lines.append(
            "Las dos cosas aportan. El delimitador pesa más que el cambio de "
            "tokenizador, pero WordPiece suma por su cuenta."
        )
    elif tokenizer_real and not brackets_real:
        lines.append(
            "Lo que mueve el AP es el tokenizador y no los paréntesis. La historia del "
            "delimitador que íbamos a contar no se sostiene."
        )
    else:
        lines.append(
            "Ninguno de los dos factores mueve el AP por encima del ruido entre "
            "semillas. Se reporta así: en este dataset el vocabulario es cerrado y el "
            "cambio de tokenización no cambia lo que el modelo puede aprender."
        )
    lines.append(
        "La fila whole-word (F − A) mide toda la puntuación, no sólo los paréntesis, "
        "porque su celda sin puntuación tiene que reproducir v1 exactamente. Es la "
        "línea base histórica, no el factor limpio."
    )
    return "\n".join(lines)


def read_d10(measured: list[Measured]) -> str:
    """La lectura de D10: la ganancia tiene que aparecer en una sola celda."""
    found = _by_key(measured)
    missing = {"B", "C", "D", "E"} - set(found)
    if missing:
        return f"D10 incompleto: faltan las celdas {sorted(missing)}"

    b, c, d, e = found["B"], found["C"], found["D"], found["E"]
    with_positions = b.mean - c.mean
    without_positions = d.mean - e.mean
    interaction = with_positions - without_positions

    lines = [
        f"paréntesis con positional=learned = {with_positions:+.4f}",
        f"paréntesis con positional=none    = {without_positions:+.4f}",
        f"interacción                       = {interaction:+.4f}",
    ]
    real = distinguishable(interaction, (b.spread, c.spread, d.spread, e.spread))
    anchored = not distinguishable(without_positions, (d.spread, e.spread))

    if real and anchored:
        lines.append(
            "La interacción apareció. Los paréntesis ganan sólo cuando hay posiciones "
            "que permitan anclarlos: el mecanismo es posicional, como predijimos."
        )
    elif not real and not anchored:
        lines.append(
            "Los paréntesis ganan también sin positional encoding, así que el mecanismo "
            "NO es el anclaje posicional y la historia de D1 es falsa. Se reporta."
        )
    else:
        lines.append(
            "El resultado es mixto: la interacción no separa limpio de los efectos "
            "principales. No alcanza para afirmar el mecanismo en ninguno de los dos "
            "sentidos."
        )
    lines.append(
        "Honestidad a declarar: con positional=none, «con paréntesis» no es idéntico a "
        "«sin paréntesis» — la bolsa tiene dos tokens más. Debería ser despreciable, "
        "pero si aparece algo mínimo en esa celda, es eso y no una señal."
    )
    return "\n".join(lines)


def interaction_table(measured: list[Measured]) -> np.ndarray:
    """El 2×2 de D10 como matriz ``[positional][keep_brackets]``, para la diapositiva."""
    found = _by_key(measured)
    return np.array(
        [
            [found["C"].mean, found["B"].mean],
            [found["E"].mean, found["D"].mean],
        ]
    )


def markdown_table(measured: list[Measured]) -> str:
    """Una fila por celda corrida, con la media entre semillas y su dispersión."""
    lines = [
        "| Celda | Variante | tokenizer | paréntesis | positional | PR-AUC |",
        "|---|---|---|---|---|---:|",
    ]
    for item in measured:
        cell = item.cell
        lines.append(
            f"| {cell.key} | {cell.label} | {cell.tokenizer} | "
            f"{'sí' if cell.keep_brackets else 'no'} | {cell.positional} | "
            f"{item.mean:.4f} ± {item.spread:.4f} |"
        )
    return "\n".join(lines)


def expected_runs(tickets: tuple[str, ...]) -> int:
    """Cuántos entrenamientos completos implica pedir estos tickets."""
    return len(cells_for(tickets)) * len(SEEDS)
