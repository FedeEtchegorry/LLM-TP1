"""La grilla que comparten los dos análisis, y cómo se lee cada uno en los dos sentidos.

El del tokenizador pregunta si la ganancia la dan los paréntesis o el tipo de
tokenizador. El de la interacción pregunta
*por qué* la dan, apagando el mecanismo: sin positional encoding la self-attention es
permutación-equivariante y ``[CLS]`` queda invariante a permutaciones del resto, así que
el modelo ve un multiconjunto de tokens y el paréntesis está presente pero es inutilizable
como delimitador.

Los dos analyses se miden sobre la misma grilla porque **comparten dos celdas**. El caché
va por digest, así que declararlas una sola vez hace que correr los dos cueste seis
configuraciones y no ocho:

======  ===========  ==============  ============  ==========
celda   tokenizer    keep_brackets   positional    análisis
======  ===========  ==============  ============  ==========
A       whole-word   False           learned       tokenizador
F       whole-word   True            learned       tokenizador
B       wordpiece    True            learned       tokenizador, interacción
C       wordpiece    False           learned       tokenizador, interacción
D       wordpiece    True            none          interacción
E       wordpiece    False           none          interacción
======  ===========  ==============  ============  ==========

El del tokenizador son las cuatro primeras: un 2x2 de ``tokenizer`` por ``keep_brackets``, no un eje de
tres brazos. Un eje de tres mueve las subpalabras y la puntuación a la vez entre v1 y la
propuesta, y ninguna diferencia queda atribuible a una sola de las dos.

Las dos filas no son simétricas y conviene decirlo antes de que lo pregunten. En
``wordpiece`` el factor es limpio: ``keep_brackets=False`` borra los paréntesis y deja el
resto de la puntuación intacta. En ``whole-word`` es más grueso, porque ``False`` tiene que
reproducir v1 exactamente y v1 borraba toda la puntuación; ``True`` devuelve la puntuación
entera, no sólo los paréntesis. La afirmación fuerte sobre los paréntesis vive en la fila
``wordpiece``; la fila ``whole-word`` aporta la línea base histórica y el contraste de
palabras enteras.

**El del tokenizador corre con ``positional = learned`` fijo, y eso no es un detalle.** Corrido con
``positional = none`` las cuatro celdas de texto miden lo mismo, daría cero y la
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

TOKENIZER = "tokenizer"
INTERACTION = "interaction"

WHOLE_WORD = "whole-word"
WORDPIECE = "wordpiece"
LEARNED = "learned"
NONE = "none"
SINUSOIDAL = "sinusoidal"


@dataclass(frozen=True)
class Cell:
    """Una configuración de la grilla, con los analyses que la usan."""

    key: str
    label: str
    tokenizer: str
    keep_brackets: bool
    positional: str
    analyses: tuple[str, ...]

    def config(self, base: RunConfig, seed: int) -> RunConfig:
        return replace(
            base,
            tokenizer=self.tokenizer,
            keep_brackets=self.keep_brackets,
            positional=self.positional,
            seed=seed,
        )


GRID = (
    Cell("A", "regex de v1 (sin puntuación)", WHOLE_WORD, False, LEARNED, (TOKENIZER,)),
    Cell("F", "palabras enteras con puntuación", WHOLE_WORD, True, LEARNED, (TOKENIZER,)),
    Cell("B", "WordPiece con paréntesis", WORDPIECE, True, LEARNED, (TOKENIZER, INTERACTION)),
    Cell("C", "WordPiece sin paréntesis (control)", WORDPIECE, False, LEARNED, (TOKENIZER, INTERACTION)),
    Cell("D", "WordPiece con paréntesis · positional=none", WORDPIECE, True, NONE, (INTERACTION,)),
    Cell("E", "WordPiece sin paréntesis · positional=none", WORDPIECE, False, NONE, (INTERACTION,)),
    # G y H extienden la interaccion a positional=sinusoidal: si el efecto de los
    # parentesis depende de "tener posiciones" en general o puntualmente de que sean
    # aprendidas. G ya estaba grabada (es la celda que usa el barrido general); H es
    # nueva.
    Cell("G", "WordPiece con paréntesis · positional=sinusoidal", WORDPIECE, True, SINUSOIDAL, (INTERACTION,)),
    Cell("H", "WordPiece sin paréntesis · positional=sinusoidal", WORDPIECE, False, SINUSOIDAL, (INTERACTION,)),
)

BY_KEY = {cell.key: cell for cell in GRID}


def cells_for(analyses: tuple[str, ...]) -> tuple[Cell, ...]:
    """Las celdas que hacen falta para los analyses pedidos, sin repetir las compartidas."""
    return tuple(cell for cell in GRID if set(cell.analyses) & set(analyses))


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


@dataclass(frozen=True)
class Contrast:
    """Una comparación entre dos celdas, resuelta semilla contra semilla.

    Las celdas comparten folds y semillas, así que la dificultad de un fold y la suerte
    de una inicialización entran igual en los dos brazos y se cancelan al restar. Lo que
    queda es el efecto. Restar las medias primero tira esa estructura: da el mismo
    número y una incertidumbre mucho más grande.
    """

    label: str
    differences: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.differences))

    @property
    def error(self) -> float:
        """Cuánto se mueve la media entre semillas, no cuánto se mueve una semilla."""
        if len(self.differences) < 2:
            return float("inf")
        return float(np.std(self.differences, ddof=1) / np.sqrt(len(self.differences)))

    @property
    def agree(self) -> int:
        return sum(1 for value in self.differences if value > 0)

    @property
    def consistent(self) -> bool:
        return self.agree in (0, len(self.differences))

    @property
    def distinguishable(self) -> bool:
        """Regla declarada, no un test: el signo no se da vuelta entre semillas, y la
        media supera lo que la propia media se mueve entre ellas.

        Sigue sin ser inferencia: tres semillas no dan un intervalo de confianza y los
        cinco folds comparten filas de entrenamiento. Lo que se reporta es la regla, el
        resultado de aplicarla, y cuántas semillas coinciden en el signo.
        """
        return self.consistent and abs(self.mean) > self.error + FLOAT_TOLERANCE

    def __str__(self) -> str:
        verdict = "distinguible" if self.distinguishable else "dentro del ruido"
        return (
            f"{self.label:<48s} = {self.mean:+.4f} ± {self.error:.4f}  "
            f"{self.agree}/{len(self.differences)} semillas  {verdict}"
        )


def contrast(label: str, left: Measured, right: Measured) -> Contrast:
    """Una diferencia por semilla, promediando los folds que las dos celdas comparten."""
    return Contrast(
        label,
        tuple(
            float(np.mean(a)) - float(np.mean(b))
            for a, b in zip(left.runs, right.runs)
        ),
    )


def _by_key(measured: list[Measured]) -> dict[str, Measured]:
    return {item.cell.key: item for item in measured}


def _interaction(label: str, first: Contrast, second: Contrast) -> Contrast:
    """La diferencia entre dos diferencias, todavía pareada por semilla."""
    return Contrast(
        label,
        tuple(a - b for a, b in zip(first.differences, second.differences)),
    )


def tokenizer_contrasts(measured: list[Measured]) -> tuple[Contrast, ...]:
    """Los cuatro contrastes del 2x2, en el orden en que se leen y se dibujan."""
    found = _by_key(measured)
    a, f, b, c = found["A"], found["F"], found["B"], found["C"]
    return (
        contrast("WordPiece: sin paréntesis → con paréntesis", b, c),
        contrast("palabras enteras: sin puntuación → con puntuación", f, a),
        contrast("sin puntuación: palabras enteras → WordPiece", c, a),
        contrast("con puntuación: palabras enteras → WordPiece", b, f),
    )


def interaction_contrasts(measured: list[Measured]) -> tuple[Contrast, Contrast, Contrast]:
    """Los dos efectos de los paréntesis y la interacción entre ellos."""
    found = _by_key(measured)
    b, c, d, e = found["B"], found["C"], found["D"], found["E"]
    with_positions = contrast("con posiciones: sin → con paréntesis", b, c)
    without_positions = contrast("sin posiciones: sin → con paréntesis", d, e)
    return (
        with_positions,
        without_positions,
        _interaction(
            "cuánto cambia ese efecto al quitar las posiciones",
            with_positions,
            without_positions,
        ),
    )


def interaction_series(measured: list[Measured]):
    """El 2x2 como dos líneas: un punto por nivel de paréntesis, y el delta pareado."""
    found = _by_key(measured)
    with_positions, without_positions, _ = interaction_contrasts(measured)
    return [
        (
            "positional = learned",
            [(found["C"].mean, found["C"].spread), (found["B"].mean, found["B"].spread)],
            (with_positions.mean, with_positions.error),
        ),
        (
            "positional = none",
            [(found["E"].mean, found["E"].spread), (found["D"].mean, found["D"].spread)],
            (without_positions.mean, without_positions.error),
        ),
    ]


def plotted(contrasts) -> list[tuple[str, float, float, bool]]:
    """Lo que la figura necesita de un contraste, sin que dibujar importe el módulo."""
    return [
        (item.label.strip(), item.mean, item.error, item.distinguishable)
        for item in contrasts
    ]


def read_tokenizer(measured: list[Measured]) -> str:
    """La conclusión del análisis del tokenizador, escrita en cualquiera de los sentidos en que salga."""
    found = _by_key(measured)
    missing = {"A", "F", "B", "C"} - set(found)
    if missing:
        return f"análisis del tokenizador incompleto: faltan las celdas {sorted(missing)}"

    brackets, punctuation, tokenizer, marked = tokenizer_contrasts(measured)
    lines = [str(item) for item in (brackets, punctuation, tokenizer, marked)]
    if brackets.distinguishable and not tokenizer.distinguishable:
        lines.append(
            "La mejora viene de conservar los paréntesis, no de WordPiece en general: "
            "el control queda al nivel del regex de v1."
        )
    elif brackets.distinguishable and tokenizer.distinguishable:
        lines.append(
            "Las dos cosas aportan. El delimitador pesa más que el cambio de "
            "tokenizador, pero WordPiece suma por su cuenta."
        )
    elif tokenizer.distinguishable and not brackets.distinguishable:
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
        "La fila de palabras enteras mide toda la puntuación, no sólo los paréntesis, "
        "porque su celda sin puntuación tiene que reproducir v1 exactamente. Es la "
        "línea base histórica, no el factor limpio."
    )
    return "\n".join(lines)


def read_interaction(measured: list[Measured]) -> str:
    """La lectura de la interacción: la ganancia tiene que aparecer en una sola celda."""
    found = _by_key(measured)
    missing = {"B", "C", "D", "E"} - set(found)
    if missing:
        return f"análisis de la interacción incompleto: faltan las celdas {sorted(missing)}"

    with_positions, without_positions, interaction = interaction_contrasts(measured)
    lines = [str(item) for item in (with_positions, without_positions, interaction)]
    anchored = not without_positions.distinguishable or without_positions.mean < 0

    if interaction.distinguishable and anchored:
        lines.append(
            "La interacción apareció. Los paréntesis ganan sólo cuando hay posiciones "
            "que permitan anclarlos: el mecanismo es posicional, como predijimos."
        )
    elif not interaction.distinguishable and without_positions.mean > 0:
        lines.append(
            "Los paréntesis ganan también sin positional encoding, así que el mecanismo "
            "NO es el anclaje posicional y la historia del tokenizador es falsa. Se reporta."
        )
    else:
        lines.append(
            "El resultado es mixto: la interacción no separa limpio de los efectos "
            "principales. No alcanza para afirmar el mecanismo en ninguno de los dos "
            "sentidos."
        )
    if without_positions.distinguishable and without_positions.mean < 0:
        lines.append(
            "Sin posiciones los paréntesis no son neutros sino levemente dañinos: son "
            "dos tokens más en la bolsa y ninguna forma de usarlos. Es más de lo que "
            "esperábamos y se dice."
        )
    return "\n".join(lines)


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


def expected_runs(analyses: tuple[str, ...]) -> int:
    """Cuántos entrenamientos completos implica pedir estos analyses."""
    return len(cells_for(analyses)) * len(SEEDS)
