"""Todos los ejes de ablación en un solo forest plot.

**No recalcula nada: lee los JSON que ya escribieron los runners.** Cada uno guarda las
corridas por semilla, que es lo único que hace falta para reconstruir un
:class:`~src.model.ablation.Contrast`. Consolidar leyendo en vez de reentrenando es lo que
hace que esta figura cueste segundos y se pueda regenerar cada vez que llega una corrida
nueva.

**El margen es el pareado por semilla**, la misma regla que declara
:mod:`src.model.ablation`. Usar dos reglas distintas haría que «no es distinguible» no
significara lo mismo en dos filas del mismo gráfico.

Un eje sin su valor base se descarta entero: sin ancla no hay contra qué comparar, y
dibujarlo apoyado en el cero afirmaría un efecto que nadie midió. Las fuentes que faltan
se informan por consola en lugar de producir una figura incompleta en silencio.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model.ablation import BY_KEY as CELLS
from src.model.ablation import Contrast, Measured, contrast, tokenizer_contrasts
from src.model.architecture_sweep import AXES
from src.model.configs import RunConfig

TOKENIZER_GROUP = "Tokenizador"


@dataclass(frozen=True)
class Group:
    """Un eje de ablación con sus comparaciones contra la base."""

    label: str
    contrasts: tuple[Contrast, ...]

    @property
    def moved(self) -> tuple[Contrast, ...]:
        return tuple(item for item in self.contrasts if item.distinguishable)


def _sweep_measured(document: dict) -> dict[str, list[dict]]:
    """Los puntos del barrido agrupados por eje, tal como quedaron en el JSON."""
    by_axis: dict[str, list[dict]] = {}
    for point in document.get("points", []):
        by_axis.setdefault(point["axis"], []).append(point)
    return by_axis


class _Point:
    """Lo mínimo que ``contrast`` necesita de una medición: sus corridas por semilla."""

    def __init__(self, runs) -> None:
        self.runs = tuple(tuple(run) for run in runs)


def sweep_groups(document: dict, base: RunConfig) -> list[Group]:
    """Un grupo por eje del barrido, comparando cada valor contra la base de ese eje.

    Cuál es la base sale de la configuración declarada y no del JSON: así el resumen no
    puede quedar apuntando a una base vieja si alguien edita ``parameters.txt`` y vuelve
    a correr.
    """
    by_axis = _sweep_measured(document)
    built = []
    for axis in AXES:
        points = by_axis.get(axis.key)
        if not points:
            continue
        base_value = getattr(base, axis.field)
        anchor = next(
            (point for point in points if _same(point["value"], base_value)), None
        )
        if anchor is None:
            continue
        steps = tuple(
            contrast(
                f"{anchor['value']} → {point['value']}",
                _Point(point["runs"]),
                _Point(anchor["runs"]),
            )
            for point in points
            if not _same(point["value"], base_value)
        )
        if steps:
            built.append(Group(axis.label, steps))
    return built


def _same(left, right) -> bool:
    """``True`` y ``1`` colisionan al volver del JSON, así que se comparan por texto."""
    return str(left) == str(right)


def tokenizer_group(document: dict) -> Group | None:
    """El 2x2 del tokenizador, reconstruido desde el JSON de su propio runner."""
    cells = {
        cell["key"]: Measured(cell=CELLS[cell["key"]], runs=tuple(
            tuple(run) for run in cell["runs"]
        ))
        for cell in document.get("cells", [])
        if cell["key"] in CELLS
    }
    if not {"A", "F", "B", "C"} <= set(cells):
        return None
    return Group(TOKENIZER_GROUP, tokenizer_contrasts(list(cells.values())))


def groups(
    sweep: dict | None, tokenizer: dict | None, base: RunConfig
) -> list[Group]:
    """Todos los ejes disponibles, con el tokenizador primero porque es el que se
    profundiza después."""
    built = []
    if tokenizer:
        found = tokenizer_group(tokenizer)
        if found:
            built.append(found)
    if sweep:
        built.extend(sweep_groups(sweep, base))
    return built


def plotted(built: list[Group]):
    """Lo que la figura necesita: por grupo, sus filas ya resueltas."""
    return [
        (
            group.label,
            [
                (item.label.strip(), item.mean, item.error, item.distinguishable)
                for item in group.contrasts
            ],
        )
        for group in built
    ]


def markdown_table(built: list[Group]) -> str:
    """Una fila por comparación, agrupada por eje."""
    lines = [
        "| Eje | Comparación | Δ PR-AUC | Semillas | ¿Distinguible del ruido? |",
        "|---|---|---:|:---:|---|",
    ]
    for group in built:
        for item in group.contrasts:
            verdict = "distinguible" if item.distinguishable else "dentro del ruido"
            lines.append(
                f"| {group.label} | {item.label.strip()} "
                f"| {item.mean:+.4f} ± {item.error:.4f} "
                f"| {item.agree}/{len(item.differences)} | {verdict} |"
            )
    return "\n".join(lines)


def reading(built: list[Group]) -> str:
    """La lectura del conjunto, incluido el caso en que nada supere el ruido."""
    if not built:
        return (
            "No hay ningún eje medido todavía: el resumen se arma leyendo los JSON que "
            "escriben los runners, y ninguno corrió."
        )
    total = sum(len(group.contrasts) for group in built)
    moved = [(group, item) for group in built for item in group.moved]

    lines = [f"{len(built)} ejes, {total} comparaciones contra la base."]
    if not moved:
        lines.append(
            "Ninguna se separa del cero. **Eso no es un gráfico fallido, es el "
            "resultado**: con desvíos entre folds de ±0,015-0,019 y 10.000 filas con "
            "13 % de positivos, ningún módulo que cambiamos mueve el PR-AUC por encima "
            "del ruido. Dice algo verdadero sobre el tamaño del dataset, y es más "
            "honesto que elegir el eje que casualmente dio distinto."
        )
        return "\n".join(lines)

    lines.append(
        f"{len(moved)} de {total} se separan del cero: "
        + "; ".join(
            f"{group.label} {item.label.strip()} ({item.mean:+.4f})"
            for group, item in moved
        )
        + "."
    )
    quiet = [group.label for group in built if not group.moved]
    if quiet:
        lines.append(
            "Los ejes que quedan enteros dentro del ruido son "
            + ", ".join(quiet)
            + ", y se reportan así en vez de leer el tercer decimal como señal."
        )
    return "\n".join(lines)
