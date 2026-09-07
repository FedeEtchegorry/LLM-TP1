"""D3: cuánto más aprendería cada modelo si hubiera más filas.

La escalera de v1 dejó al Transformer por debajo de la baseline lineal (AP 0,771) y muy
por debajo del techo con la clave extraída a mano (0,813). Este ticket no intenta dar
vuelta ese resultado: lo interpreta. **Lo que se lee acá no es quién está más arriba, es
la pendiente.**

- Si la del Transformer sube más rápido, el límite no es la arquitectura sino tener
  10.000 filas con 13 % de positivos, y eso se dice.
- Si las dos pendientes son iguales, el Transformer no tiene nada más que extraer de este
  texto, y eso también se dice.

**El submuestreo es por query, no por fila.** La unidad del problema es la búsqueda: varias
filas comparten ``query_id`` y recortar por fila simularía un dataset que nunca existiría,
además de dejar queries partidas entre lo que entra y lo que no. Recortar por query
simula lo que de verdad significaría tener menos datos: haber registrado menos búsquedas.

**Las filas puntuadas no se tocan.** Sólo encoge el conjunto de entrenamiento de cada
fold; la validación del fold queda idéntica en los cinco tamaños, que es lo que hace que
los puntos de la curva sean comparables entre sí.

**El mismo recorte para los dos modelos.** El generador se siembra con el fold y el
tamaño, nunca con la semilla del modelo, así que el Transformer y la baseline ven
exactamente las mismas filas en cada punto. Si se sembrara con ``config.seed``, cada
modelo vería un subconjunto distinto y la diferencia entre curvas mezclaría dos efectos.

Un aviso sobre el caché: ``RunConfig.digest`` no incluye el tamaño del entrenamiento, así
que dos puntos de la curva con la misma configuración colisionan en el mismo archivo. El
runner escribe cada tamaño en su propio directorio; no es un detalle de prolijidad, es lo
que evita que la curva salga plana por leer cinco veces el mismo resultado.

La regla para decidir si dos pendientes difieren es la misma que declara
:mod:`src.model.ablation`: una diferencia cuenta si supera la suma de las dispersiones
entre semillas. Se importa en lugar de reescribirla para que las dos lecturas del trabajo
no puedan divergir en qué consideran ruido.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean, stdev

import numpy as np

from src.model.ablation import Contrast
from src.model.representation_selection import SEEDS, seed_mean, seed_spread
from src.partitions import DataPartitions, FoldIndices

SIZES = (500, 1000, 2000, 4000, 6400)
"""Filas de entrenamiento por punto. El último es prácticamente el fold completo, así que
ancla la curva contra el número que ya reporta el resto del trabajo."""

TRANSFORMER = "transformer"
LOGISTIC = "logistic"


def subsample(
    partitions: DataPartitions, query_ids: list, target_rows: int
) -> DataPartitions:
    """Encoger el entrenamiento de cada fold a ``target_rows`` filas, por query entera."""
    folds = tuple(
        replace(fold, train_indices=_take_queries(fold, query_ids, target_rows))
        for fold in partitions.folds
    )
    return replace(partitions, folds=folds)


def _take_queries(fold: FoldIndices, query_ids: list, target_rows: int) -> tuple[int, ...]:
    """Queries barajadas de forma determinística hasta llegar al presupuesto de filas."""
    by_query: dict[object, list[int]] = {}
    for index in fold.train_indices:
        by_query.setdefault(query_ids[index], []).append(index)

    order = sorted(by_query)
    generator = np.random.default_rng((fold.fold_index + 1) * 100_003 + target_rows)
    generator.shuffle(order)

    taken: list[int] = []
    for query in order:
        if len(taken) >= target_rows:
            break
        taken.extend(by_query[query])
    return tuple(sorted(taken))


@dataclass(frozen=True)
class Point:
    """Un modelo en un tamaño, medido con las tres semillas."""

    model: str
    size: int
    runs: tuple[tuple[float, ...], ...]

    @property
    def mean(self) -> float:
        return seed_mean([list(run) for run in self.runs], label=f"{self.model}@{self.size}")

    @property
    def spread(self) -> float:
        return seed_spread([list(run) for run in self.runs], label=f"{self.model}@{self.size}")


def curve(points: list[Point], model: str) -> list[Point]:
    """Los puntos de un modelo, ordenados por tamaño."""
    return sorted((p for p in points if p.model == model), key=lambda p: p.size)


def slopes_by_seed(points: list[Point], model: str) -> list[float]:
    """Una pendiente de AP contra log10(filas) por semilla.

    Ajustar una pendiente por semilla y dispersar entre semillas usa el mismo protocolo
    de tres repeticiones que el resto del trabajo, en lugar de inventar una incertidumbre
    para un único ajuste sobre las medias.
    """
    ordered = curve(points, model)
    if len(ordered) < 2:
        return []
    log_sizes = np.log10([point.size for point in ordered])
    fitted = []
    for seed_index in range(len(SEEDS)):
        values = [float(np.mean(point.runs[seed_index])) for point in ordered]
        fitted.append(float(np.polyfit(log_sizes, values, 1)[0]))
    return fitted


def read_slopes(points: list[Point]) -> str:
    """La conclusión sobre la pendiente, en cualquiera de los dos sentidos en que salga."""
    transformer = slopes_by_seed(points, TRANSFORMER)
    logistic = slopes_by_seed(points, LOGISTIC)
    if not transformer or not logistic:
        return "curva incompleta: hacen falta al menos dos tamaños por modelo"

    t_mean, t_spread = fmean(transformer), _spread(transformer)
    l_mean, l_spread = fmean(logistic), _spread(logistic)
    gap = Contrast(
        "diferencia           ",
        tuple(t - l for t, l in zip(transformer, logistic)),
    )
    steeper, difference = gap.distinguishable, gap.mean

    lines = [
        f"pendiente Transformer = {t_mean:+.4f} ± {t_spread:.4f} AP por década de filas",
        f"pendiente baseline    = {l_mean:+.4f} ± {l_spread:.4f}",
        str(gap),
    ]
    if steeper and difference > 0:
        lines.append(
            "La curva del Transformer sube más rápido: el límite no es la arquitectura "
            "sino el tamaño del dataset. Con más búsquedas registradas la diferencia "
            "contra la baseline se achicaría."
        )
    elif steeper:
        lines.append(
            "La baseline sube más rápido que el Transformer. Más datos no acercarían al "
            "Transformer a la baseline, la alejarían."
        )
    else:
        lines.append(
            "Las dos pendientes son indistinguibles: el Transformer no tiene nada más "
            "que extraer de este texto que la baseline no extraiga ya. Es un resultado "
            "negativo y se reporta como tal."
        )
    lines.append(
        "La pendiente se mide sobre la validación cruzada, no sobre el holdout: la "
        "conclusión tiene que quedar escrita antes de gastarlo."
    )
    return "\n".join(lines)


def _spread(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def markdown_table(points: list[Point]) -> str:
    """Una fila por tamaño, una columna por modelo."""
    sizes = sorted({point.size for point in points})
    by_key = {(point.model, point.size): point for point in points}
    lines = ["| Filas de train | Transformer | Baseline lineal |", "|---:|---:|---:|"]
    for size in sizes:
        cells = []
        for model in (TRANSFORMER, LOGISTIC):
            point = by_key.get((model, size))
            cells.append("—" if point is None else f"{point.mean:.4f} ± {point.spread:.4f}")
        lines.append(f"| {size} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)
