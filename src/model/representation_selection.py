"""Decisiones deterministas a partir de las corridas de validación cruzada.

**La regla es la media más alta.** No hay margen que gatee el movimiento ni desempate
por dispersión: entre las alternativas de un bloque gana la que tiene mayor AP media, y
punto.

Eso sólo es defendible por el protocolo que la acompaña: **cada configuración se mide
con tres semillas**, y el estadístico que se compara es el promedio de las tres. Medimos
que cambiar la semilla mueve el AP unas 0.0150, más que la mediana de los efectos de
arquitectura (0.0065). Con una sola semilla, "la media más alta" es perseguir ruido; con
tres, ese ruido baja alrededor de 0.0087 y la media pasa a significar algo.

La dispersión se sigue reportando —entre folds y entre semillas— porque describe cuánto
depende el resultado de qué queries y de qué inicialización tocaron. Pero **describe, no
decide**: un candidato con mayor media gana aunque sea más disperso.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SEEDS = (1337, 7, 99)
"""Las tres semillas con las que se mide cada configuración antes de compararla."""


def _folds(values: Sequence[float] | np.ndarray, *, label: str) -> np.ndarray:
    """Los AP por fold de una corrida, validados."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} debe contener AP numéricos por fold") from error
    if array.shape != (5,):
        raise ValueError(f"{label} requiere exactamente cinco folds")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} debe contener sólo valores finitos")
    return array


def seed_mean(runs: Sequence[Sequence[float]], *, label: str = "configuración") -> float:
    """El estadístico que decide: promedio sobre semillas del AP medio de cada una.

    Se piden las tres semillas y no se acepta un subconjunto. Comparar una
    configuración medida con tres contra otra medida con una reintroduce exactamente
    el ruido que el protocolo existe para promediar.
    """
    if len(runs) != len(SEEDS):
        raise ValueError(
            f"{label} tiene {len(runs)} semillas y el protocolo declara {len(SEEDS)}"
        )
    return float(np.mean([_folds(r, label=label).mean() for r in runs]))


def seed_spread(runs: Sequence[Sequence[float]], *, label: str = "configuración") -> float:
    """Dispersión entre semillas. Se reporta, no decide."""
    if len(runs) != len(SEEDS):
        raise ValueError(
            f"{label} tiene {len(runs)} semillas y el protocolo declara {len(SEEDS)}"
        )
    return float(np.std([_folds(r, label=label).mean() for r in runs], ddof=1))


def fold_spread(runs: Sequence[Sequence[float]], *, label: str = "configuración") -> float:
    """Dispersión entre folds, promediada sobre semillas. Se reporta, no decide."""
    return float(np.mean([_folds(r, label=label).std(ddof=1) for r in runs]))


def paired_margin(differences: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Media de la diferencia pareada, y media menos y más un desvío.

    Ya **no decide nada**: se conserva porque el informe reporta la dispersión de cada
    contraste, y porque distinguir una diferencia grande de una chica sigue siendo
    información aunque no sea la que elige.
    """
    values = _folds(differences, label="comparación pareada")
    mean = float(values.mean())
    spread = float(values.std(ddof=1))
    return mean, mean - spread, mean + spread


def compare(
    incumbent: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> str:
    """``"candidate"`` si su media sobre las tres semillas es mayor; si no, ``"incumbent"``.

    Un empate exacto conserva al incumbente, que es siempre la opción declarada antes
    o la más simple.
    """
    return (
        "candidate"
        if seed_mean(candidate, label="candidato") > seed_mean(incumbent, label="incumbente")
        else "incumbent"
    )


def choose(
    reference: str, candidates: Mapping[str, Sequence[Sequence[float]]]
) -> str:
    """La alternativa con mayor AP medio sobre las tres semillas.

    Los desempates exactos se resuelven por nombre para que la decisión sea
    reproducible, y la referencia gana cualquier empate con ella.
    """
    if reference not in candidates:
        raise ValueError(f"la referencia {reference!r} no tiene corridas")
    scored = {
        name: seed_mean(runs, label=f"candidato {name!r}")
        for name, runs in candidates.items()
    }
    best = max(scored.values())
    if scored[reference] >= best:
        return reference
    return min(name for name, value in scored.items() if value == best)


def choose_deterministic(
    reference: str, candidates: Mapping[str, Sequence[float] | np.ndarray]
) -> str:
    """La misma regla —mayor AP media— para modelos sin semilla.

    El barrido lineal usa ``LogisticRegression`` con ``lbfgs`` y sin ``random_state``:
    es determinista, así que medirlo con tres semillas daría tres números idénticos y
    el promedio no compraría nada. Se compara una corrida por caso, y gana la media
    más alta igual que en el resto del trabajo.
    """
    if reference not in candidates:
        raise ValueError(f"la referencia {reference!r} no tiene folds")
    scored = {
        name: float(_folds(values, label=f"caso {name!r}").mean())
        for name, values in candidates.items()
    }
    best = max(scored.values())
    if scored[reference] >= best:
        return reference
    return min(name for name, value in scored.items() if value == best)


def compare_folds(
    incumbent_folds: Sequence[float] | np.ndarray,
    candidate_folds: Sequence[float] | np.ndarray,
) -> str:
    """La misma regla de mayor media, pero sobre los cinco folds de una sola corrida.

    Existe para los lugares que comparan dos corridas concretas y no dos familias de
    tres semillas: la lectura de la escalera y las figuras de decisión. Devuelve el
    vocabulario que esas figuras usan para colorear (``improves`` / ``loses``), sin
    ``tie-break`` ni ``inconclusive``, porque bajo la regla de mayor media no hay
    zona intermedia: o la media es mayor o no lo es.

    Cuando se comparan familias medidas con las tres semillas, la función correcta
    es :func:`compare`, que promedia primero y así no decide sobre el ruido de una
    inicialización.
    """
    incumbent = _folds(incumbent_folds, label="incumbente")
    candidate = _folds(candidate_folds, label="candidato")
    return "improves" if float(candidate.mean()) > float(incumbent.mean()) else "loses"
