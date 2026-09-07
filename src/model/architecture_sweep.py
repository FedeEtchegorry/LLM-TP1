"""D4: los cinco ejes de capacidad del Transformer, un factor por vez.

Es lo que la devolución pidió priorizar --*"heads, cantidad de encoders apilados, d_model,
dimension de la ffn, etc"*-- y los valores son los que el grupo se comprometió por mail.

**La conclusión probable hay que anticiparla, no descubrirla en vivo.** Con desvíos de
±0,015--0,019 entre folds, y sabiendo que en v1 quitar la autoatención entera ya quedó
dentro del ruido (L1 0,757 contra L2 0,752), un barrido de anchos difícilmente produzca
diferencias distinguibles. Por eso la tabla tiene una columna que dice si la diferencia
supera el ruido, y por eso la regla que la decide es la misma que declara
:mod:`src.model.ablation`: si cada análisis del trabajo usara su propio umbral, «no es
distinguible» no querría decir lo mismo en dos diapositivas seguidas.

**Un factor por vez, con el resto congelado en la base.** El valor base aparece en los
cinco ejes, así que se mide una sola vez: es la misma configuración, y el caché por digest
la reconoce. Doce configuraciones, no dieciséis.

``d_model`` tiene que ser divisible por ``n_heads`` o el bloque de atención se niega a
construirse. Como los dos son ejes del barrido, la grilla se valida antes de entrenar en
lugar de descubrirlo a mitad de la tercera hora.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.model.ablation import distinguishable
from src.model.configs import RunConfig
from src.model.representation_selection import SEEDS, seed_mean, seed_spread

BASE = "base"
IMPROVES = "improves"
LOSES = "loses"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Axis:
    """Un eje de capacidad y los valores que se le prueban."""

    key: str
    field: str
    label: str
    values: tuple

    def config(self, base: RunConfig, value, seed: int) -> RunConfig:
        return replace(base, **{self.field: value}, seed=seed)


AXES = (
    Axis("heads", "n_heads", "Cantidad de heads", (2, 4, 8)),
    Axis("layers", "n_layers", "Encoders apilados", (1, 2, 4)),
    Axis("width", "d_model", "d_model", (64, 96, 128, 256)),
    Axis("ffn", "ffn_multiplier", "Dimensión del FFN (× d_model)", (1, 2, 4)),
    Axis("dropout", "dropout", "Dropout", (0.0, 0.1, 0.3)),
)

BY_KEY = {axis.key: axis for axis in AXES}


def axes_for(keys: tuple[str, ...] | None) -> tuple[Axis, ...]:
    if not keys:
        return AXES
    unknown = set(keys) - set(BY_KEY)
    if unknown:
        raise ValueError(f"ejes desconocidos: {sorted(unknown)}; hay {sorted(BY_KEY)}")
    return tuple(BY_KEY[key] for key in keys)


def validate(base: RunConfig, axes: tuple[Axis, ...]) -> None:
    """Rechazar la grilla entera antes de entrenar si alguna celda no se puede construir."""
    problems = []
    for axis in axes:
        for value in axis.values:
            config = axis.config(base, value, base.seed)
            if config.d_model % config.n_heads:
                problems.append(
                    f"{axis.key}={value}: d_model={config.d_model} no es divisible "
                    f"por n_heads={config.n_heads}"
                )
    if problems:
        raise ValueError("la grilla tiene celdas imposibles:\n  " + "\n  ".join(problems))


def unique_configs(base: RunConfig, axes: tuple[Axis, ...]) -> dict[str, RunConfig]:
    """Las configuraciones distintas de la grilla, indexadas por digest.

    El valor base se repite en los cinco ejes y colapsa a una sola entrada, que es lo que
    hace que el barrido cueste doce entrenamientos por semilla y no dieciséis.
    """
    found: dict[str, RunConfig] = {}
    for axis in axes:
        for value in axis.values:
            config = axis.config(base, value, base.seed)
            found.setdefault(config.digest, config)
    return found


@dataclass(frozen=True)
class Measured:
    """Un valor de un eje, medido con las tres semillas."""

    axis: Axis
    value: object
    runs: tuple[tuple[float, ...], ...]

    @property
    def label(self) -> str:
        return f"{self.axis.field}={self.value}"

    @property
    def mean(self) -> float:
        return seed_mean([list(run) for run in self.runs], label=self.label)

    @property
    def spread(self) -> float:
        return seed_spread([list(run) for run in self.runs], label=self.label)


def outcome(item: Measured, base_value: object, base_mean: float, base_spread: float) -> str:
    """Cómo se compara un valor contra la base del mismo eje."""
    if item.value == base_value:
        return BASE
    delta = item.mean - base_mean
    if not distinguishable(delta, (item.spread, base_spread)):
        return INCONCLUSIVE
    return IMPROVES if delta > 0 else LOSES


def stages(measured: list[Measured], base: RunConfig) -> list[dict]:
    """La grilla en la forma que ``figures.architecture_grid`` ya sabe dibujar."""
    built = []
    for axis in AXES:
        items = [item for item in measured if item.axis.key == axis.key]
        if not items:
            continue
        base_value = getattr(base, axis.field)
        anchor = next((item for item in items if item.value == base_value), None)
        base_mean = anchor.mean if anchor else 0.0
        base_spread = anchor.spread if anchor else 0.0
        points = [
            {
                "label": str(item.value),
                "ap": item.mean,
                "ap_std": item.spread,
                "outcome": outcome(item, base_value, base_mean, base_spread),
            }
            for item in sorted(items, key=lambda i: i.mean, reverse=True)
        ]
        built.append(
            {
                "stage": axis.label,
                "points": points,
                "selected": max(points, key=lambda point: point["ap"])["label"],
            }
        )
    return built


VERDICTS = {
    BASE: "base",
    IMPROVES: "mejora",
    LOSES: "empeora",
    INCONCLUSIVE: "dentro del ruido",
}


def markdown_table(measured: list[Measured], base: RunConfig) -> str:
    """Una fila por configuración y la columna que el ticket pide explícitamente."""
    lines = [
        "| Eje | Valor | PR-AUC | Δ vs base | ¿Distinguible del ruido? |",
        "|---|---|---:|---:|---|",
    ]
    for axis in AXES:
        items = [item for item in measured if item.axis.key == axis.key]
        if not items:
            continue
        base_value = getattr(base, axis.field)
        anchor = next((item for item in items if item.value == base_value), None)
        base_mean = anchor.mean if anchor else 0.0
        base_spread = anchor.spread if anchor else 0.0
        for item in sorted(items, key=lambda i: str(i.value)):
            verdict = outcome(item, base_value, base_mean, base_spread)
            delta = "—" if verdict == BASE else f"{item.mean - base_mean:+.4f}"
            lines.append(
                f"| {axis.label} | {item.value} | {item.mean:.4f} ± {item.spread:.4f} "
                f"| {delta} | {VERDICTS[verdict]} |"
            )
    return "\n".join(lines)


def reading(measured: list[Measured], base: RunConfig) -> str:
    """Qué se dice sobre el barrido, en cualquiera de los dos sentidos en que salga."""
    if not measured:
        return "barrido vacío"
    moved = []
    for axis in AXES:
        items = [item for item in measured if item.axis.key == axis.key]
        if not items:
            continue
        base_value = getattr(base, axis.field)
        anchor = next((item for item in items if item.value == base_value), None)
        if anchor is None:
            continue
        for item in items:
            verdict = outcome(item, base_value, anchor.mean, anchor.spread)
            if verdict in (IMPROVES, LOSES):
                moved.append((axis, item, verdict))

    if not moved:
        return (
            "Ningún eje mueve el PR-AUC por encima del ruido entre semillas. Es el "
            "resultado esperado y se reporta como tal: con ±0,015-0,019 entre folds, y "
            "sabiendo que en v1 quitar la autoatención entera ya quedó dentro del ruido, "
            "el tamaño del modelo no es lo que limita este problema. Subir capacidad no "
            "compra nada acá."
        )
    lines = ["Ejes que sí mueven el PR-AUC por encima del ruido:"]
    for axis, item, verdict in moved:
        lines.append(
            f"  {axis.label}: {item.value} {VERDICTS[verdict]} "
            f"({item.mean - _anchor_mean(measured, axis, base):+.4f})"
        )
    lines.append(
        "El resto queda dentro del ruido y se reporta así, en vez de leer una diferencia "
        "de tercer decimal como si fuera señal."
    )
    return "\n".join(lines)


def _anchor_mean(measured: list[Measured], axis: Axis, base: RunConfig) -> float:
    base_value = getattr(base, axis.field)
    anchor = next(
        (i for i in measured if i.axis.key == axis.key and i.value == base_value), None
    )
    return anchor.mean if anchor else 0.0


def expected_runs(base: RunConfig, axes: tuple[Axis, ...]) -> int:
    return len(unique_configs(base, axes)) * len(SEEDS)
