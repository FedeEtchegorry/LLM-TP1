"""Barrido de arquitectura: un factor por vez, con el resto congelado en la base.

Dos familias de ejes. Los de **capacidad** cambian cuánto mide el encoder --heads, bloques
apilados, ``d_model``, dimensión del FFN y dropout--. Los de **módulo** cambian qué se usa
en cada lugar: pooler, positional encoding, la capa de cierre del embedding, la torre
tabular y el MLP de salida.

**El valor base aparece en todos los ejes y se mide una sola vez**: es la misma
configuración y el caché por digest la reconoce. Sin ese colapso el barrido costaría 28
entrenamientos por semilla en lugar de 19.

**Cada valor se compara contra la base de su eje, pareado por semilla**, con la
:class:`~src.model.ablation.Contrast` que declara ese módulo en lugar de una regla propia.
Restar las medias primero da el mismo número con una incertidumbre mucho más grande,
porque tira la estructura pareada que comparten dos configuraciones al correr sobre los
mismos folds y las mismas semillas.

Con desvíos entre folds de ±0,015-0,019 es esperable que casi ningún eje se separe del
ruido, así que la tabla lleva una columna que lo dice explícitamente y la lectura cubre
ese caso.

``d_model`` tiene que ser divisible por ``n_heads`` o el bloque de atención se niega a
construirse. Como los dos son ejes del barrido, la grilla se valida antes de entrenar en
lugar de descubrirlo a mitad de la tercera hora.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.model.ablation import Contrast, contrast
from src.model.configs import RunConfig
from src.model.representation_selection import SEEDS, seed_mean, seed_spread


@dataclass(frozen=True)
class Axis:
    """Un eje de capacidad y los valores que se le prueban."""

    key: str
    field: str
    label: str
    values: tuple

    def config(self, base: RunConfig, value, seed: int) -> RunConfig:
        return replace(base, **{self.field: value}, seed=seed)


CAPACITY_AXES = (
    Axis("heads", "n_heads", "Cantidad de heads", (2, 4, 8)),
    Axis("layers", "n_layers", "Encoders apilados", (1, 2, 4)),
    Axis("width", "d_model", "d_model", (64, 96, 128, 256)),
    Axis("ffn", "ffn_multiplier", "Dimensión del FFN (× d_model)", (1, 2, 4)),
    Axis("dropout", "dropout", "Dropout", (0.0, 0.1, 0.3)),
)
"""Cuánta capacidad tiene el encoder."""

MODULE_AXES = (
    Axis("pooling", "pooling", "Pooler", ("cls", "mean", "attention")),
    Axis("positional", "positional", "Positional encoding",
         ("none", "sinusoidal", "learned")),
    Axis("embnorm", "embedding_norm", "LayerNorm + Dropout de cierre", (False, True)),
    Axis("tab", "tab_tower", "Torre tabular", ("linear", "mlp")),
    Axis("head", "fusion_head", "MLP de salida", ("linear", "mlp")),
)
"""Los que cambian qué módulo se usa, no cuánto mide.

Van en el mismo barrido que los de capacidad porque son la misma pregunta con la misma
forma --un factor por vez contra la base, pareado por semilla-- y porque así comparten el
valor base, que se mide una sola vez para los diez ejes."""

AXES = CAPACITY_AXES + MODULE_AXES

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

    El valor base se repite en los diez ejes y colapsa a una sola entrada, que es lo que
    hace que el barrido cueste 19 entrenamientos por semilla y no 28.
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


def anchor_of(measured: list[Measured], axis: Axis, base: RunConfig) -> Measured | None:
    """La medición del valor base de un eje, que es contra quien se compara todo lo demás."""
    base_value = getattr(base, axis.field)
    return next(
        (item for item in measured if item.axis.key == axis.key and item.value == base_value),
        None,
    )


def contrasts(measured: list[Measured], base: RunConfig) -> tuple[Contrast, ...]:
    """Un contraste por valor no-base, pareado semilla contra semilla contra su base."""
    built = []
    for axis in AXES:
        anchor = anchor_of(measured, axis, base)
        if anchor is None:
            continue
        for item in measured:
            if item.axis.key != axis.key or item.value == anchor.value:
                continue
            built.append(
                contrast(f"{axis.field}: {anchor.value} → {item.value}", item, anchor)
            )
    return tuple(built)


def markdown_table(measured: list[Measured], base: RunConfig) -> str:
    """Una fila por configuración, con su margen contra la base y el veredicto."""
    lines = [
        "| Eje | Valor | PR-AUC | Δ vs base | Semillas de acuerdo | "
        "¿Distinguible del ruido? |",
        "|---|---|---:|---:|:---:|---|",
    ]
    for axis in AXES:
        items = [item for item in measured if item.axis.key == axis.key]
        anchor = anchor_of(measured, axis, base)
        if not items or anchor is None:
            continue
        for item in sorted(items, key=lambda i: str(i.value)):
            if item.value == anchor.value:
                lines.append(
                    f"| {axis.label} | {item.value} | {item.mean:.4f} ± {item.spread:.4f} "
                    f"| — | — | base |"
                )
                continue
            step = contrast(axis.field, item, anchor)
            verdict = "distinguible" if step.distinguishable else "dentro del ruido"
            lines.append(
                f"| {axis.label} | {item.value} | {item.mean:.4f} ± {item.spread:.4f} "
                f"| {step.mean:+.4f} ± {step.error:.4f} "
                f"| {step.agree}/{len(step.differences)} | {verdict} |"
            )
    return "\n".join(lines)


def reading(measured: list[Measured], base: RunConfig) -> str:
    """Qué se dice sobre el barrido, en cualquiera de los dos sentidos en que salga."""
    steps = contrasts(measured, base)
    if not steps:
        return "barrido vacío"

    lines = [str(step) for step in steps]
    moved = [step for step in steps if step.distinguishable]
    if not moved:
        lines.append(
            "Ningún eje mueve el PR-AUC por encima del ruido entre semillas. Es el "
            "resultado esperado y se reporta como tal: con ±0,015-0,019 entre folds, y "
            "sabiendo que en v1 quitar la autoatención entera ya quedó dentro del ruido, "
            "el tamaño del modelo no es lo que limita este problema. Subir capacidad no "
            "compra nada acá."
        )
    else:
        lines.append(
            f"{len(moved)} de {len(steps)} comparaciones superan el ruido: "
            + "; ".join(f"{step.label} ({step.mean:+.4f})" for step in moved)
            + ". El resto queda dentro del ruido y se reporta así, en vez de leer una "
            "diferencia de tercer decimal como si fuera señal."
        )
    return "\n".join(lines)


def expected_runs(base: RunConfig, axes: tuple[Axis, ...]) -> int:
    return len(unique_configs(base, axes)) * len(SEEDS)
