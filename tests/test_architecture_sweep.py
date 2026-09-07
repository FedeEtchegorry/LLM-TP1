"""El barrido no puede duplicar la base ni prometer celdas que el bloque rechaza."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.model.architecture_sweep import (
    AXES,
    BY_KEY,
    Measured,
    anchor_of,
    axes_for,
    contrasts,
    expected_runs,
    markdown_table,
    reading,
    unique_configs,
    validate,
)
from src.model.configs import load_parameters


@pytest.fixture(scope="module")
def base():
    return load_parameters("parameters.txt")["RUN"]


def measured(axis_key: str, value, per_seed: tuple[float, float, float]) -> Measured:
    """Tres semillas, cada una con cinco folds planos en el valor pedido.

    Los valores se dan por semilla y no como una media, porque los contrastes se
    resuelven semilla contra semilla: dos celdas con la misma media pero distinto orden
    entre semillas no dan el mismo veredicto, y eso es exactamente lo que hay que poder
    probar.
    """
    return Measured(
        axis=BY_KEY[axis_key],
        value=value,
        runs=tuple(tuple([value] * 5) for value in per_seed),
    )


FLAT = (0.700, 0.700, 0.700)


def test_the_base_is_measured_once_instead_of_once_per_axis(base):
    """Cinco ejes con 16 valores declarados colapsan a doce configuraciones."""
    assert sum(len(axis.values) for axis in AXES) == 16
    assert len(unique_configs(base, AXES)) == 12
    assert expected_runs(base, AXES) == 36


def test_every_declared_cell_can_actually_be_built(base):
    validate(base, AXES)


def test_validate_rejects_a_width_that_is_not_divisible_by_the_heads(base):
    with pytest.raises(ValueError, match="no es divisible"):
        validate(replace(base, n_heads=5), (BY_KEY["width"],))


def test_axes_for_rejects_an_unknown_axis():
    with pytest.raises(ValueError, match="ejes desconocidos"):
        axes_for(("heads", "inventado"))


def test_axes_for_defaults_to_all_of_them():
    assert axes_for(None) == AXES
    assert axes_for(("ffn",)) == (BY_KEY["ffn"],)


def test_anchor_is_the_declared_base_value(base):
    rows = [measured("heads", value, FLAT) for value in (2, 4, 8)]
    assert anchor_of(rows, BY_KEY["heads"], base).value == base.n_heads


def test_contrasts_skip_the_base_and_name_the_step(base):
    rows = [measured("heads", value, FLAT) for value in (2, 4, 8)]
    steps = contrasts(rows, base)
    assert len(steps) == 2
    assert {step.label for step in steps} == {"n_heads: 4 → 2", "n_heads: 4 → 8"}


def test_a_consistent_shift_across_seeds_is_distinguishable(base):
    """Las tres semillas se mueven en el mismo sentido y la media supera su error."""
    rows = [
        measured("heads", 4, (0.700, 0.710, 0.720)),
        measured("heads", 8, (0.750, 0.760, 0.770)),
        measured("heads", 2, FLAT),
    ]
    step = next(s for s in contrasts(rows, base) if s.label.endswith("8"))
    assert step.agree == 3
    assert step.distinguishable


def test_a_shift_that_flips_sign_between_seeds_is_not(base):
    """Mismo tamaño de efecto medio, pero el signo se da vuelta: no cuenta."""
    rows = [
        measured("heads", 4, (0.700, 0.700, 0.700)),
        measured("heads", 8, (0.760, 0.640, 0.760)),
        measured("heads", 2, FLAT),
    ]
    step = next(s for s in contrasts(rows, base) if s.label.endswith("8"))
    assert step.agree == 2
    assert not step.distinguishable


def test_reading_reports_a_flat_sweep_as_the_expected_result(base):
    rows = [measured("heads", value, FLAT) for value in (2, 4, 8)]
    text = reading(rows, base)
    assert "Ningún eje mueve" in text
    assert "no compra nada" in text


def test_reading_names_the_comparisons_that_moved(base):
    rows = [
        measured("heads", 2, FLAT),
        measured("heads", 4, FLAT),
        measured("heads", 8, (0.800, 0.810, 0.820)),
    ]
    text = reading(rows, base)
    assert "n_heads: 4 → 8" in text
    assert "superan el ruido" in text


def test_markdown_table_carries_the_column_the_ticket_asks_for(base):
    rows = [measured("heads", value, FLAT) for value in (2, 4, 8)]
    rendered = markdown_table(rows, base)
    assert "¿Distinguible del ruido?" in rendered
    assert "Semillas de acuerdo" in rendered
    assert "dentro del ruido" in rendered
    assert rendered.count("| Cantidad de heads |") == 3


def test_the_base_row_has_no_delta(base):
    rows = [measured("heads", value, FLAT) for value in (2, 4, 8)]
    base_row = [
        line for line in markdown_table(rows, base).splitlines() if line.endswith("base |")
    ]
    assert len(base_row) == 1
