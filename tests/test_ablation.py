"""La grilla y, sobre todo, que cada lectura salga en el sentido que corresponde."""

from __future__ import annotations

import pytest

from src.model.ablation import (
    BY_KEY,
    GRID,
    LEARNED,
    Contrast,
    Measured,
    cells_for,
    contrast,
    tokenizer_contrasts,
    interaction_contrasts,
    expected_runs,
    interaction_series,
    markdown_table,
    plotted,
    read_tokenizer,
    read_interaction,
)


def measured(key: str, per_seed: tuple[float, float, float]) -> Measured:
    """Tres semillas, cada una con cinco folds planos.

    Los valores van por semilla y no como media, porque los contrastes se resuelven
    semilla contra semilla: dos celdas con la misma media pero distinto orden entre
    semillas no dan el mismo veredicto.
    """
    return Measured(
        cell=BY_KEY[key],
        runs=tuple(tuple([value] * 5) for value in per_seed),
    )


def flat(value: float) -> tuple[float, float, float]:
    return (value, value, value)


def shifted(value: float) -> tuple[float, float, float]:
    """Semillas que varían entre sí, para que el pareo tenga algo que cancelar."""
    return (value - 0.01, value, value + 0.01)


def test_the_tokenizer_grid_runs_with_learned_positions():
    """Con positional=none no habría nada que medir: la restricción vive en la grilla."""
    for cell in GRID:
        if "tokenizer" in cell.analyses:
            assert cell.positional == LEARNED


def test_the_two_analyses_share_cells_instead_of_duplicating_them():
    """El del tokenizador es un 2x2; B y C las comparte con el de la interacción."""
    assert len(cells_for(("tokenizer",))) == 4
    assert len(cells_for(("interaction",))) == 4
    assert len(cells_for(("tokenizer", "interaction"))) == 6
    assert expected_runs(("tokenizer", "interaction")) == 18


def test_pairing_cancels_the_variation_shared_between_seeds():
    """Dos celdas que se mueven juntas entre semillas: la diferencia es limpia."""
    left = measured("B", (0.70, 0.72, 0.74))
    right = measured("C", (0.60, 0.62, 0.64))
    step = contrast("prueba", left, right)
    assert step.differences == pytest.approx((0.10, 0.10, 0.10))
    assert step.error == pytest.approx(0.0, abs=1e-12)
    assert step.distinguishable


def test_a_contrast_whose_sign_flips_is_not_distinguishable():
    step = Contrast("prueba", (0.05, -0.04, 0.05))
    assert step.agree == 2
    assert not step.consistent
    assert not step.distinguishable


def test_tokenizer_reads_brackets_when_the_control_falls_back_to_v1():
    reading = read_tokenizer(
        [
            measured("A", shifted(0.600)), measured("F", shifted(0.770)),
            measured("B", shifted(0.780)), measured("C", shifted(0.600)),
        ]
    )
    assert "viene de conservar los paréntesis" in reading


def test_tokenizer_reads_tokenizer_when_the_control_keeps_the_gain():
    reading = read_tokenizer(
        [
            measured("A", shifted(0.600)), measured("F", shifted(0.600)),
            measured("B", shifted(0.780)), measured("C", shifted(0.780)),
        ]
    )
    assert "el tokenizador y no los paréntesis" in reading
    assert "no se sostiene" in reading


def test_tokenizer_reports_a_null_result_as_a_result():
    rows = [measured(key, shifted(0.700)) for key in ("A", "F", "B", "C")]
    assert "Ninguno de los dos factores" in read_tokenizer(rows)


def test_tokenizer_says_when_both_factors_move_the_metric():
    reading = read_tokenizer(
        [
            measured("A", shifted(0.600)), measured("F", shifted(0.650)),
            measured("B", shifted(0.800)), measured("C", shifted(0.700)),
        ]
    )
    assert "Las dos cosas aportan" in reading


def test_tokenizer_always_reports_the_whole_word_row_as_coarser():
    rows = [measured(key, shifted(0.700)) for key in ("A", "F", "B", "C")]
    assert "no el factor limpio" in read_tokenizer(rows)


def test_tokenizer_has_the_four_contrasts_of_the_square():
    rows = [measured(key, shifted(0.700)) for key in ("A", "F", "B", "C")]
    assert len(tokenizer_contrasts(rows)) == 4


def test_interaction_confirms_the_mechanism_when_the_gain_needs_positions():
    reading = read_interaction(
        [
            measured("B", shifted(0.780)), measured("C", shifted(0.600)),
            measured("D", shifted(0.600)), measured("E", shifted(0.600)),
        ]
    )
    assert "La interacción apareció" in reading
    assert "el mecanismo es posicional" in reading


def test_interaction_refutes_the_story_when_the_gain_survives_without_positions():
    reading = read_interaction(
        [
            measured("B", shifted(0.780)), measured("C", shifted(0.600)),
            measured("D", shifted(0.780)), measured("E", shifted(0.600)),
        ]
    )
    assert "NO es el anclaje posicional" in reading
    assert "es falsa" in reading


def test_interaction_has_two_effects_and_their_interaction():
    rows = [measured(key, shifted(0.700)) for key in ("B", "C", "D", "E")]
    assert len(interaction_contrasts(rows)) == 3


def test_an_incomplete_grid_says_so_instead_of_inventing_a_reading():
    assert "incompleto" in read_tokenizer([measured("A", flat(0.6))])
    assert "incompleto" in read_interaction([measured("B", flat(0.6))])


def test_interaction_series_has_a_line_per_positional_level():
    """Dos líneas, cada una con sus dos puntos y el delta pareado de esa fila."""
    rows = [measured(key, shifted(0.700)) for key in ("B", "C", "D", "E")]
    series = interaction_series(rows)
    assert [label for label, _, _ in series] == [
        "positional = learned",
        "positional = none",
    ]
    for _, points, (delta, error) in series:
        assert len(points) == 2
        assert all(len(point) == 2 for point in points)
        assert isinstance(delta, float) and isinstance(error, float)


def test_plotted_carries_what_the_figure_needs():
    rows = [measured(key, shifted(0.700)) for key in ("A", "F", "B", "C")]
    for label, mean, error, stands in plotted(tokenizer_contrasts(rows)):
        assert isinstance(label, str) and label
        assert isinstance(mean, float)
        assert isinstance(error, float)
        assert isinstance(stands, bool)


def test_markdown_table_lists_every_measured_cell():
    rows = [measured("A", shifted(0.600)), measured("B", shifted(0.780))]
    rendered = markdown_table(rows)
    assert "whole-word" in rendered
    assert "0.7800" in rendered
