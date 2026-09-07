"""El resumen consolida leyendo, así que lo que hay que probar es que lea bien."""

from __future__ import annotations

import pytest

from src.model.ablation_summary import (
    TOKENIZER_GROUP,
    groups,
    markdown_table,
    plotted,
    reading,
    sweep_groups,
    tokenizer_group,
)
from src.model.configs import load_parameters


@pytest.fixture(scope="module")
def base():
    return load_parameters("parameters.txt")["RUN"]


def runs(per_seed: tuple[float, float, float]) -> list[list[float]]:
    """Como quedan en el JSON: una lista por semilla, cinco folds cada una."""
    return [[value] * 5 for value in per_seed]


def flat(value: float) -> tuple[float, float, float]:
    return (value - 0.01, value, value + 0.01)


def sweep_document(**values) -> dict:
    """Un JSON de barrido con un eje de pooling y otro de heads."""
    return {
        "points": [
            {"axis": "pooling", "field": "pooling", "value": "cls",
             "runs": runs(flat(values.get("cls", 0.770)))},
            {"axis": "pooling", "field": "pooling", "value": "mean",
             "runs": runs(flat(values.get("mean", 0.770)))},
            {"axis": "heads", "field": "n_heads", "value": 4,
             "runs": runs(flat(0.770))},
            {"axis": "heads", "field": "n_heads", "value": 8,
             "runs": runs(flat(values.get("heads8", 0.770)))},
        ]
    }


def tokenizer_document(**values) -> dict:
    return {
        "cells": [
            {"key": key, "runs": runs(flat(values.get(key, 0.700)))}
            for key in ("A", "F", "B", "C", "D", "E")
        ]
    }


def test_sweep_skips_the_base_and_names_the_step(base):
    built = sweep_groups(sweep_document(), base)
    labels = {group.label: [c.label.strip() for c in group.contrasts] for group in built}
    assert labels["Pooler"] == ["cls → mean"]
    assert labels["Cantidad de heads"] == ["4 → 8"]


def test_an_axis_without_its_base_is_skipped_instead_of_anchored_at_zero(base):
    document = {"points": [
        {"axis": "pooling", "field": "pooling", "value": "mean", "runs": runs(flat(0.7))},
    ]}
    assert sweep_groups(document, base) == []


def test_boolean_values_survive_the_json_round_trip(base):
    """``embedding_norm`` vuelve como ``True``, y su base también lo es."""
    document = {"points": [
        {"axis": "embnorm", "field": "embedding_norm", "value": True,
         "runs": runs(flat(0.770))},
        {"axis": "embnorm", "field": "embedding_norm", "value": False,
         "runs": runs(flat(0.740))},
    ]}
    built = sweep_groups(document, base)
    assert len(built) == 1
    assert [c.label.strip() for c in built[0].contrasts] == ["True → False"]


def test_tokenizer_group_needs_the_whole_square():
    assert tokenizer_group({"cells": [{"key": "A", "runs": runs(flat(0.7))}]}) is None
    assert tokenizer_group(tokenizer_document()) is not None


def test_tokenizer_goes_first_because_it_is_the_one_that_gets_deepened(base):
    built = groups(sweep_document(), tokenizer_document(), base)
    assert built[0].label == TOKENIZER_GROUP


def test_a_missing_source_just_drops_its_axes(base):
    only_sweep = groups(sweep_document(), None, base)
    assert TOKENIZER_GROUP not in {group.label for group in only_sweep}
    assert only_sweep


def test_reading_has_the_sentence_e5_asks_for_when_nothing_moves(base):
    built = groups(sweep_document(), tokenizer_document(), base)
    text = reading(built)
    assert "no es un gráfico fallido" in text
    assert "el resultado" in text


def test_reading_names_what_moved(base):
    built = groups(sweep_document(mean=0.700), None, base)
    text = reading(built)
    assert "cls → mean" in text
    assert "se separan del cero" in text


def test_reading_without_any_source_says_so():
    assert "ninguno corrió" in reading([])


def test_plotted_gives_the_figure_one_entry_per_group(base):
    built = groups(sweep_document(), tokenizer_document(), base)
    shaped = plotted(built)
    assert len(shaped) == len(built)
    for label, items in shaped:
        assert isinstance(label, str) and label
        for row, mean, error, stands in items:
            assert isinstance(row, str) and row
            assert isinstance(mean, float) and isinstance(error, float)
            assert isinstance(stands, bool)


def test_markdown_table_has_a_row_per_comparison(base):
    built = groups(sweep_document(), None, base)
    body = [
        line
        for line in markdown_table(built).splitlines()
        if line.startswith("| ") and not line.startswith("| Eje |")
    ]
    assert len(body) == sum(len(group.contrasts) for group in built)
    assert "Semillas" in markdown_table(built)
