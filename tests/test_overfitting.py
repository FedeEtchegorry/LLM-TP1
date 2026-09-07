"""The four levels are read off a stored record, so they are testable without training."""

from __future__ import annotations

import pytest

from src.model.overfitting import (
    FIT,
    STOP,
    TEST,
    VALIDATION,
    epoch_gap_frame,
    fold_levels,
    gaps,
    levels,
    markdown_table,
    stopped_early,
)


def epoch(number: int, train_ap: float, validation_ap: float) -> dict:
    return {
        "epoch": number,
        "train_loss": 1.0 / number,
        "train_ap": train_ap,
        "validation_loss": 2.0 / number,
        "validation_ap": validation_ap,
    }


def record(*, best_epoch: int = 2, epochs: int = 3, folds: int = 2) -> dict:
    """A two-fold run whose AP rises on fit and peaks early on the stopping split."""
    return {
        "schema": 3,
        "name": "synthetic",
        "digest": "deadbeef",
        "config": {"epochs": epochs},
        "folds": [
            {
                "fold_index": index,
                "roc_auc": 0.90 + 0.01 * index,
                "average_precision": 0.60 + 0.02 * index,
                "n_train": 6400,
                "n_scored": 1600,
                "seconds": 1.0,
            }
            for index in range(folds)
        ],
        "curves": [
            {
                "fold_index": index,
                "best_epoch": best_epoch,
                "parameters": 1000,
                "epochs": [
                    epoch(1, 0.70, 0.65),
                    epoch(2, 0.80, 0.70),
                    epoch(3, 0.95, 0.66),
                ][:epochs],
            }
            for index in range(folds)
        ],
    }


def test_fold_levels_reads_the_kept_epoch_not_the_last():
    frame = fold_levels(record(best_epoch=2))
    assert list(frame["best_epoch"]) == [2, 2]
    # Epoch 3 has the highest fit AP (0.95) but was discarded by early stopping.
    assert list(frame["fit_ap"]) == [0.80, 0.80]
    assert list(frame["stop_ap"]) == [0.70, 0.70]


def test_validation_comes_from_the_fold_score_not_the_curve():
    frame = fold_levels(record())
    assert list(frame["validation_ap"]) == [0.60, 0.62]


def test_levels_without_holdout_stop_at_validation():
    found = levels(record())
    assert [level.name for level in found] == [FIT, STOP, VALIDATION]


def test_levels_with_holdout_add_test():
    test_document = {
        "folds": [{"average_precision": 0.55, "roc_auc": 0.88}],
    }
    found = levels(record(), test_document)
    assert [level.name for level in found] == [FIT, STOP, VALIDATION, TEST]
    assert found[-1].average_precision == pytest.approx(0.55)
    # A single-fold holdout has no spread to report.
    assert found[-1].deviation == 0.0


def test_gaps_are_signed_drops_between_consecutive_levels():
    steps = gaps(levels(record()))
    assert [(step.origin, step.destination) for step in steps] == [
        (FIT, STOP),
        (STOP, VALIDATION),
    ]
    assert steps[0].delta == pytest.approx(0.70 - 0.80)
    assert steps[1].delta == pytest.approx(0.61 - 0.70)


def test_a_run_without_curves_reports_only_what_it_can_support():
    """The logistic bar has no epochs, so fit and the stopping split do not exist."""
    curveless = record()
    curveless["curves"] = []
    found = levels(curveless)
    assert [level.name for level in found] == [VALIDATION]
    assert gaps(found) == []


def test_epoch_gap_frame_differences_both_metrics():
    frame = epoch_gap_frame(record())
    first = frame[(frame.fold_index == 0) & (frame.epoch == 3)].iloc[0]
    assert first["ap_gap"] == pytest.approx(0.95 - 0.66)
    # Loss is subtracted the other way round: the held-out loss is the higher one.
    assert first["loss_gap"] == pytest.approx(2.0 / 3 - 1.0 / 3)


def test_stopped_early_flags_a_fold_that_used_the_whole_budget():
    exhausted = stopped_early(record(epochs=3, best_epoch=2))
    assert not exhausted["stopped_early"].any()

    patient = stopped_early(record(epochs=3, best_epoch=2) | {"config": {"epochs": 60}})
    assert patient["stopped_early"].all()


def test_markdown_table_names_every_level_and_gap():
    found = levels(record())
    rendered = markdown_table(found, gaps(found))
    for level in found:
        assert level.name in rendered
    assert "memorización" in rendered
    assert "generalización" in rendered
