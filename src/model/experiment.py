"""Running one declared configuration: train, score, record, and never twice.

Both entrypoints go through here, so the ladder and the module sweep cannot drift
apart in how they train or in what they store.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.model.baseline import logistic_scorer, target_of
from src.model.configs import (
    FINETUNE,
    FROZEN,
    LOGISTIC,
    PROTOCOL,
    TRAINING,
    TRANSFER,
    TRANSFORMER,
    RunConfig,
)
from src.model.protocol import (
    EvaluationResult,
    ScoreFold,
    evaluate_across_folds,
    evaluate_on_test,
)
from src.model.records import TrainedFold
from src.model.results import (
    RESULTS_DIR,
    load,
    load_predictions,
    save,
    save_predictions,
    save_weights,
)
from src.model.training import transformer_scorer
from src.partitions import DataPartitions, build_query_partitions


def partition(frame: pd.DataFrame) -> DataPartitions:
    """The split from ``configs.PROTOCOL``; no run is allowed a different one."""
    return build_query_partitions(
        target_of(frame).tolist(),
        frame["query_id"].tolist(),
        n_folds=PROTOCOL.folds,
        test_fraction=PROTOCOL.test_fraction,
        random_state=PROTOCOL.random_state,
    )


def describe(frame: pd.DataFrame, partitions: DataPartitions) -> None:
    target = target_of(frame)
    print(
        f"{len(frame)} rows, {frame['query_id'].nunique()} queries, "
        f"positive rate {target.mean():.4f}"
    )
    print(f"test holdout {len(partitions.test_indices)} rows, never scored here")
    for fold in partitions.folds:
        print(
            f"  fold {fold.fold_index}: train {len(fold.train_indices)} "
            f"validation {len(fold.validation_indices)}"
        )


def folds_for(config: RunConfig, partitions: DataPartitions) -> DataPartitions:
    """The folds this run reports into.

    Every regime gets the same five, with one stated exception: fine-tuning a 22M
    parameter checkpoint is a training run per epoch, so it reports fold 0 only. The
    restriction lives here, once, rather than in the entrypoint, so a fine-tune
    launched from anywhere gets the same budget -- and so the stored record, which
    carries its folds, always says how many it actually ran.
    """
    if config.model != FINETUNE:
        return partitions
    return replace(partitions, folds=partitions.folds[: TRANSFER.finetune_folds])


def scorer_for(
    config: RunConfig, frame: pd.DataFrame, trained: list[TrainedFold]
) -> ScoreFold:
    """The one place a model name becomes a model. Four regimes, one signature."""
    if config.model == LOGISTIC:
        return logistic_scorer(
            frame,
            text_fields=config.text_fields,
            categorical_fields=config.categorical_fields,
            numeric_fields=config.numeric_fields,
            n_buckets=TRAINING.n_buckets,
            c=TRAINING.regularisation,
            folds=trained,
        )
    if config.model == FROZEN:
        from src.model.pretrained import embeddings_for, frozen_scorer

        return frozen_scorer(
            config,
            frame,
            embeddings_for(frame, config.text_fields),
            n_buckets=TRAINING.n_buckets,
            c=TRAINING.regularisation,
            folds=trained,
        )
    if config.model == FINETUNE:
        from src.model.finetuning import finetune_scorer

        return finetune_scorer(config, frame, folds=trained)
    return transformer_scorer(config, frame, folds=trained)


def run_one(
    config: RunConfig,
    frame: pd.DataFrame,
    partitions: DataPartitions,
    *,
    directory=RESULTS_DIR,
    force: bool = False,
) -> tuple[EvaluationResult, str]:
    """Return the run's result and a one-word note saying where it came from."""
    if not force:
        recorded = load(config, directory)
        if recorded is not None:
            return recorded, "recorded"

    trained: list[TrainedFold] = []
    started = time.perf_counter()
    result = evaluate_across_folds(
        config.name,
        target_of(frame),
        folds_for(config, partitions),
        scorer_for(config, frame, trained),
    )
    seconds = time.perf_counter() - started

    save(config, result, seconds=seconds, curves=_curves(trained), directory=directory)
    # Only our own model's weights are kept. The linear regimes have nothing worth
    # storing, and the fine-tuned checkpoint is 22M parameters that the same recipe
    # reproduces -- ``run_final`` reads attention out of the Transformer alone.
    if PROTOCOL.save_weights and config.model == TRANSFORMER:
        for fold_index, fold in enumerate(trained):
            save_weights(config, fold_index, fold.model.state_dict(), directory=directory)

    note = f"{seconds:.0f}s"
    if trained:
        note += f", {trained[0].parameters:,} params"
        if trained[0].curve:
            note += f", best epoch {trained[0].best_epoch}"
    return result, note


FINAL_DIR = RESULTS_DIR / "final"
"""The held-out run lands here, apart from the cross-validation records.

Separate on purpose. ``results.documents`` globs ``results/*.json`` to build the
selection table, and a test number must never be able to walk into the table a model
is selected from.
"""


def run_test(
    config: RunConfig,
    frame: pd.DataFrame,
    partitions: DataPartitions,
    *,
    directory=FINAL_DIR,
    force: bool = False,
) -> tuple[EvaluationResult, np.ndarray, str]:
    """Train on every development row, score the holdout, once.

    The scores are kept beside the metrics so that every Stage 6 table and figure is a
    view of one evaluation rather than an excuse to run another. A configuration that
    already has a record here is *not* re-run: ``--force`` exists, but reaching for it
    means deciding to spend the holdout a second time, which is a decision and not a
    default.
    """
    recorded = None if force else load(config, directory)
    predicted = None if force else load_predictions(config, directory)
    if recorded is not None and predicted is not None:
        return recorded, predicted, "recorded"

    trained: list[TrainedFold] = []
    captured: dict[str, object] = {}
    scorer = scorer_for(config, frame, trained)

    def capturing(train_indices, scored_indices):
        scores = scorer(train_indices, scored_indices)
        captured["predicted"] = scores
        return scores

    started = time.perf_counter()
    result = evaluate_on_test(config.name, target_of(frame), partitions, capturing)
    seconds = time.perf_counter() - started

    predicted = captured["predicted"]
    save(config, result, seconds=seconds, curves=_curves(trained), directory=directory)
    save_predictions(config, predicted, directory=directory)
    if PROTOCOL.save_weights and config.model == TRANSFORMER and trained:
        save_weights(config, -1, trained[0].model.state_dict(), directory=directory)

    note = f"{seconds:.0f}s"
    if trained:
        note += f", {trained[0].parameters:,} params"
    return result, predicted, note


def _curves(trained: list[TrainedFold]) -> list[dict]:
    return [
        {
            "fold_index": fold_index,
            "best_epoch": fold.best_epoch,
            "parameters": fold.parameters,
            "epochs": [asdict(record) for record in fold.curve],
        }
        for fold_index, fold in enumerate(trained)
    ]
