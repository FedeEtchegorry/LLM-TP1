"""Running one declared configuration: train, score, record, and never twice.

Both entrypoints go through here, so the ladder and the module sweep cannot drift
apart in how they train or in what they store.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from src.model.baseline import logistic_scorer, target_of
from src.model.configs import LOGISTIC, PROTOCOL, TRAINING, RunConfig
from src.model.protocol import EvaluationResult, ScoreFold, evaluate_across_folds
from src.model.results import RESULTS_DIR, load, save, save_weights
from src.model.training import TrainedFold, transformer_scorer
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


def scorer_for(
    config: RunConfig, frame: pd.DataFrame, trained: list[TrainedFold]
) -> ScoreFold:
    if config.model == LOGISTIC:
        return logistic_scorer(
            frame,
            categorical_fields=config.categorical_fields,
            numeric_fields=config.numeric_fields,
            n_buckets=TRAINING.n_buckets,
            c=TRAINING.regularisation,
        )
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
        config.name, target_of(frame), partitions, scorer_for(config, frame, trained)
    )
    seconds = time.perf_counter() - started

    save(config, result, seconds=seconds, curves=_curves(trained), directory=directory)
    if PROTOCOL.save_weights:
        for fold_index, fold in enumerate(trained):
            save_weights(config, fold_index, fold.model.state_dict(), directory=directory)

    note = f"{seconds:.0f}s"
    if trained:
        note += f", {trained[0].parameters:,} params, best epoch {trained[0].best_epoch}"
    return result, note


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
