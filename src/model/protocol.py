"""The one protocol every model reports into: same folds, same metrics.

A model enters as a ``ScoreFold``. Nothing here knows whether the scores came from
a logistic regression or a Transformer, which is what makes the table comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.partitions import DataPartitions

ScoreFold = Callable[[Sequence[int], Sequence[int]], np.ndarray]
"""Fits on the first argument only, then scores the second one, in that order."""


@dataclass(frozen=True)
class FoldScore:
    """What one fold produced, with the sizes that produced it."""

    fold_index: int
    roc_auc: float
    average_precision: float
    n_train: int
    n_scored: int


@dataclass(frozen=True)
class EvaluationResult:
    """One model over every fold, summarised the way the write-up quotes it."""

    name: str
    folds: tuple[FoldScore, ...]

    @property
    def roc_auc_mean(self) -> float:
        return fmean(fold.roc_auc for fold in self.folds)

    @property
    def roc_auc_std(self) -> float:
        return _std([fold.roc_auc for fold in self.folds])

    @property
    def average_precision_mean(self) -> float:
        return fmean(fold.average_precision for fold in self.folds)

    @property
    def average_precision_std(self) -> float:
        return _std([fold.average_precision for fold in self.folds])

    def summary_row(self) -> str:
        return (
            f"{self.name:<58s}  "
            f"ROC {self.roc_auc_mean:.4f} +/- {self.roc_auc_std:.4f}   "
            f"AP {self.average_precision_mean:.4f} +/- {self.average_precision_std:.4f}"
        )


def evaluate_across_folds(
    name: str,
    target: np.ndarray,
    partitions: DataPartitions,
    score_fold: ScoreFold,
) -> EvaluationResult:
    """Run ``score_fold`` on every fold. The test set is never passed to it."""
    scores = [
        _score_one(
            fold.fold_index,
            target,
            fold.train_indices,
            fold.validation_indices,
            score_fold,
        )
        for fold in partitions.folds
    ]
    return EvaluationResult(name=name, folds=tuple(scores))


def evaluate_on_test(
    name: str,
    target: np.ndarray,
    partitions: DataPartitions,
    score_fold: ScoreFold,
) -> EvaluationResult:
    """Train on all development rows and score the holdout. Call this once, ever."""
    score = _score_one(
        -1,
        target,
        partitions.development_indices,
        partitions.test_indices,
        score_fold,
    )
    return EvaluationResult(name=name, folds=(score,))


def _score_one(
    fold_index: int,
    target: np.ndarray,
    train_indices: Sequence[int],
    scored_indices: Sequence[int],
    score_fold: ScoreFold,
) -> FoldScore:
    """Fit, predict and measure, checking the model returned one score per row."""
    predicted = np.asarray(score_fold(train_indices, scored_indices), dtype=np.float64)
    actual = target[list(scored_indices)]
    if predicted.shape != actual.shape:
        raise ValueError(
            f"fold {fold_index}: expected {actual.shape} scores, got {predicted.shape}"
        )
    return FoldScore(
        fold_index=fold_index,
        roc_auc=float(roc_auc_score(actual, predicted)),
        average_precision=float(average_precision_score(actual, predicted)),
        n_train=len(train_indices),
        n_scored=len(scored_indices),
    )


def _std(values: Sequence[float]) -> float:
    """Sample deviation, defined as zero for the single-fold test result."""
    return stdev(values) if len(values) > 1 else 0.0


def markdown_table(results: Sequence[EvaluationResult], positive_rate: float) -> str:
    """The results as the write-up's table, with the random floor as the first row."""
    lines = [
        "| Model | ROC-AUC | PR-AUC (AP) |",
        "|---|---:|---:|",
        f"| random | 0.500 | {positive_rate:.3f} |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} "
            f"| {result.roc_auc_mean:.3f} ± {result.roc_auc_std:.3f} "
            f"| {result.average_precision_mean:.3f} ± {result.average_precision_std:.3f} |"
        )
    return "\n".join(lines)
