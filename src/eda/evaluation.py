"""Fold-level evaluation shared by every model in the project.

The harness is deliberately model-agnostic. A candidate supplies one callable::

    score_fold(train_indices, validation_indices) -> np.ndarray

which fits on the training rows and returns a score per validation row, higher
meaning more likely to be bought. A logistic-regression baseline and a
Transformer therefore report through the same protocol, on the same folds, with
the same metrics -- which is what makes an ablation table comparable.

Metric definitions, stated once so the numbers are unambiguous:

``roc_auc``
    :func:`sklearn.metrics.roc_auc_score`.

``average_precision``
    :func:`sklearn.metrics.average_precision_score` -- the step-wise sum
    ``sum_n (R_n - R_{n-1}) * P_n``. Reported as "PR-AUC" in prose; it is average
    precision, not a trapezoidal area under an interpolated curve.

The fixed test set is never touched by :func:`evaluate_across_folds`. Score it
once, with :func:`evaluate_on_test`, after a configuration has been chosen.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eda.dataset import BtrData
from src.eda.features import FeatureSpec
from src.partitions import DataPartitions

ScoreFold = Callable[[Sequence[int], Sequence[int]], np.ndarray]


@dataclass(frozen=True)
class FoldScore:
    """Validation metrics for a single cross-validation fold."""

    fold_index: int
    roc_auc: float
    average_precision: float
    n_train: int
    n_validation: int


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one candidate across every fold."""

    name: str
    folds: tuple[FoldScore, ...]
    uses_oracle: bool = False

    @property
    def roc_auc_mean(self) -> float:
        return statistics.fmean(fold.roc_auc for fold in self.folds)

    @property
    def roc_auc_std(self) -> float:
        return _std([fold.roc_auc for fold in self.folds])

    @property
    def average_precision_mean(self) -> float:
        return statistics.fmean(fold.average_precision for fold in self.folds)

    @property
    def average_precision_std(self) -> float:
        return _std([fold.average_precision for fold in self.folds])

    def summary_row(self) -> str:
        marker = "  [oracle]" if self.uses_oracle else ""
        return (
            f"{self.name:<52s} "
            f"ROC {self.roc_auc_mean:.4f} +/- {self.roc_auc_std:.4f}   "
            f"AP {self.average_precision_mean:.4f} +/- {self.average_precision_std:.4f}"
            f"{marker}"
        )


def evaluate_across_folds(
    name: str,
    target: np.ndarray,
    partitions: DataPartitions,
    score_fold: ScoreFold,
    *,
    uses_oracle: bool = False,
) -> EvaluationResult:
    """Run ``score_fold`` on every fold and collect validation metrics."""

    scores = []
    for fold in partitions.folds:
        predicted = np.asarray(
            score_fold(fold.train_indices, fold.validation_indices), dtype=np.float64
        )
        actual = target[list(fold.validation_indices)]
        if predicted.shape != actual.shape:
            raise ValueError(
                f"{name}: score_fold returned {predicted.shape} scores for "
                f"{actual.shape} validation rows"
            )
        scores.append(
            FoldScore(
                fold_index=fold.fold_index,
                roc_auc=float(roc_auc_score(actual, predicted)),
                average_precision=float(average_precision_score(actual, predicted)),
                n_train=len(fold.train_indices),
                n_validation=len(fold.validation_indices),
            )
        )
    return EvaluationResult(name=name, folds=tuple(scores), uses_oracle=uses_oracle)


def evaluate_on_test(
    name: str,
    target: np.ndarray,
    partitions: DataPartitions,
    score_fold: ScoreFold,
) -> FoldScore:
    """Score the fixed test set once, training on all development rows."""

    development = partitions.development_indices
    predicted = np.asarray(
        score_fold(development, partitions.test_indices), dtype=np.float64
    )
    actual = target[list(partitions.test_indices)]
    return FoldScore(
        fold_index=-1,
        roc_auc=float(roc_auc_score(actual, predicted)),
        average_precision=float(average_precision_score(actual, predicted)),
        n_train=len(development),
        n_validation=len(partitions.test_indices),
    )


def logistic_scorer(
    data: BtrData, spec: FeatureSpec, *, c: float = 1.0, max_iter: int = 2000
) -> ScoreFold:
    """Return a :data:`ScoreFold` fitting L2 logistic regression on ``spec``."""

    def score_fold(
        train_indices: Sequence[int], other_indices: Sequence[int]
    ) -> np.ndarray:
        train_matrix, other_matrix = spec.fit_transform(
            data, train_indices, other_indices
        )
        model = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
        model.fit(train_matrix, data.target[list(train_indices)])
        return model.predict_proba(other_matrix)[:, 1]

    return score_fold


def _std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0
