"""Leakage-safe query-grouped indices for BTR model selection."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import isclose

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldIndices:
    """Disjoint training and validation indices for one model-selection run."""

    fold_index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class DataPartitions:
    """A fixed test holdout plus cross-validation folds over development data."""

    test_indices: tuple[int, ...]
    folds: tuple[FoldIndices, ...]

    @property
    def development_indices(self) -> tuple[int, ...]:
        """Return every non-test row index in deterministic order."""

        first_fold = self.folds[0]
        return tuple(sorted(first_fold.train_indices + first_fold.validation_indices))


def build_query_partitions(
    targets: Sequence[bool | int],
    query_ids: Sequence[Hashable],
    *,
    n_folds: int = 5,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> DataPartitions:
    """Split row indices while keeping each query entirely in one subset.

    The fixed test set is one fold of an outer grouped split. A separate inner
    grouped split creates train/validation folds from the development rows.
    """

    if len(targets) != len(query_ids):
        raise ValueError("targets and query_ids must have the same length")
    if len(targets) == 0:
        raise ValueError("targets and query_ids must not be empty")
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    outer_split_count = _split_count_for_fraction(test_fraction)
    target_array = _validate_and_normalize_targets(targets)
    query_array = _normalize_query_ids(query_ids)

    if len(np.unique(query_array)) < outer_split_count:
        raise ValueError(
            "there must be at least one distinct query_id per outer split"
        )

    features = np.zeros((len(target_array), 1), dtype=np.uint8)
    _require_class_group_support(
        target_array,
        query_array,
        required_groups=outer_split_count,
        context="outer test split",
    )

    outer_splitter = StratifiedGroupKFold(
        n_splits=outer_split_count,
        shuffle=True,
        random_state=random_state,
    )
    development_indices, test_indices = next(
        outer_splitter.split(features, target_array, groups=query_array)
    )

    development_targets = target_array[development_indices]
    development_queries = query_array[development_indices]
    development_features = features[development_indices]
    if len(np.unique(development_queries)) < n_folds:
        raise ValueError(
            "the development set must contain at least one query_id per fold"
        )
    _require_class_group_support(
        development_targets,
        development_queries,
        required_groups=n_folds,
        context="development folds",
    )

    inner_splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state + 1,
    )
    folds: list[FoldIndices] = []
    for fold_index, (train_relative, validation_relative) in enumerate(
        inner_splitter.split(
            development_features,
            development_targets,
            groups=development_queries,
        )
    ):
        folds.append(
            FoldIndices(
                fold_index=fold_index,
                train_indices=tuple(
                    sorted(int(index) for index in development_indices[train_relative])
                ),
                validation_indices=tuple(
                    sorted(
                        int(index) for index in development_indices[validation_relative]
                    )
                ),
            )
        )

    return DataPartitions(
        test_indices=tuple(sorted(int(index) for index in test_indices)),
        folds=tuple(folds),
    )


def _split_count_for_fraction(test_fraction: float) -> int:
    """Return the grouped-CV split count represented by a test fraction."""

    if isinstance(test_fraction, bool) or not 0.0 < test_fraction <= 0.5:
        raise ValueError("test_fraction must be greater than 0 and at most 0.5")
    inverse = 1.0 / test_fraction
    split_count = round(inverse)
    if not isclose(inverse, split_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "test_fraction must be the reciprocal of an integer, such as 0.2"
        )
    return split_count


def _validate_and_normalize_targets(
    targets: Sequence[bool | int],
) -> np.ndarray:
    """Validate binary targets and convert them to scikit-learn's integer form."""

    accepted_types = (bool, int, np.bool_, np.integer)
    if any(not isinstance(target, accepted_types) for target in targets):
        raise ValueError("targets must contain 0 and 1 only")

    if any(target not in (False, True, 0, 1) for target in targets):
        raise ValueError("targets must contain 0 and 1 only")
    target_array = np.asarray(targets, dtype=np.int8)
    if set(target_array.tolist()) != {0, 1}:
        raise ValueError("targets must contain 0 and 1 only")
    return target_array


def _normalize_query_ids(query_ids: Sequence[Hashable]) -> np.ndarray:
    """Map arbitrary hashable query identifiers to stable integer group labels."""

    group_codes: dict[Hashable, int] = {}
    normalized: list[int] = []
    try:
        for query_id in query_ids:
            code = group_codes.setdefault(query_id, len(group_codes))
            normalized.append(code)
    except TypeError as error:
        raise TypeError("every query_id must be hashable") from error
    return np.asarray(normalized, dtype=np.int64)


def _require_class_group_support(
    targets: np.ndarray,
    query_ids: np.ndarray,
    *,
    required_groups: int,
    context: str,
) -> None:
    """Ensure each class spans enough query groups for grouped stratification."""

    for target in (0, 1):
        supporting_groups = np.unique(query_ids[targets == target])
        if len(supporting_groups) < required_groups:
            raise ValueError(
                f"{context} requires class {target} in at least "
                f"{required_groups} query groups"
            )
