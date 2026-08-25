"""Does the target contain feature interactions a linear model cannot reach?"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from src.eda.dataset import DEFAULT_DATASET_PATH, BtrData, load_btr_data
from src.eda.evaluation import (
    EvaluationResult,
    evaluate_across_folds,
    logistic_scorer,
)
from src.eda.features import (
    CategoricalOneHot,
    Crossed,
    FeatureSpec,
    NumericBuckets,
)
from src.partitions import DataPartitions, build_query_partitions

PRICE_BINS = 10


def _popularity(field: str) -> CategoricalOneHot:
    return CategoricalOneHot((field,))


def _price() -> NumericBuckets:
    return NumericBuckets(("price_pct",), n_bins=PRICE_BINS)


def _category() -> CategoricalOneHot:
    return CategoricalOneHot(("category",))


def interaction_specs(field: str, *, uses_oracle: bool) -> tuple[FeatureSpec, ...]:
    """The additive/crossed ladder over ``field``, ``price_pct`` and ``category``."""
    label = "tier" if uses_oracle else "phrase"
    return (
        FeatureSpec(f"{label} alone", (_popularity(field),), uses_oracle),
        FeatureSpec(
            f"{label} + price buckets (additive)",
            (_popularity(field), _price()),
            uses_oracle,
        ),
        FeatureSpec(
            f"{label} x price buckets (interaction)",
            (_popularity(field), _price(), Crossed(_popularity(field), _price())),
            uses_oracle,
        ),
        FeatureSpec(
            f"{label} + price + category (additive)",
            (_popularity(field), _price(), _category()),
            uses_oracle,
        ),
        FeatureSpec(
            f"{label} x price, {label} x category (interaction)",
            (
                _popularity(field),
                _price(),
                _category(),
                Crossed(_popularity(field), _price()),
                Crossed(_popularity(field), _category()),
            ),
            uses_oracle,
        ),
        FeatureSpec(
            f"{label} x price x category, all pairs (interaction)",
            (
                _popularity(field),
                _price(),
                _category(),
                Crossed(_popularity(field), _price()),
                Crossed(_popularity(field), _category()),
                Crossed(_price(), _category()),
            ),
            uses_oracle,
        ),
    )


def evaluate_specs(
    data: BtrData, partitions: DataPartitions, specs: Iterable[FeatureSpec]
) -> list[EvaluationResult]:
    return [
        evaluate_across_folds(
            spec.name,
            data.target,
            partitions,
            logistic_scorer(data, spec),
            uses_oracle=spec.uses_oracle,
        )
        for spec in specs
    ]


def markdown_table(results: Iterable[EvaluationResult]) -> str:
    lines = [
        "| Feature set | ROC-AUC | PR-AUC (AP) | Δ AP vs additive |",
        "|---|---:|---:|---:|",
    ]
    additive = None
    for result in results:
        if "additive" in result.name:
            additive = result.average_precision_mean
        delta = ""
        if "interaction" in result.name and additive is not None:
            delta = f"{result.average_precision_mean - additive:+.3f}"
        lines.append(
            f"| {result.name} "
            f"| {result.roc_auc_mean:.3f} ± {result.roc_auc_std:.3f} "
            f"| {result.average_precision_mean:.3f} ± {result.average_precision_std:.3f} "
            f"| {delta} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="also run the ladder over the hand-assigned tier",
    )
    args = parser.parse_args(argv)

    data = load_btr_data(args.dataset)
    partitions = build_query_partitions(
        data.target.tolist(),
        data.query_ids,
        n_folds=args.folds,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )
    fold = partitions.folds[0]
    print(
        f"rows {len(data)}  queries {len(set(data.query_ids))}  "
        f"positive rate {data.positive_rate:.4f}\n"
        f"per fold: train {len(fold.train_indices)} / "
        f"validation {len(fold.validation_indices)} (seed {args.random_state})\n"
    )

    honest = evaluate_specs(
        data, partitions, interaction_specs("popularity_phrase", uses_oracle=False)
    )
    print("=== ADDITIVE vs INTERACTION (impression-time popularity phrase) ===")
    for result in honest:
        print(" ", result.summary_row())
    print("\n--- Markdown ---\n")
    print(markdown_table(honest))

    if args.include_oracle:
        oracle = evaluate_specs(
            data, partitions, interaction_specs("oracle_tier", uses_oracle=True)
        )
        print("\n\n=== same ladder over the hand-assigned tier (oracle) ===")
        for result in oracle:
            print(" ", result.summary_row())
        print("\n--- Markdown ---\n")
        print(markdown_table(oracle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
