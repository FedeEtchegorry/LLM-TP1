"""Baseline and headroom tables for the BTR task."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from src.eda.dataset import (
    DEFAULT_DATASET_PATH,
    BtrData,
    load_btr_data,
)
from src.eda.evaluation import (
    EvaluationResult,
    evaluate_across_folds,
    logistic_scorer,
)
from src.eda.features import (
    BagOfWords,
    CategoricalOneHot,
    MissingIndicators,
    NumericBuckets,
    NumericScaled,
    FeatureSpec,
)
from src.partitions import DataPartitions, build_query_partitions

TABULAR_CATEGORICAL = ("category", "storage_type", "allergens", "unit_of_measure")
ALL_CATEGORICAL = TABULAR_CATEGORICAL + ("brand", "country_of_origin")
ALL_NUMERIC = ("price", "price_pct", "net_weight_oz", "nutrition_score")


def baseline_specs() -> tuple[FeatureSpec, ...]:
    """Feature sets that use nothing beyond what a search page knows."""
    return (
        FeatureSpec("numerics only, linear", (NumericScaled(ALL_NUMERIC),)),
        FeatureSpec(
            "tabular only (numerics + categoricals)",
            (NumericScaled(ALL_NUMERIC), CategoricalOneHot(ALL_CATEGORICAL),
             MissingIndicators()),
        ),
        FeatureSpec("bag-of-words, unigrams", (BagOfWords(),)),
        FeatureSpec(
            "bag-of-words + bigrams",
            (BagOfWords(ngram_range=(1, 2), min_df=5),),
        ),
        FeatureSpec(
            "bag-of-words + tabular",
            (BagOfWords(), NumericScaled(ALL_NUMERIC),
             CategoricalOneHot(ALL_CATEGORICAL), MissingIndicators()),
        ),
        FeatureSpec(
            "bag-of-words + tabular + bucketed numerics",
            (BagOfWords(), NumericScaled(ALL_NUMERIC),
             NumericBuckets(ALL_NUMERIC), CategoricalOneHot(ALL_CATEGORICAL),
             MissingIndicators()),
        ),
        FeatureSpec(
            "popularity phrase only (learned from train labels)",
            (CategoricalOneHot(("popularity_phrase",)),),
        ),
        FeatureSpec(
            "popularity phrase + bucketed price_pct + category + allergens",
            (CategoricalOneHot(("popularity_phrase", "category", "allergens")),
             NumericBuckets(("price_pct",))),
        ),
    )


def oracle_specs() -> tuple[FeatureSpec, ...]:
    """Feature sets containing the hand-assigned tier -- upper bounds only."""
    return (
        FeatureSpec(
            "oracle tier only",
            (CategoricalOneHot(("oracle_tier",)),),
            uses_oracle=True,
        ),
        FeatureSpec(
            "oracle tier + price_pct, linear",
            (CategoricalOneHot(("oracle_tier",)), NumericScaled(("price_pct",))),
            uses_oracle=True,
        ),
        FeatureSpec(
            "oracle tier + price_pct in 10 buckets",
            (CategoricalOneHot(("oracle_tier",)), NumericBuckets(("price_pct",))),
            uses_oracle=True,
        ),
        FeatureSpec(
            "oracle tier + buckets + category",
            (CategoricalOneHot(("oracle_tier", "category")),
             NumericBuckets(("price_pct",))),
            uses_oracle=True,
        ),
        FeatureSpec(
            "oracle tier + buckets + category + allergens",
            (CategoricalOneHot(("oracle_tier", "category", "allergens")),
             NumericBuckets(("price_pct",))),
            uses_oracle=True,
        ),
        FeatureSpec(
            "oracle tier + buckets + category + allergens + brand + country",
            (CategoricalOneHot(
                ("oracle_tier", "category", "allergens", "brand",
                 "country_of_origin")),
             NumericBuckets(("price_pct",))),
            uses_oracle=True,
        ),
    )


def evaluate_specs(
    data: BtrData, partitions: DataPartitions, specs: Iterable[FeatureSpec]
) -> list[EvaluationResult]:
    """Score every spec across the cross-validation folds."""
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


def markdown_table(results: Iterable[EvaluationResult], positive_rate: float) -> str:
    """Render results as a Markdown table, mean +/- std across folds."""
    lines = [
        "| Model | ROC-AUC | PR-AUC (AP) |",
        "|---|---:|---:|",
        f"| random | 0.500 | {positive_rate:.3f} |",
    ]
    for result in results:
        label = f"*{result.name}*" if result.uses_oracle else result.name
        lines.append(
            f"| {label} "
            f"| {result.roc_auc_mean:.3f} ± {result.roc_auc_std:.3f} "
            f"| {result.average_precision_mean:.3f} ± {result.average_precision_std:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
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
        f"split: test {len(partitions.test_indices)} rows, "
        f"per fold train {len(fold.train_indices)} / "
        f"validation {len(fold.validation_indices)} "
        f"(seed {args.random_state})\n"
    )

    baselines = evaluate_specs(data, partitions, baseline_specs())
    oracles = evaluate_specs(data, partitions, oracle_specs())

    print("=== BASELINES (impression-time features, fitted per fold) ===")
    for result in baselines:
        print(" ", result.summary_row())
    print("\n=== ORACLE (hand-assigned tier: upper bound, not achievable) ===")
    for result in oracles:
        print(" ", result.summary_row())

    print("\n\n--- Markdown: baselines ---\n")
    print(markdown_table(baselines, data.positive_rate))
    print("\n--- Markdown: oracle headroom ---\n")
    print(markdown_table(oracles, data.positive_rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
