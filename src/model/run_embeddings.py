from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.eda.loading import load_dataset
from src.model.baseline import OneHotLevels, WordIndicators, target_of
from src.model.configs import PROTOCOL
from src.model.embeddings import (
    Continuous,
    ContinuousAndBuckets,
    HashedLevels,
    OrdinalLevels,
    Periodic,
    PiecewiseLinear,
    QuantileBucketsBlock,
    TargetEncoded,
    TfidfWords,
)
from src.model.experiment import partition
from src.model.protocol import ScoreFold, evaluate_across_folds
from src.model.results import RESULTS_DIR

BAR_CAT = ("popularity_phrase", "category", "allergens")
PRICE = "price_position"
TEXT = ("title", "description")
REGULARISATION = 1.0

EMBEDDINGS_DIR = "embeddings"
SWEEP_FILE = "linear-sweep.csv"


def sweep_path(results: Path | str) -> Path:
    return Path(results) / EMBEDDINGS_DIR / SWEEP_FILE


def read_sweep(results: Path | str) -> pd.DataFrame:
    path = sweep_path(results)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_sweep(table: pd.DataFrame, results: Path | str) -> Path:
    path = sweep_path(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def blocks_scorer(make_blocks, frame, *, c: float = REGULARISATION) -> ScoreFold:
    target = target_of(frame)

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        blocks = make_blocks()
        for block in blocks:
            block.fit(frame, train_indices)
        train_matrix = np.hstack([b.transform(frame, train_indices) for b in blocks])
        scored_matrix = np.hstack([b.transform(frame, scored_indices) for b in blocks])
        model = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
        model.fit(train_matrix, target[list(train_indices)])
        score_fold.width = train_matrix.shape[1]
        return model.predict_proba(scored_matrix)[:, 1]

    return score_fold


def bar_categoricals():
    return [OneHotLevels(BAR_CAT)]


NUMERIC_CASES = [
    ("sin price_position", lambda: bar_categoricals()),
    ("continuo (1 columna)", lambda: bar_categoricals() + [Continuous(PRICE)]),
    (
        "percentiles one-hot (la barra)",
        lambda: bar_categoricals() + [QuantileBucketsBlock(PRICE)],
    ),
    (
        "continuo + percentiles (2 codificaciones)",
        lambda: bar_categoricals() + [ContinuousAndBuckets(PRICE)],
    ),
    ("piecewise-linear", lambda: bar_categoricals() + [PiecewiseLinear(PRICE)]),
    ("periodico (Fourier, 4 frecuencias)", lambda: bar_categoricals() + [Periodic(PRICE)]),
    ("percentiles 5 tramos", lambda: bar_categoricals() + [QuantileBucketsBlock(PRICE, 5)]),
    ("percentiles 20 tramos", lambda: bar_categoricals() + [QuantileBucketsBlock(PRICE, 20)]),
]

CATEGORICAL_CASES = [
    (
        "one-hot (lo que dice el informe)",
        lambda: [OneHotLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
    (
        "target encoding suavizado",
        lambda: [TargetEncoded(name) for name in BAR_CAT] + [QuantileBucketsBlock(PRICE)],
    ),
    (
        "one-hot + target (2 codificaciones)",
        lambda: [OneHotLevels(BAR_CAT)]
        + [TargetEncoded(name) for name in BAR_CAT]
        + [QuantileBucketsBlock(PRICE)],
    ),
    (
        "hashing a 32 columnas",
        lambda: [HashedLevels(BAR_CAT, 32), QuantileBucketsBlock(PRICE)],
    ),
    (
        "ordinal (el control que debe perder)",
        lambda: [OrdinalLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
]

TEXT_CASES = [
    (
        "frase extraida one-hot (la barra)",
        lambda: [OneHotLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
    (
        "bolsa de palabras binaria",
        lambda: [
            WordIndicators(TEXT),
            OneHotLevels(("category", "allergens")),
            QuantileBucketsBlock(PRICE),
        ],
    ),
    (
        "tf-idf",
        lambda: [
            TfidfWords(TEXT),
            OneHotLevels(("category", "allergens")),
            QuantileBucketsBlock(PRICE),
        ],
    ),
    (
        "frase + bolsa de palabras (2 codificaciones)",
        lambda: [WordIndicators(TEXT), OneHotLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
    (
        "frase + tf-idf (2 codificaciones)",
        lambda: [TfidfWords(TEXT), OneHotLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
]

TX_CAT = ("category", "storage_type", "allergens", "unit_of_measure")
TX_NUM = ("price", "price_position", "net_weight_oz", "nutrition_score")

FIELD_CASES = [
    (
        "las 4 del informe (frase+category+allergens+price_pct)",
        lambda: [OneHotLevels(BAR_CAT), QuantileBucketsBlock(PRICE)],
    ),
    (
        "+ las 2 categoricas que el informe descarta",
        lambda: [
            OneHotLevels(BAR_CAT + ("storage_type", "unit_of_measure")),
            QuantileBucketsBlock(PRICE),
        ],
    ),
    (
        "+ las 3 numericas que el informe descarta",
        lambda: [OneHotLevels(BAR_CAT)] + [QuantileBucketsBlock(n) for n in TX_NUM],
    ),
    (
        "las 9 columnas que parameters.txt le da al Transformer",
        lambda: [
            WordIndicators(("title", "description", "ingredients")),
            OneHotLevels(TX_CAT),
        ]
        + [QuantileBucketsBlock(n) for n in TX_NUM],
    ),
]

BLOCKS = {
    "numeric": ("LA PREGUNTA ABIERTA: como codificar price_position", NUMERIC_CASES),
    "categorical": ("CATEGORICAS: contra que se eligio one-hot", CATEGORICAL_CASES),
    "text": ("TEXTO: frase extraida contra el texto crudo", TEXT_CASES),
    "fields": ("COLUMNAS: las 4 del informe contra las 9 que recibe el modelo", FIELD_CASES),
}


def run_block(title, cases, frame, partitions, target, block: str) -> list[dict]:
    print(f"\n=== {title} ===")
    print(f"{'codificacion':<46s} {'ROC':>17s} {'AP':>17s} {'cols':>7s}")
    print("-" * 92)
    records: list[dict] = []
    for name, make_blocks in cases:
        scorer = blocks_scorer(make_blocks, frame)
        result = evaluate_across_folds(name, target, partitions, scorer)
        width = getattr(scorer, "width", 0)
        for fold in result.folds:
            records.append(
                {
                    "block": block,
                    "encoding": name,
                    "fold": fold.fold_index,
                    "roc_auc": fold.roc_auc,
                    "average_precision": fold.average_precision,
                    "columns": width,
                }
            )
        print(
            f"{name:<46s} {result.roc_auc_mean:.4f} +/- {result.roc_auc_std:.4f} "
            f"{result.average_precision_mean:.4f} +/- {result.average_precision_std:.4f} "
            f"{width:>7,d}"
        )
        sys.stdout.flush()
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block", type=str, default="", choices=[""] + list(BLOCKS),
        help="run one block only",
    )
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    target = target_of(frame)
    print(
        f"{len(frame)} rows, {frame['query_id'].nunique()} queries, "
        f"positive rate {target.mean():.4f}, {PROTOCOL.folds} folds, "
        "logistic head held fixed"
    )

    wanted = [args.block] if args.block else list(BLOCKS)
    records: list[dict] = []
    for key in wanted:
        title, cases = BLOCKS[key]
        records += run_block(title, cases, frame, partitions, target, key)

    table = pd.DataFrame(records)
    if args.block:
        stored = read_sweep(args.results)
        if not stored.empty:
            table = pd.concat(
                [stored[stored["block"] != args.block], table], ignore_index=True
            )
    path = write_sweep(table, args.results)
    print(f"\n{len(records)} filas por fold escritas en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
