"""Compare a small set of encodings while the EDA columns stay together.

The classifier, grouped folds and all non-tested representations are fixed inside
each block. This is not a univariate feature ranking: every case represents the
complete EDA input contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.eda.loading import load_dataset
from src.model.baseline import OneHotLevels, WordIndicators, target_of
from src.model.configs import PROTOCOL
from src.model.eda_contract import CATEGORICAL_FIELDS, NUMERIC_FIELDS, TEXT_FIELDS
from src.model.embeddings import (
    Continuous,
    ContinuousAndBuckets,
    Periodic,
    PiecewiseLinear,
    QuantileBucketsBlock,
    TargetEncoded,
    TfidfWords,
)
from src.model.experiment import partition
from src.model.protocol import ScoreFold, evaluate_across_folds
from src.model.representation_selection import choose_deterministic, paired_margin
from src.model.results import RESULTS_DIR

TEXT = TEXT_FIELDS
CATEGORICAL = CATEGORICAL_FIELDS
PRICE = NUMERIC_FIELDS[0]
REGULARISATION = 1.0

EMBEDDINGS_DIR = "embeddings"
SWEEP_FILE = "linear-sweep.csv"
SELECTION_FILE = "selection.json"

TEXT_REFERENCE = "bolsa binaria"
CATEGORICAL_REFERENCE = "one-hot"
NUMERIC_REFERENCE = "buckets por cuantiles"
REFERENCES = {
    "text": TEXT_REFERENCE,
    "categorical": CATEGORICAL_REFERENCE,
    "numeric": NUMERIC_REFERENCE,
}


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


def text_blocks(name: str):
    builders = {
        "bolsa binaria": lambda: WordIndicators(TEXT),
        "tf-idf": lambda: TfidfWords(TEXT),
    }
    return [builders[name]()]


def categorical_blocks(name: str):
    if name == "one-hot":
        return [OneHotLevels(CATEGORICAL)]
    if name == "target encoding suavizado":
        return [TargetEncoded(field) for field in CATEGORICAL]
    raise KeyError(name)


def numeric_blocks(name: str):
    builders = {
        "continuo estandarizado": lambda: Continuous(PRICE),
        "buckets por cuantiles": lambda: QuantileBucketsBlock(PRICE),
        "continuo + buckets": lambda: ContinuousAndBuckets(PRICE),
        "piecewise-linear": lambda: PiecewiseLinear(PRICE),
        # Para que el eje numerico lineal cubra los mismos cinco modos que el del
        # Transformer. No es un equivalente exacto: el Transformer aprende las
        # frecuencias y aca son fijas (diadicas), porque ajustarlas no es convexo.
        "periodic": lambda: Periodic(PRICE),
    }
    return [builders[name]()]


def composed_blocks(text: str, categorical: str, numeric: str):
    """Build all three families, changing representation rather than fields."""
    return text_blocks(text) + categorical_blocks(categorical) + numeric_blocks(numeric)


def cases_for(
    block: str, selected: dict[str, str] | None = None
) -> list[tuple[str, object]]:
    """Return one-coordinate cases under the decisions made by prior blocks."""
    selected = selected or {}
    text = selected.get("text", TEXT_REFERENCE)
    categorical = selected.get("categorical", CATEGORICAL_REFERENCE)

    if block == "text":
        names = ("bolsa binaria", "tf-idf")
        return [
            (
                name,
                lambda name=name: composed_blocks(
                    name, CATEGORICAL_REFERENCE, NUMERIC_REFERENCE
                ),
            )
            for name in names
        ]
    if block == "categorical":
        names = ("one-hot", "target encoding suavizado")
        return [
            (
                name,
                lambda name=name: composed_blocks(text, name, NUMERIC_REFERENCE),
            )
            for name in names
        ]
    if block == "numeric":
        names = (
            "continuo estandarizado",
            "buckets por cuantiles",
            "continuo + buckets",
            "piecewise-linear",
            "periodic",
        )
        return [
            (
                name,
                lambda name=name: composed_blocks(text, categorical, name),
            )
            for name in names
        ]
    raise KeyError(block)


TEXT_CASES = cases_for("text")
CATEGORICAL_CASES = cases_for("categorical")
NUMERIC_CASES = cases_for("numeric")

BLOCKS = {
    "text": ("TEXTO: bolsa binaria contra TF-IDF", TEXT_CASES),
    "categorical": ("CATEGORICAS: one-hot contra target encoding", CATEGORICAL_CASES),
    "numeric": ("NUMERICA: como codificar price_position", NUMERIC_CASES),
}


def represented_fields(make_blocks) -> frozenset[str]:
    """Dataset fields represented by one case, used by the contract test."""
    represented: set[str] = set()
    for block in make_blocks():
        represented.update(getattr(block, "fields", ()))
        name = getattr(block, "name", None)
        if name:
            represented.add(name)
    return frozenset(represented)


def run_block(
    title,
    cases,
    frame,
    partitions,
    target,
    block: str,
    selected: dict[str, str],
) -> list[dict]:
    print(f"\n=== {title} ===")
    print(f"{'codificacion':<32s} {'ROC':>17s} {'AP':>17s} {'cols':>7s}")
    print("-" * 78)
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
                    "text_representation": (
                        name
                        if block == "text"
                        else selected.get("text", TEXT_REFERENCE)
                    ),
                    "categorical_representation": (
                        name
                        if block == "categorical"
                        else selected.get("categorical", CATEGORICAL_REFERENCE)
                    ),
                    "numeric_representation": (
                        name if block == "numeric" else NUMERIC_REFERENCE
                    ),
                }
            )
        print(
            f"{name:<32s} {result.roc_auc_mean:.4f} +/- {result.roc_auc_std:.4f} "
            f"{result.average_precision_mean:.4f} +/- "
            f"{result.average_precision_std:.4f} {width:>7,d}"
        )
        sys.stdout.flush()
    return records


def fold_scores(records: list[dict]) -> dict[str, np.ndarray]:
    """Collect AP in fold order for one representation block."""
    table = pd.DataFrame(records)
    scores: dict[str, np.ndarray] = {}
    for name, rows in table.groupby("encoding", sort=False):
        ordered = rows.sort_values("fold")
        scores[str(name)] = ordered["average_precision"].to_numpy(dtype=float)
    return scores


def decide(block: str, records: list[dict]) -> tuple[str, dict[str, str]]:
    """Apply the predeclared paired rule and make its reason auditable."""
    reference = REFERENCES[block]
    scores = fold_scores(records)
    selected = choose_deterministic(reference, scores)
    if selected == reference:
        reason = f"no alternative beat {reference} on mean AP"
    else:
        mean, low, high = paired_margin(scores[selected] - scores[reference])
        reason = (
            f"{selected} selected by higher mean AP versus {reference} "
            f"(delta={mean:.6f}, dispersion=[{low:.6f}, {high:.6f}])"
        )
    return selected, {
        "reference": reference,
        "selected": selected,
        "reason": reason,
    }


def write_selection(decisions: dict[str, dict[str, str]], results: Path | str) -> Path:
    path = Path(results) / EMBEDDINGS_DIR / SELECTION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        type=str,
        default="",
        choices=[""] + list(BLOCKS),
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
        "logistic head and complete EDA input held fixed"
    )

    wanted = [args.block] if args.block else list(BLOCKS)
    if args.block and args.block != "text":
        raise ValueError(
            "directed representation selection must start with text; "
            "run without --block to preserve the declared chain"
        )
    records: list[dict] = []
    selected: dict[str, str] = {}
    decisions: dict[str, dict[str, str]] = {}
    for key in wanted:
        title, _ = BLOCKS[key]
        cases = cases_for(key, selected)
        block_records = run_block(
            title, cases, frame, partitions, target, key, selected
        )
        records += block_records
        selected[key], decisions[key] = decide(key, block_records)
        print(
            f"selected {key}: {selected[key]} -- {decisions[key]['reason']}",
            flush=True,
        )

    table = pd.DataFrame(records)
    if args.block:
        stored = read_sweep(args.results)
        if not stored.empty:
            table = pd.concat(
                [stored[stored["block"] != args.block], table], ignore_index=True
            )
    path = write_sweep(table, args.results)
    print(f"\n{len(records)} fold rows written to {path}")
    if not args.block:
        selection_path = write_selection(decisions, args.results)
        print(f"representation decisions written to {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
