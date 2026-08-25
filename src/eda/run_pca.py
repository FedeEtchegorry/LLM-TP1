"""PCA over the text bag-of-words, ranked by predictive value."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eda.dataset import DEFAULT_DATASET_PATH, load_btr_data, tokenize
from src.partitions import build_query_partitions

def build_bag_of_words(
    texts: Sequence[str], vocabulary: Sequence[str]
) -> np.ndarray:
    """Binary occurrence matrix over a fixed vocabulary."""
    position = {word: index for index, word in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(position)), dtype=np.float64)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            column = position.get(word)
            if column is not None:
                matrix[row, column] = 1.0
    return matrix


def fit_pca(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component vectors, explained-variance ratios and the column means."""
    means = matrix.mean(axis=0)
    centred = matrix - means
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    ratios = singular**2 / np.sum(singular**2)
    return components[:n_components], ratios[:n_components], means


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--figure", type=Path, default=None, help="write a PNG scatter here"
    )
    args = parser.parse_args(argv)

    data = load_btr_data(args.dataset)
    partitions = build_query_partitions(
        data.target.tolist(), data.query_ids, random_state=args.random_state
    )
    fold = partitions.folds[0]
    train, validation = list(fold.train_indices), list(fold.validation_indices)

    vocabulary = sorted(
        {word for index in train for word in tokenize(data.text[index])}
    )
    train_matrix = build_bag_of_words([data.text[i] for i in train], vocabulary)
    validation_matrix = build_bag_of_words(
        [data.text[i] for i in validation], vocabulary
    )
    components, ratios, means = fit_pca(train_matrix, args.components)

    projected = (validation_matrix - means) @ components.T
    actual = data.target[validation]

    print(
        f"vocabulary fitted on {len(train):,} training rows: {len(vocabulary)} words\n"
        f"components scored on {len(validation):,} validation rows\n"
    )
    print("| PC | Variance | AUC vs bought | Heaviest words |")
    print("|---:|---:|---:|---|")
    for index in range(args.components):
        auc = roc_auc_score(actual, projected[:, index])
        heaviest = [
            vocabulary[position]
            for position in np.argsort(-np.abs(components[index]))[:5]
        ]
        print(
            f"| {index + 1} | {100 * ratios[index]:.1f}% | {max(auc, 1 - auc):.3f} "
            f"| {', '.join(heaviest)} |"
        )
    print(f"\ncumulative variance of PC1+PC2: {100 * ratios[:2].sum():.1f}%")

    if args.figure is not None:
        _write_figure(args.figure, projected, actual, [data.oracle_tier[i] for i in validation], ratios)
    return 0


def _write_figure(
    destination: Path,
    projected: np.ndarray,
    actual: np.ndarray,
    tiers: Sequence[str],
    ratios: np.ndarray,
) -> None:
    """The same points twice, coloured by tier and then by the label."""
    try:
        import pandas as pd

        from src.eda.charts import save_figure, scatter_panels
    except ImportError:
        print(f"\nmatplotlib is not installed; skipped {destination}")
        return

    frame = pd.DataFrame(
        {
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
            "tier": list(tiers),
            "bought": np.where(actual == 1, "bought", "not bought"),
        }
    )
    scatter_panels(
        frame,
        x="PC1",
        y="PC2",
        groupings=(
            ("tier", "Coloured by the hand-assigned tier"),
            ("bought", "Coloured by the label"),
        ),
        title="PCA organises the text by product type, not by buyer",
        subtitle=(
            f"the first two components explain {100 * ratios[:2].sum():.1f}% of "
            f"variance between them. The clusters are product categories and each "
            f"one contains all three tiers, so a rotation that maximises variance "
            f"is not a rotation that separates the label"
        ),
        xlabel=f"PC1 — {100 * ratios[0]:.1f}% of variance",
        ylabel=f"PC2 — {100 * ratios[1]:.1f}%",
    )
    save_figure(destination)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
