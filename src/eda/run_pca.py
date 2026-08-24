"""PCA over the text bag-of-words, ranked by predictive value.

Run with ``python -m src.eda.run_pca``.

The point of the exhibit is that PCA maximises *variance*, not separability. The
components are fitted on one fold's training rows and each is then scored, on
that fold's validation rows, by how well it alone ranks ``bought`` -- so the
ranking is a measurement rather than a description of the data it was fitted on.

The figure is optional: matplotlib is not a project dependency, so the numbers
print with or without it and ``--figure`` is skipped when it is unavailable.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eda.dataset import DEFAULT_DATASET_PATH, load_btr_data, tokenize
from src.partitions import build_query_partitions

TIER_COLOURS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib is not installed; skipped {destination}")
        return

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor="white")
    tier_array = np.array(tiers)
    for tier in ("C", "B", "A"):
        mask = tier_array == tier
        axes[0].scatter(
            projected[mask, 0], projected[mask, 1], s=5, alpha=0.4,
            c=TIER_COLOURS[tier], label=f"tier {tier}", linewidths=0,
        )
    axes[0].set_title("Coloured by the hand-assigned tier", loc="left")
    axes[0].legend(markerscale=3, fontsize=8)
    axes[1].scatter(
        projected[actual == 0, 0], projected[actual == 0, 1], s=5, alpha=0.3,
        c="#c4cbd3", label="not bought", linewidths=0,
    )
    axes[1].scatter(
        projected[actual == 1, 0], projected[actual == 1, 1], s=6, alpha=0.6,
        c="#2a78d6", label="bought", linewidths=0,
    )
    axes[1].set_title("Coloured by the label", loc="left")
    axes[1].legend(markerscale=3, fontsize=8)
    for axis in axes:
        axis.set_xlabel(f"PC1 — {100 * ratios[0]:.1f}% of variance")
        axis.set_ylabel(f"PC2 — {100 * ratios[1]:.1f}%")
        axis.grid(alpha=0.15)
        axis.set_facecolor("white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=105, facecolor="white")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
