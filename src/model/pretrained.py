"""Transfer learning, regime one: a frozen sentence encoder.

``all-MiniLM-L6-v2`` turns each row's text into 384 numbers. Nothing about it moves:
the weights are somebody else's, they stay where they are, and the only thing fitted
per fold is the standardiser and the logistic head on top. That is what makes this the
cheap half of Stage 4 -- one pass over the ten thousand rows, cached to ``.npy``, and
every fold afterwards is a logistic regression.

**Encoding all ten thousand rows in one pass is not leakage.** The encoder never saw
``bought``; it is a fixed function of the text, computed before any split, the same
way ``len(title)`` would be. What must be fitted on training rows only is anything
that reads the label or the fold's own distribution, and that is the standardiser and
the classifier -- both of which are, below.

**Why we expect this to lose.** The signal in this dataset is the parenthesis at the
end of the title, and it is a code, not a sentence: ``(Customer Favorite)`` buys at
67.7% and ``(Shopper Favorite)`` at 2.8%. A semantic encoder is built to map those two
onto nearly the same point, which is the correct behaviour for the task it was trained
on and the wrong behaviour for this one. :func:`phrase_similarity` measures exactly
that, and it is the slide that explains *when transfer learning does not help*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model.baseline import (
    N_BUCKETS,
    design_matrix,
    feature_blocks,
    fit_blocks,
    target_of,
)
from src.model.configs import TRANSFER, RunConfig
from src.model.protocol import ScoreFold
from src.model.records import TrainedFold
from src.model.results import RESULTS_DIR

EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
"""Where the one pass over the corpus is kept, so it is never paid for twice."""

ENCODE_BATCH = 64


def cache_path(
    fields: tuple[str, ...],
    *,
    checkpoint: str = TRANSFER.checkpoint,
    directory: Path | str = EMBEDDINGS_DIR,
) -> Path:
    """Named by checkpoint and by which columns were concatenated into the sentence."""
    model_slug = checkpoint.rsplit("/", 1)[-1]
    field_slug = "-".join(fields) or "empty"
    return Path(directory) / f"{model_slug}.{field_slug}.npy"


def sentences(frame: pd.DataFrame, fields: tuple[str, ...]) -> list[str]:
    """The declared text columns of each row, joined into one string per row."""
    if not fields:
        raise ValueError("a frozen encoding needs at least one text field")
    columns = frame[list(fields)].astype(str)
    return columns.agg(" ".join, axis=1).tolist()


def encode(
    texts: list[str],
    *,
    checkpoint: str = TRANSFER.checkpoint,
    batch_size: int = ENCODE_BATCH,
    show_progress: bool = False,
) -> np.ndarray:
    """Run the sentence encoder. Imported here so the module loads without torch."""
    from sentence_transformers import SentenceTransformer

    from src.model.hardware import device

    model = SentenceTransformer(checkpoint, device=str(device()))
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embeddings_for(
    frame: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    checkpoint: str = TRANSFER.checkpoint,
    directory: Path | str = EMBEDDINGS_DIR,
    show_progress: bool = True,
) -> np.ndarray:
    """The corpus encoded once, read from the cache on every later call.

    The cache is keyed by checkpoint and columns, and its row count is checked against
    the frame's, so a stale file from a different dataset is an error rather than a
    silently misaligned matrix.
    """
    path = cache_path(fields, checkpoint=checkpoint, directory=directory)
    if path.exists():
        stored = np.load(path)
        if len(stored) == len(frame):
            return stored
        raise ValueError(
            f"{path} holds {len(stored)} rows but the frame has {len(frame)}; "
            "delete it and encode again"
        )

    vectors = encode(
        sentences(frame, fields), checkpoint=checkpoint, show_progress=show_progress
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


@dataclass
class FrozenEmbedding:
    """The cached vectors as a fit/transform block, so they sit beside the tabular ones.

    ``fit`` sees only training rows and learns nothing but a mean and a scale. The
    vectors themselves were computed before it was ever called.
    """

    vectors: np.ndarray
    _centre: np.ndarray | None = field(default=None, init=False)
    _scale: np.ndarray | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, train_indices) -> Self:
        training = self.vectors[list(train_indices)]
        self._centre = training.mean(axis=0)
        scale = training.std(axis=0)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        return self

    def transform(self, frame: pd.DataFrame, indices) -> np.ndarray:
        if self._centre is None or self._scale is None:
            raise RuntimeError("the frozen embedding block was never fitted")
        return (self.vectors[list(indices)] - self._centre) / self._scale


def frozen_scorer(
    config: RunConfig,
    frame: pd.DataFrame,
    vectors: np.ndarray,
    *,
    n_buckets: int = N_BUCKETS,
    c: float = 1.0,
    folds: list | None = None,
) -> ScoreFold:
    """384 frozen dimensions, plus whatever tabular columns the section declares.

    The head is the same ``LogisticRegression`` the bar uses, on purpose: the only
    difference between this row of the table and the bar's row is where the features
    came from, so the comparison is about the features and not about the classifier.
    """
    target = target_of(frame)

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        blocks: list = [FrozenEmbedding(vectors)]
        if config.categorical_fields or config.numeric_fields:
            blocks += feature_blocks(
                categorical_fields=config.categorical_fields,
                numeric_fields=config.numeric_fields,
                n_buckets=n_buckets,
            )
        fit_blocks(blocks, frame, train_indices)

        train_matrix = design_matrix(blocks, frame, train_indices)
        scored_matrix = design_matrix(blocks, frame, scored_indices)
        model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        model.fit(train_matrix, target[list(train_indices)])
        if folds is not None:
            folds.append(TrainedFold(parameters=train_matrix.shape[1] + 1))
        return model.predict_proba(scored_matrix)[:, 1]

    return score_fold


# ---------------------------------------------------------------------------
# The slide: what the encoder thinks the popularity phrases mean.
# ---------------------------------------------------------------------------

CONTRAST = ("Customer Favorite", "Shopper Favorite")
"""The pair the write-up singles out: 67.7% against 2.8%, one word apart.

Those are the two phrases' own rates. ``docs/informe-ejercicio-1.md`` quotes 64.7% and
2.6% for the *tiers* they belong to, which is a different number for a different thing.
"""


def cosine(vectors: np.ndarray) -> np.ndarray:
    """All pairwise cosine similarities of a small matrix of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms > 0.0, norms, 1.0)
    return unit @ unit.T


def phrase_similarity(
    frame: pd.DataFrame,
    *,
    column: str = "popularity_phrase",
    checkpoint: str = TRANSFER.checkpoint,
) -> pd.DataFrame:
    """Every pair of popularity phrases: how close the encoder puts them, and how far
    apart their buy rates actually are.

    If the semantics carried the signal, a high cosine would go with a small gap in
    BTR. The scatter this returns is the evidence for or against that, and the row
    for :data:`CONTRAST` is the single number worth saying out loud.
    """
    rates = frame.groupby(column)["bought"].agg(rows="size", rate="mean")
    phrases = list(rates.index)
    similarity = cosine(encode(phrases, checkpoint=checkpoint))

    pairs = []
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            pairs.append(
                {
                    "left": phrases[i],
                    "right": phrases[j],
                    "cosine": float(similarity[i, j]),
                    "left_rate": float(rates.iloc[i]["rate"]),
                    "right_rate": float(rates.iloc[j]["rate"]),
                    "rate_gap": abs(
                        float(rates.iloc[i]["rate"]) - float(rates.iloc[j]["rate"])
                    ),
                }
            )
    return pd.DataFrame(pairs).sort_values("cosine", ascending=False).reset_index(drop=True)


def contrast_row(pairs: pd.DataFrame, contrast: tuple[str, str] = CONTRAST) -> pd.Series | None:
    """The :data:`CONTRAST` pair out of the table, whichever way round it was stored."""
    left, right = contrast
    match = pairs[
        ((pairs["left"] == left) & (pairs["right"] == right))
        | ((pairs["left"] == right) & (pairs["right"] == left))
    ]
    return None if match.empty else match.iloc[0]
