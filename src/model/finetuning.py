"""Transfer learning, regime two: fine-tuning MiniLM on the serialised row.

A sentence model reads sentences, so the row has to become one. The serialisation is
a design decision and it goes on a slide::

    <title> <description> <ingredients> |
    category: Frozen | storage type: Frozen | allergens: Wheat | unit of measure: oz |
    price: 8.30 | price position: 0.45 | net weight oz: 10.14 | nutrition score: 36

Three choices inside it, each defensible and each arguable:

- **The tabular columns are written as ``name: value`` pairs**, not as bare numbers,
  so the 22M-parameter model can tell which number is which without a positional
  convention it was never trained on.
- **Numbers stay in their own units**, rounded to two decimals. Standardising them
  would produce ``-0.37``, a string the tokenizer has no useful prior over; ``8.30``
  at least looks like a price to a model trained on the web.
- **A missing value is the word ``unknown``**, not an empty field, so the pair
  structure survives and the absence is something the model can attend to.

**The budget, stated rather than hidden.** 22M parameters at ``max_length=96`` is a
training run per epoch, so this regime reports **fold 0 only** -- see
``Transfer.finetune_folds`` in ``configs.py``. Every other row of the results table is
five folds with a standard deviation; this one is a single fold and the table says so.

The epoch is chosen the same way our own model chooses it -- lowest loss on training
queries held out for the purpose, never on the rows about to be scored. Three epochs
rarely makes that choice interesting, but the rule is the same rule, which is the
point.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.model.baseline import target_of
from src.model.configs import TRANSFER, RunConfig
from src.model.encoding import categorical_column, numeric_column
from src.model.protocol import ScoreFold
from src.model.records import EpochRecord, TrainedFold

MISSING = "unknown"
"""What an absent value is written as, so the ``name: value`` pair survives it."""

SEPARATOR = " | "


def label_of(column: str) -> str:
    """``price_position`` becomes ``price position``: readable to a language model."""
    return column.replace("_", " ")


def serialise(frame: pd.DataFrame, indices, config: RunConfig) -> list[str]:
    """One string per row, from exactly the columns the section declares.

    The same sentinel handling as ``encoding.py``: a ``nutrition_score`` of zero is a
    "not applicable", so it is written ``unknown`` rather than as a very bad score.
    """
    rows = frame.iloc[list(indices)]

    text = [
        " ".join(str(value) for value in row)
        for row in rows[list(config.text_fields)].to_numpy()
    ] if config.text_fields else [""] * len(rows)

    pairs: list[list[str]] = [[] for _ in range(len(rows))]
    for name in config.categorical_fields:
        for position, value in enumerate(categorical_column(rows, name)):
            pairs[position].append(f"{label_of(name)}: {value}")
    for name in config.numeric_fields:
        values = numeric_column(rows, name)
        for position, value in enumerate(values):
            written = MISSING if np.isnan(value) else f"{value:.2f}"
            pairs[position].append(f"{label_of(name)}: {written}")

    return [
        SEPARATOR.join([sentence, *fields]) if fields else sentence
        for sentence, fields in zip(text, pairs)
    ]


def _tokenize(tokenizer, texts: list[str], device):
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=TRANSFER.max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def train_fold(
    config: RunConfig,
    frame: pd.DataFrame,
    train_indices,
    *,
    seed: int = TRANSFER.seed,
    device=None,
) -> TrainedFold:
    """Fine-tune the checkpoint on one fold's training rows."""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.model.hardware import deterministic
    from src.model.hardware import device as best_device
    from src.model.network import count_parameters
    from src.model.training import early_stopping_split

    deterministic()
    torch.manual_seed(seed)
    device = device or best_device()

    target = target_of(frame)
    fit_indices, stop_indices = early_stopping_split(
        target, frame["query_id"].tolist(), train_indices
    )

    tokenizer = AutoTokenizer.from_pretrained(TRANSFER.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFER.checkpoint, num_labels=1
    ).to(device)

    fit_batch = _tokenize(tokenizer, serialise(frame, fit_indices, config), device)
    stop_batch = _tokenize(tokenizer, serialise(frame, stop_indices, config), device)
    fit_target = torch.tensor(target[fit_indices], dtype=torch.float32, device=device)
    stop_target = torch.tensor(target[stop_indices], dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=TRANSFER.learning_rate,
        weight_decay=TRANSFER.weight_decay,
    )
    loss_of = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    curve: list[EpochRecord] = []
    best_loss, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, TRANSFER.epochs + 1):
        model.train()
        order = torch.randperm(len(fit_indices), generator=generator)
        for start in range(0, len(order), TRANSFER.batch_size):
            rows = order[start : start + TRANSFER.batch_size].to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(
                **{key: value[rows] for key, value in fit_batch.items()}
            ).logits.squeeze(-1)
            loss_of(logits, fit_target[rows]).backward()
            optimiser.step()

        train_loss, train_ap = _measure(model, fit_batch, fit_target, loss_of)
        stop_loss, stop_ap = _measure(model, stop_batch, stop_target, loss_of)
        curve.append(EpochRecord(epoch, train_loss, train_ap, stop_loss, stop_ap))

        if stop_loss < best_loss:
            best_loss, best_epoch = stop_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TrainedFold(
        parameters=count_parameters(model),
        model=model,
        encoder=tokenizer,
        curve=curve,
        best_epoch=best_epoch,
    )


def predict(model, tokenizer, frame: pd.DataFrame, indices, config: RunConfig, device) -> np.ndarray:
    """Probabilities for the given rows, in batches so a fold fits in memory."""
    import torch

    model.eval()
    texts = serialise(frame, indices, config)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), TRANSFER.batch_size):
            batch = _tokenize(
                tokenizer, texts[start : start + TRANSFER.batch_size], device
            )
            logits = model(**batch).logits.squeeze(-1)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0)


def _measure(model, batch: dict, target, loss_of) -> tuple[float, float]:
    """Loss and average precision on one split, without touching the gradients."""
    import torch
    from sklearn.metrics import average_precision_score

    model.eval()
    with torch.no_grad():
        logits = torch.cat(
            [
                model(
                    **{
                        key: value[start : start + TRANSFER.batch_size]
                        for key, value in batch.items()
                    }
                ).logits.squeeze(-1)
                for start in range(0, len(target), TRANSFER.batch_size)
            ]
        )
        loss = float(loss_of(logits, target))
        labels = target.cpu().numpy()
        if labels.min() == labels.max():
            return loss, float("nan")
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
    return loss, float(average_precision_score(labels, probabilities))


def finetune_scorer(
    config: RunConfig, frame: pd.DataFrame, *, folds: list | None = None
) -> ScoreFold:
    """The fine-tuned checkpoint as a ``ScoreFold``, on the same folds as everything else."""

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        from src.model.hardware import device as best_device

        target_device = best_device()
        index = len(folds) if folds is not None else 0
        trained = train_fold(
            config, frame, train_indices, seed=TRANSFER.seed + index, device=target_device
        )
        if folds is not None:
            folds.append(trained)
        return predict(
            trained.model, trained.encoder, frame, scored_indices, config, target_device
        )

    return score_fold
