"""Training one fold, and the ``ScoreFold`` that puts the Transformer in the table.

The one decision worth stating: **early stopping never looks at the fold's own
validation rows.** Stopping on the rows we are about to score would fold them into
model selection and quietly inflate the reported number. Instead whole queries are
held out of the *training* rows for the patience signal, so the fold's validation set
stays untouched until it is scored once.

The vocabulary, levels, medians and bucket edges are still fitted on the whole training
fold, early-stopping rows included: those rows are training data, and the contract that
matters is that nothing is fitted on the rows being scored.

**The epoch is chosen by loss, not by average precision.** On a stopping set of about a
thousand rows carrying some 140 positives, AP swings by ±0.07 between neighbouring
epochs with no trend, so taking its argmax over thirty epochs selects a lucky draw
rather than a good model. The loss on the same rows is far smoother, bottoms cleanly,
and is the quantity being optimised. AP is still recorded every epoch -- it belongs in
the curve, just not in the decision.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.model.baseline import target_of
from src.model.configs import TRAINING, RunConfig
from src.model.encoding import EncodedRows, EncodingSpec, RowEncoder
from src.model.network import BtrTransformer, count_parameters
from src.model.protocol import ScoreFold
from src.model.records import EpochRecord, TrainedFold

EARLY_STOPPING_SPLITS = 6
"""One sixth of the training queries carry the patience signal; the rest train."""

EARLY_STOPPING_SPLIT_SEED = 1337
"""Fixed on purpose, independent of ``config.seed``: which queries carry the patience
signal is part of the fold, not part of the initialisation. A repeat of a config under
a different ``seed`` (axis S) should vary the weights and the batch order and nothing
else -- if this moved with ``config.seed`` too, a seed repeat would also be training on
a different fit/stop split, and the two effects would be impossible to tell apart."""

INFERENCE_BATCH = 256

MEMBER_STRIDE = 1000


def spec_for(config: RunConfig) -> EncodingSpec:
    return EncodingSpec(
        text_fields=config.text_fields,
        categorical_fields=config.categorical_fields,
        numeric_fields=config.numeric_fields,
        n_buckets=TRAINING.n_buckets,
        max_text_tokens=TRAINING.max_text_tokens,
    )


def early_stopping_split(
    target: np.ndarray, query_ids: list, train_indices
) -> tuple[list[int], list[int]]:
    """Split the training rows by whole query into rows to fit on and rows to stop on."""
    train_indices = list(train_indices)
    labels = target[train_indices]
    groups = [query_ids[index] for index in train_indices]
    splitter = StratifiedGroupKFold(
        n_splits=EARLY_STOPPING_SPLITS, shuffle=True, random_state=EARLY_STOPPING_SPLIT_SEED
    )
    features = np.zeros((len(train_indices), 1), dtype=np.uint8)
    fit_rows, stop_rows = next(splitter.split(features, labels, groups=groups))
    return (
        [train_indices[row] for row in fit_rows],
        [train_indices[row] for row in stop_rows],
    )


def train_fold(
    config: RunConfig,
    frame: pd.DataFrame,
    train_indices,
    *,
    seed: int,
    device=None,
) -> TrainedFold:
    """Fit one model, stopping on held-out training queries rather than on the fold."""
    from src.model.hardware import deterministic
    from src.model.hardware import device as best_device

    deterministic()

    torch.manual_seed(seed)
    device = device or best_device()
    target = target_of(frame)
    fit_indices, stop_indices = early_stopping_split(
        target, frame["query_id"].tolist(), train_indices
    )

    encoder = RowEncoder(spec_for(config)).fit(frame, train_indices)
    fit_rows = encoder.transform(frame, fit_indices).to(device)
    stop_rows = encoder.transform(frame, stop_indices).to(device)
    fit_target = torch.tensor(target[fit_indices], dtype=torch.float32, device=device)
    stop_target = torch.tensor(target[stop_indices], dtype=torch.float32, device=device)

    model = BtrTransformer(encoder, config, TRAINING.n_buckets).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_of = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    curve: list[EpochRecord] = []
    best_loss, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    best_states: list[tuple[float, int, dict]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(fit_rows), generator=generator).to(device)
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_of(model(fit_rows.select(rows)), fit_target[rows])
            loss.backward()
            optimiser.step()

        train_loss, train_ap = _measure(model, fit_rows, fit_target, loss_of)
        stop_loss, stop_ap = _measure(model, stop_rows, stop_target, loss_of)
        curve.append(
            EpochRecord(epoch, train_loss, train_ap, stop_loss, stop_ap)
        )

        if config.checkpoints > 1:
            best_states.append((stop_loss, epoch, _on_cpu(model.state_dict())))
            best_states.sort(key=lambda kept: kept[0])
            del best_states[config.checkpoints :]

        if stop_loss < best_loss:
            best_loss, best_epoch = stop_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainedFold(
        model=model,
        encoder=encoder,
        curve=curve,
        best_epoch=best_epoch,
        parameters=count_parameters(model),
        states=[state for _, _, state in best_states],
    )


def _on_cpu(state: dict) -> dict:
    return {name: value.detach().to("cpu", copy=True) for name, value in state.items()}


@torch.no_grad()
def _member_scores(trained: TrainedFold, frame, scored_indices) -> list[np.ndarray]:
    """One vector per kept epoch, or just the best one when only that was kept."""
    rows = trained.encoder.transform(frame, scored_indices)
    if not trained.states:
        return [predict(trained.model, rows)]
    device = next(trained.model.parameters()).device
    scores = []
    for state in trained.states:
        trained.model.load_state_dict({k: v.to(device) for k, v in state.items()})
        scores.append(predict(trained.model, rows))
    return scores


@torch.no_grad()
def predict(model: BtrTransformer, rows: EncodedRows) -> np.ndarray:
    """Return probabilities in bounded inference batches."""
    model.eval()
    device = next(model.parameters()).device
    rows = rows.to(device)
    scores = [
        torch.sigmoid(
            model(
                rows.select(
                    torch.arange(
                        start,
                        min(start + INFERENCE_BATCH, len(rows)),
                        device=device,
                    )
                )
            )
        )
        for start in range(0, len(rows), INFERENCE_BATCH)
    ]
    return torch.cat(scores).cpu().numpy()


@torch.no_grad()
def _measure(
    model: BtrTransformer,
    rows: EncodedRows,
    target: torch.Tensor,
    loss_of: nn.Module,
) -> tuple[float, float]:
    """Loss and average precision on one split, without touching the gradients."""
    model.eval()
    device = next(model.parameters()).device
    rows = rows.to(device)
    logits = torch.cat(
        [
            model(
                rows.select(
                    torch.arange(
                        start,
                        min(start + INFERENCE_BATCH, len(rows)),
                        device=device,
                    )
                )
            )
            for start in range(0, len(rows), INFERENCE_BATCH)
        ]
    )
    loss = float(loss_of(logits, target))
    labels = target.cpu().numpy()
    if labels.min() == labels.max():
        return loss, float("nan")
    return loss, float(
        average_precision_score(labels, torch.sigmoid(logits).cpu().numpy())
    )


def transformer_scorer(
    config: RunConfig,
    frame: pd.DataFrame,
    *,
    folds: list | None = None,
    members: list | None = None,
) -> ScoreFold:
    """Build a fold scorer with optional seed and checkpoint averaging."""

    def score_fold(train_indices, scored_indices) -> np.ndarray:
        index = len(folds) if folds is not None else 0
        base = config.seed + index
        trained = [
            train_fold(config, frame, train_indices, seed=base + member * MEMBER_STRIDE)
            for member in range(config.seeds)
        ]
        if folds is not None:
            folds.append(trained[0])
        scored = np.stack(
            [
                output
                for m in trained
                for output in _member_scores(m, frame, scored_indices)
            ]
        )
        if members is not None:
            members.append({"scored": np.asarray(scored_indices), "members": scored})
        return scored.mean(axis=0)

    return score_fold
