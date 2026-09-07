"""Train, validation and test in one table, and the three gaps between them.

The results a run produces are usually quoted as two numbers -- cross-validation and
holdout -- and the distance between them is called "overfitting". That is one gap where
this protocol has three, because a fold's rows are split further than the word "train"
suggests.

**The four row sets.** ``training.train_fold`` takes a fold's training indices and holds
a sixth of its *queries* out as the patience signal, so the rows a run touches fall into
four groups rather than two:

===============  =================================================================
``fit``          The gradients saw these. ``EpochRecord.train_loss`` and
                 ``train_ap`` are measured here.
``stop``         Held out of training for early stopping. The encoder was still
                 fitted on them, but no gradient step ever used them.
                 ``EpochRecord.validation_loss`` and ``validation_ap`` are these
                 rows -- **not** the fold's validation set.
``validation``   The fold's own rows, scored once, out of fold.
                 ``FoldScore.average_precision`` is this.
``test``         The 20% holdout, opened once by ``experiment.run_test``.
===============  =================================================================

The naming inside ``EpochRecord`` is the trap this module exists to defuse: a curve
labelled "validation" is the stopping split, and reading it as the fold's validation
score understates the gap by exactly the part that matters -- the step to unseen
queries.

**The three gaps**, each answering a different question:

``fit -> stop``          Memorisation. Same queries, same encoder; the only
                         difference is whether a gradient step consumed the row.
``stop -> validation``   Generalisation. The first step to queries the model never
                         saw, and the one an early-stopping curve cannot show.
``validation -> test``   Whether the number the model was selected on survives being
                         spent. Only present once the holdout has been opened.

A model can have a wide first gap and a flat second one -- memorising the training
rows while still ranking unseen queries as well as it ever did -- and that reads very
differently from the reverse. Collapsing them into one number hides which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev

import pandas as pd

FIT = "fit"
STOP = "corte"
VALIDATION = "validation"
TEST = "test"

LEVEL_ORDER = (FIT, STOP, VALIDATION, TEST)

WHAT_THE_MODEL_SAW = {
    FIT: "gradientes",
    STOP: "solo el encoder",
    VALIDATION: "nada",
    TEST: "nada",
}
"""What each row set contributed to the fitted model, for the table's last column."""


@dataclass(frozen=True)
class Level:
    """One row set's average precision, averaged over the folds that produced it."""

    name: str
    average_precision: float
    deviation: float
    folds: int
    seen: str

    @property
    def label(self) -> str:
        """Una sola medicion no tiene dispersion medida, y un ``± 0.0000`` dice lo
        contrario: se reporta como una medicion unica."""
        if self.folds < 2:
            return f"{self.average_precision:.4f} (una medicion)"
        return f"{self.average_precision:.4f} ± {self.deviation:.4f}"


@dataclass(frozen=True)
class Gap:
    """The drop between two consecutive levels."""

    origin: str
    destination: str
    delta: float
    question: str

    @property
    def label(self) -> str:
        return f"{self.delta:+.4f}"


GAP_QUESTIONS = {
    (FIT, STOP): "memorización de las filas que vieron gradiente",
    (STOP, VALIDATION): "generalización a queries no vistas",
    (VALIDATION, TEST): "si el número de selección sobrevive al holdout",
}


def fold_levels(document: dict) -> pd.DataFrame:
    """One row per fold: the epoch that was kept and the scores at that epoch.

    ``fit_ap`` and ``stop_ap`` are read out of the curve at ``best_epoch`` rather than
    at the last epoch, because the kept model is the one from ``best_epoch`` -- the
    epochs after it were trained through and then discarded, so their scores describe
    a model that was never used.
    """
    curves = {curve["fold_index"]: curve for curve in document.get("curves", [])}
    rows = []
    for fold in document["folds"]:
        curve = curves.get(fold["fold_index"])
        record = _record_at_best(curve)
        rows.append(
            {
                "fold_index": fold["fold_index"],
                "best_epoch": None if curve is None else curve["best_epoch"],
                "epochs_run": 0 if curve is None else len(curve["epochs"]),
                "fit_ap": None if record is None else record["train_ap"],
                "fit_loss": None if record is None else record["train_loss"],
                "stop_ap": None if record is None else record["validation_ap"],
                "stop_loss": None if record is None else record["validation_loss"],
                "validation_ap": fold["average_precision"],
                "validation_roc": fold["roc_auc"],
                "n_train": fold["n_train"],
                "n_scored": fold["n_scored"],
            }
        )
    return pd.DataFrame(rows)


def _record_at_best(curve: dict | None) -> dict | None:
    """The epoch record the kept weights came from, or ``None`` for a curveless run."""
    if not curve or not curve.get("epochs"):
        return None
    best = curve["best_epoch"]
    return next(
        (epoch for epoch in curve["epochs"] if epoch["epoch"] == best),
        curve["epochs"][-1],
    )


def levels(document: dict, test_document: dict | None = None) -> list[Level]:
    """The three cross-validated levels, plus ``test`` once the holdout is opened.

    A run whose folds carry no curve -- the logistic bar has no epochs -- yields only
    the levels it can support, rather than padding the table with zeros.
    """
    frame = fold_levels(document)
    found = [
        _level(FIT, frame["fit_ap"]),
        _level(STOP, frame["stop_ap"]),
        _level(VALIDATION, frame["validation_ap"]),
    ]
    if test_document is not None:
        found.append(
            _level(
                TEST,
                pd.Series(
                    [fold["average_precision"] for fold in test_document["folds"]]
                ),
            )
        )
    return [level for level in found if level is not None]


def _level(name: str, values: pd.Series) -> Level | None:
    present = [float(value) for value in values.dropna()]
    if not present:
        return None
    return Level(
        name=name,
        average_precision=fmean(present),
        deviation=stdev(present) if len(present) > 1 else 0.0,
        folds=len(present),
        seen=WHAT_THE_MODEL_SAW[name],
    )


def gaps(found: list[Level]) -> list[Gap]:
    """The drop between each consecutive pair of levels that is actually present."""
    by_name = {level.name: level for level in found}
    steps = []
    for origin, destination in zip(LEVEL_ORDER, LEVEL_ORDER[1:]):
        if origin in by_name and destination in by_name:
            steps.append(
                Gap(
                    origin=origin,
                    destination=destination,
                    delta=by_name[destination].average_precision
                    - by_name[origin].average_precision,
                    question=GAP_QUESTIONS[(origin, destination)],
                )
            )
    return steps


def markdown_table(found: list[Level], steps: list[Gap]) -> str:
    """The two tables the write-up quotes: the levels, then the gaps between them."""
    lines = [
        "| Conjunto | PR-AUC | Qué vio el modelo | Folds |",
        "|---|---:|---|---:|",
    ]
    for level in found:
        lines.append(
            f"| {level.name} | {level.label} | {level.seen} | {level.folds} |"
        )
    lines.append("")
    lines.append("| Brecha | Δ PR-AUC | Qué mide |")
    lines.append("|---|---:|---|")
    for step in steps:
        lines.append(
            f"| {step.origin} → {step.destination} | {step.label} | {step.question} |"
        )
    return "\n".join(lines)


def epoch_gap_frame(document: dict) -> pd.DataFrame:
    """One row per epoch per fold, with the fit-to-stop gap already differenced.

    This is the curve that shows *when* the memorisation gap opens, which the level
    table -- a single snapshot at ``best_epoch`` -- cannot.
    """
    rows = []
    for curve in document.get("curves", []):
        for epoch in curve["epochs"]:
            rows.append(
                {
                    "fold_index": curve["fold_index"],
                    "best_epoch": curve["best_epoch"],
                    "epoch": epoch["epoch"],
                    "fit_ap": epoch["train_ap"],
                    "stop_ap": epoch["validation_ap"],
                    "fit_loss": epoch["train_loss"],
                    "stop_loss": epoch["validation_loss"],
                    "ap_gap": epoch["train_ap"] - epoch["validation_ap"],
                    "loss_gap": epoch["validation_loss"] - epoch["train_loss"],
                }
            )
    return pd.DataFrame(rows)


def stopped_early(document: dict) -> pd.DataFrame:
    """Whether each fold stopped on patience or ran out of epochs.

    A fold that used every epoch never showed the patience signal, so its curve is a
    lower bound on the gap rather than a picture of it -- worth stating next to the
    table instead of leaving the reader to check epoch counts.
    """
    budget = document["config"]["epochs"]
    frame = fold_levels(document)
    return frame.assign(
        epoch_budget=budget,
        stopped_early=frame["epochs_run"] < budget,
    )[["fold_index", "best_epoch", "epochs_run", "epoch_budget", "stopped_early"]]
