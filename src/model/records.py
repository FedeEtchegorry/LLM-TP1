"""What one fold produced, in a shape every model can fill.

The logistic bar, the Transformer, the frozen encoder and the fine-tuned checkpoint
are four very different objects, but the results table asks all four the same three
questions: how many parameters were trained, how the loss moved epoch by epoch, and
which epoch was kept. A model that has no epochs answers with an empty curve rather
than with a gap in the table.

This lives apart from ``training`` so that ``baseline`` can fill one too without the
two modules importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EpochRecord:
    """One row of the over- and underfitting curve."""

    epoch: int
    train_loss: float
    train_ap: float
    validation_loss: float
    validation_ap: float


@dataclass(frozen=True)
class TrainedFold:
    """A fitted model, ready to be scored, recorded and -- if it is ours -- saved.

    ``model`` and ``encoder`` are typed loosely because the four regimes carry
    different objects there; only ``scripts.run_final`` reaches into them, and it
    does so for the Transformer alone.
    """

    parameters: int
    model: object | None = None
    encoder: object | None = None
    curve: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    states: list[dict] = field(default_factory=list)
