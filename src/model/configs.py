"""What varies between runs lives in ``parameters.txt``; what does not lives here.

``parameters.txt`` carries only the knobs an experiment actually turns -- the fields
that enter, the shape of the encoder, and how numbers are embedded. Everything else is
a constant below.

Those constants still reach every run's digest, so editing one here invalidates the
cached results it would have changed. A constant that stops being constant moves up
into :class:`RunConfig` and into the file; nothing else has to change.
"""

from __future__ import annotations

import hashlib
import re
from configparser import ConfigParser
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PARAMETERS_PATH = Path("parameters.txt")

LOGISTIC = "logistic"
TRANSFORMER = "transformer"

NUMERIC_EMBEDDINGS = ("none", "affine", "buckets", "affine+buckets")
POSITIONAL_ENCODINGS = ("none", "learned", "sinusoidal")
POOLINGS = ("cls", "mean", "attention")

LADDER_NAME = re.compile(r"^L\d")
"""``[L0 ...]`` through ``[L4 ...]``: the rungs ``run_ladder`` walks, in file order."""

AXIS_NAME = re.compile(r"^[A-H] ")
"""``[B 1 layer]``, ``[C 8 heads]``, ...: the alternatives ``run_modules`` sweeps."""


class ParameterError(ValueError):
    """A malformed ``parameters.txt``, reported with the section that caused it."""


@dataclass(frozen=True)
class Training:
    """The same for every run, so no run can win by training longer than another."""

    n_buckets: int = 10
    max_text_tokens: int = 64
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    patience: int = 5
    seed: int = 1337
    regularisation: float = 1.0


@dataclass(frozen=True)
class Protocol:
    """The split every run reports into, fixed so the table stays comparable.

    ``save_weights`` is the one field that does not reach a digest: keeping the
    trained parameters is an operational choice, not a modelling one.
    """

    dataset: str = "data/supermarket_products.csv"
    folds: int = 5
    test_fraction: float = 0.2
    random_state: int = 42
    save_weights: bool = True


TRAINING = Training()
PROTOCOL = Protocol()


@dataclass(frozen=True)
class RunConfig:
    """One row of the results table: everything an experiment is allowed to change."""

    name: str
    model: str

    text_fields: tuple[str, ...]
    categorical_fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]

    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    positional: str
    pooling: str
    numeric_embedding: str

    @property
    def digest(self) -> str:
        """Stable across processes, and sensitive to the constants above."""
        payload = repr(
            (
                sorted(asdict(self).items()),
                sorted(asdict(TRAINING).items()),
                (
                    PROTOCOL.dataset,
                    PROTOCOL.folds,
                    PROTOCOL.test_fraction,
                    PROTOCOL.random_state,
                ),
            )
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def load_parameters(path: Path | str = PARAMETERS_PATH) -> dict[str, RunConfig]:
    """Read the declared runs, in file order."""
    path = Path(path)
    if not path.exists():
        raise ParameterError(f"{path} does not exist")

    parser = ConfigParser()
    parser.optionxform = str
    with path.open(encoding="utf-8") as handle:
        parser.read_file(handle)

    runs = {name: _run(name, parser[name]) for name in parser.sections()}
    if not runs:
        raise ParameterError(f"{path} declares no runs")
    return runs


def ladder_runs(runs: dict[str, RunConfig]) -> dict[str, RunConfig]:
    return {name: run for name, run in runs.items() if LADDER_NAME.match(name)}


def axis_runs(runs: dict[str, RunConfig]) -> dict[str, RunConfig]:
    return {name: run for name, run in runs.items() if AXIS_NAME.match(name)}


def _run(name: str, section) -> RunConfig:
    known = {field.name for field in fields(RunConfig)} - {"name"}
    unknown = set(section.keys()) - known
    if unknown:
        raise ParameterError(
            f"[{name}] sets unknown keys: {sorted(unknown)}. "
            "Fixed values live in src/model/configs.py, not here."
        )

    try:
        config = RunConfig(
            name=name,
            model=section.get("model"),
            text_fields=_tuple(section.get("text_fields")),
            categorical_fields=_tuple(section.get("categorical_fields")),
            numeric_fields=_tuple(section.get("numeric_fields")),
            d_model=section.getint("d_model"),
            n_layers=section.getint("n_layers"),
            n_heads=section.getint("n_heads"),
            dropout=section.getfloat("dropout"),
            positional=section.get("positional"),
            pooling=section.get("pooling"),
            numeric_embedding=section.get("numeric_embedding"),
        )
    except (TypeError, ValueError) as error:
        raise ParameterError(f"[{name}] is malformed: {error}") from error

    _validate(config)
    return config


def _validate(config: RunConfig) -> None:
    """Catch a typo in the file rather than three hours into a sweep."""
    name = config.name
    if config.model not in (LOGISTIC, TRANSFORMER):
        raise ParameterError(f"[{name}] model must be {LOGISTIC} or {TRANSFORMER}")
    if not (config.text_fields or config.categorical_fields or config.numeric_fields):
        raise ParameterError(f"[{name}] declares no input fields")
    if config.model == LOGISTIC:
        return
    for value, allowed, label in (
        (config.numeric_embedding, NUMERIC_EMBEDDINGS, "numeric_embedding"),
        (config.positional, POSITIONAL_ENCODINGS, "positional"),
        (config.pooling, POOLINGS, "pooling"),
    ):
        if value not in allowed:
            raise ParameterError(f"[{name}] {label}={value!r} is not one of {allowed}")
    if config.d_model % config.n_heads:
        raise ParameterError(
            f"[{name}] d_model={config.d_model} is not divisible by "
            f"n_heads={config.n_heads}"
        )
    if config.n_layers == 0 and config.pooling == "cls":
        raise ParameterError(
            f"[{name}] with n_layers=0 the [CLS] position is never updated, "
            "so pooling must be mean or attention"
        )


def _tuple(value: str | None) -> tuple[str, ...]:
    """``a, b, c`` becomes a tuple; an empty value becomes an empty one."""
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())
