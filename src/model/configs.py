"""What varies between runs lives in the parameters file; what does not lives here.

The file carries only the knobs an experiment actually turns -- the fields
that enter and the shape of the two towers. Everything else is a constant below.

Those constants still reach every run's digest, so editing one here invalidates the
cached results it would have changed. A constant that stops being constant moves up
into :class:`RunConfig` and into the file; nothing else has to change.
"""

from __future__ import annotations

import hashlib
import re
from configparser import ConfigParser
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import ClassVar

PARAMETERS_PATH = Path("parameters.txt")
"""The two-tower architecture. The only file that is still written to."""

V1_PARAMETERS = Path("parameters-v1.txt")
V1_MODULES = Path("parameters-v1-modules.txt")
"""Frozen: what the first submission declared, under the two names it used."""

LEGACY_KEYS = frozenset({"numeric_embedding"})
"""Keys the frozen files still declare and no run reads. Ignored rather than
rejected, so those files stay the record of what the first submission swept."""

LOGISTIC = "logistic"
TRANSFORMER = "transformer"
FROZEN = "frozen"
FINETUNE = "finetune"
MODELS = (LOGISTIC, TRANSFORMER, FROZEN, FINETUNE)

PRETRAINED = (FROZEN, FINETUNE)

POSITIONAL_ENCODINGS = ("none", "learned", "sinusoidal")
POOLINGS = ("cls", "mean", "attention")

TOKENIZERS = ("whole-word", "wordpiece")
TAB_TOWERS = ("linear", "mlp")
FUSION_HEADS = ("linear", "mlp")

LADDER_NAME = re.compile(r"^L\d")
"""``[L0 ...]`` through ``[L4 ...]``: the rungs ``run_ladder`` walks, in file order."""

AXIS_NAME = re.compile(r"^(?![TQ] )[A-Z] ")
"""``[B 1 layer]``, ``[C 8 heads]``, ``[K lr 3e-4]``, ...: the alternatives
``run_modules`` sweeps. Any capital letter and a space -- except ``T`` (the withdrawn
transfer runs) and ``Q`` (diagnostic controls), so neither counts as an axis point."""


class ParameterError(ValueError):
    """A malformed parameters file, reported with the section that caused it."""


@dataclass(frozen=True)
class Training:
    """What no experiment may move, so every run is comparable on the parts that count."""

    n_buckets: int = 10
    max_text_tokens: int = 64
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
    pooling: str

    positional: str = "learned"
    embedding_norm: bool = True
    pooler_projection: bool = True
    tokenizer: str = "wordpiece"
    keep_brackets: bool = True
    tab_tower: str = "mlp"
    fusion_head: str = "mlp"
    tabular_dim: int = 16

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 60
    patience: int = 10
    batch_size: int = 64

    seed: int = 1337
    checkpoints: int = 1
    seeds: int = 1

    _DIGEST_TRAINING_FIELDS: ClassVar[tuple[str, ...]] = ("seed",)
    _LEGACY_TRAINING_FIELDS: ClassVar[tuple[str, ...]] = (
        "learning_rate",
        "weight_decay",
        "epochs",
        "patience",
        "batch_size",
        "seed",
    )

    @property
    def digest(self) -> str:
        """Stable across processes, and sensitive to the constants above.

        Two sections with different names but byte-identical parameters are the same
        experiment and must land on the same cache file: ``name`` is stored inside the
        JSON record for display, but never enters the hash. An override probe and a
        declared section that resolve to the same configuration therefore share one
        record instead of training it twice.
        """
        config_fields = asdict(self)
        del config_fields["name"]
        # Preserve cache keys for the default single-model run.
        for name in ("seeds", "checkpoints"):
            if config_fields[name] == 1:
                del config_fields[name]
        training_fields = asdict(TRAINING)
        for name in self._DIGEST_TRAINING_FIELDS:
            training_fields[name] = config_fields.pop(name)
        return self._digest_from(config_fields, training_fields)

    @property
    def compatible_digests(self) -> tuple[str, ...]:
        """Return current and equivalent historical cache keys."""
        current = self.digest
        if self.seeds != 1 or self.checkpoints != 1:
            return (current,)

        config_fields = asdict(self)
        del config_fields["name"]
        del config_fields["seeds"]
        del config_fields["checkpoints"]
        training_fields = asdict(TRAINING)
        for name in self._LEGACY_TRAINING_FIELDS:
            training_fields[name] = config_fields.pop(name)
        legacy = self._digest_from(config_fields, training_fields)
        return tuple(dict.fromkeys((current, legacy)))

    def _digest_from(self, config_fields: dict, training_fields: dict) -> str:
        payload = [
            sorted(config_fields.items()),
            sorted(training_fields.items()),
            (
                PROTOCOL.dataset,
                PROTOCOL.folds,
                PROTOCOL.test_fraction,
                PROTOCOL.random_state,
            ),
        ]
        return hashlib.sha256(repr(tuple(payload)).encode()).hexdigest()[:12]


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


def apply_overrides(config: RunConfig, assignments: list[str]) -> RunConfig:
    """Build a variant from ``key=value`` strings, so probing costs no declared section.

    The overrides land in the run's name, which is what the results record shows. They
    reach the digest only through the fields they change, so the same values spelled in
    a different order stay one cache entry.
    """
    if not assignments:
        return config
    annotations = {
        field.name: field.type for field in fields(RunConfig) if field.name != "name"
    }
    changes: dict[str, object] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in annotations:
            raise ParameterError(
                f"--set {assignment!r} does not name a field of a run; expected one of "
                f"{sorted(annotations)}"
            )
        changes[key] = _coerced(key, annotations[key], value)
    label = ", ".join(f"{key}={changes[key]}" for key in sorted(changes))
    changed = replace(config, name=f"{config.name} [{label}]", **changes)
    _validate(changed)
    return changed


def _coerced(key: str, annotation: str, value: str):
    """Read one override the way ``ConfigParser`` would have read the same line."""
    try:
        if annotation.startswith("tuple"):
            return _tuple(value)
        if annotation == "bool":
            return ConfigParser.BOOLEAN_STATES[value.lower()]
        if annotation == "int":
            return int(value)
        if annotation == "float":
            return float(value)
        return value
    except (KeyError, ValueError) as error:
        raise ParameterError(
            f"--set {key}={value!r} is not a valid {annotation}"
        ) from error


def _run(name: str, section) -> RunConfig:
    known = {field.name for field in fields(RunConfig)} - {"name"}
    unknown = set(section.keys()) - known - LEGACY_KEYS
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
            pooling=section.get("pooling"),
            positional=section.get("positional", fallback="learned"),
            embedding_norm=section.getboolean("embedding_norm", fallback=True),
            pooler_projection=section.getboolean("pooler_projection", fallback=True),
            tokenizer=section.get("tokenizer", fallback="wordpiece"),
            keep_brackets=section.getboolean("keep_brackets", fallback=True),
            tab_tower=section.get("tab_tower", fallback="mlp"),
            fusion_head=section.get("fusion_head", fallback="mlp"),
            tabular_dim=section.getint("tabular_dim", fallback=16),
            learning_rate=section.getfloat("learning_rate", fallback=1e-4),
            weight_decay=section.getfloat("weight_decay", fallback=0.01),
            epochs=section.getint("epochs", fallback=60),
            patience=section.getint("patience", fallback=10),
            batch_size=section.getint("batch_size", fallback=64),
            seed=section.getint("seed", fallback=1337),
            seeds=section.getint("seeds", fallback=1),
            checkpoints=section.getint("checkpoints", fallback=1),
        )
    except (TypeError, ValueError) as error:
        raise ParameterError(f"[{name}] is malformed: {error}") from error

    _validate(config)
    return config


def _validate(config: RunConfig) -> None:
    """Catch a typo in the file rather than three hours into a sweep."""
    name = config.name
    if config.model not in MODELS:
        raise ParameterError(f"[{name}] model must be one of {MODELS}")
    if not (config.text_fields or config.categorical_fields or config.numeric_fields):
        raise ParameterError(f"[{name}] declares no input fields")
    if config.model in PRETRAINED and not config.text_fields:
        raise ParameterError(
            f"[{name}] {config.model} declares no text_fields, and a pretrained "
            "language model has nothing to transfer without text"
        )
    for value, label in (
        (config.epochs, "epochs"),
        (config.patience, "patience"),
        (config.batch_size, "batch_size"),
        (config.seeds, "seeds"),
        (config.checkpoints, "checkpoints"),
    ):
        if value < 1:
            raise ParameterError(f"[{name}] {label}={value} must be at least 1")
    if config.learning_rate <= 0:
        raise ParameterError(f"[{name}] learning_rate must be positive")
    if config.model != TRANSFORMER:
        return
    for value, allowed, label in (
        (config.positional, POSITIONAL_ENCODINGS, "positional"),
        (config.pooling, POOLINGS, "pooling"),
        (config.tokenizer, TOKENIZERS, "tokenizer"),
        (config.tab_tower, TAB_TOWERS, "tab_tower"),
        (config.fusion_head, FUSION_HEADS, "fusion_head"),
    ):
        if value not in allowed:
            raise ParameterError(f"[{name}] {label}={value!r} is not one of {allowed}")
    if config.tabular_dim < 1:
        raise ParameterError(
            f"[{name}] tabular_dim={config.tabular_dim} must be at least 1"
        )
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


def changed_fields(left: RunConfig, right: RunConfig) -> tuple[str, ...]:
    """The dataclass fields where ``right`` departs from ``left``, name excluded.

    Shared by every task that needs to prove a comparison changed exactly one
    thing: the architecture search (Task 5), its greedy-order validation
    (Task 11), and the stability seeds (Task 7), which change only ``seed``.
    """
    a, b = asdict(left), asdict(right)
    return tuple(sorted(key for key in a if key != "name" and a[key] != b[key]))


def _tuple(value: str | None) -> tuple[str, ...]:
    """``a, b, c`` becomes a tuple; an empty value becomes an empty one."""
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())
