"""The input contract established by the EDA for Exercise 2.

The constants in this module are deliberately boring: they make it possible to
fail before training when a candidate silently inherits a discarded column.
Diagnostic runs may expose ``popularity_phrase`` because measuring the value of
that manual extraction is part of the analysis; it is never a candidate input.
"""

from __future__ import annotations

from dataclasses import asdict

from src.model.configs import RunConfig, ladder_runs

TEXT_FIELDS = ("title", "description", "ingredients")
CATEGORICAL_FIELDS = ("category", "allergens")
NUMERIC_FIELDS = ("price_position",)

CONTRACT_FIELDS = frozenset(TEXT_FIELDS + CATEGORICAL_FIELDS + NUMERIC_FIELDS)
DIAGNOSTIC_FIELDS = frozenset(("popularity_phrase",))
DIAGNOSTIC_PREFIX = "Q "

TABULAR_FIELDS = frozenset(CATEGORICAL_FIELDS + NUMERIC_FIELDS)

# Diagnostic brackets: neither run is a candidate, and nothing downstream may be
# chosen with them.  Their deliberately reduced inputs are the only exceptions to
# the complete-field contract.
#
# ``L0b`` si se puntuo sobre el holdout (``run_ceiling_holdout``), despues de que la
# comparacion final ya estaba escrita, para dibujar la cota en las figuras del test.
# Sigue fuera de FINALISTS y ninguna decision lo mira.
BRACKET_RUNS: dict[str, frozenset[str]] = {
    "L0a linear, no text": TABULAR_FIELDS,
    "L0b linear, extracted key only": TABULAR_FIELDS | DIAGNOSTIC_FIELDS,
}

FINALISTS: tuple[str, ...] = ("FINAL bracket d96 L2 h4 piecewise do0.3 lr2e-4 seed99",)

EXPECTED_LADDER_MOVES = (("L1", "L2", ("n_layers",)),)


def configured_fields(run: RunConfig) -> frozenset[str]:
    """Every dataset column named by one run."""
    return frozenset(run.text_fields + run.categorical_fields + run.numeric_fields)


def changed_fields(left: RunConfig, right: RunConfig) -> tuple[str, ...]:
    """Configuration fields changed between two runs, excluding their labels."""
    before, after = asdict(left), asdict(right)
    return tuple(
        sorted(
            name
            for name in before
            if name != "name" and before[name] != after[name]
        )
    )


def find_prefix(declared: dict[str, RunConfig], prefix: str) -> RunConfig:
    """Find exactly one declared run by its stable rung/control prefix."""
    matches = [run for name, run in declared.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one run starting with {prefix!r}, found {len(matches)}")
    return matches[0]


def validation_errors(declared: dict[str, RunConfig]) -> list[str]:
    """Return every contract violation instead of stopping at the first one."""
    errors: list[str] = []
    for name, run in declared.items():
        fields = configured_fields(run)
        if name in BRACKET_RUNS:
            expected = BRACKET_RUNS[name]
            if fields != expected:
                errors.append(
                    f"[{name}] diagnostic bracket must use {sorted(expected)}, "
                    f"got {sorted(fields)}"
                )
        elif name.startswith(DIAGNOSTIC_PREFIX):
            expected = CONTRACT_FIELDS | DIAGNOSTIC_FIELDS
            if fields != expected:
                errors.append(
                    f"[{name}] diagnostic must use the complete EDA contract plus "
                    f"{sorted(DIAGNOSTIC_FIELDS)}, got {sorted(fields)}"
                )
        elif fields != CONTRACT_FIELDS:
            errors.append(
                f"[{name}] must keep all EDA fields together, got {sorted(fields)}"
            )

    for prefix in ("L0 linear raw EDA", "L2"):
        try:
            run = find_prefix(declared, prefix)
        except ValueError as error:
            errors.append(str(error))
            continue
        if configured_fields(run) != CONTRACT_FIELDS:
            errors.append(
                f"[{run.name}] must use the complete EDA contract, got "
                f"{sorted(configured_fields(run))}"
            )

    rungs = ladder_runs(declared)
    for left_prefix, right_prefix, expected in EXPECTED_LADDER_MOVES:
        try:
            left = find_prefix(rungs, left_prefix)
            right = find_prefix(rungs, right_prefix)
        except ValueError as error:
            errors.append(str(error))
            continue
        actual = changed_fields(left, right)
        if actual != expected:
            errors.append(
                f"{left_prefix} -> {right_prefix} moves {actual}, expected {expected}"
            )
    return errors


def require_valid(declared: dict[str, RunConfig]) -> None:
    """Raise one actionable error before any invalid experimental family runs."""
    errors = validation_errors(declared)
    if errors:
        raise ValueError("invalid EDA experiment family:\n- " + "\n- ".join(errors))
