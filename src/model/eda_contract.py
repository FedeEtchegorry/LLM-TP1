"""The columns the EDA handed to Ejercicio 2, frozen so no run can quietly drop one.

Every candidate -- ladder rung, architecture point, seed repeat, transfer run -- must
carry exactly ``CONTRACT_FIELDS``. The only two exceptions are the diagnostic bounds
declared in ``BRACKET_RUNS`` (Task 4): they measure the floor and the ceiling of what
the text is worth, and are never candidates. ``require_valid`` is called right after
``load_parameters`` in every entrypoint that reads ``parameters-eda.txt``, so a
malformed section fails before the dataset is even loaded.
"""

from __future__ import annotations

from src.model.configs import RunConfig

TEXT_FIELDS = ("title", "description", "ingredients")
CATEGORICAL_FIELDS = ("category", "allergens")
NUMERIC_FIELDS = ("price_position",)
CONTRACT_FIELDS = frozenset(TEXT_FIELDS + CATEGORICAL_FIELDS + NUMERIC_FIELDS)
DIAGNOSTIC_FIELDS = frozenset(("popularity_phrase",))

TABULAR_FIELDS = frozenset(CATEGORICAL_FIELDS + NUMERIC_FIELDS)

# The two diagnostic bounds of Task 4. They frame what the text is worth; they
# are never candidates, never enter the architecture search and never see test.
BRACKET_RUNS: dict[str, frozenset[str]] = {
    "L0a linear, no text": TABULAR_FIELDS,
    "L0b linear, extracted key only": TABULAR_FIELDS | DIAGNOSTIC_FIELDS,
}

# Declared here rather than in run_final.py so the contract test can assert that
# no diagnostic bound is ever a finalist. Task 7 imports this tuple, it does not
# redefine it.
FINALISTS: tuple[str, ...] = (
    "L0 linear raw EDA",
    "M selected from directed comparisons",
)


def configured_fields(run: RunConfig) -> frozenset[str]:
    """Every input field a run declares, text, categorical and numeric together."""
    return frozenset(run.text_fields) | frozenset(run.categorical_fields) | frozenset(
        run.numeric_fields
    )


def validation_errors(declared: dict[str, RunConfig]) -> list[str]:
    """Every contract violation in ``declared``, accumulated rather than raised early.

    A "Q " section is a diagnostic control: the full contract plus exactly the
    diagnostic fields, never fewer and never a different extra field. A section
    named in ``BRACKET_RUNS`` must carry exactly that bracket's fields. Everything
    else -- every real candidate -- must carry exactly ``CONTRACT_FIELDS``.
    """
    errors: list[str] = []
    for name, run in declared.items():
        fields = configured_fields(run)
        if name.startswith("Q "):
            if not (CONTRACT_FIELDS < fields):
                errors.append(
                    f"[{name}] is a diagnostic control and must carry the full "
                    f"contract plus something extra, got {sorted(fields)}"
                )
            elif fields - CONTRACT_FIELDS != DIAGNOSTIC_FIELDS:
                errors.append(
                    f"[{name}] adds {sorted(fields - CONTRACT_FIELDS)} beyond the "
                    f"contract, expected exactly {sorted(DIAGNOSTIC_FIELDS)}"
                )
        elif name in BRACKET_RUNS:
            expected = BRACKET_RUNS[name]
            if fields != expected:
                errors.append(
                    f"[{name}] is a declared diagnostic bound and must carry exactly "
                    f"{sorted(expected)}, got {sorted(fields)}"
                )
        else:
            if fields != CONTRACT_FIELDS:
                errors.append(
                    f"[{name}] must carry exactly the contract fields "
                    f"{sorted(CONTRACT_FIELDS)}, got {sorted(fields)}"
                )
    for name in BRACKET_RUNS:
        if name in FINALISTS:
            errors.append(f"[{name}] is a diagnostic bound and cannot be a finalist")
    return errors


def require_valid(declared: dict[str, RunConfig]) -> None:
    """Raise one ``ValueError`` naming every contract violation, or return quietly."""
    errors = validation_errors(declared)
    if errors:
        raise ValueError(
            "parameters-eda.txt violates the EDA contract:\n  " + "\n  ".join(errors)
        )
