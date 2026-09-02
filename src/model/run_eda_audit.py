"""Audit declared CV evidence without reading the held-out test split.

The audit intentionally reads only the configured results directory.  A declared
run is covered only when its exact CV JSON and fold-prediction archive are both
present, and its stored fields still obey the EDA contract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.model.configs import PARAMETERS_PATH, RunConfig, load_parameters
from src.model.eda_contract import (
    BRACKET_RUNS,
    CONTRACT_FIELDS,
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_PREFIX,
    configured_fields,
    require_valid,
)
from src.model.representation_selection import paired_margin
from src.model.results import RESULTS_DIR, fold_predictions_path, result_path


@dataclass(frozen=True)
class Coverage:
    """Completeness and contract violations for the declared CV family."""

    missing_json: tuple[str, ...]
    missing_predictions: tuple[str, ...]
    invalid_fields: tuple[str, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        """Names lacking required evidence, regardless of which artifact failed."""
        return self.missing_json + self.missing_predictions

    @property
    def errors(self) -> tuple[str, ...]:
        return self.missing + self.invalid_fields

    @property
    def complete(self) -> bool:
        return not self.errors


def expected_fields(name: str) -> frozenset[str]:
    """The permitted field set for a declared run name."""
    if name in BRACKET_RUNS:
        return BRACKET_RUNS[name]
    if name.startswith(DIAGNOSTIC_PREFIX):
        return CONTRACT_FIELDS | DIAGNOSTIC_FIELDS
    return CONTRACT_FIELDS


def stored_fields(document: Mapping) -> frozenset[str]:
    """Read all named dataset fields from a serialized result configuration."""
    config = document.get("config", {})
    return frozenset(
        field
        for key in ("text_fields", "categorical_fields", "numeric_fields")
        for field in config.get(key, [])
    )


def coverage(declared: Mapping[str, RunConfig], results: Path | str) -> Coverage:
    """Check JSON, fold predictions, and stored input fields for every declaration.

    A missing JSON is reported separately from an existing JSON missing its `.npz`.
    This makes an interrupted run actionable without pretending that it has complete
    evidence.  The audit always uses the exact current digest: a historical record
    under an older configuration is not evidence for the current declaration.
    """
    root = Path(results)
    missing_json: list[str] = []
    missing_predictions: list[str] = []
    invalid_fields: list[str] = []

    for name, run in declared.items():
        json_path = result_path(run, root)
        if not json_path.exists():
            missing_json.append(name)
            continue

        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid_fields.append(f"[{name}] unreadable result JSON: {error}")
            continue

        actual = stored_fields(document)
        expected = expected_fields(name)
        if actual != expected:
            invalid_fields.append(
                f"[{name}] stored fields {sorted(actual)} differ from the allowed "
                f"contract {sorted(expected)}"
            )
        if document.get("name") != name:
            invalid_fields.append(
                f"[{name}] result JSON names {document.get('name')!r} instead"
            )

        predictions = fold_predictions_path(run, root)
        if not predictions.exists():
            missing_predictions.append(name)

    return Coverage(
        missing_json=tuple(missing_json),
        missing_predictions=tuple(missing_predictions),
        invalid_fields=tuple(invalid_fields),
    )


def paired_comparison(
    left: Mapping[int, float], right: Mapping[int, float]
) -> tuple[float, float, float]:
    """Return the declared paired margin for ``right - left`` over five folds."""
    if set(left) != set(right) or len(left) != 5:
        raise ValueError("paired comparison requires the same folds (exactly five)")
    folds = sorted(left)
    return paired_margin(
        np.asarray([right[fold] for fold in folds], dtype=float)
        - np.asarray([left[fold] for fold in folds], dtype=float)
    )


def _fold_scores(document: Mapping) -> dict[int, float]:
    return {
        int(fold["fold_index"]): float(fold["average_precision"])
        for fold in document.get("folds", [])
    }


def _stored_documents(declared: Mapping[str, RunConfig], root: Path) -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for name, run in declared.items():
        path = result_path(run, root)
        if path.exists():
            try:
                documents[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return documents


def _margin_against(
    name: str, documents: Mapping[str, Mapping], reference: str
) -> str:
    if name == reference:
        return "base"
    if name not in documents or reference not in documents:
        return "not available"
    try:
        mean, low, high = paired_comparison(
            _fold_scores(documents[reference]), _fold_scores(documents[name])
        )
    except ValueError:
        return "not comparable (not the same five folds)"
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def _mean_std(document: Mapping, metric: str) -> str:
    metrics = document.get("metrics", {})
    mean, spread = metrics.get(f"{metric}_mean"), metrics.get(f"{metric}_std")
    if mean is None or spread is None:
        return "not recorded"
    return f"{float(mean):.4f} +/- {float(spread):.4f}"


def _diagnostic_mean(document: Mapping, key: str) -> str:
    values = [fold.get(key) for fold in document.get("folds", [])]
    finite = [float(value) for value in values if value is not None]
    return f"{float(np.mean(finite)):.4f}" if finite else "not recorded"


def render_audit(
    declared: Mapping[str, RunConfig], results: Path | str, report: Coverage
) -> str:
    """Render a markdown audit from stored CV records; it never opens the holdout."""
    root = Path(results)
    documents = _stored_documents(declared, root)
    lines = [
        "# EDA contract audit",
        "",
        "This report reads cross-validation JSON and fold-prediction archives only; "
        "it never reads the holdout directory.",
        "",
        "## Coverage",
        "",
        f"- Missing JSON: {', '.join(report.missing_json) or 'none'}",
        f"- Missing fold predictions: {', '.join(report.missing_predictions) or 'none'}",
        f"- Invalid stored fields: {'; '.join(report.invalid_fields) or 'none'}",
        "",
        "## Recorded cross-validation evidence",
        "",
        "| Name | Fields | Parameters | Seconds | Best epoch | Train AP | Validation AP | PR-AUC mean +/- sd | ROC-AUC mean +/- sd | Delta vs L0 | Delta vs L2 |",
        "|---|---|---|---:|---|---:|---:|---|---|---|---|",
    ]
    for name, run in declared.items():
        document = documents.get(name)
        if document is None:
            lines.append(
                f"| {name} | {', '.join(sorted(configured_fields(run)))} | not recorded "
                "| — | — | — | — | — | — | — | — |"
            )
            continue
        config = document.get("config", {})
        parameters = ", ".join(
            f"{key}={config[key]}"
            for key in ("model", "d_model", "n_layers", "n_heads", "dropout", "positional", "pooling", "numeric_embedding", "seed")
            if key in config
        )
        best_epochs = [fold.get("best_epoch") for fold in document.get("folds", [])]
        best_epoch = ", ".join(str(value) for value in best_epochs if value is not None) or "not recorded"
        lines.append(
            "| {name} | {fields} | {parameters} | {seconds:.3f} | {best_epoch} | "
            "{train_ap} | {validation_ap} | {ap} | {roc} | {l0} | {l2} |".format(
                name=name,
                fields=", ".join(sorted(stored_fields(document))),
                parameters=parameters,
                seconds=float(document.get("seconds", 0.0)),
                best_epoch=best_epoch,
                train_ap=_diagnostic_mean(document, "train_ap"),
                validation_ap=_diagnostic_mean(document, "average_precision"),
                ap=_mean_std(document, "average_precision"),
                roc=_mean_std(document, "roc_auc"),
                l0=_margin_against(name, documents, "L0 linear raw EDA"),
                l2=_margin_against(name, documents, "L2 learned embeddings with attention"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strict", action="store_true", help="fail on incomplete evidence")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    report = coverage(declared, args.results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_audit(declared, args.results, report), encoding="utf-8")
    print(f"audit written to {output}")
    if args.strict and not report.complete:
        raise SystemExit("audit is incomplete:\n- " + "\n- ".join(report.errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
