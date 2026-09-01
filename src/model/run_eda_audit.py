"""Task 6: one table, from stored evidence alone, naming everything that is missing.

    .venv/bin/python -m src.model.run_eda_audit --parameters parameters-eda.txt \
        --results results/eda-contract --output results/eda-contract/audit.md

Nothing here trains. It reads every declared section's JSON record and per-fold
predictions and fails loudly -- rather than silently skip a row -- when a JSON is
missing, its predictions are missing, or its stored configuration reads a column the
EDA contract forbids.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.model.configs import PARAMETERS_PATH, load_parameters
from src.model.eda_contract import BRACKET_RUNS, CONTRACT_FIELDS, DIAGNOSTIC_FIELDS, require_valid
from src.model.representation_selection import compare, paired_margin
from src.model.results import RESULTS_DIR, document, load_fold_predictions

L0_NAME = "L0 linear raw EDA"
L2_NAME = "L2 learned embeddings with attention"


@dataclass
class CoverageReport:
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    missing_predictions: list[str] = field(default_factory=list)
    forbidden_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.missing_predictions or self.forbidden_columns)


def _allowed_fields(name: str) -> frozenset[str]:
    if name in BRACKET_RUNS:
        return BRACKET_RUNS[name]
    if name.startswith("Q "):
        return CONTRACT_FIELDS | DIAGNOSTIC_FIELDS
    return CONTRACT_FIELDS


def coverage(declared: dict, directory: Path | str) -> CoverageReport:
    """Every declared section against what is actually recorded in ``directory``."""
    report = CoverageReport()
    for name, config in declared.items():
        stored = document(config, directory)
        if stored is None:
            report.missing.append(name)
            continue
        report.present.append(name)
        cfg = stored["config"]
        used = frozenset(cfg["text_fields"]) | frozenset(cfg["categorical_fields"]) | frozenset(
            cfg["numeric_fields"]
        )
        extra = used - _allowed_fields(name)
        if extra:
            report.forbidden_columns[name] = sorted(extra)
        if load_fold_predictions(config, directory) is None and len(stored["folds"]) > 1:
            report.missing_predictions.append(name)
    return report


def paired_comparison(left: dict[int, float], right: dict[int, float]) -> dict:
    """Compare two runs' per-fold AP, keyed by fold index. Both sides must report
    the same folds -- comparing a five-fold run against a four-fold one is not a
    paired comparison at all."""
    if set(left) != set(right):
        raise ValueError(
            "paired comparison requires the same folds on both sides, got "
            f"{sorted(left)} and {sorted(right)}"
        )
    folds = sorted(left)
    left_ap = np.array([left[f] for f in folds], dtype=float)
    right_ap = np.array([right[f] for f in folds], dtype=float)
    mean, low, high = paired_margin(right_ap - left_ap)
    return {
        "folds": folds,
        "mean": mean,
        "low": low,
        "high": high,
        "outcome": compare(left_ap, right_ap),
    }


def _fold_ap(stored: dict) -> dict[int, float]:
    return {fold["fold_index"]: fold["average_precision"] for fold in stored["folds"]}


def audit_rows(declared: dict, directory: Path | str) -> list[dict]:
    rows = []
    documents = {name: document(config, directory) for name, config in declared.items()}
    l0_ap = _fold_ap(documents[L0_NAME]) if documents.get(L0_NAME) else None
    l2_ap = _fold_ap(documents[L2_NAME]) if documents.get(L2_NAME) else None
    for name, stored in documents.items():
        if stored is None:
            continue
        metrics = stored["metrics"]
        curves = stored.get("curves") or []
        best_epochs = [c["best_epoch"] for c in curves if c.get("best_epoch") is not None]
        train_aps = [
            e["train_ap"]
            for c in curves
            for e in c.get("epochs", [])
            if e["epoch"] == c.get("best_epoch")
        ]
        row = {
            "name": name,
            "fields": sorted(
                set(stored["config"]["text_fields"])
                | set(stored["config"]["categorical_fields"])
                | set(stored["config"]["numeric_fields"])
            ),
            "parameters": curves[0]["parameters"] if curves else None,
            "seconds": stored["seconds"],
            "best_epochs": best_epochs,
            "train_ap": float(np.mean(train_aps)) if train_aps else None,
            "validation_ap": metrics["average_precision_mean"],
            "roc_mean": metrics["roc_auc_mean"],
            "roc_std": metrics["roc_auc_std"],
            "ap_mean": metrics["average_precision_mean"],
            "ap_std": metrics["average_precision_std"],
        }
        own_ap = _fold_ap(stored)
        for label, other in (("l0", l0_ap), ("l2", l2_ap)):
            if other is not None and name not in (L0_NAME, L2_NAME) and set(other) == set(own_ap):
                comparison = paired_comparison(other, own_ap)
                row[f"delta_vs_{label}"] = comparison["mean"]
                row[f"margin_vs_{label}"] = (comparison["low"], comparison["high"])
                row[f"outcome_vs_{label}"] = comparison["outcome"]
            else:
                row[f"delta_vs_{label}"] = None
                row[f"margin_vs_{label}"] = None
                row[f"outcome_vs_{label}"] = None
        rows.append(row)
    return rows


def render_markdown(rows: list[dict], report: CoverageReport) -> str:
    lines = [
        "# Auditoria de resultados bajo el contrato EDA",
        "",
        "| name | fields | parameters | seconds | best epochs | train AP | validation AP "
        "| ROC-AUC | AP (PR-AUC) | delta vs L0 | delta vs L2 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best_epochs = ",".join(str(e) for e in row["best_epochs"]) or "-"
        train_ap = "-" if row["train_ap"] is None else f"{row['train_ap']:.4f}"
        delta_l0 = "-" if row["delta_vs_l0"] is None else f"{row['delta_vs_l0']:+.4f}"
        delta_l2 = "-" if row["delta_vs_l2"] is None else f"{row['delta_vs_l2']:+.4f}"
        lines.append(
            f"| {row['name']} | {', '.join(row['fields'])} | {row['parameters'] or '-'} "
            f"| {row['seconds']:.0f} | {best_epochs} | {train_ap} "
            f"| {row['validation_ap']:.4f} | {row['roc_mean']:.4f} ± {row['roc_std']:.4f} "
            f"| {row['ap_mean']:.4f} ± {row['ap_std']:.4f} | {delta_l0} | {delta_l2} |"
        )
    lines.append("")
    lines.append("## Cobertura")
    lines.append(f"- presentes: {len(report.present)}")
    lines.append(f"- faltantes: {report.missing or 'ninguna'}")
    lines.append(f"- sin predicciones por fold: {report.missing_predictions or 'ninguna'}")
    lines.append(f"- columnas prohibidas encontradas: {report.forbidden_columns or 'ninguna'}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero if coverage is incomplete"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)

    report = coverage(declared, args.results)
    rows = audit_rows(declared, args.results)
    text = render_markdown(rows, report)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"audit written to {path}")
    else:
        print(text)

    if report.missing:
        print(f"\nmissing runs: {report.missing}")
    if report.missing_predictions:
        print(f"missing predictions: {report.missing_predictions}")
    if report.forbidden_columns:
        print(f"forbidden columns: {report.forbidden_columns}")

    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
