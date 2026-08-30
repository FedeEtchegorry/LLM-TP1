"""Regenerate every figure the model side draws, from ``results/`` alone.

    .venv/bin/python -m src.model.run_figures
    .venv/bin/python -m src.model.run_figures --only 09-final-roc-pr

Nothing here trains and nothing here calls ``evaluate_on_test``: every number comes
from a record already on disk -- the cross-validation and holdout JSON documents, the
holdout predictions, the saved Transformer weights. A figure whose corridas are not
recorded yet is skipped, with the missing run named, rather than trained on the spot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import PARAMETERS_PATH, PROTOCOL, RunConfig, load_parameters
from src.model.diagnostics import (
    Scored,
    calibration,
    calibration_error,
    errors_by_level,
    pr_points,
    ranking_gains,
    roc_points,
)
from src.model import figures as fig
from src.model.experiment import partition
from src.model.results import RESULTS_DIR, load, load_predictions, summary_frame
from src.model.run_final import explainable, interpretability

BAR = "L0"
ERROR_COLUMN = "popularity_phrase"

INTERPRETABILITY_FIGURES = (
    "09-final-curvas-entrenamiento",
    "09-final-atencion-cls",
    "09-final-buckets-precio",
)
"""Drawn together by ``run_final.interpretability``, which reports its own gaps."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(fig.FIGURES_DIR))
    parser.add_argument("--only", type=str, default="", help="draw just the matching figure(s)")
    return parser.parse_args(argv)


def wanted(name: str, only: str) -> bool:
    return not only or only in name


def report(name: str, path: Path) -> None:
    print(f"  [{name}] {path}")


def report_missing(name: str, reasons: list[str]) -> None:
    print(f"  [{name}] falta: {'; '.join(reasons)}")


def best_recorded(declared: dict[str, RunConfig], directory: str) -> RunConfig | None:
    """The declared configuration with the best cross-validated AP, or ``None``.

    Mirrors ``run_final.select`` but returns rather than raising, so a figure that
    needs it can be reported as missing instead of aborting the whole run.
    """
    summary = summary_frame(directory)
    if summary.empty:
        return None
    eligible = summary[summary["name"].isin(declared)]
    if eligible.empty:
        return None
    ranked = eligible.sort_values("average_precision_mean", ascending=False)
    return declared[ranked.iloc[0]["name"]]


def cached_test(config: RunConfig, directory: str) -> tuple | None:
    """The stored holdout result and predictions for this config, or ``None``.

    Reads only what ``run_final`` already wrote; a config with no such record is not
    drawable yet, not a reason to score the holdout here.
    """
    result = load(config, directory)
    predicted = load_predictions(config, directory)
    if result is None or predicted is None:
        return None
    return result, predicted


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    figures_dir = Path(args.figures)
    results_dir = args.results
    final_dir = str(Path(results_dir) / "final")

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    actual = target_of(frame)[list(partitions.test_indices)]
    positive_rate = float(actual.mean())

    print(f"=== FIGURAS DESDE {results_dir}/ (sin entrenar) ===")

    winner = best_recorded(declared, results_dir)
    bar_config = next((run for name, run in declared.items() if name.startswith(BAR)), None)

    scores: list[Scored] = []
    stage6_missing: list[str] = []
    if winner is None:
        stage6_missing.append(
            f"sin corridas de cross-validation registradas en {results_dir}/ "
            "(correr run_ladder)"
        )
    else:
        cached = cached_test(winner, final_dir)
        if cached is None:
            stage6_missing.append(f"falta la corrida final de {winner.name} (correr run_final)")
        else:
            scores.append(Scored(winner.name, actual, np.asarray(cached[1], dtype=float)))

    if bar_config is not None and (winner is None or bar_config.name != winner.name):
        cached = cached_test(bar_config, final_dir)
        if cached is not None:
            scores.append(Scored(bar_config.name, actual, np.asarray(cached[1], dtype=float)))

    if wanted("09-final-roc-pr", args.only):
        if stage6_missing:
            report_missing("09-final-roc-pr", stage6_missing)
        else:
            path = fig.roc_and_pr(
                [
                    (s.name, roc_points(s), pr_points(s), s.roc_auc, s.average_precision)
                    for s in scores
                ],
                positive_rate=positive_rate,
                title=f"Test retenido ({len(actual)} filas, BTR {positive_rate:.3f})",
                path=figures_dir / "09-final-roc-pr.png",
            )
            report("09-final-roc-pr", path)

    if wanted("09-final-calibracion", args.only):
        if stage6_missing:
            report_missing("09-final-calibracion", stage6_missing)
        else:
            table = calibration(scores[0])
            path = fig.calibration(
                table,
                title=f"Calibracion de [{scores[0].name}] en el test",
                path=figures_dir / "09-final-calibracion.png",
                error=calibration_error(table),
            )
            report("09-final-calibracion", path)

    if wanted("09-final-ranking-lift", args.only):
        if stage6_missing:
            report_missing("09-final-ranking-lift", stage6_missing)
        else:
            gains = [(s.name, ranking_gains(s)) for s in scores]
            path = fig.ranking_gains(
                gains,
                title="Precision y lift en el tope del ranking (test)",
                path=figures_dir / "09-final-ranking-lift.png",
            )
            report("09-final-ranking-lift", path)

    if wanted("09-final-error-por-frase", args.only):
        if stage6_missing:
            report_missing("09-final-error-por-frase", stage6_missing)
        else:
            errors = errors_by_level(frame, partitions.test_indices, scores[0], ERROR_COLUMN)
            path = fig.errors_by_level(
                errors,
                title=f"BTR observado y predicho por {ERROR_COLUMN} (test)",
                path=figures_dir / "09-final-error-por-frase.png",
            )
            report("09-final-error-por-frase", path)

    if any(wanted(name, args.only) for name in INTERPRETABILITY_FIGURES):
        explained = (
            None
            if winner is None
            else explainable(winner, declared, partitions, results_dir, final_dir)
        )
        if explained is None:
            report_missing(
                " / ".join(INTERPRETABILITY_FIGURES),
                ["ningun Transformer registrado para explicar (correr run_ladder o run_final)"],
            )
        else:
            interpretability(explained, frame, figures_dir)

    if wanted("08-transfer-similitud-frases", args.only):
        from src.model.pretrained import CONTRAST, contrast_row, phrase_similarity

        pairs = phrase_similarity(frame)
        contrast = contrast_row(pairs, CONTRAST)
        path = fig.similarity_against_gap(
            pairs,
            contrast=contrast,
            title="Similitud semantica frente a diferencia de BTR (MiniLM congelado)",
            path=figures_dir / "08-transfer-similitud-frases.png",
        )
        report("08-transfer-similitud-frases", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
