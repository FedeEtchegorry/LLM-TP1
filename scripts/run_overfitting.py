"""Los resultados en train, validation y test, y las brechas entre ellos.

    .venv/bin/python -m scripts.run_overfitting
    .venv/bin/python -m scripts.run_overfitting --config RUN --test

Cross-validation alone answers three of the four levels: ``fit``, the stopping split,
and the fold's own rows. The holdout is the fourth, and this script does **not** open
it unless ``--test`` says so. That is deliberate -- the analysis of how much a model
memorises is worth running many times while tuning, and every one of those runs would
otherwise spend the one evaluation the protocol allows.

See :mod:`src.model.overfitting` for what the four row sets are and why the curve's
``validation_ap`` is not the fold's validation score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one, run_test
from src.model.figures import (
    FIGURES_DIR,
    gap_by_epoch,
    generalisation_ladder,
    overfitting_curve,
)
from src.model.overfitting import (
    epoch_gap_frame,
    fold_levels,
    gaps,
    levels,
    markdown_table,
    stopped_early,
)
from src.model.results import RESULTS_DIR, document
from src.model.training import early_stopping_split
from src.model.baseline import target_of


def row_counts(frame, partitions) -> dict[str, int]:
    """How many rows each level holds, averaged over the folds where it varies.

    The fit/stop sizes are recomputed rather than stored: the split is a deterministic
    function of the fold and the fixed ``EARLY_STOPPING_SPLIT_SEED``, so recomputing it
    here cannot disagree with what training did, and nothing has to be migrated into
    the record for the table to quote a size.
    """
    target = target_of(frame)
    query_ids = frame["query_id"].tolist()
    fit_sizes, stop_sizes = [], []
    for fold in partitions.folds:
        fit_rows, stop_rows = early_stopping_split(target, query_ids, fold.train_indices)
        fit_sizes.append(len(fit_rows))
        stop_sizes.append(len(stop_rows))
    return {
        "fit": round(sum(fit_sizes) / len(fit_sizes)),
        "corte": round(sum(stop_sizes) / len(stop_sizes)),
        "validation": round(
            sum(len(fold.validation_indices) for fold in partitions.folds)
            / len(partitions.folds)
        ),
        "test": len(partitions.test_indices),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "overfitting"))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument(
        "--test",
        action="store_true",
        help="abrir el holdout y agregar el cuarto nivel; se gasta una sola vez",
    )
    parser.add_argument(
        "--force", action="store_true", help="reentrenar aunque haya resultado guardado"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="clave=valor",
        help="cambiar un campo sin declarar una seccion; repetible",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if args.config not in declared:
        raise SystemExit(
            f"{args.config!r} is not declared in {args.parameters}; "
            f"found {sorted(declared)}"
        )
    config = apply_overrides(declared[args.config], args.overrides)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\n=== CROSS-VALIDATION: {config.name} ===")
    result, note = run_one(
        config, frame, partitions, directory=args.results, force=args.force
    )
    print(f"  {config.digest}  {result.summary_row()}   [{note}]")

    record = document(config, args.results)
    if record is None:
        raise SystemExit(
            f"no record landed in {args.results}/ for {config.digest}; nothing to read"
        )

    test_record = None
    if args.test:
        final_dir = Path(args.results) / "final"
        print(f"\n=== HOLDOUT: {config.name} ===")
        print("  se abre una sola vez; --force lo vuelve a gastar a proposito")
        test_result, _, test_note = run_test(
            config, frame, partitions, directory=final_dir, force=args.force
        )
        print(f"  {test_result.summary_row()}   [{test_note}]")
        test_record = document(config, final_dir)

    per_fold = fold_levels(record)
    found = levels(record, test_record)
    steps = gaps(found)
    sizes = row_counts(frame, partitions)

    print("\n=== POR FOLD ===")
    print(per_fold.to_string(index=False))

    print("\n=== EARLY STOPPING ===")
    print(stopped_early(record).to_string(index=False))
    if not stopped_early(record)["stopped_early"].all():
        print(
            "  aviso: algun fold agoto el presupuesto de epocas, asi que su brecha\n"
            "  es una cota inferior y no un maximo observado"
        )

    print("\n=== NIVELES Y BRECHAS ===")
    print(markdown_table(found, steps))
    print("\nfilas por nivel: " + ", ".join(f"{k} {v}" for k, v in sizes.items()))
    if test_record is None:
        print("el nivel test no se midio; correr con --test para agregarlo")

    figures = Path(args.figures)
    written = [
        generalisation_ladder(
            found,
            title=f"De fit al holdout — {config.name}",
            path=figures / "generalisation-ladder.png",
        ),
        gap_by_epoch(
            epoch_gap_frame(record),
            title=f"Brecha por epoca — {config.name}",
            path=figures / "gap-by-epoch.png",
        ),
        overfitting_curve(
            epoch_gap_frame(record),
            title=f"Sobreajuste por epoca — {config.name}",
            path=figures / "overfitting-curve.png",
        ),
    ]

    summary_path = Path(args.results) / "overfitting" / f"{config.digest}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "name": config.name,
                "digest": config.digest,
                "opened_holdout": test_record is not None,
                "rows_per_level": sizes,
                "levels": [
                    {
                        "name": level.name,
                        "average_precision": level.average_precision,
                        "deviation": level.deviation,
                        "folds": level.folds,
                        "seen": level.seen,
                    }
                    for level in found
                ],
                "gaps": [
                    {
                        "origin": step.origin,
                        "destination": step.destination,
                        "delta": step.delta,
                        "question": step.question,
                    }
                    for step in steps
                ],
                "folds": per_fold.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    for path in written:
        print(f"figura: {path}")
    print(f"resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
