"""El techo diagnóstico (``L0b``) medido sobre el holdout, para enmarcar el resultado.

    .venv/Scripts/python -m src.model.run_ceiling_holdout \
        --parameters parameters-eda.txt --results results/eda-contract

``L0b`` no compite ni entra a ``eda_contract.FINALISTS``: sólo dibuja la cota superior
en las figuras del test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import EDA_PARAMETERS, PROTOCOL, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import FINALISTS, require_valid
from src.model.experiment import describe, partition, run_test
from src.model.results import RESULTS_DIR

CEILING = "L0b linear, extracted key only"
CEILING_FILE = "ceiling.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--final-results", type=str, default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    final = Path(args.final_results or (Path(args.results) / "final"))
    final.mkdir(parents=True, exist_ok=True)

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if CEILING not in declared:
        raise SystemExit(f"[{CEILING}] no está declarado en {args.parameters}")
    if CEILING in FINALISTS:
        raise SystemExit(f"[{CEILING}] es una cota y no puede ser finalista")

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)
    target = target_of(frame)

    print(f"\n=== EL TECHO DIAGNOSTICO EN EL HOLDOUT ({len(partitions.test_indices)} filas) ===")
    result, _, note = run_test(
        declared[CEILING], frame, partitions, directory=str(final), force=args.force
    )
    fold = result.folds[0]
    print(f"  {CEILING:<38s} ROC {fold.roc_auc:.4f}   AP {fold.average_precision:.4f}   [{note}]")

    document = {
        "name": CEILING,
        "rol": "cota superior diagnostica, no candidata",
        "rows": len(partitions.test_indices),
        "positive_rate": float(target[list(partitions.test_indices)].mean()),
        "roc_auc": float(fold.roc_auc),
        "average_precision": float(fold.average_precision),
    }
    out = final / CEILING_FILE
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
