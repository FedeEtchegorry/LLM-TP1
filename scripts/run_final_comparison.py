"""La comparación final: el mejor lineal contra el mejor Transformer, en el holdout.

    .venv/Scripts/python -m scripts.run_final_comparison \
        --parameters parameters-eda.txt --results results/eda-contract

**Se corre una sola vez**, sobre dos modelos congelados antes de abrir el conjunto.

``LINEAR_FINALIST`` no puede ser una sección de ``parameters-eda.txt``: la
representación no es un campo de ``RunConfig`` sino una composición de bloques de
``run_embeddings``, así que se reconstruye acá con el mismo protocolo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import EDA_PARAMETERS, PROTOCOL, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_test
from src.model.protocol import evaluate_on_test
from src.model.results import RESULTS_DIR
from scripts.run_embeddings import blocks_scorer, composed_blocks

TRANSFORMER_FINALIST = "FINAL bracket d96 L2 h4 piecewise do0.3 lr2e-4 seed99"

LINEAR_FINALIST = "L0* swept-best linear"
LINEAR_REPRESENTATION = ("tf-idf", "one-hot", "piecewise-linear")
"""Exactamente lo que eligió embeddings/selection.json, y lo que la segunda pasada
confirmó estable frente a un cambio en cualquiera de las tres familias."""


def score_linear(frame, partitions, target) -> tuple[np.ndarray, np.ndarray, float]:
    """Ajusta el lineal en todo el desarrollo y puntúa el holdout, una vez."""
    text, categorical, numeric = LINEAR_REPRESENTATION
    scorer = blocks_scorer(
        lambda: composed_blocks(text, categorical, numeric), frame
    )
    captured: dict[str, np.ndarray] = {}

    def capturing(train_indices, scored_indices):
        scores = scorer(train_indices, scored_indices)
        captured["predicted"] = scores
        return scores

    started = time.perf_counter()
    result = evaluate_on_test(LINEAR_FINALIST, target, partitions, capturing)
    seconds = time.perf_counter() - started
    fold = result.folds[0]
    return (
        np.asarray([fold.roc_auc, fold.average_precision], dtype=float),
        captured["predicted"],
        seconds,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--final-results", type=str, default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="volver a gastar el holdout; es una decisión, no un default",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    final = Path(args.final_results or (Path(args.results) / "final"))
    final.mkdir(parents=True, exist_ok=True)

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if TRANSFORMER_FINALIST not in declared:
        raise SystemExit(f"[{TRANSFORMER_FINALIST}] no está declarado en {args.parameters}")

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)
    target = target_of(frame)

    print(f"\n=== EL HOLDOUT, {len(partitions.test_indices)} FILAS, UNA SOLA VEZ ===")
    print(f"    ajustando sobre {len(partitions.development_indices)} filas de desarrollo\n")

    roc_ap, linear_scores, linear_seconds = score_linear(frame, partitions, target)
    print(
        f"  {LINEAR_FINALIST:<34s} ROC {roc_ap[0]:.4f}   AP {roc_ap[1]:.4f}"
        f"   [{linear_seconds:.0f}s, 453 params]"
    )

    config = declared[TRANSFORMER_FINALIST]
    result, predicted, note = run_test(
        config, frame, partitions, directory=str(final), force=args.force
    )
    fold = result.folds[0]
    print(
        f"  {TRANSFORMER_FINALIST:<34s} ROC {fold.roc_auc:.4f}   AP "
        f"{fold.average_precision:.4f}   [{note}]"
    )

    prevalence = float(target[list(partitions.test_indices)].mean())
    delta = float(fold.average_precision - roc_ap[1])
    document = {
        "rows": len(partitions.test_indices),
        "positive_rate": prevalence,
        "linear": {
            "name": LINEAR_FINALIST,
            "representation": list(LINEAR_REPRESENTATION),
            "roc_auc": float(roc_ap[0]),
            "average_precision": float(roc_ap[1]),
            "parameters": 453,
        },
        "transformer": {
            "name": TRANSFORMER_FINALIST,
            "roc_auc": float(fold.roc_auc),
            "average_precision": float(fold.average_precision),
            "config": config.digest,
        },
        "delta_transformer_minus_linear": delta,
        "cross_validation_delta": -0.0229,
    }
    (final / "comparison.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        final / "linear-predictions.npz",
        indices=np.asarray(list(partitions.test_indices), dtype=np.int64),
        scores=linear_scores,
    )

    print()
    print(f"  tasa positiva del holdout: {prevalence:.4f}")
    print(f"  delta Transformer − lineal: {delta:+.4f}")
    print(f"  el mismo delta en validación cruzada: {document['cross_validation_delta']:+.4f}")
    print(f"\n  escrito en {final}/comparison.json")
    print("  el holdout queda gastado: no volver a correr esto sin --force y sin motivo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
