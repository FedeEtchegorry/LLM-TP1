"""The ladder: the linear bar, then each component the Transformer adds to reach it.

    .venv/bin/python -m src.model.run_ladder

Every rung is declared in ``parameters.txt`` and is one run that doubles as a slide and
as a point in the ablation table:

    L1 -> L2   axis B, does attention buy anything over an average?
    L2 -> L3   axis G, do the tabular columns add to the text?
    L3 -> L4   axis A, is the bucket term worth the +0.137 AP the EDA predicts?
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import PARAMETERS_PATH, PROTOCOL, ladder_runs, load_parameters
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition, run_one, sweep_note
from src.model.protocol import EvaluationResult, markdown_table
from src.model.representation_selection import compare_folds, paired_margin
from src.model.results import RESULTS_DIR, fold_frame


RUNGS: tuple[tuple[str, str], ...] = (
    ("L0a linear, no text", "sin texto -> piso"),
    ("L0 linear raw EDA", "texto crudo, lineal"),
    ("L1 learned embeddings, no attention", "embeddings aprendidos"),
    ("L2 learned embeddings with attention", "con autoatencion"),
    ("L0b linear, extracted key only", "clave extraida a mano -> techo"),
)
"""El orden de la lectura es el argumento: piso, lineal, aprendido, atencion, techo."""

FLOOR, CEILING = RUNGS[0][0], RUNGS[-1][0]
LEARNED, CANDIDATE = RUNGS[2][0], RUNGS[3][0]

READING_DIR = "ladder"
READING_FILE = "reading.json"


def fold_ap(directory: Path | str = RESULTS_DIR) -> dict[str, np.ndarray]:
    """AP por fold, en orden de fold, para cada peldano ya registrado."""
    frame = fold_frame(directory)
    if frame.empty:
        return {}
    scores: dict[str, np.ndarray] = {}
    for name, _ in RUNGS:
        rows = frame[frame["name"] == name]
        if rows.empty:
            continue
        ordered = rows.sort_values("fold_index")
        scores[name] = ordered["average_precision"].to_numpy(dtype=float)
    return scores


def reading_table(scores: Mapping[str, Sequence[float]]) -> str:
    """Una sola tabla, en el orden declarado, con AP media y AP por fold."""
    lines = [
        "| Peldano | Lectura | AP media | AP por fold |",
        "|---|---|---:|---|",
    ]
    for name, label in RUNGS:
        values = np.asarray(scores.get(name, ()), dtype=float)
        if values.size == 0:
            lines.append(f"| {name} | {label} | sin registrar | -- |")
            continue
        folds = ", ".join(f"{value:.4f}" for value in values)
        lines.append(
            f"| {name} | {label} "
            f"| {values.mean():.4f} +/- {values.std(ddof=1):.4f} | {folds} |"
        )
    return "\n".join(lines)


def recovered_fraction(scores: Mapping[str, Sequence[float]]) -> dict:
    """Que fraccion del tramo entre las dos cotas recupera el candidato desde el texto.

    Se lee fold a fold, porque las tres corridas comparten la misma particion: cada
    fold aporta un cociente y el margen pareado resume los cinco.  Si el denominador
    -- el techo por encima del piso -- no es claramente positivo, la metrica no se
    publica: dividir por un tramo que puede ser cero no informa nada.
    """
    missing = [name for name in (FLOOR, CEILING, CANDIDATE) if name not in scores]
    if missing:
        return {
            "published": False,
            "candidate": CANDIDATE,
            "reason": f"missing fold scores for {', '.join(missing)}",
        }
    floor = np.asarray(scores[FLOOR], dtype=float)
    ceiling = np.asarray(scores[CEILING], dtype=float)
    candidate = np.asarray(scores[CANDIDATE], dtype=float)

    span = ceiling - floor
    span_mean, span_low, span_high = paired_margin(span)
    if span_low <= 0.0:
        return {
            "published": False,
            "candidate": CANDIDATE,
            "reason": (
                "denominator is not clearly positive: the extracted-key ceiling "
                f"exceeds the no-text floor by {span_mean:.6f} "
                f"[{span_low:.6f}, {span_high:.6f}]"
            ),
        }

    mean, low, high = paired_margin((candidate - floor) / span)
    return {
        "published": True,
        "candidate": CANDIDATE,
        "floor": FLOOR,
        "ceiling": CEILING,
        "mean": mean,
        "low": low,
        "high": high,
        "span_mean": span_mean,
        "span_low": span_low,
        "span_high": span_high,
    }


def attention_reading(scores: Mapping[str, Sequence[float]]) -> dict:
    """La primera regla narrativa: que se puede decir del salto L1 -> L2."""
    missing = [name for name in (LEARNED, CANDIDATE) if name not in scores]
    if missing:
        raise ValueError(f"missing fold scores for {', '.join(missing)}")
    learned = np.asarray(scores[LEARNED], dtype=float)
    candidate = np.asarray(scores[CANDIDATE], dtype=float)
    outcome = compare_folds(learned, candidate)
    mean, low, high = paired_margin(candidate - learned)

    narratives = {
        "improves": (
            "la autoatencion mejora la representacion promediada por margen pareado "
            "y se incorpora a la base"
        ),
        "tie-break": (
            "la autoatencion se incorpora por desempate -- media mas alta y folds mas "
            "estables -- y no se narra como demostracion"
        ),
        "inconclusive": (
            "la autoatencion no mostro una mejora concluyente: el intervalo incluye "
            "cero; L2 se conserva como Transformer requerido para las comparaciones "
            "modulares, pero no se declara superior"
        ),
        "loses": (
            "resultado negativo: la autoatencion pierde contra la representacion "
            "aprendida y L1 queda como referencia de representacion aprendida"
        ),
    }
    adopted = outcome in ("improves", "tie-break")
    return {
        "outcome": outcome,
        "attention_adopted": adopted,
        # L2 sigue siendo la referencia salvo que pierda: un intervalo que incluye
        # cero no lo declara superior, pero tampoco lo reemplaza por un modelo sin
        # atencion que las comparaciones modulares no podrian usar.
        "reference": LEARNED if outcome == "loses" else CANDIDATE,
        "delta": mean,
        "low": low,
        "high": high,
        "narrative": narratives[outcome],
    }


def read_ladder(scores: Mapping[str, Sequence[float]]) -> dict:
    """Lo que la escalera dice una vez corrida: folds, recuperacion y veredicto."""
    payload = {
        "order": [name for name, _ in RUNGS],
        "folds": {
            name: [float(value) for value in values] for name, values in scores.items()
        },
        "recovery": recovered_fraction(scores),
    }
    if LEARNED in scores and CANDIDATE in scores:
        payload["attention"] = attention_reading(scores)
    return payload


def write_reading(payload: dict, directory: Path | str = RESULTS_DIR) -> Path:
    path = Path(directory) / READING_DIR / READING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def print_reading(scores: Mapping[str, Sequence[float]]) -> dict:
    """La lectura completa por consola, en el orden en que se argumenta."""
    print("\n=== LECTURA DE LA ESCALERA CONTRA LAS COTAS ===")
    print(reading_table(scores))

    payload = read_ladder(scores)
    recovery = payload["recovery"]
    print()
    if recovery["published"]:
        print("recuperacion = (AP(L2) - AP(L0a)) / (AP(L0b) - AP(L0a))")
        print(
            f"  {recovery['mean']:.4f} "
            f"[{recovery['low']:.4f}, {recovery['high']:.4f}] (margen pareado)"
        )
    else:
        print(f"recuperacion no publicada -- {recovery['reason']}")

    attention = payload.get("attention")
    if attention is not None:
        print(
            f"\nL1 -> L2: {attention['outcome']} "
            f"(delta={attention['delta']:.6f}, "
            f"margen=[{attention['low']:.6f}, {attention['high']:.6f}])"
        )
        print(f"  {attention['narrative']}")
        print(f"  referencia declarada tras el contraste: {attention['reference']}")
    return payload


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument(
        "--force", action="store_true", help="retrain even when a result is recorded"
    )
    parser.add_argument("--only", type=str, default="", help="run just this section")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="read the recorded ladder against its brackets without training",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    require_valid(declared)
    runs = ladder_runs(declared)
    if args.only:
        runs = {name: run for name, run in runs.items() if args.only in name}
    if args.read_only:
        payload = print_reading(fold_ap(args.results))
        print(f"\nlectura escrita en {write_reading(payload, args.results)}")
        return 0

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    print(f"\n=== LADDER ({len(runs)} rungs from {args.parameters}) ===")
    results: list[EvaluationResult] = []
    trained_seconds, trained = 0.0, 0
    for position, (name, config) in enumerate(runs.items(), start=1):
        started = time.perf_counter()
        print(f"  [{position}/{len(runs)}] {name}", flush=True)
        result, note = run_one(
            config, frame, partitions, directory=args.results, force=args.force
        )
        if note != "recorded":
            trained_seconds += time.perf_counter() - started
            trained += 1
        results.append(result)
        print(
            f"  {config.digest}  {result.summary_row()}   [{note}]"
            + sweep_note(trained_seconds, trained, len(runs) - position)
        )

    if results:
        print()
        print(markdown_table(results, float(target_of(frame).mean())))
        print(f"\nrecorded in {args.results}/ -- read them with src.model.results")

    scores = fold_ap(args.results)
    if all(name in scores for name, _ in RUNGS):
        payload = print_reading(scores)
        print(f"\nlectura escrita en {write_reading(payload, args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
