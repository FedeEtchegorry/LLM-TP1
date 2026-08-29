"""Stages 4 and 5: what somebody else's weights are worth on this problem.

    .venv/Scripts/python -m src.model.run_transfer
    .venv/Scripts/python -m src.model.run_transfer --only frozen

Three things happen, in this order:

1. **The declared ``[T ...]`` runs**, through the same ``run_one`` as every other row
   of the table -- same folds, same metrics, same cache. A bag of words, the frozen
   MiniLM encoder with and without the tabular columns, and the fine-tuned checkpoint.
2. **The three regimes side by side on fold 0** -- from scratch, frozen, fine-tuned --
   with the parameters each one actually trained and the time it took. Fine-tuning
   runs one fold, so the comparison is made on the fold all three share rather than
   against a five-fold mean it never earned.
3. **The similarity slide.** The frozen encoder is *expected* to lose here, and the
   measurement says why. ``Customer Favorite`` and ``Shopper Favorite`` sit at cosine
   0.733 and buy at 67.7% against 2.8%; the *closest* pair the encoder sees at all,
   ``Highly Rated`` and ``Top Rated`` at cosine 0.873, is 60.6 points apart in
   behaviour. A semantic encoder is built to collapse exactly that distinction, which
   is the correct behaviour for the task it was trained on and the wrong one for this
   dataset, where the phrase is a code and not a meaning. This is the concrete answer
   to "when does transfer learning not help", and it is worth more than the AP it costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import (
    FINETUNE,
    FROZEN,
    PARAMETERS_PATH,
    PROTOCOL,
    TRANSFER,
    RunConfig,
    ladder_runs,
    load_parameters,
    transfer_runs,
)
from src.model.experiment import describe, partition, run_one
from src.model.figures import FIGURES_DIR, similarity_against_gap
from src.model.protocol import EvaluationResult, markdown_table
from src.model.results import RESULTS_DIR, document

SCRATCH = "L4"
"""The rung the two pretrained regimes are measured against: our own Transformer."""

REGIME_LABELS = {
    "logistic": "bolsa de palabras (sin preentrenar)",
    "transformer": "propio, desde cero",
    FROZEN: "MiniLM congelado + cabezal logistico",
    FINETUNE: "MiniLM fine-tuneado",
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", type=str, default="", help="run just the matching sections")
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="skip the phrase-similarity slide, which needs the encoder loaded",
    )
    return parser.parse_args(argv)


def fold_zero(config: RunConfig, directory: str) -> dict | None:
    """One run's fold-0 record: the metrics, the parameters trained and the seconds.

    ``seconds`` in the stored document covers every fold the run did, so it is divided
    back down. That is an average, not a stopwatch on fold 0, and the column says so.
    """
    stored = document(config, directory)
    if stored is None:
        return None
    folds = stored["folds"]
    first = next((fold for fold in folds if fold["fold_index"] == 0), None)
    if first is None:
        return None
    curves = {curve["fold_index"]: curve for curve in stored["curves"]}
    return {
        "name": stored["name"],
        "regime": REGIME_LABELS.get(stored["config"]["model"], stored["config"]["model"]),
        "folds": len(folds),
        "roc_auc": first["roc_auc"],
        "average_precision": first["average_precision"],
        "parameters": curves.get(0, {}).get("parameters"),
        "seconds_per_fold": stored["seconds"] / max(len(folds), 1),
    }


def regime_table(rows: list[dict]) -> str:
    """The comparison the plan asks for: three regimes, one fold, one table."""
    lines = [
        "| Regimen | Corrida | Folds | ROC-AUC (fold 0) | PR-AUC (fold 0) "
        "| Parametros entrenados | Segundos por fold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        parameters = "—" if row["parameters"] is None else f"{row['parameters']:,}"
        lines.append(
            f"| {row['regime']} | {row['name']} | {row['folds']} "
            f"| {row['roc_auc']:.3f} | {row['average_precision']:.3f} "
            f"| {parameters} | {row['seconds_per_fold']:.0f} |"
        )
    return "\n".join(lines)


def similarity_slide(frame: pd.DataFrame, figures: Path) -> None:
    """Measure and draw what the frozen encoder makes of the popularity phrases."""
    from src.model.pretrained import CONTRAST, contrast_row, phrase_similarity

    print("\n=== SEMANTIC SIMILARITY OF THE POPULARITY PHRASES ===")
    pairs = phrase_similarity(frame)
    contrast = contrast_row(pairs, CONTRAST)

    closest = pairs.iloc[0]
    print(
        f"\nThe closest pair the encoder sees is {closest['left']} / {closest['right']} "
        f"at cosine {closest['cosine']:.3f}, and those two buy at "
        f"{closest['left_rate'] * 100:.1f}% and {closest['right_rate'] * 100:.1f}%: "
        f"{closest['rate_gap'] * 100:.1f} points apart."
    )
    print("\nClosest pairs the encoder sees, with the gap they actually carry:")
    display = pairs.head(8).assign(
        cosine=lambda d: d["cosine"].map("{:.3f}".format),
        rate_gap=lambda d: (d["rate_gap"] * 100).map("{:.1f} pp".format),
    )
    print(display[["left", "right", "cosine", "rate_gap"]].to_string(index=False))

    if contrast is None:
        print(f"\n{CONTRAST[0]} / {CONTRAST[1]} are not both present in this dataset")
    else:
        print(
            f"\n{CONTRAST[0]} vs {CONTRAST[1]}: "
            f"cosine {contrast['cosine']:.3f}, "
            f"BTR {contrast['left_rate'] * 100:.1f}% against "
            f"{contrast['right_rate'] * 100:.1f}% "
            f"({contrast['rate_gap'] * 100:.1f} points apart)"
        )
        rank = int((pairs["cosine"] > contrast["cosine"]).sum()) + 1
        print(
            f"That pair is the {rank}th closest of {len(pairs)} and the "
            f"{int((pairs['rate_gap'] > contrast['rate_gap']).sum()) + 1}th widest: "
            "the encoder's notion of meaning and this dataset's notion of a code "
            "are not the same notion."
        )

    path = similarity_against_gap(
        pairs,
        contrast=contrast,
        title="Similitud semantica frente a diferencia de BTR (MiniLM congelado)",
        path=figures / "08-transfer-similitud-frases.png",
    )
    print(f"\nfigure: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    runs = transfer_runs(declared)
    if args.only:
        runs = {name: run for name, run in runs.items() if args.only in name}

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)
    print(
        f"\ncheckpoint {TRANSFER.checkpoint}, max_length {TRANSFER.max_length}, "
        f"batch {TRANSFER.batch_size}, {TRANSFER.epochs} epochs, "
        f"{TRANSFER.finetune_folds} fold(s) for the fine-tune"
    )

    print(f"\n=== TRANSFER ({len(runs)} runs from {args.parameters}) ===")
    results: list[EvaluationResult] = []
    for name, config in runs.items():
        result, note = run_one(
            config, frame, partitions, directory=args.results, force=args.force
        )
        results.append(result)
        print(f"  {config.digest}  {result.summary_row()}   [{note}]")

    if results:
        print()
        print(markdown_table(results, float(target_of(frame).mean())))

    print("\n=== THE THREE REGIMES, ON THE FOLD ALL THREE RAN ===")
    wanted = [
        next((run for name, run in ladder_runs(declared).items() if name.startswith(SCRATCH)), None),
        *[run for name, run in transfer_runs(declared).items() if run.model in (FROZEN, FINETUNE)],
    ]
    rows = [fold_zero(config, args.results) for config in wanted if config is not None]
    present = [row for row in rows if row is not None]
    if len(present) < len(rows):
        print(
            f"  {len(rows) - len(present)} of the runs this table needs are not "
            "recorded yet; run src.model.run_ladder first for the scratch row"
        )
    if present:
        print()
        print(regime_table(present))

    if not args.skip_similarity:
        similarity_slide(frame, Path(args.figures))

    print(f"\nrecorded in {args.results}/ -- read them with src.model.results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
