"""Stage 6: spend the holdout once, then ask the model why.

    .venv/Scripts/python -m src.model.run_final
    .venv/Scripts/python -m src.model.run_final --config "L4"

The 20% test split has not been read by anything up to here -- not by the ladder, not
by the sweep, not by the transfer runs, and not by the early stopping inside any of
them. This is the one script allowed to touch it, and it touches it once per model:
the configuration that won on cross-validation, and the linear bar it has to beat.
Both scores are stored, so every table and figure below is a view of that single
evaluation and never a reason to run another.

**The winner is whatever the cross-validation says**, read out of ``results/``. If
that turns out to be the logistic bar rather than the Transformer, this script reports
it that way; the ladder measured L4 at AP 0.710 against a bar of 0.813, and that is a
result to report rather than to hide.

What comes out, in order:

1. ROC and PR on the holdout, final against bar.
2. **Calibration by decile** -- ``BTR`` is the mean predicted probability, so this is
   the business metric itself, not a stand-in for it.
3. **Precision@k and lift@k**, which is the brief's actual use: identify the best
   products and promote them, on a budget.
4. **Where ``[CLS]`` looks** and **whether the price buckets recovered the inverted
   U**, both read off stored weights rather than a fresh training run. When the bar
   wins the selection, these still get drawn -- from the Transformer's own
   cross-validation fold 0, on rows it never trained on, so explaining the
   architecture never costs a third pass over the holdout. See :func:`explainable`.
5. **Where it fails**, by popularity phrase -- the plan predicts the tier that buys at
   2.6% and reads, in text, exactly like the tier that buys at 64.7%.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import (
    EDA_PARAMETERS,
    PROTOCOL,
    TRAINING,
    TRANSFORMER,
    RunConfig,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import FINALISTS
from src.model.diagnostics import (
    Scored,
    bucket_embedding_axis,
    calibration,
    calibration_error,
    cls_attention,
    errors_by_level,
    price_bucket_recovery,
    pr_points,
    ranking_gains,
    roc_points,
)
from src.model.experiment import FINAL_DIR, describe, partition, run_test
from src.model.figures import FIGURES_DIR
from src.model.results import RESULTS_DIR, curve_frame, summary_frame

BAR = "L0"
"""The rung the final model is drawn against: the best honest linear baseline."""

PRICE_COLUMN = "price_position"
ERROR_COLUMN = "popularity_phrase"

FLAT_RESPONSE = 0.005
"""Half a point of BTR. Below this the price sweep is a flat line, not a shape."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument(
        "--final", "--final-results", dest="final", type=str, default=str(FINAL_DIR)
    )
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR))
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="name the final configuration instead of taking the best recorded one",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="score the holdout again for a model that already has a record",
    )
    return parser.parse_args(argv)


def finalists(declared: dict[str, RunConfig]) -> list[RunConfig]:
    """The frozen finalist list, ordered as the holdout scores it.

    ``FINALISTS`` is declared in ``eda_contract`` before any holdout run and is
    never re-ranked here.  That is the whole point: ``L0b``, a diagnostic bracket
    that receives the hand-extracted popularity phrase, has the highest recorded
    AP of any run in the family, so a sort by cross-validated AP would promote it
    to the holdout.  A parameter file that declares every finalist gets this list;
    one that does not -- the historical ``parameters.txt`` -- keeps ``select``.

    The model comes first because the diagnostics below read ``scores[0]``; the
    linear bar follows it.
    """
    if any(name not in declared for name in FINALISTS):
        return []
    runs = [declared[name] for name in FINALISTS]
    return [run for run in runs if not run.name.startswith(BAR)] + [
        run for run in runs if run.name.startswith(BAR)
    ]


def select(
    declared: dict[str, RunConfig], directory: str
) -> tuple[RunConfig, pd.DataFrame]:
    """The declared run with the best cross-validated AP, and the table it came from.

    Only declared sections are eligible: a record left over from a parameter file that
    has since changed is history, not a candidate.
    """
    summary = summary_frame(directory)
    if summary.empty:
        raise SystemExit(
            f"no cross-validation runs recorded in {directory}/ -- run "
            "src.model.run_ladder (and run_modules, run_transfer) before selecting"
        )
    eligible = summary[summary["name"].isin(declared)]
    if eligible.empty:
        raise SystemExit(
            f"none of the runs recorded in {directory}/ is still declared in the "
            "parameter file; nothing can be selected"
        )
    ranked = eligible.sort_values("average_precision_mean", ascending=False)
    return declared[ranked.iloc[0]["name"]], ranked


def rebuild(
    config: RunConfig,
    frame: pd.DataFrame,
    train_indices,
    fold_index: int,
    directory: str,
):
    """A stored Transformer back in memory, without retraining anything.

    The encoder is a deterministic function of the rows it was fitted on and of the
    configuration, so refitting it on the same ``train_indices`` reproduces exactly the
    vocabulary, levels, medians and bucket edges the stored weights were trained
    against. Pass ``fold_index=-1`` for the held-out run's model, or a fold number for
    a cross-validation one.
    """
    from src.model.encoding import RowEncoder
    from src.model.network import BtrTransformer
    from src.model.results import load_weights
    from src.model.training import spec_for

    state = load_weights(config, fold_index, directory)
    if state is None:
        return None, None
    encoder = RowEncoder(spec_for(config)).fit(frame, train_indices)
    model = BtrTransformer(encoder, config, TRAINING.n_buckets)
    model.load_state_dict(state)
    model.eval()
    return model, encoder


@dataclass(frozen=True)
class Explained:
    """Which Transformer the interpretability section reads, and off which rows."""

    config: RunConfig
    fold_index: int
    fitted_on: tuple[int, ...]
    read_on: tuple[int, ...]
    directory: str
    label: str


def explainable(
    winner: RunConfig,
    declared: dict[str, RunConfig],
    partitions,
    results_directory: str,
    final_directory: str,
) -> Explained | None:
    """Pick that Transformer, in the two cases that can arise.

    If the Transformer won the cross-validation, the model to explain is the one
    already trained for the holdout, read on the holdout rows.

    If the **linear bar won** -- which is what the plan's Stage 2 measured -- the
    architecture still has to be explained, because "where does the attention go" is a
    claim about our design and not about the winner. We prefer whichever Transformer
    already has a holdout record: that is the one a previous run of this script
    actually selected and spent the holdout on, and re-explaining it is free. A sweep
    point added to ``parameters.txt`` afterwards -- another axis knob, or a seed
    repeat under axis S that exists only to measure variance, never to compete for
    selection -- must not silently swap out which model gets explained just because it
    happens to score a hair higher on cross-validation; ``seed_variance`` exists
    precisely because those differences are not reliable enough to decide anything.
    Only when nothing has a holdout record yet do we fall back to the best
    cross-validation score, read on that Transformer's own fold 0, on rows it never
    trained on -- **not the holdout, which is not spent a third time to draw a slide.**
    """
    if winner.model == TRANSFORMER:
        return Explained(
            config=winner,
            fold_index=-1,
            fitted_on=partitions.development_indices,
            read_on=partitions.test_indices,
            directory=final_directory,
            label="test",
        )

    final_summary = summary_frame(final_directory)
    if not final_summary.empty:
        finalists = final_summary[
            (final_summary["model"] == TRANSFORMER) & (final_summary["name"].isin(declared))
        ]
        if not finalists.empty:
            already_selected = finalists.sort_values(
                "average_precision_mean", ascending=False
            ).iloc[0]
            return Explained(
                config=declared[already_selected["name"]],
                fold_index=-1,
                fitted_on=partitions.development_indices,
                read_on=partitions.test_indices,
                directory=final_directory,
                label="test",
            )

    summary = summary_frame(results_directory)
    if summary.empty:
        return None
    transformers = summary[
        (summary["model"] == TRANSFORMER) & (summary["name"].isin(declared))
    ]
    if transformers.empty:
        return None

    best = transformers.sort_values("average_precision_mean", ascending=False).iloc[0]
    fold = partitions.folds[0]
    return Explained(
        config=declared[best["name"]],
        fold_index=0,
        fitted_on=fold.train_indices,
        read_on=fold.validation_indices,
        directory=results_directory,
        label="validacion del fold 0",
    )


def holdout_table(scores: list[Scored], positive_rate: float) -> str:
    lines = [
        "| Modelo | ROC-AUC | PR-AUC (AP) | Lift sobre el azar |",
        "|---|---:|---:|---:|",
        f"| azar | 0.500 | {positive_rate:.3f} | 1.00x |",
    ]
    for scored in scores:
        lines.append(
            f"| {scored.name} | {scored.roc_auc:.3f} "
            f"| {scored.average_precision:.3f} "
            f"| {scored.average_precision / positive_rate:.2f}x |"
        )
    return "\n".join(lines)


def gains_table(tables: list[tuple[str, pd.DataFrame]]) -> str:
    lines = ["| Modelo | k | Productos | Aciertos | Precision@k | Recall@k | Lift@k |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, table in tables:
        for _, row in table.iterrows():
            lines.append(
                f"| {name} | {row['fraction'] * 100:.0f}% | {int(row['k'])} "
                f"| {int(row['hits'])} | {row['precision'] * 100:.1f}% "
                f"| {row['recall'] * 100:.1f}% | {row['lift']:.2f}x |"
            )
    return "\n".join(lines)


def interpretability(
    explained: Explained, frame: pd.DataFrame, figures: Path
) -> None:
    """The two questions only the trained Transformer can answer."""
    from src.model.figures import attention_by_group, price_recovery

    from src.model.figures import training_curves

    config, label = explained.config, explained.label
    model, encoder = rebuild(
        config, frame, explained.fitted_on, explained.fold_index, explained.directory
    )
    if model is None:
        print(f"\n  no stored weights for [{config.name}], so no attention slide")
        return
    print(
        f"\n(leyendo [{config.name}] sobre {len(explained.read_on)} filas de {label})"
    )

    # The over- and underfitting evidence belongs to the model being explained, not to
    # the model that won: a logistic regression has no epochs to draw.
    curves = curve_frame(explained.directory)
    if not curves.empty and (curves["digest"] == config.digest).any():
        path = training_curves(
            curves[curves["digest"] == config.digest],
            title=f"Curvas de entrenamiento de [{config.name}]",
            path=figures / "09-final-curvas-entrenamiento.png",
        )
        print(f"figure: {path}")

    print("\n=== WHERE [CLS] LOOKS ===")
    attention = cls_attention(model, encoder, frame, explained.read_on)
    if attention.empty:
        print("  this configuration has no attention blocks to read")
    else:
        display = attention.assign(
            layer=lambda d: d["layer"] + 1,
            tokens=lambda d: d["tokens"].map("{:.1f}".format),
            mass=lambda d: d["mass"].map("{:.4f}".format),
            per_token=lambda d: d["per_token"].map("{:.4f}".format),
        ).sort_values(["layer", "per_token"], ascending=[True, False])
        print(display.to_string(index=False))
        path = attention_by_group(
            attention,
            title=f"Atencion del [CLS] por grupo de posiciones ({label})",
            path=figures / "09-final-atencion-cls.png",
        )
        print(f"figure: {path}")

    if PRICE_COLUMN not in config.numeric_fields:
        return
    if model.numbers is None or model.numbers.buckets is None:
        print(f"\n  {config.name} has no bucket table, so there is no U to recover")
        return

    print("\n=== DID THE PRICE BUCKETS RECOVER THE INVERTED U? ===")
    sweep = price_bucket_recovery(
        model, encoder, frame, explained.read_on, column=PRICE_COLUMN
    )
    axis = bucket_embedding_axis(model, encoder, PRICE_COLUMN)
    display = sweep.assign(
        observed=lambda d: (d["observed"] * 100).map("{:.1f}%".format),
        counterfactual=lambda d: (d["counterfactual"] * 100).map("{:.1f}%".format),
        as_is=lambda d: (d["as_is"] * 100).map("{:.1f}%".format),
        centre=lambda d: d["centre"].map("{:.3f}".format),
    )
    print(display.to_string(index=False))

    # How far the response moves matters before how well its shape matches: a flat
    # line correlates with anything you like, and correlating against one is noise
    # dressed as a finding.
    spread = float(sweep["counterfactual"].max() - sweep["counterfactual"].min())
    print(
        f"\nmoving the price bucket across all ten deciles moves the predicted BTR by "
        f"{spread * 100:.2f} points"
    )
    if spread < FLAT_RESPONSE:
        print(
            "which is flat: this model barely reads the price token at all, so it did "
            "not recover the hump -- it is riding whatever else correlates with price. "
            "The as_is column moves and the counterfactual one does not, and that gap "
            "is the finding."
        )
    else:
        correlation = float(np.corrcoef(sweep["observed"], sweep["counterfactual"])[0, 1])
        print(
            "correlation between the observed hump and the model's response: "
            f"{correlation:+.3f}"
        )

    path = price_recovery(
        sweep,
        axis,
        title=f"El modelo frente a la U invertida de price_position ({label})",
        path=figures / "09-final-buckets-precio.png",
    )
    print(f"figure: {path}")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    from src.model.figures import calibration as calibration_figure
    from src.model.figures import errors_by_level as errors_figure
    from src.model.figures import ranking_gains as gains_figure
    from src.model.figures import roc_and_pr

    args = parse_args(argv)
    figures = Path(args.figures)
    declared = load_parameters(args.parameters)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    frozen = finalists(declared)
    chosen: list[RunConfig] | None = None
    if args.config:
        matching = [run for name, run in declared.items() if args.config in name]
        if not matching:
            raise SystemExit(f"no declared section matches {args.config!r}")
        winner, ranking = matching[0], summary_frame(args.results)
    elif frozen:
        winner, ranking, chosen = frozen[0], summary_frame(args.results), frozen
    else:
        winner, ranking = select(declared, args.results)

    print("\n=== SELECTED ON CROSS-VALIDATION, NEVER ON THE HOLDOUT ===")
    if chosen is not None:
        print("finalists frozen before the holdout was opened: " + ", ".join(FINALISTS))
    if not ranking.empty:
        display = ranking.head(8).assign(
            AP=lambda d: d.average_precision_mean.map("{:.4f}".format)
            + " ± "
            + d.average_precision_std.map("{:.4f}".format),
            ROC=lambda d: d.roc_auc_mean.map("{:.4f}".format),
        )
        print(display[["name", "digest", "model", "ROC", "AP"]].to_string(index=False))
    print(f"\nfinal model: [{winner.name}] ({winner.model}, digest {winner.digest})")

    if chosen is None:
        bar = next((run for name, run in declared.items() if name.startswith(BAR)), None)
        chosen = [winner] + ([bar] if bar is not None and bar.name != winner.name else [])

    print(f"\n=== THE HOLDOUT, {len(partitions.test_indices)} ROWS, SCORED ONCE ===")
    target = target_of(frame)
    actual = target[list(partitions.test_indices)]
    scores: list[Scored] = []
    for config in chosen:
        result, predicted, note = run_test(
            config, frame, partitions, directory=args.final, force=args.force
        )
        scores.append(Scored(config.name, actual, np.asarray(predicted, dtype=float)))
        print(f"  {config.digest}  {result.summary_row()}   [{note}]")

    positive_rate = float(actual.mean())
    print()
    print(holdout_table(scores, positive_rate))

    path = roc_and_pr(
        [
            (
                scored.name,
                roc_points(scored),
                pr_points(scored),
                scored.roc_auc,
                scored.average_precision,
            )
            for scored in scores
        ],
        positive_rate=positive_rate,
        title=f"Test retenido ({len(actual)} filas, BTR {positive_rate:.3f})",
        path=figures / "09-final-roc-pr.png",
    )
    print(f"\nfigure: {path}")

    print("\n=== CALIBRATION: PREDICTED BTR AGAINST OBSERVED BTR ===")
    table = calibration(scores[0])
    error = calibration_error(table)
    display = table.assign(
        predicted=lambda d: (d["predicted"] * 100).map("{:.1f}%".format),
        observed=lambda d: (d["observed"] * 100).map("{:.1f}%".format),
        interval=lambda d: (d["low"] * 100).map("{:.1f}".format)
        + " – "
        + (d["high"] * 100).map("{:.1f}%".format),
    )
    print(display[["bin", "rows", "predicted", "observed", "interval"]].to_string(index=False))
    print(f"\nexpected calibration error: {error:.4f}")
    path = calibration_figure(
        table,
        title=f"Calibracion de [{scores[0].name}] en el test",
        path=figures / "09-final-calibracion.png",
        error=error,
    )
    print(f"figure: {path}")

    print("\n=== RANKING FOR PROMOTION ===")
    gains = [(scored.name, ranking_gains(scored)) for scored in scores]
    print()
    print(gains_table(gains))
    path = gains_figure(
        gains,
        title="Precision y lift en el tope del ranking (test)",
        path=figures / "09-final-ranking-lift.png",
    )
    print(f"\nfigure: {path}")

    print(f"\n=== WHERE IT FAILS, BY {ERROR_COLUMN} ===")
    errors = errors_by_level(frame, partitions.test_indices, scores[0], ERROR_COLUMN)
    display = errors.assign(
        observed=lambda d: (d["observed"] * 100).map("{:.1f}%".format),
        predicted=lambda d: (d["predicted"] * 100).map("{:.1f}%".format),
        gap=lambda d: (d["gap"] * 100).map("{:+.1f} pp".format),
        average_precision=lambda d: d["average_precision"].map("{:.3f}".format),
    )
    print(display.to_string(index=False))
    path = errors_figure(
        errors,
        title=f"BTR observado y predicho por {ERROR_COLUMN} (test)",
        path=figures / "09-final-error-por-frase.png",
    )
    print(f"figure: {path}")

    explained = explainable(winner, declared, partitions, args.results, args.final)
    if explained is None:
        print("\n  no Transformer recorded, so there is no architecture to explain")
    else:
        interpretability(explained, frame, figures)

    print(f"\nrecorded in {args.final}/ -- the holdout has now been spent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
