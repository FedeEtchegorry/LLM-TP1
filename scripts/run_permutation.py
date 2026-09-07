"""Cuánto pierde el modelo cuando se le rompe el orden de los tokens.

    .venv/bin/python -m scripts.run_permutation
    .venv/bin/python -m scripts.run_permutation --fold 2 --repeats 10

Se permuta en inferencia el 0, 25, 50, 75 y 100% de los tokens de cada fila y se mide la
caída de PR-AUC. La segunda curva es el mismo modelo con ``positional = none``, que no
puede distinguir un orden de otro y por lo tanto tiene que salir plana: si se inclina, lo
que hay que arreglar es el código. Si la curva con posiciones tampoco cae, el veredicto es
que aprendió una bolsa de palabras a pesar de tenerlas, y se reporta igual.

Los pesos se guardan por digest, así que la segunda corrida no reentrena nada.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.diagnostics import (
    PERMUTED_FRACTIONS,
    order_invariance,
    permutation_curve,
)
from src.model.eda_contract import require_valid
from src.model.experiment import describe, partition
from src.model.figures import FIGURES_DIR, permutation_curves
from src.model.results import RESULTS_DIR, load_weights, save_weights

POSITIONAL = ("learned", "none")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR / "permutation"))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--force", action="store_true", help="reentrenar aunque haya pesos guardados"
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


def fitted(config, frame, train_indices, fold_index, directory, *, force):
    """El modelo de un fold, del disco si está y entrenado una sola vez si no."""
    from src.model.encoding import RowEncoder
    from src.model.network import BtrTransformer
    from src.model.training import spec_for, train_fold

    encoder = RowEncoder(spec_for(config)).fit(frame, train_indices)
    state = None if force else load_weights(config, fold_index, directory)
    if state is not None:
        model = BtrTransformer(encoder, config)
        model.load_state_dict(state)
        return model.eval(), encoder, "cache"

    trained = train_fold(config, frame, train_indices, seed=config.seed)
    save_weights(
        config, fold_index, trained.model.state_dict(), directory=directory
    )
    return trained.model.eval(), trained.encoder, f"entrenado, epoca {trained.best_epoch}"


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
    base = apply_overrides(declared[args.config], args.overrides)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    describe(frame, partitions)

    fold = partitions.folds[args.fold]
    # Los indices son una tupla, y numpy lee una tupla como indice multidimensional.
    scored_on = list(fold.validation_indices)
    actual = target_of(frame)[scored_on]

    curves, notes, drift = [], [], {}
    for positional in POSITIONAL:
        config = replace(base, positional=positional)
        model, encoder, note = fitted(
            config, frame, fold.train_indices, fold.fold_index, args.results,
            force=args.force,
        )
        notes.append(f"positional={positional:<8s} {config.digest}  [{note}]")
        print(f"\n=== {notes[-1]} ===")
        curve = permutation_curve(
            model, encoder, frame, scored_on, actual,
            repeats=args.repeats, seed=base.seed,
        )
        curves.append(curve.assign(model=positional))
        drift[positional] = order_invariance(
            model, encoder, frame, scored_on, seed=base.seed
        )

    table = pd.concat(curves, ignore_index=True)
    summary = (
        table.groupby(["model", "fraction"])["average_precision"]
        .agg(["mean", "std"])
        .reset_index()
    )

    print(f"\n=== ORDEN DE LOS TOKENS — fold {fold.fold_index}, {len(actual)} filas, "
          f"{args.repeats} permutaciones por punto ===")
    print(summary.to_string(index=False))

    print("\n=== CAIDA DESDE EL 0% ===")
    for positional in POSITIONAL:
        rows = summary[summary["model"] == positional].set_index("fraction")["mean"]
        drop = float(rows.loc[1.0] - rows.loc[0.0])
        print(f"  positional={positional:<8s} {rows.loc[0.0]:.4f} -> "
              f"{rows.loc[1.0]:.4f}   {drop:+.4f}")

    control = summary[summary["model"] == "none"]["mean"]
    flat = float(control.max() - control.min())
    print("\n=== CONTROL ===")
    print(f"  rango del AP con positional=none      {flat:.2e}")
    for positional in POSITIONAL:
        print(f"  torre de texto, positional={positional:<8s} {drift[positional]:.2e}")
    if drift["none"] > 1e-5:
        print("  AVISO: el control no salio plano, asi que el arnes mueve algo que no\n"
              "  deberia mover y la otra curva todavia no se puede leer")
    elif drift["learned"] < 100 * drift["none"]:
        print("  AVISO: el modelo con posiciones casi no reacciona al orden, asi que\n"
              "  su curva mide ruido de redondeo y no perdida de informacion")

    figures = Path(args.figures)
    path = permutation_curves(
        table,
        title=f"Orden de los tokens — fold {fold.fold_index} de [{base.name}]",
        path=figures / "permutation-curves.png",
    )

    summary_path = Path(args.results) / "permutation" / f"{base.digest}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "name": base.name,
                "fold_index": fold.fold_index,
                "rows_scored": len(actual),
                "repeats": args.repeats,
                "fractions": list(PERMUTED_FRACTIONS),
                "models": notes,
                "control_range": flat,
                "text_tower_drift": drift,
                "points": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nfigura: {path}")
    print(f"resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
