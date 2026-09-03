from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.figures import (
    FIGURES_DIR,
    encoding_families,
    encoding_ladder,
    noise_against_effect,
    one_versus_two,
    paired_differences,
    seed_collapse,
    training_curves,
)
from src.model.results import RESULTS_DIR, curve_frame, fold_frame
from scripts.run_embeddings import read_sweep

SEEDS_FILE = "embeddings/seeds-numeric-axis.csv"
BAR_NAME = "L0 linear bar"

AXIS_A = {
    "A no numeric reading": "none  (no lee el numero)",
    "L3 tabular, numbers affine only": "affine  (termino ordenado)",
    "A buckets only": "buckets  (one-hot por decil)",
    "L4 tabular, numbers affine and bucketed": "affine+buckets  (dos a la vez)",
    "A piecewise-linear": "piecewise-linear",
    "A periodic": "periodic  (Fourier aprendido)",
}

BUCKETS_ONLY = "A buckets only"
AFFINE_ONLY = "L3 tabular, numbers affine only"
AFFINE_BUCKETS = "L4 tabular, numbers affine and bucketed"
NO_NUMERIC = "A no numeric reading"

FAMILY_TITLES = {
    "numeric": "Numericas - la pregunta que el informe del Ejercicio 1 deja abierta",
    "categorical": "Categoricas - contra que se eligio one-hot",
    "text": "Texto - la frase extraida contra el texto crudo",
    "fields": "Que columnas entran - las 4 del informe contra las 9 del modelo",
}

TWO_ENCODING_CONTRASTS = [
    ("numeric", "percentiles one-hot (la barra)",
     "continuo + percentiles (2 codificaciones)",
     "Lineal: percentiles  ->  + continuo"),
    ("categorical", "one-hot (lo que dice el informe)",
     "one-hot + target (2 codificaciones)",
     "Lineal: one-hot  ->  + target encoding"),
    ("text", "frase extraida one-hot (la barra)",
     "frase + bolsa de palabras (2 codificaciones)",
     "Lineal: frase  ->  + bolsa de palabras"),
    ("text", "frase extraida one-hot (la barra)",
     "frase + tf-idf (2 codificaciones)",
     "Lineal: frase  ->  + tf-idf"),
]


def axis_a_folds(results: Path | str) -> pd.DataFrame:
    folds = fold_frame(results)
    if folds.empty or "name" not in folds:
        return pd.DataFrame()
    subset = folds[folds["name"].isin(AXIS_A)].copy()
    subset["encoding"] = subset["name"].map(AXIS_A)
    return subset


def seeds_table(results: Path | str) -> pd.DataFrame:
    path = Path(results) / SEEDS_FILE
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def paired_summary(frame, index, column, value, base, other, label) -> dict | None:
    if frame.empty or value not in frame or column not in frame:
        return None
    wide = frame.pivot_table(index=index, columns=column, values=value)
    if base not in wide or other not in wide:
        return None
    difference = (wide[other] - wide[base]).dropna().to_numpy()
    if len(difference) < 2:
        return None
    return {
        "contrast": label,
        "delta": float(difference.mean()),
        "half": float(1.96 * difference.std(ddof=1) / np.sqrt(len(difference))),
        "pairs": len(difference),
    }


def two_encoding_contrasts(axis, seeds, sweep) -> pd.DataFrame:
    rows = [
        paired_summary(
            axis, ["fold_index"], "encoding", "average_precision",
            AXIS_A[BUCKETS_ONLY], AXIS_A[AFFINE_BUCKETS],
            "Transformer: buckets  ->  + termino afin",
        )
    ]
    if not seeds.empty:
        rows.append(
            paired_summary(
                seeds, ["offset", "fold"], "mode", "ap", "buckets", "periodic",
                "Transformer: buckets  ->  periodic (4 semillas)",
            )
        )
    if not sweep.empty:
        for block, base, other, label in TWO_ENCODING_CONTRASTS:
            rows.append(
                paired_summary(
                    sweep[sweep["block"] == block], ["fold"], "encoding",
                    "average_precision", base, other, label,
                )
            )
    return pd.DataFrame([row for row in rows if row is not None])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    figures = Path(args.figures)
    axis = axis_a_folds(args.results)
    seeds = seeds_table(args.results)
    sweep = read_sweep(args.results)
    summary = fold_frame(args.results)
    if summary.empty or "name" not in summary:
        bar = None
    else:
        bar = summary[summary["name"] == BAR_NAME]["average_precision"].mean()
        bar = None if pd.isna(bar) else float(bar)

    written: list[Path] = []
    missing: list[str] = []

    if axis.empty:
        missing.append("run_ladder  (para el eje A en results/*.json)")
    else:
        repeats = seeds[seeds["mode"] == "buckets"]["ap"] if not seeds.empty else None
        written.append(
            encoding_ladder(
                axis,
                title="Eje A: como se codifica un numero dentro del Transformer",
                path=figures / "10-embeddings-eje-a.png",
                reference=bar,
                noise=None
                if repeats is None or repeats.empty
                else float(repeats.std(ddof=1)),
            )
        )
        curves = curve_frame(args.results)
        curves = curves[curves["name"].isin(AXIS_A)]
        if not curves.empty:
            written.append(
                training_curves(
                    curves,
                    title="Eje A: curvas de entrenamiento por codificacion numerica",
                    path=figures / "13-embeddings-curvas.png",
                )
            )

    if seeds.empty:
        missing.append("el barrido de semillas  (results/embeddings/seeds-numeric-axis.csv)")
    else:
        written.append(
            paired_differences(
                seeds.rename(columns={"offset": "seed"}),
                baseline="buckets",
                group="mode",
                value="ap",
                title="Diferencias pareadas contra one-hot por decil (4 semillas x 5 folds)",
                path=figures / "11-embeddings-pareado.png",
            )
        )
        written.append(
            seed_collapse(
                seeds,
                baseline="buckets",
                candidate="periodic",
                title="Por que una sola semilla miente: periodic contra buckets",
                path=figures / "12-embeddings-semillas.png",
            )
        )
        wide = seeds.pivot_table(index=["offset", "fold"], columns="mode", values="ap")
        effects = {
            f"{mode} - buckets": float((wide[mode] - wide["buckets"]).mean())
            for mode in ("periodic", "piecewise")
            if mode in wide
        }
        if not axis.empty:
            axis_wide = axis.pivot_table(
                index="fold_index", columns="encoding", values="average_precision"
            )
            for label, other, base in (
                ("affine+buckets - buckets", AFFINE_BUCKETS, BUCKETS_ONLY),
                ("buckets - affine", BUCKETS_ONLY, AFFINE_ONLY),
                ("buckets - none", BUCKETS_ONLY, NO_NUMERIC),
            ):
                names = (AXIS_A[other], AXIS_A[base])
                if all(name in axis_wide for name in names):
                    effects[label] = float(
                        (axis_wide[names[0]] - axis_wide[names[1]]).mean()
                    )
        written.append(
            noise_against_effect(
                seeds[seeds["mode"] == "buckets"],
                effects,
                title="El ruido de reentrenar contra el tamano de lo que medimos",
                path=figures / "14-embeddings-ruido.png",
            )
        )

    if sweep.empty:
        missing.append("run_embeddings  (para results/embeddings/linear-sweep.csv)")
    else:
        written.append(
            encoding_families(
                sweep,
                title="Todas las codificaciones medidas, con el clasificador fijo",
                path=figures / "15-embeddings-familias.png",
                labels=FAMILY_TITLES,
            )
        )

    contrasts = two_encoding_contrasts(axis, seeds, sweep)
    if not contrasts.empty:
        written.append(
            one_versus_two(
                contrasts,
                title="Sirve una segunda codificacion de la misma columna?",
                path=figures / "16-embeddings-una-vs-dos.png",
            )
        )

    for path in written:
        print(f"  {path}")
    print(f"\n{len(written)} figuras escritas en {figures}/")
    for item in missing:
        print(f"  falta: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
