"""Regenerate every figure the model side draws, from ``results/`` alone.

    .venv/bin/python -m src.model.run_figures
    .venv/bin/python -m src.model.run_figures --only 09-final-roc-pr

Nothing here trains and nothing here calls ``evaluate_on_test``: every number comes
from a record already on disk -- the cross-validation and holdout JSON documents, the
holdout predictions, the saved Transformer weights. A figure whose corridas are not
recorded yet is skipped, with the missing run named, rather than trained on the spot.
"""

from __future__ import annotations

import json

import argparse
from pathlib import Path

import numpy as np

from src.eda.loading import load_dataset
from src.model.baseline import target_of
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    TRAINING,
    RunConfig,
    axis_runs,
    ladder_runs,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.diagnostics import (
    Scored,
    calibration,
    calibration_error,
    cls_attention,
    errors_by_level,
    pr_points,
    ranking_gains,
    roc_points,
)
from src.model import figures as fig
from src.model.experiment import partition
from src.model.model_alias import alias_label
from src.model.protocol import EvaluationResult, FoldScore
from src.model.results import (
    RESULTS_DIR,
    fold_frame,
    load,
    load_predictions,
    summary_frame,
)
from src.model.run_final import explainable, interpretability, rebuild

BAR = "L0"
ERROR_COLUMN = "popularity_phrase"
TEXT_ATTENTION_CONFIG = "L2 text with attention"
"""Text-only, no tabular fields: the config where "where does [CLS] look" is a
question about text tokens. The final winner's own attention slide (drawn by
run_final, 09-final-atencion-cls) answers a different question -- where the deployed
model's attention actually goes, tabular fields included, which is why it shows most
of the mass on popularity_phrase rather than on words."""
ABLATION_BASE = "L4 tabular, numbers affine and bucketed"
ABLATION_MULTI_KNOB_PREFIXES = ("Y ", "K ", "V ", "S ")
"""Same exemption as ``test_every_axis_point_changes_exactly_one_thing_from_the_base``:
these axes build on more than one prior knob, so a paired delta against L4 alone would
not isolate what they measure."""
ABLATION_MULTI_KNOB_NAMES = frozenset({"G tabular con popularity"})
LEARNING_RATE_SWEEP_NAMES = (
    "K lr 3e-5",
    "V d_model 32, 1 capa",  # the 1e-4 point; not repeated as a "K" run
    "K lr 3e-4",
    "K lr 1e-3",
    "K lr 3e-3",
)
SEED_VARIANCE_SEED_RUNS = ("K lr 3e-4", "S ganadora seed 7", "S ganadora seed 99")
SEED_VARIANCE_CONFIG_RUNS = ("V sin dropout", "K lr 3e-4", "V d_model 32, 1 capa")

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
    """El finalista congelado, no el mejor AP de todo lo declarado.

    Antes se ordenaba por AP entre las secciones declaradas, y eso elegía
    ``L0b linear, extracted key only``: la cota diagnóstica que recibe la frase de
    popularidad ya extraída a mano. Es la de mayor AP del archivo por construcción, y
    justamente por eso nunca fue candidata. El finalista lo declara ``FINALISTS`` antes
    de abrir el holdout, y es el único que corresponde dibujar.
    """
    from src.model.eda_contract import FINALISTS

    for name in FINALISTS:
        if name in declared:
            return declared[name]
    summary = summary_frame(directory)
    if summary.empty:
        return None
    eligible = summary[summary["name"].isin(declared)]
    if eligible.empty:
        return None
    ranked = eligible.sort_values("average_precision_mean", ascending=False)
    return declared[ranked.iloc[0]["name"]]

def cached_test(config: RunConfig, directory: str) -> tuple | None:
    """El holdout guardado para esta configuración, buscado por CONFIGURACIÓN.

    Buscar por digest deja de funcionar apenas cambia la fórmula que lo calcula, y
    eso ya pasó: un commit sacó el nombre de sección del hash -- un arreglo correcto,
    porque evitaba entrenar la misma red dos veces con dos nombres -- y con eso los
    archivos guardados quedaron con nombres que la fórmula nueva no reproduce. Los
    resultados están intactos; lo que se rompió es el índice.

    Comparar la configuración almacenada contra la pedida no depende de ninguna
    fórmula de hash, así que encuentra la corrida con cualquiera de las dos.
    """
    result = load(config, directory)
    predicted = load_predictions(config, directory)
    if result is not None and predicted is not None:
        return result, predicted

    from dataclasses import asdict

    wanted = {k: list(v) if isinstance(v, tuple) else v
              for k, v in asdict(config).items() if k != "name"}
    for path in sorted(Path(directory).glob("*.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        if "config" not in stored or "folds" not in stored:
            continue  # comparison.json y otros resumenes no son corridas
        if {k: v for k, v in stored["config"].items() if k != "name"} != wanted:
            continue
        npy = path.with_suffix(".predictions.npy")
        if not npy.exists():
            continue
        return (
            EvaluationResult(
                name=stored["name"],
                folds=tuple(
                    FoldScore(
                        fold_index=f["fold_index"], roc_auc=f["roc_auc"],
                        average_precision=f["average_precision"],
                        n_train=f["n_train"], n_scored=f["n_scored"], seconds=f["seconds"],
                    )
                    for f in stored["folds"]
                ),
            ),
            np.load(npy),
        )
    return None

def main(argv: list[str] | None = None) -> int:
    utf8_console()
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
            scores.append(Scored(alias_label(winner.name), actual, np.asarray(cached[1], dtype=float)))

    if bar_config is not None and (winner is None or bar_config.name != winner.name):
        cached = cached_test(bar_config, final_dir)
        if cached is not None:
            scores.append(Scored(alias_label(bar_config.name), actual, np.asarray(cached[1], dtype=float)))

    # A -- el lineal que compitio contra C* -- no se dibuja aca por decision explicita:
    # estas figuras muestran C* contra la cota A*, y la comparacion entre finalistas
    # vive en ``final/comparison.json``. Para devolverlo, leer sus predicciones de
    # ``final/linear-predictions.npz`` y sumarlo a ``scores`` antes de la cota.

    # A*, la cota diagnostica: la logistica sin texto crudo pero con
    # ``popularity_phrase`` extraida a mano. No compitio y nada se eligio con ella. Va
    # ultima para que ``scores[0]`` siga siendo el finalista.
    techo = next((run for name, run in declared.items() if name.startswith("L0b")), None)
    if techo is not None:
        cached = cached_test(techo, final_dir)
        if cached is None:
            print(
                "  [aviso] A* no tiene holdout registrado "
                "(correr src.model.run_ceiling_holdout); las 09 salen sin la cota"
            )
        else:
            scores.append(
                Scored(alias_label(techo.name), actual, np.asarray(cached[1], dtype=float))
            )

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
            tablas = [
                (s.name, calibration(s), calibration_error(calibration(s))) for s in scores
            ]
            path = fig.calibration(
                tablas,
                title="Calibracion en el test: BTR predicho contra observado",
                path=figures_dir / "09-final-calibracion.png",
                annotate=alias_label(techo.name) if techo is not None else None,
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

    if wanted("07-escalera-cascada", args.only):
        ladder_names = ladder_runs(declared)
        ladder_summary = summary_frame(results_dir)
        rows = ladder_summary[ladder_summary["name"].isin(ladder_names)]
        missing_rungs = sorted(set(ladder_names) - set(rows["name"]))
        if missing_rungs:
            report_missing("07-escalera-cascada", [f"faltan los peldaños: {missing_rungs}"])
        else:
            path = fig.ladder_waterfall(
                rows,
                title="La escalera: L0 a L4, con el AP de la barra lineal marcado",
                path=figures_dir / "07-escalera-cascada.png",
            )
            report("07-escalera-cascada", path)

    if wanted("07-ablacion-bosque", args.only):
        variant_names = [
            name
            for name in axis_runs(declared)
            if not name.startswith(ABLATION_MULTI_KNOB_PREFIXES)
            and name not in ABLATION_MULTI_KNOB_NAMES
        ]
        if ABLATION_BASE not in declared:
            report_missing("07-ablacion-bosque", [f"no está declarada la base {ABLATION_BASE!r}"])
        else:
            folds = fold_frame(results_dir)
            recorded_names = set(folds["name"]) if not folds.empty else set()
            if ABLATION_BASE not in recorded_names:
                report_missing(
                    "07-ablacion-bosque",
                    [f"falta la corrida de la base {ABLATION_BASE!r} (correr run_ladder)"],
                )
            else:
                recorded_variants = [name for name in variant_names if name in recorded_names]
                not_yet_run = sorted(set(variant_names) - set(recorded_variants))
                if not_yet_run:
                    print(f"  [07-ablacion-bosque] sin registrar todavia: {not_yet_run}")
                if not recorded_variants:
                    report_missing(
                        "07-ablacion-bosque",
                        ["ninguna variante de la ablacion esta registrada todavia"],
                    )
                else:
                    path = fig.ablation_forest(
                        folds,
                        ABLATION_BASE,
                        recorded_variants,
                        title=f"Deltas pareados de AP contra [{ABLATION_BASE}] (IC 95%, 5 folds)",
                        path=figures_dir / "07-ablacion-bosque.png",
                    )
                    report("07-ablacion-bosque", path)

    if wanted("07-tasa-aprendizaje", args.only):
        undeclared = [name for name in LEARNING_RATE_SWEEP_NAMES if name not in declared]
        if undeclared:
            report_missing("07-tasa-aprendizaje", [f"no declaradas en parameters.txt: {undeclared}"])
        else:
            folds = fold_frame(results_dir)
            recorded_names = set(folds["name"]) if not folds.empty else set()
            present = [name for name in LEARNING_RATE_SWEEP_NAMES if name in recorded_names]
            missing = sorted(set(LEARNING_RATE_SWEEP_NAMES) - set(present))
            if missing:
                print(f"  [07-tasa-aprendizaje] sin registrar todavia: {missing}")
            if not present:
                report_missing("07-tasa-aprendizaje", ["ningun punto del barrido esta registrado"])
            else:
                path = fig.learning_rate_sweep(
                    folds,
                    present,
                    epoch_ceiling=TRAINING.epochs,
                    title="Barrido de learning rate: AP y best_epoch por fold",
                    path=figures_dir / "07-tasa-aprendizaje.png",
                )
                report("07-tasa-aprendizaje", path)

    if wanted("07-varianza-semilla", args.only):
        wanted_names = set(SEED_VARIANCE_SEED_RUNS) | set(SEED_VARIANCE_CONFIG_RUNS)
        undeclared = sorted(name for name in wanted_names if name not in declared)
        if undeclared:
            report_missing("07-varianza-semilla", [f"no declaradas en parameters.txt: {undeclared}"])
        else:
            seed_summary = summary_frame(results_dir)
            recorded_names = set(seed_summary["name"]) if not seed_summary.empty else set()
            missing = sorted(wanted_names - recorded_names)
            if missing:
                report_missing("07-varianza-semilla", [f"sin registrar: {missing}"])
            else:
                path = fig.seed_variance(
                    seed_summary,
                    list(SEED_VARIANCE_SEED_RUNS),
                    list(SEED_VARIANCE_CONFIG_RUNS),
                    seed_row_label="K lr 3e-4 (semillas 1337 / 7 / 99)",
                    config_row_label="V sin dropout / K lr 3e-4 / V d_model 32, 1 capa",
                    title="Varianza por semilla contra varianza entre configuraciones",
                    path=figures_dir / "07-varianza-semilla.png",
                )
                report("07-varianza-semilla", path)

    if wanted("07-atencion-l2-texto", args.only):
        if TEXT_ATTENTION_CONFIG not in declared:
            report_missing(
                "07-atencion-l2-texto", [f"no declarada en parameters.txt: {TEXT_ATTENTION_CONFIG!r}"]
            )
        else:
            text_config = declared[TEXT_ATTENTION_CONFIG]
            fold0 = partitions.folds[0]
            model, encoder = rebuild(
                text_config, frame, fold0.train_indices, 0, results_dir
            )
            if model is None:
                report_missing(
                    "07-atencion-l2-texto",
                    [f"faltan los pesos guardados del fold 0 de {TEXT_ATTENTION_CONFIG!r} "
                     "(correr run_ladder con save_weights)"],
                )
            else:
                attention = cls_attention(model, encoder, frame, fold0.validation_indices)
                if attention.empty:
                    report_missing(
                        "07-atencion-l2-texto", [f"{TEXT_ATTENTION_CONFIG!r} no tiene bloques de atencion"]
                    )
                else:
                    path = fig.attention_by_group(
                        attention,
                        title=(
                            f"Atencion del [CLS] sobre texto puro ([{TEXT_ATTENTION_CONFIG}], "
                            "validacion del fold 0)"
                        ),
                        path=figures_dir / "07-atencion-l2-texto.png",
                    )
                    report("07-atencion-l2-texto", path)

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
