"""The four pre-holdout decision charts of the Ejercicio 2 flow.

    .venv/bin/python -m scripts.run_eda_contract_figures \
        --results results/v1-una-torre/eda-contract --figures figures/eda-contract

All four charts use recorded evidence. The architecture path and greedy neighbourhood
are reconstructed from the real five-fold run records when the summary JSONs predate
the richer ``stages``/``all_probes`` schema. No holdout result is read or plotted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import figures as fig
from src.model.representation_selection import compare_folds, paired_margin
from src.model.results import RESULTS_DIR, fold_frame, summary_frame
from scripts.run_embeddings import read_sweep

LADDER_ORDER = [
    "L0a linear, no text",
    "L0* swept-best linear",
    "L1 learned embeddings, no attention",
    "L2 learned embeddings with attention",
    "L0b linear, extracted key only",
]
"""El peldano lineal es ``L0*`` y no ``L0``: misma logistica, pero con la
representacion que eligio el barrido (``embeddings/selection.json``)."""
FLOOR = "L0a linear, no text"
CEILING = "L0b linear, extracted key only"
SWEPT = "L0* swept-best linear"
SWEPT_REPRESENTATION = ("tf-idf", "one-hot", "piecewise-linear")
BRACKET_RESULTS = Path("iteracion-bracket/resultados")


def _config_key(fields: dict) -> str:
    """La configuracion sin ``name`` ni ``seed``: identifica una arquitectura.

    El bracket regrabo cada corrida con su nombre canonico, asi que buscar ``L2`` por
    nombre encuentra una sola de sus tres semillas y por configuracion las tres.
    """
    return json.dumps(
        {
            key: (list(value) if isinstance(value, (list, tuple)) else value)
            for key, value in fields.items()
            if key not in ("name", "seed")
        },
        sort_keys=True,
    )


def _seed_means(key: str, directories) -> dict[int, float]:
    """AP medio sobre los folds, una entrada por semilla entrenada."""
    found: dict[int, float] = {}
    for directory in directories:
        for archivo in sorted(Path(directory).glob("*.json")):
            document = json.loads(archivo.read_text(encoding="utf-8"))
            if "config" not in document or "folds" not in document:
                continue
            if _config_key(document["config"]) != key:
                continue
            folds = [fold["average_precision"] for fold in document["folds"]]
            found[int(document["config"].get("seed", 0))] = float(np.mean(folds))
    return found


def _ladder_frame(results: str) -> pd.DataFrame:
    """La escalera con la misma estadistica que decidio todo lo demas.

    Cada peldano con semilla es la media de las medias por semilla, y su dispersion es
    la de entre semillas, no la de entre folds. Los lineales no tienen semilla, asi que
    la suya queda en ``NaN``: mezclar las dos dispersiones en un mismo eje las vuelve
    incomparables.
    """
    from src.model.configs import load_parameters

    declared = load_parameters("parameters-eda.txt")
    directories = [Path(results)] + ([BRACKET_RESULTS] if BRACKET_RESULTS.exists() else [])

    rows = []
    for name in LADDER_ORDER:
        if name == SWEPT:
            sweep = read_sweep(results)
            text, categorical, numeric = SWEPT_REPRESENTATION
            chosen = sweep[
                (sweep["text_representation"] == text)
                & (sweep["categorical_representation"] == categorical)
                & (sweep["numeric_representation"] == numeric)
            ]
            if chosen.empty:
                raise RuntimeError(f"{SWEPT}: falta {SWEPT_REPRESENTATION} en linear-sweep.csv")
            rows.append({
                "name": name,
                "average_precision_mean": float(chosen["average_precision"].mean()),
                "average_precision_std": float("nan"),
                "semillas": 0,
            })
            continue

        config = declared[name]
        key = _config_key(asdict(config))
        seeds = _seed_means(key, directories)
        if not seeds:
            raise RuntimeError(f"{name}: ninguna corrida registrada con esa configuracion")
        values = np.asarray(list(seeds.values()), dtype=float)
        # La logistica es convexa: el registro trae un seed pero no lo usa.
        repeticiones = 0 if config.model == "logistic" else len(values)
        rows.append({
            "name": name,
            "average_precision_mean": float(values.mean()),
            "average_precision_std": (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            ),
            "semillas": repeticiones,
        })
    return pd.DataFrame(rows)


def _ladder_label(name: str) -> str:
    """``alias_label``, salvo la cota superior: sin leyenda donde aclarar que ``*`` es
    "sin texto crudo, con la clave extraida a mano", la barra lo lleva escrito."""
    from src.model.model_alias import alias_label

    if name == CEILING:
        return "A* (sin texto + clave extraída)"
    return alias_label(name)


def chart_2_ladder(results: str, figures: Path) -> Path | None:
    """Chart 2: the text ladder between its floor and ceiling, real data from Task 4."""
    try:
        ladder = _ladder_frame(results)
    except (RuntimeError, KeyError) as problem:
        print(f"  [2] {problem}")
        return None

    for _, row in ladder.iterrows():
        semillas = int(row["semillas"])
        cuantas = f"{semillas} semillas" if semillas else "deterministico"
        print(f"       {row['name']:<38s} {row['average_precision_mean']:.4f}  ({cuantas})")
    path = fig.eda_ladder_waterfall(
        ladder,
        order=LADDER_ORDER,
        floor=FLOOR,
        ceiling=CEILING,
        label_fn=_ladder_label,
        # The report's recovery formula is specifically about L2 (the attention
        # Transformer), not whichever rung happens to score highest -- L0, the
        # plain linear model, currently scores higher than L1/L2 on this dataset,
        # and annotating *that* recovery would misrepresent what the ladder is for.
        recover_for="L2 learned embeddings with attention",
        title="2. La escalera del texto, entre piso (sin texto) y techo (clave extraida)",
        path=figures / "02-escalera.png",
    )
    print(f"  [2] {path}")
    return path


FAMILY_TITLES = {
    "text": "Texto: bolsa binaria vs tf-idf",
    "categorical": "Categoricas: one-hot vs target encoding",
    "numeric": "Numerica: affine vs buckets vs piecewise vs periodic",
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures", type=str, default="figures/eda-contract")
    return parser.parse_args(argv)


def chart_1_representations(results: str, figures: Path) -> Path | None:
    """Chart 1: which encoding won in each family, real data from Task 3's sweep."""
    sweep = read_sweep(results)
    if sweep.empty:
        print(
            "  [1] falta results/v1-una-torre/eda-contract/embeddings/linear-sweep.csv "
            "(correr scripts.run_embeddings --results <resultados>)"
        )
        return None
    path = fig.encoding_families(
        sweep,
        title="1. Como representar cada columna (barrido con el conjunto completo)",
        path=figures / "01-representaciones.png",
        labels=FAMILY_TITLES,
    )
    print(f"  [1] {path}")
    return path


AXIS_LABEL = {
    "numeric_embedding": "embedding numerico",
    "depth": "profundidad",
    "d_model": "ancho",
    "n_heads": "heads",
    "modules": "posicion / pooling / dropout",
}

ARCHITECTURE_STAGES = [
    {
        "stage": "embedding numerico",
        "base": ("L2 learned embeddings with attention", "affine+buckets (base)"),
        "candidates": [
            ("A numeric affine", "affine"),
            ("A numeric buckets", "buckets"),
            ("A numeric piecewise", "piecewise"),
        ],
        "selected": "buckets",
    },
    {
        "stage": "profundidad",
        "base": ("A numeric buckets", "1 capa (base)"),
        "candidates": [("B depth 2", "2 capas"), ("B depth 3", "3 capas")],
        "selected": "1 capa (base)",
    },
    {
        "stage": "ancho",
        "base": ("A numeric buckets", "64 (base)"),
        "candidates": [("D d_model 32", "32"), ("D d_model 96", "96")],
        "selected": "64 (base)",
    },
    {
        "stage": "heads",
        "base": ("A numeric buckets", "4 (base)"),
        "candidates": [("C 2 heads", "2"), ("C 8 heads", "8")],
        "selected": "4 (base)",
    },
    {
        "stage": "modulos finales",
        "base": ("A numeric buckets", "base"),
        "candidates": [
            ("E learned positional", "positional aprendido"),
            ("F attention pooling", "attention pooling"),
        ],
        "selected": "base",
    },
]

GREEDY_PROBES = [
    ("V depth 2 d_model 32", "capa 2 · ancho 32"),
    ("V depth 2 d_model 96", "capa 2 · ancho 96"),
    ("V depth 2 heads 2", "capa 2 · 2 heads"),
    ("V depth 2 heads 8", "capa 2 · 8 heads"),
    ("B depth 2", "vecino · 2 capas"),
    ("B depth 3", "vecino · 3 capas"),
    ("D d_model 32", "vecino · ancho 32"),
    ("D d_model 96", "vecino · ancho 96"),
    ("C 2 heads", "vecino · 2 heads"),
    ("C 8 heads", "vecino · 8 heads"),
]


def _summary_ap(summary: pd.DataFrame, name: str) -> float:
    rows = summary[summary["name"] == name]
    if rows.empty:
        raise RuntimeError(f"falta la corrida real {name!r}")
    return float(rows["average_precision_mean"].iloc[-1])


def _fold_ap(folds: pd.DataFrame, name: str) -> np.ndarray:
    rows = folds[folds["name"] == name].sort_values("fold_index")
    if len(rows) != 5 or set(rows["fold_index"]) != set(range(5)):
        raise RuntimeError(f"{name!r} no tiene exactamente cinco folds reales")
    return rows["average_precision"].to_numpy(dtype=float)


def _recorded_stages(results: str) -> list[dict]:
    summary = summary_frame(results)
    folds = fold_frame(results)
    drawn: list[dict] = []
    for stage in ARCHITECTURE_STAGES:
        base_name, base_label = stage["base"]
        base_ap = _fold_ap(folds, base_name)
        points = [{"label": base_label, "ap": _summary_ap(summary, base_name), "outcome": "base"}]
        for candidate_name, label in stage["candidates"]:
            candidate_ap = _fold_ap(folds, candidate_name)
            points.append(
                {
                    "label": label,
                    "ap": _summary_ap(summary, candidate_name),
                    "outcome": compare_folds(base_ap, candidate_ap),
                }
            )
        drawn.append(
            {"stage": stage["stage"], "points": points, "selected": stage["selected"]}
        )
    return drawn


def _recorded_greedy_moves(results: str) -> pd.DataFrame:
    folds = fold_frame(results)
    selected = _fold_ap(folds, "M selected from directed comparisons")
    rows = []
    for name, label in GREEDY_PROBES:
        delta, low, high = paired_margin(_fold_ap(folds, name) - selected)
        rows.append({"name": label, "delta": delta, "low": low, "high": high})
    return pd.DataFrame(rows)


def _stages_from_selection(document: dict) -> list[dict]:
    """Convert run_architecture.py's persisted ``stages`` into chart records."""
    drawn = []
    for stage in document["stages"]:
        base = stage["base"]
        base_label = f"{base['name']} (base)"
        points = [{"label": base_label, "ap": base["ap_mean"], "outcome": "base"}]
        selected_label = base_label
        for candidate in stage["candidates"]:
            label = candidate["name"]
            points.append(
                {
                    "label": label,
                    "ap": candidate["ap_mean"],
                    "outcome": candidate["outcome"],
                }
            )
            if candidate["outcome"] in ("improves", "tie-break"):
                selected_label = label
        drawn.append(
            {
                "stage": AXIS_LABEL.get(stage["axis"], stage["axis"]),
                "points": points,
                "selected": selected_label,
            }
        )
    return drawn


AXIS_TITLE = {
    "numeric_embedding": "embedding numerico", "n_layers": "profundidad",
    "d_model": "ancho", "n_heads": "heads", "learning_rate": "learning rate",
}
AXIS_ORDER = ("numeric_embedding", "n_layers", "d_model", "n_heads", "learning_rate")

PANEL_TITLE = {
    "dropout": "dropout",
    # "pooling por atencion" a secas se confunde con los bloques de autoatencion,
    # que son otro mecanismo y apuntan al lado contrario: los bloques aportan
    # +0.018 y este pooling resta 0.002. Se nombra por lo que hace.
    "pooling": "pooling: promedio vs ponderado",
    "positional": "codificacion posicional",
}


def _seed_index(results: str) -> dict[str, dict[int, float]]:
    """Configuración (sin nombre ni semilla) -> {semilla: AP medio de sus folds}.

    Guarda el AP de **cada** semilla y no un desvío ya agregado, porque el error de
    una diferencia hay que calcularlo pareando: la misma semilla de un lado y del
    otro. El desvío del valor absoluto no sirve para eso.
    """
    import glob
    from collections import defaultdict

    por_arq: dict[str, dict[int, float]] = defaultdict(dict)
    for archivo in glob.glob(str(Path(results) / "*.json")):
        doc = json.loads(Path(archivo).read_text(encoding="utf-8"))
        if "config" not in doc or "folds" not in doc:
            continue
        clave = json.dumps(
            {k: (list(v) if isinstance(v, list) else v)
             for k, v in doc["config"].items() if k not in ("name", "seed")},
            sort_keys=True,
        )
        por_arq[clave][int(doc["config"]["seed"])] = float(
            np.mean([f["average_precision"] for f in doc["folds"]])
        )
    return dict(por_arq)


def paired_delta(
    candidato: dict[int, float], base: dict[int, float]
) -> tuple[float, float]:
    """Media y desvío de la diferencia, pareada semilla por semilla.

    La base contra sí misma da exactamente ``(0.0, 0.0)``: no hay incertidumbre en
    comparar algo consigo mismo, y dibujarle una barra de error sugeriría que sí.
    """
    comunes = sorted(set(candidato) & set(base))
    if not comunes:
        return 0.0, 0.0
    deltas = np.array([candidato[s] - base[s] for s in comunes], dtype=float)
    desvio = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    return float(deltas.mean()), desvio


def _clave(config) -> str:
    from dataclasses import asdict

    return json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in asdict(config).items() if k not in ("name", "seed")},
        sort_keys=True,
    )


def _bracket_stages(results: str) -> tuple[list[dict], dict]:
    """Las etapas del bracket con su AP y su desvío entre semillas.

    Devuelve la forma que comparten ``architecture_path`` y ``architecture_grid``.
    Cada punto se reconstruye contra la base vigente **en su etapa**: cuando se
    recorrió la profundidad el ancho todavía era 64, así que buscar los puntos contra
    la configuración final no los encontraría.
    """
    from dataclasses import replace

    from src.model.configs import load_parameters

    search_path = Path(results) / "architecture" / "bracket-search.json"
    if not search_path.exists():
        raise RuntimeError(f"falta {search_path}: correr scripts.run_bracket_search")
    search = json.loads(search_path.read_text(encoding="utf-8"))
    spreads = _seed_index(results)

    declared = load_parameters("parameters-eda.txt")
    current = next(run for name, run in declared.items() if name.startswith("L2"))

    def tipado(axis: str, value: str):
        if axis == "numeric_embedding":
            return value
        return float(value) if axis == "learning_rate" else int(value)

    stages: list[dict] = []
    for axis in AXIS_ORDER:
        block = search[axis]
        elegido = str(block["selected"])
        points = []
        for value, ap in block["evaluated"].items():
            probe = replace(current, **{axis: tipado(axis, value)})
            semillas = spreads.get(_clave(probe), {})
            mismo = (value == elegido if axis == "numeric_embedding"
                     else float(value) == float(elegido))
            etiqueta = f"{float(value):g}" if axis == "learning_rate" else str(value)
            points.append({"label": etiqueta, "ap": float(ap), "ap_std": 0.0,
                           "seeds": semillas,
                           "outcome": "improves" if mismo else "inconclusive"})
        points.sort(key=lambda p: p["ap"])
        stages.append({
            "stage": AXIS_TITLE[axis], "points": points,
            "selected": (f"{float(elegido):g}" if axis == "learning_rate" else elegido),
            "base": (f"{float(getattr(current, axis)):g}" if axis == "learning_rate"
                     else str(getattr(current, axis))),
        })
        current = replace(current, **{axis: tipado(axis, elegido)})

    # los modulos: un solo cambio contra la arquitectura ya resuelta.
    # ``current`` quedo con los cinco ejes aplicados, que es justo la base contra la
    # que se probaron, asi que sus semillas se buscan reconstruyendo cada variante.
    modules = search["modules"]
    base_ap = modules["positional"]["base"]
    base_seeds = spreads.get(_clave(current), {})
    for field, valor, base_label, alt_label in (
        ("dropout", 0.3, "0.1 (base)", "0.3"),
        ("pooling", "attention", "promedio (base)", "ponderado aprendido"),
        ("positional", "learned", "ninguno (base)", "aprendido"),
    ):
        alt = modules[field]["alternativa"]
        gana = alt > base_ap
        alt_seeds = spreads.get(_clave(replace(current, **{field: valor})), {})
        stages.append({
            "stage": PANEL_TITLE.get(field, field), "base": base_label,
            "selected": alt_label if gana else base_label,
            "points": [
                {"label": base_label, "ap": base_ap, "ap_std": 0.0,
                 "seeds": base_seeds, "outcome": "base" if gana else "improves"},
                {"label": alt_label, "ap": alt, "ap_std": 0.0,
                 "seeds": alt_seeds,
                 "outcome": "improves" if gana else "inconclusive"},
            ],
        })

    # la autoatencion aislada: la misma red con cero bloques
    ablation_path = Path(results) / "architecture" / "attention-ablation.json"
    if ablation_path.exists():
        ab = json.loads(ablation_path.read_text(encoding="utf-8"))
        elegida = replace(current, dropout=0.3) if modules["dropout"]["alternativa"] > base_ap else current
        stages.append({
            "stage": "bloques de autoatencion", "base": "0 bloques",
            "selected": f"{ab['con_atencion']['n_layers']} bloques",
            "points": [
                {"label": "0 bloques", "ap": ab["sin_atencion"]["ap"], "ap_std": 0.0,
                 "seeds": spreads.get(_clave(replace(elegida, n_layers=0)), {}),
                 "outcome": "base"},
                {"label": f"{ab['con_atencion']['n_layers']} bloques",
                 "ap": ab["con_atencion"]["ap"], "ap_std": 0.0,
                 "seeds": spreads.get(_clave(elegida), {}),
                 "outcome": "improves"},
            ],
        })

    return stages, search


def chart_3_architecture_path(results: str, figures: Path) -> Path:
    """Chart 3: el recorrido por bracket como camino, un eje por fila.

    Lee ``bracket-search.json`` y no ``selection.json``: el recorrido dirigido que
    escribía el segundo se reemplazó por el bracket adaptativo con tres semillas por
    configuración, que además agregó el eje de ``learning_rate``.
    """
    stages, search = _bracket_stages(results)
    print(f"  [3] bracket adaptativo, {len(stages)} ejes, "
          f"{search['cost']['entrenadas']} corridas entrenadas y "
          f"{search['cost']['reusadas']} reusadas")
    path = fig.architecture_path(
        stages,
        title="3. El recorrido por bracket: cada eje cerro con optimo interior",
        path=figures / "03-arquitectura.png",
    )
    print(f"  [3] {path}")
    return path


def chart_3b_architecture_grid(results: str, figures: Path) -> Path:
    """Chart 3b: todas las perillas, un panel por eje, en AP medio ± desvío entre semillas.

    El valor es el promedio de las tres semillas y la barra su desvío: el mismo
    estadístico con el que se tomaron todas las decisiones del recorrido.

    ``weight_decay`` no tiene panel: se midió en el protocolo anterior y sobre otra
    arquitectura, así que ponerlo al lado sugeriría una comparación que no se hizo.
    """
    stages, _ = _bracket_stages(results)
    listos = []
    for stage in stages:
        points = []
        for point in stage["points"]:
            semillas = list(point.get("seeds", {}).values())
            media = float(np.mean(semillas)) if semillas else point["ap"]
            desvio = float(np.std(semillas, ddof=1)) if len(semillas) > 1 else 0.0
            points.append({**point, "ap": media, "ap_std": desvio})
        listos.append({**stage, "points": points})

    movidos = [s["stage"] for s in stages if s["selected"] != s["base"]]
    print(f"  [3b] {len(stages)} perillas; se movieron: {', '.join(movidos) or 'ninguna'}")
    path = fig.architecture_grid(
        listos,
        title="3b. Cada perilla en su escala: AP medio y desvio entre las tres semillas",
        path=figures / "03b-arquitectura-barras.png",
        value_label="AP (media de 3 semillas)",
    )
    print(f"  [3b] {path}")
    return path


def chart_4_mechanisms(results: str, figures: Path) -> Path:
    """Chart 4: cuánto aporta cada mecanismo del Transformer, con la atención aislada.

    Reemplaza a la validación del recorrido greedy, que existía para comprobar que el
    orden del descenso por coordenadas no decidiera cuando los efectos eran del tamaño
    del ruido de una sola semilla. Promediando tres semillas ese ruido baja lo
    suficiente como para que la comparación directa sea informativa, así que la
    pregunta útil pasa a ser cuál de los mecanismos aporta y cuál no.

    La fila que importa es la autoatención: la arquitectura elegida contra esa misma
    arquitectura con ``n_layers = 0``, un solo campo de diferencia.
    """
    ablation_path = Path(results) / "architecture" / "attention-ablation.json"
    search_path = Path(results) / "architecture" / "bracket-search.json"
    for needed in (ablation_path, search_path):
        if not needed.exists():
            raise RuntimeError(f"falta {needed}")
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    modules = json.loads(search_path.read_text(encoding="utf-8"))["modules"]

    base = modules["positional"]["base"]
    filas = [{"name": "autoatencion (2 bloques vs 0)", "delta": ablation["delta"]}]
    for field, etiqueta in (
        ("dropout", "dropout 0.3 vs 0.1"),
        ("pooling", "pooling atencion vs promedio"),
        ("positional", "positional aprendido vs ninguno"),
    ):
        filas.append({"name": etiqueta, "delta": modules[field]["alternativa"] - base})
    for fila in filas:
        fila["low"] = fila["delta"]
        fila["high"] = fila["delta"]

    adoptados = [f["name"] for f in filas if f["delta"] > 0]
    print(f"  [4] {len(filas)} mecanismos; aportan: {', '.join(adoptados) or 'ninguno'}")
    path = fig.greedy_neighbourhood_forest(
        pd.DataFrame(filas).sort_values("delta"),
        selected_name="la arquitectura elegida",
        title="4. Que aporta cada mecanismo, sobre tres semillas",
        path=figures / "04-mecanismos.png",
    )
    print(f"  [4] {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figures = Path(args.figures)

    print("=== 1-2: datos reales ===")
    chart_1_representations(args.results, figures)
    chart_2_ladder(args.results, figures)

    print("\n=== 3-4: el recorrido por bracket y sus mecanismos ===")
    chart_3_architecture_path(args.results, figures)
    chart_3b_architecture_grid(args.results, figures)
    chart_4_mechanisms(args.results, figures)

    print(f"\nfiguras en {figures}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
