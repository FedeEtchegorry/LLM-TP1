"""The five decision charts of the Ejercicio 2 flow, drawn from ``results/eda-contract/``.

    .venv/bin/python -m src.model.run_eda_contract_figures \
        --results results/eda-contract --figures figures/eda-contract

Charts 1 and 2 (representation selection, and the L0a/L0/L1/L2/L0b ladder) are drawn
from real recorded runs and fail loudly if that evidence is missing. Charts 3-5
(architecture path, greedy-validation neighbourhood, final holdout comparison) need
evidence this repository has not produced yet -- the full architecture search, its
greedy-order validation, and an opened holdout -- so when their real inputs are
missing this script draws them from a clearly-labelled synthetic example instead of
skipping them, so the chart code itself is exercised and reviewable before the real
runs exist. A real run automatically takes over: once
``results/eda-contract/architecture/selection.json`` (etc.) exists, this script reads
it instead of the example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import figures as fig
from src.model.model_alias import alias_label, variant_label
from src.model.results import RESULTS_DIR, summary_frame
from src.model.run_architecture import DEPTHS, HEADS, WIDTHS
from src.model.run_embeddings import read_sweep

LADDER_ORDER = [
    "L0a linear, no text",
    "L0 linear raw EDA",
    "L1 learned embeddings, no attention",
    "L2 learned embeddings with attention",
    "L0b linear, extracted key only",
]
FLOOR = "L0a linear, no text"
CEILING = "L0b linear, extracted key only"

FAMILY_TITLES = {
    "text": "Texto",
    "categorical": "Categoricas",
    "numeric": "Numerica",
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
        print("  [1] falta results/eda-contract/embeddings/linear-sweep.csv "
              "(correr src.model.run_embeddings --results <resultados>)")
        return None
    path = fig.encoding_families(
        sweep,
        title="1. Como representar cada columna, con regresion logistica (familia de Modelo A)",
        path=figures / "01-representaciones.png",
        labels=FAMILY_TITLES,
    )
    print(f"  [1] {path}")
    return path


def chart_2_ladder(results: str, figures: Path) -> Path | None:
    """Chart 2: the text ladder between its floor and ceiling, real data from Task 4."""
    summary = summary_frame(results)
    if summary.empty:
        print("  [2] no hay corridas registradas en results/eda-contract/ "
              "(correr src.model.run_ladder --parameters parameters-eda.txt)")
        return None
    missing = [name for name in LADDER_ORDER if name not in set(summary["name"])]
    if missing:
        print(f"  [2] faltan corridas de la escalera: {missing}")
        return None
    path = fig.eda_ladder_waterfall(
        summary,
        order=LADDER_ORDER,
        floor=FLOOR,
        ceiling=CEILING,
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


# ---------------------------------------------------------------------------
# Example fixtures for charts 3-5: used only when the real evidence they need is
# not on disk yet. Kept in this file (not in tests) so the same numbers back both
# the smoke test and the "-EJEMPLO" figure a reader might open by mistake -- the
# filename and the title both say so, and the console prints a warning either way.
# ---------------------------------------------------------------------------

EXAMPLE_ARCHITECTURE_STAGES = [
    {
        "stage": "embedding numerico",
        "points": [
            {"label": "affine+buckets (base)", "ap": 0.752, "outcome": "base"},
            {"label": "affine", "ap": 0.744, "outcome": "loses"},
            {"label": "buckets", "ap": 0.748, "outcome": "inconclusive"},
            {"label": "piecewise", "ap": 0.751, "outcome": "inconclusive"},
        ],
        "selected": "affine+buckets (base)",
    },
    {
        "stage": "profundidad",
        "points": [
            {"label": "1 capa (base)", "ap": 0.752, "outcome": "base"},
            {"label": "2 capas", "ap": 0.768, "outcome": "improves"},
            {"label": "3 capas", "ap": 0.769, "outcome": "inconclusive"},
        ],
        "selected": "2 capas",
    },
    {
        "stage": "ancho",
        "points": [
            {"label": "64 (base)", "ap": 0.768, "outcome": "base"},
            {"label": "32", "ap": 0.760, "outcome": "loses"},
            {"label": "96", "ap": 0.771, "outcome": "tie-break"},
        ],
        "selected": "96",
    },
    {
        "stage": "heads",
        "points": [
            {"label": "4 (base)", "ap": 0.771, "outcome": "base"},
            {"label": "2", "ap": 0.766, "outcome": "loses"},
            {"label": "8", "ap": 0.774, "outcome": "improves"},
        ],
        "selected": "8",
    },
]

EXAMPLE_GREEDY_MOVES = pd.DataFrame(
    [
        {"name": "V anchor depth 1 (d_model 32)", "delta": -0.006, "low": -0.014, "high": 0.002},
        {"name": "V anchor depth 1 (d_model 96)", "delta": -0.002, "low": -0.010, "high": 0.006},
        {"name": "V anchor depth 1 (2 heads)", "delta": -0.009, "low": -0.017, "high": -0.001},
        {"name": "V anchor depth 1 (8 heads)", "delta": 0.001, "low": -0.006, "high": 0.008},
        {"name": "V neighbour n_layers 1", "delta": -0.006, "low": -0.013, "high": 0.001},
        {"name": "V neighbour n_layers 3", "delta": 0.001, "low": -0.005, "high": 0.007},
        {"name": "V neighbour d_model 32", "delta": -0.011, "low": -0.018, "high": -0.004},
        {"name": "V neighbour d_model 64", "delta": -0.003, "low": -0.009, "high": 0.003},
        {"name": "V neighbour n_heads 2", "delta": -0.009, "low": -0.017, "high": -0.001},
        {"name": "V neighbour n_heads 4", "delta": -0.003, "low": -0.010, "high": 0.004},
    ]
)

EXAMPLE_FINAL_ROWS = [
    {"name": "L0 linear raw EDA", "ap": 0.771, "std": 0.019},
    {"name": "M selected from directed comparisons", "ap": 0.774, "std": 0.017},
]


AXIS_LABEL = {
    "numeric_embedding": "embedding numerico",
    "depth": "profundidad",
    "d_model": "ancho",
    "n_heads": "heads",
    "modules": "posicion / pooling / dropout",
}


GRID_AXES = ("numeric_embedding", "depth", "d_model", "n_heads")
"""The four axes chart 3 draws as its own subplot -- module knobs (position,
pooling, dropout) are a separate, later stage and not part of this grid."""

_AXIS_DOMAIN = {"depth": [str(v) for v in DEPTHS], "d_model": [str(v) for v in WIDTHS], "n_heads": [str(v) for v in HEADS]}


def _axis_value(axis: str, name: str) -> str:
    """The bare value a candidate's codename encodes -- no field name, since the
    panel title already says which field this is ("A numeric affine" -> "affine",
    "B depth 2" -> "2", "D d_model 96" -> "96", "C 8 heads" -> "8")."""
    tokens = name.split()
    return tokens[1] if axis == "n_heads" else tokens[-1]


def _stages_from_selection(document: dict, axes: tuple[str, ...] = GRID_AXES) -> list[dict]:
    """Convert run_architecture.py's persisted ``stages`` (base + candidates with
    real AP, per axis) into the ``{"stage", "points", "selected"}`` shape the chart
    functions draw. Restricted to ``axes`` -- by default the four capacity axes,
    excluding the later position/pooling/dropout stage. Every point (base included)
    is labelled with the bare value of that axis's field -- not "Base" and not a
    repeated "campo=" prefix, since the panel title already names the field."""
    drawn = []
    for stage in document["stages"]:
        axis = stage["axis"]
        if axis not in axes:
            continue
        base = stage["base"]
        candidate_values = [_axis_value(axis, candidate["name"]) for candidate in stage["candidates"]]
        if axis == "numeric_embedding":
            base_value = document["numeric_embedding"]["base"]
        elif axis in _AXIS_DOMAIN:
            remaining = [value for value in _AXIS_DOMAIN[axis] if value not in candidate_values]
            base_value = remaining[0] if remaining else base["name"]
        else:
            base_value = base["name"]
        base_label = str(base_value)
        points = [{"label": base_label, "ap": base["ap_mean"], "ap_std": base["ap_std"], "outcome": "base"}]
        selected_label = base_label
        for candidate, value in zip(stage["candidates"], candidate_values):
            points.append({
                "label": value,
                "ap": candidate["ap_mean"],
                "ap_std": candidate["ap_std"],
                "outcome": candidate["outcome"],
            })
            if candidate["outcome"] in ("improves", "tie-break"):
                selected_label = value
        drawn.append({
            "stage": AXIS_LABEL.get(axis, axis),
            "points": points,
            "selected": selected_label,
        })
    return drawn


def chart_3_architecture_path(results: str, figures: Path) -> Path:
    """Chart 3: one subplot per capacity axis (embedding numerico, profundidad,
    ancho, heads), all four in the same figure. Real data needs run_architecture.py
    to have actually run (Task 5) with per-candidate AP persisted; until then, drawn
    from a labelled example."""
    selection_path = Path(results) / "architecture" / "selection.json"
    document = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    if document.get("stages"):
        print(f"  [3] {selection_path} tiene el AP real de cada candidato por etapa")
        stages = _stages_from_selection(document)
        title = "3. La busqueda de arquitectura, eje por eje"
        out = figures / "03-arquitectura.png"
    else:
        if selection_path.exists():
            print(f"  [3] {selection_path} existe, pero no guarda 'stages' (version vieja "
                  "de run_architecture.py) -- usando datos de EJEMPLO")
        else:
            print(f"  [3] falta {selection_path} (correr src.model.run_architecture) "
                  "-- usando datos de EJEMPLO, no reales")
        stages = EXAMPLE_ARCHITECTURE_STAGES
        title = "3. [EJEMPLO, no datos reales] La busqueda de arquitectura, eje por eje"
        out = figures / "03-arquitectura-EJEMPLO.png"
    path = fig.architecture_grid(stages, title=title, path=out)
    print(f"  [3] {path}")
    return path


def chart_4_greedy_neighbourhood(results: str, figures: Path) -> Path:
    """Chart 4: Task 11's neighbourhood probes -- every probe tried, not only the
    ones that would have won. Real data needs run_greedy_validation.py to have
    actually run; until then, a labelled example."""
    greedy_path = Path(results) / "architecture" / "greedy-validation.json"
    document = json.loads(greedy_path.read_text(encoding="utf-8")) if greedy_path.exists() else {}
    all_probes = document.get("layer_1_depth_probe", {}).get("all_probes")
    if all_probes is not None:
        all_probes = all_probes + document["layer_2_neighbourhood"]["all_probes"]
        moves = pd.DataFrame(all_probes)
        moves["name"] = moves["name"].map(variant_label)
        stable = document["stable"]
        print(f"  [4] {greedy_path} tiene las {len(moves)} sondas reales "
              f"({'ninguna superó a M -- estable' if stable else 'al menos una superó a M'})")
        title = ("4. Validacion del recorrido: vecinos de un solo cambio, "
                 f"todas las sondas ({'estable' if stable else 'inestable'})")
        out = figures / "04-validacion-recorrido.png"
    else:
        if greedy_path.exists():
            print(f"  [4] {greedy_path} existe, pero no guarda 'all_probes' (version vieja "
                  "de run_greedy_validation.py) -- usando datos de EJEMPLO")
        else:
            print(f"  [4] falta {greedy_path} (correr src.model.run_greedy_validation) "
                  "-- usando datos de EJEMPLO, no reales")
        moves = EXAMPLE_GREEDY_MOVES
        title = "4. [EJEMPLO, no datos reales] Validacion del recorrido: vecinos de un solo cambio"
        out = figures / "04-validacion-recorrido-EJEMPLO.png"
    path = fig.greedy_neighbourhood_forest(
        moves,
        selected_name=alias_label("M selected from directed comparisons"),
        title=title,
        path=out,
    )
    print(f"  [4] {path}")
    return path


def chart_5_final(results: str, figures: Path) -> Path:
    """Chart 5: L0 vs M on the holdout. Real data needs the holdout to have actually
    been opened (Task 7 Step 6 / run_final.py); until then, a labelled example."""
    final_dir = Path(results) / "final"
    final_summary = summary_frame(final_dir) if final_dir.exists() else pd.DataFrame()
    if not final_summary.empty and set(("L0 linear raw EDA", "M selected from directed comparisons")) <= set(final_summary["name"]):
        rows = [
            {
                "name": alias_label(name),
                "ap": float(final_summary.loc[final_summary["name"] == name, "average_precision_mean"].iloc[0]),
                "std": float(final_summary.loc[final_summary["name"] == name, "average_precision_std"].iloc[0]),
            }
            for name in ("L0 linear raw EDA", "M selected from directed comparisons")
        ]
        print(f"  [4] {final_dir} tiene el holdout real de L0 y M")
        label = "5. L0 vs M, en el holdout (una sola vez)"
        out = figures / "05-final-holdout.png"
    else:
        print(f"  [5] el holdout todavia no se abrio para L0 y M en {final_dir} "
              "(correr src.model.run_final) -- usando datos de EJEMPLO, no reales")
        rows = EXAMPLE_FINAL_ROWS
        label = "5. [EJEMPLO, no datos reales] L0 vs M, en el holdout"
        out = figures / "05-final-holdout-EJEMPLO.png"
    path = fig.final_candidates_bar(rows, title=label, path=out)
    print(f"  [5] {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figures = Path(args.figures)

    print("=== 1-2: datos reales ===")
    chart_1_representations(args.results, figures)
    chart_2_ladder(args.results, figures)

    print("\n=== 3-5: implementados y probados con fixtures; leen datos reales si existen ===")
    chart_3_architecture_path(args.results, figures)
    chart_4_greedy_neighbourhood(args.results, figures)
    chart_5_final(args.results, figures)

    print(f"\nfiguras en {figures}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
