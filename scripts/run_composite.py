"""El maximo de cada eje, junto: ¿los efectos se suman?

El barrido mueve un factor por vez, asi que no dice nada sobre que pasa al mover dos a
la vez. De los diez ejes, solo dos tienen su maximo fuera de la configuracion declarada
--n_heads y dropout--, asi que la pregunta se contesta con una sola configuracion mas.

Se mide con el MISMO protocolo que el barrido: 5 folds x 3 semillas sobre las filas de
desarrollo. NO toca el holdout, que ya se abrio una vez con el finalista congelado.

    .venv/bin/python -m scripts.run_composite
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from src.eda.loading import load_dataset
from src.model.configs import (
    PARAMETERS_PATH,
    PROTOCOL,
    apply_overrides,
    load_parameters,
)
from src.model.console import utf8_console
from src.model.eda_contract import require_valid
from src.model.experiment import partition, run_one
from src.model.representation_selection import SEEDS
from src.model.results import RESULTS_DIR

ARMS = (
    ("declarada", {}),
    ("n_heads = 1", {"n_heads": 1}),
    ("dropout = 0,3", {"dropout": 0.3}),
    ("las dos juntas", {"n_heads": 1, "dropout": 0.3}),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=str, default=str(PARAMETERS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--config", type=str, default="RUN")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="clave=valor")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    utf8_console()
    args = parse_args(argv)

    declared = load_parameters(args.parameters)
    require_valid(declared)
    if args.config not in declared:
        raise SystemExit(f"{args.config!r} no esta en {args.parameters}")
    base = apply_overrides(declared[args.config], args.overrides)

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)

    measured = []
    for label, changes in ARMS:
        per_seed = []
        print(f"\n=== {label} ===", flush=True)
        for seed in SEEDS:
            config = replace(base, seed=seed, **changes)
            result, note = run_one(config, frame, partitions,
                                   directory=args.results, force=args.force)
            per_seed.append(result.average_precision_mean)
            print(f"  seed {seed:<5d} AP {result.average_precision_mean:.4f} [{note}]",
                  flush=True)
        measured.append({
            "label": label,
            "changes": changes,
            "mean": statistics.fmean(per_seed),
            "spread": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
            "per_seed": per_seed,
        })

    by_label = {item["label"]: item for item in measured}
    declarada = by_label["declarada"]["mean"]
    additive = (declarada
                + (by_label["n_heads = 1"]["mean"] - declarada)
                + (by_label["dropout = 0,3"]["mean"] - declarada))
    juntas = by_label["las dos juntas"]["mean"]

    print("\n=== LECTURA ===")
    for item in measured:
        delta = item["mean"] - declarada
        print(f"  {item['label']:<16s} {item['mean']:.4f} +/- {item['spread']:.4f}"
              f"   ({delta:+.4f} contra la declarada)")
    print(f"\n  si los efectos se sumaran: {additive:.4f}")
    print(f"  medido junto:               {juntas:.4f}   ({juntas - additive:+.4f})")
    print("\n  Los efectos " + ("se suman de mas" if juntas > additive + 0.002
                                else "NO se suman" if juntas < additive - 0.002
                                else "se suman, dentro del ruido") + ".")
    print("  Medido sobre validacion cruzada. El holdout no se toco.")

    out = Path(args.results) / "architecture" / "composite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "base_digest": base.digest,
        "arms": measured,
        "additive_prediction": additive,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresumen: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
