"""Un peldaño declarado, medido con las tres semillas del protocolo.

    .venv/Scripts/python -m scripts.run_seeds --prefix L1

Los peldaños se declaran con una sola semilla y las decisiones comparan medias de tres.
Reusa lo grabado: sólo entrena las que falten.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from src.eda.loading import load_dataset
from src.model.configs import EDA_PARAMETERS, PROTOCOL, load_parameters
from src.model.console import utf8_console
from src.model.experiment import partition, run_one
from src.model.representation_selection import SEEDS, seed_mean, seed_spread
from src.model.results import RESULTS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=str, required=True, help="p. ej. L1, L2")
    parser.add_argument("--parameters", type=str, default=str(EDA_PARAMETERS))
    parser.add_argument("--results", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    declared = load_parameters(args.parameters)
    matches = [run for name, run in declared.items() if name.startswith(args.prefix)]
    if len(matches) != 1:
        raise SystemExit(f"[{args.prefix}] identifica {len(matches)} corridas, se necesita una")
    config = matches[0]

    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    print(f"{config.name}: n_layers={config.n_layers} d{config.d_model} "
          f"{config.numeric_embedding} lr{config.learning_rate:g}")

    runs = []
    for seed in SEEDS:
        result, note = run_one(
            replace(config, seed=seed), frame, partitions,
            directory=args.results, force=args.force,
        )
        ordered = sorted(result.folds, key=lambda fold: fold.fold_index)
        runs.append(np.asarray([fold.average_precision for fold in ordered]))
        print(f"  seed {seed:<5d} AP {result.average_precision_mean:.4f} [{note}]", flush=True)

    print(f"\n{args.prefix} = {seed_mean(runs):.4f} +- {seed_spread(runs):.4f} "
          f"({len(SEEDS)} semillas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
