"""Every run's numbers land in ``results/`` and are read back from there.

Two jobs at once. As a **record**, one JSON per run holds the resolved configuration,
the per-fold metrics and the per-epoch curves, so the tables and figures for the
presentation regenerate without retraining anything. As a **cache**, a run whose
digest already has a file is not run again, which is what makes a sweep something you
can interrupt and resume.

    .venv/bin/python -m src.model.results          # everything recorded so far
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.model.configs import PROTOCOL, TRAINING, RunConfig
from src.model.protocol import EvaluationResult, FoldScore

RESULTS_DIR = Path("results")
WEIGHTS_DIR = "weights"
"""Trained parameters, one file per fold. Regenerable, so ``.gitignore`` skips them."""

SCHEMA = 1
"""Bumped when the stored shape changes, so an old file is skipped, not misread."""


def result_path(config: RunConfig, directory: Path | str = RESULTS_DIR) -> Path:
    """One file per run, named by digest so a changed parameter is a different file."""
    return Path(directory) / f"{config.digest}.json"


def save(
    config: RunConfig,
    result: EvaluationResult,
    *,
    seconds: float,
    curves: list[dict] | None = None,
    device: str | None = None,
    directory: Path | str = RESULTS_DIR,
) -> Path:
    """Write the run's record, creating ``results/`` if this is the first one.

    The fixed training and protocol constants are stored beside the run's own
    parameters, so a record stays readable after one of them is changed in code.
    """
    path = result_path(config, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SCHEMA,
        "name": config.name,
        "digest": config.digest,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 3),
        "config": asdict(config),
        "training": asdict(TRAINING),
        "device": device or "unrecorded",
        "protocol": asdict(PROTOCOL),
        "metrics": {
            "roc_auc_mean": result.roc_auc_mean,
            "roc_auc_std": result.roc_auc_std,
            "average_precision_mean": result.average_precision_mean,
            "average_precision_std": result.average_precision_std,
        },
        "folds": [asdict(fold) for fold in result.folds],
        "curves": curves or [],
    }
    path.write_text(json.dumps(document, indent=2, default=list), encoding="utf-8")
    return path


def load(
    config: RunConfig, directory: Path | str = RESULTS_DIR
) -> EvaluationResult | None:
    """The recorded result for this exact configuration, or ``None`` if there is none."""
    path = result_path(config, directory)
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        return None
    return EvaluationResult(
        name=document["name"],
        folds=tuple(FoldScore(**fold) for fold in document["folds"]),
    )


def document(
    config: RunConfig, directory: Path | str = RESULTS_DIR
) -> dict | None:
    """The whole stored record for one run: metrics, curves, parameter counts.

    ``load`` returns only what the protocol needs; the transfer and final tables also
    quote how many parameters were trained and how long it took, which live here.
    """
    path = result_path(config, directory)
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    return stored if stored.get("schema") == SCHEMA else None


def weights_path(
    config: RunConfig, fold_index: int, directory: Path | str = RESULTS_DIR
) -> Path:
    """``results/weights/<digest>/fold-2.pt``; fold ``-1`` is the final test model."""
    label = "test" if fold_index < 0 else f"fold-{fold_index}"
    return Path(directory) / WEIGHTS_DIR / config.digest / f"{label}.pt"


def save_weights(
    config: RunConfig,
    fold_index: int,
    state: dict,
    *,
    directory: Path | str = RESULTS_DIR,
) -> Path:
    """Keep the trained parameters so interpretability never needs a retrain."""
    import torch

    path = weights_path(config, fold_index, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"digest": config.digest, "fold_index": fold_index, "state": state}, path)
    return path


def load_weights(
    config: RunConfig, fold_index: int, directory: Path | str = RESULTS_DIR
) -> dict | None:
    """The stored parameters for one fold, or ``None`` if that fold was never saved."""
    import torch

    path = weights_path(config, fold_index, directory)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)["state"]


def predictions_path(
    config: RunConfig, directory: Path | str = RESULTS_DIR
) -> Path:
    """``results/final/<digest>.predictions.npy``.

    The held-out run happens once, so its scores are kept: calibration, lift and the
    error tables are all views of the same vector, and none of them should cost a
    retrain -- or, worse, tempt one.
    """
    return Path(directory) / f"{config.digest}.predictions.npy"


def save_predictions(
    config: RunConfig, predicted, *, directory: Path | str = RESULTS_DIR
) -> Path:
    import numpy as np

    path = predictions_path(config, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(predicted, dtype=np.float64))
    return path


def load_predictions(config: RunConfig, directory: Path | str = RESULTS_DIR):
    import numpy as np

    path = predictions_path(config, directory)
    return np.load(path) if path.exists() else None


def documents(directory: Path | str = RESULTS_DIR) -> list[dict]:
    """Every recorded run, newest last."""
    paths = sorted(Path(directory).glob("*.json")) if Path(directory).exists() else []
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(
        (doc for doc in loaded if doc.get("schema") == SCHEMA),
        key=lambda doc: doc["recorded_at"],
    )


def fold_frame(directory: Path | str = RESULTS_DIR) -> pd.DataFrame:
    """One row per fold per run, with the configuration flattened alongside it.

    This is the frame the analysis works from: group by ``name`` for the summary,
    or keep the folds to draw the error bars.
    """
    rows = []
    for document in documents(directory):
        config = {f"config.{key}": _flat(value) for key, value in document["config"].items()}
        for fold in document["folds"]:
            rows.append(
                {
                    "name": document["name"],
                    "digest": document["digest"],
                    "recorded_at": document["recorded_at"],
                    "seconds": document["seconds"],
                    **fold,
                    **config,
                }
            )
    return pd.DataFrame(rows)


def curve_frame(directory: Path | str = RESULTS_DIR) -> pd.DataFrame:
    """One row per epoch per fold per run: the over- and underfitting evidence."""
    rows = []
    for document in documents(directory):
        for curve in document["curves"]:
            for epoch in curve["epochs"]:
                rows.append(
                    {
                        "name": document["name"],
                        "digest": document["digest"],
                        "fold_index": curve["fold_index"],
                        **epoch,
                    }
                )
    return pd.DataFrame(rows)


def summary_frame(directory: Path | str = RESULTS_DIR) -> pd.DataFrame:
    """One row per run: what the results table in the write-up quotes."""
    rows = [
        {
            "name": document["name"],
            "digest": document["digest"],
            "model": document["config"]["model"],
            "seconds": document["seconds"],
            **document["metrics"],
        }
        for document in documents(directory)
    ]
    return pd.DataFrame(rows)


def _flat(value: object) -> object:
    """Tuples come back from JSON as lists; join them so a column stays scalar."""
    return ", ".join(value) if isinstance(value, list) else value


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    summary = summary_frame(args.directory)
    if summary.empty:
        print(f"no runs recorded in {args.directory}/ yet")
        return 0

    display = summary.assign(
        ROC=lambda d: d.roc_auc_mean.map("{:.4f}".format)
        + " ± "
        + d.roc_auc_std.map("{:.4f}".format),
        AP=lambda d: d.average_precision_mean.map("{:.4f}".format)
        + " ± "
        + d.average_precision_std.map("{:.4f}".format),
    ).sort_values("average_precision_mean", ascending=False)
    print(display[["name", "digest", "model", "seconds", "ROC", "AP"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
