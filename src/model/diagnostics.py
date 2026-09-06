"""What the final model is asked once it is chosen: is it right, is it honest, is it
useful, and where does it look. Everything here computes; ``figures.py`` draws.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.eda.loading import NO_PHRASE
from src.model.encoding import text_or_empty

Z_95 = 1.959963984540054


def wilson(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson: the normal interval goes below zero at rates as low as 2.6%."""
    if total <= 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Scored:
    name: str
    actual: np.ndarray
    predicted: np.ndarray

    def __post_init__(self) -> None:
        if self.actual.shape != self.predicted.shape:
            raise ValueError(
                f"{self.name}: {self.actual.shape} labels against "
                f"{self.predicted.shape} predictions"
            )

    @property
    def roc_auc(self) -> float:
        return float(roc_auc_score(self.actual, self.predicted))

    @property
    def average_precision(self) -> float:
        return float(average_precision_score(self.actual, self.predicted))

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.actual))


def roc_points(scored: Scored) -> pd.DataFrame:
    false_positive, true_positive, _ = roc_curve(scored.actual, scored.predicted)
    return pd.DataFrame({"fpr": false_positive, "tpr": true_positive})


def pr_points(scored: Scored) -> pd.DataFrame:
    precision, recall, _ = precision_recall_curve(scored.actual, scored.predicted)
    return pd.DataFrame({"recall": recall, "precision": precision})


def calibration(scored: Scored, *, n_bins: int = 10) -> pd.DataFrame:
    """Predicted against observed BTR in equal-count bins: scores pile up near zero."""
    frame = pd.DataFrame({"actual": scored.actual, "predicted": scored.predicted})
    frame["bin"] = pd.qcut(
        frame["predicted"].rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    rows = []
    for index, group in frame.groupby("bin", observed=True):
        successes = int(group["actual"].sum())
        total = len(group)
        low, high = wilson(successes, total)
        rows.append(
            {
                "bin": int(index) + 1,
                "rows": total,
                "predicted": float(group["predicted"].mean()),
                "observed": successes / total,
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)


def calibration_error(table: pd.DataFrame) -> float:
    weights = table["rows"] / table["rows"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())


DEFAULT_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
"""Promotion budgets, as a share of the catalogue shown."""


def ranking_gains(
    scored: Scored, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> pd.DataFrame:
    """Precision, recall and lift at each budget, sorting by the predicted score."""
    order = np.argsort(-scored.predicted, kind="stable")
    labels = scored.actual[order]
    positives = int(labels.sum())
    base = scored.positive_rate

    rows = []
    for fraction in fractions:
        k = max(1, int(round(fraction * len(labels))))
        hits = int(labels[:k].sum())
        rows.append(
            {
                "fraction": fraction,
                "k": k,
                "hits": hits,
                "precision": hits / k,
                "recall": hits / positives if positives else float("nan"),
                "lift": (hits / k) / base if base else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def errors_by_level(
    frame: pd.DataFrame, indices, scored: Scored, column: str
) -> pd.DataFrame:
    """Observed against predicted BTR per level: the tiers that look alike in text."""
    rows = frame.iloc[list(indices)]
    table = pd.DataFrame(
        {
            "level": rows[column].to_numpy(),
            "actual": scored.actual,
            "predicted": scored.predicted,
        }
    )
    records = []
    for level, group in table.groupby("level", observed=True):
        observed = float(group["actual"].mean())
        predicted = float(group["predicted"].mean())
        records.append(
            {
                "level": level,
                "rows": len(group),
                "observed": observed,
                "predicted": predicted,
                "gap": predicted - observed,
                "average_precision": float("nan")
                if group["actual"].nunique() < 2
                else float(average_precision_score(group["actual"], group["predicted"])),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("observed", ascending=False)
        .reset_index(drop=True)
    )


PHRASE_GROUP = "frase de popularidad"
TITLE_REST = "titulo (resto)"
CLS_GROUP = "[CLS]"
SEP_GROUP = "[SEP]"
PADDING_GROUP = "(padding)"
TABULAR_PRICE = "price_position"
ATTENTION_COLUMNS = ["layer", "group", "tokens", "mass", "per_token"]


def _phrase_tokens(encoder, row) -> int:
    """Tokens of the title's closing parenthesis; ``keep_brackets`` may add two."""
    title = text_or_empty(getattr(row, "title", ""))
    if getattr(row, "popularity_phrase", NO_PHRASE) == NO_PHRASE or "(" not in title:
        return 0
    return len(encoder.encode(title)) - len(encoder.encode(title[: title.rindex("(")]))


def position_groups(encoder, frame: pd.DataFrame, indices) -> list[list[str]]:
    """One name per position: the phrase closes the title, before the first ``[SEP]``."""
    rows = frame.iloc[list(indices)]
    named: list[list[str]] = []
    for row, spans in zip(rows.itertuples(index=False), encoder.layout(rows)):
        labels = [CLS_GROUP]
        for span in spans:
            field = [span.name] * span.kept
            if span.name == "title":
                phrase = min(_phrase_tokens(encoder, row), span.kept)
                field[span.kept - phrase :] = [PHRASE_GROUP] * phrase
                field[: span.kept - phrase] = [TITLE_REST] * (span.kept - phrase)
            labels.extend([*field, SEP_GROUP])
        labels.extend([PADDING_GROUP] * (encoder.sequence_length - len(labels)))
        named.append(labels)
    return named


def _batches(model, encoder, frame: pd.DataFrame, indices):
    """The encoded rows in bounded batches, on whatever device the model sits on."""
    import torch

    from src.model.training import INFERENCE_BATCH

    text, tabular = encoder.transform(frame, indices)
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(text), INFERENCE_BATCH):
        rows = torch.arange(start, min(start + INFERENCE_BATCH, len(text)))
        yield text.select(rows).to(device), tabular.select(rows).to(device)


def _named_positions(encoder, frame: pd.DataFrame, indices, shape) -> np.ndarray:
    groups = np.array(position_groups(encoder, frame, indices))
    if groups.shape != shape:
        raise ValueError(
            f"the names are {groups.shape} and the weights are {shape}: "
            "position_groups and the encoder disagree on the sequence layout"
        )
    return groups


def _by_group(per_position: pd.DataFrame, keys: list[str], n_rows: int) -> pd.DataFrame:
    totals = (
        per_position[per_position["group"] != PADDING_GROUP]
        .groupby([*keys, "group"], as_index=False)
        .agg(tokens=("mass", "size"), mass=("mass", "sum"))
    )
    return totals.assign(
        per_token=totals["mass"] / totals["tokens"],
        tokens=totals["tokens"] / n_rows,
        mass=totals["mass"] / n_rows,
    )[[*keys, "group", "tokens", "mass", "per_token"]]


def cls_attention(model, encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """Where ``[CLS]`` looks: ``mass`` is the total, ``per_token`` the average."""
    import torch

    chunks = []
    with torch.no_grad():
        for batch in _batches(model, encoder, frame, indices):
            weights = model.attention_of_cls(batch)
            if weights.numel() == 0:  # a run with no blocks has no attention to read
                return pd.DataFrame(columns=ATTENTION_COLUMNS)
            chunks.append(weights.mean(dim=2).cpu().numpy())

    attention = np.concatenate(chunks)
    n_rows, n_layers, n_positions = attention.shape
    groups = _named_positions(encoder, frame, indices, (n_rows, n_positions))

    return _by_group(
        pd.DataFrame(
            {
                "layer": np.tile(np.repeat(np.arange(n_layers), n_positions), n_rows),
                "group": np.repeat(groups[:, None, :], n_layers, axis=1).reshape(-1),
                "mass": attention.reshape(-1),
            }
        ),
        ["layer"],
        n_rows,
    )


def pooling_by_group(model, encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """The final reduction to one vector per row, which ``cls_attention`` never sees."""
    import torch

    with torch.no_grad():
        weights = np.concatenate(
            [
                model.pooling_weights(batch).cpu().numpy()
                for batch in _batches(model, encoder, frame, indices)
            ]
        )

    n_rows, n_positions = weights.shape
    groups = _named_positions(encoder, frame, indices, weights.shape)

    return _by_group(
        pd.DataFrame({"group": groups.reshape(-1), "mass": weights.reshape(-1)}),
        [],
        n_rows,
    )


def price_bucket_recovery(model, encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """The model's own price response: every row rescored in each bucket in turn."""
    import torch

    from src.model.encoding import PRICE_SLICE
    from src.model.training import predict

    if TABULAR_PRICE not in encoder.spec.numeric_fields:
        raise ValueError(f"this run does not encode {TABULAR_PRICE}: no U to recover")

    rows = frame.iloc[list(indices)]
    raw = rows[TABULAR_PRICE].to_numpy(dtype=np.float64)
    edges = encoder.bucket_edges(TABULAR_PRICE)
    assigned = np.digitize(raw, edges)
    observed = rows["bought"].to_numpy().astype(float)

    text, tabular = encoder.transform(frame, indices)
    as_is = predict(model, (text, tabular))

    records = []
    for bucket in range(len(edges) + 1):
        members = assigned == bucket
        if not members.any():
            continue
        centre = float(raw[members].mean())
        x_tab = tabular.x_tab.clone()
        x_tab[:, PRICE_SLICE] = torch.from_numpy(
            encoder.tabular_price_ratios(np.array([centre]))
        ).to(x_tab)
        counterfactual = predict(model, (text, replace(tabular, x_tab=x_tab)))
        records.append(
            {
                "bucket": bucket,
                "centre": centre,
                "rows": int(members.sum()),
                "observed": float(observed[members].mean()),
                "counterfactual": float(counterfactual.mean()),
                "as_is": float(as_is[members].mean()),
            }
        )
    return pd.DataFrame(records)


TRUNCATION_COLUMNS = ["field", "tokens_mean", "tokens_max", "dropped", "rows_truncated"]
TOTAL_FIELD = "(total)"


def _truncation_line(field: str, tokens: pd.Series, dropped: pd.Series) -> dict:
    return {
        "field": field,
        "tokens_mean": float(tokens.mean()),
        "tokens_max": int(tokens.max()),
        "dropped": int(dropped.sum()),
        "rows_truncated": float((dropped > 0).mean()),
    }


def truncation(encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """What the declared budget cut: ``dropped`` totals, ``rows_truncated`` is a share."""
    if not encoder.spec.text_fields:
        return pd.DataFrame(columns=TRUNCATION_COLUMNS)

    spans = pd.DataFrame(
        [
            {
                "row": row,
                "field": span.name,
                "tokens": span.total,
                "dropped": span.dropped,
            }
            for row, layout in enumerate(encoder.layout(frame.iloc[list(indices)]))
            for span in layout
        ]
    )
    tokens = spans.pivot(index="row", columns="field", values="tokens")
    dropped = spans.pivot(index="row", columns="field", values="dropped")
    return pd.DataFrame(
        [
            *(
                _truncation_line(name, tokens[name], dropped[name])
                for name in encoder.spec.text_fields
            ),
            _truncation_line(TOTAL_FIELD, tokens.sum(axis=1), dropped.sum(axis=1)),
        ]
    )


def tower_norms(model, encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """Whether the two towers reach the fusion within an order of magnitude."""
    import torch

    outputs: dict[str, list] = {"h_text": [], "h_tab": []}
    with torch.no_grad():
        for text_rows, tab_rows in _batches(model, encoder, frame, indices):
            outputs["h_text"].append(model.text_tower(text_rows))
            outputs["h_tab"].append(model.tabular_tower(tab_rows.x_tab))

    records = []
    for name, parts in outputs.items():
        stacked = torch.cat(parts)
        lengths = stacked.norm(dim=-1).cpu().numpy()
        records.append(
            {
                "tower": name,
                "dimensions": stacked.shape[-1],
                "mean": float(lengths.mean()),
                "sd": float(lengths.std()),
                "min": float(lengths.min()),
                "max": float(lengths.max()),
            }
        )
    return pd.DataFrame(records)
