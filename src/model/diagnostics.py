"""What the final model is asked once it has been chosen: is it right, is it honest,
and can we see why.

Stage 6 of the plan is four different questions and they are not the same question:

- **Is it right?** ROC and PR against the linear bar, on rows nothing ever fitted on.
- **Is it honest?** Calibration. ``BTR`` is defined as the mean of the predicted
  probability, so a model whose 0.4 does not buy at 40% is reporting a business number
  that is wrong even when its ranking is right. This is not a proxy metric; it is the
  metric.
- **Is it useful?** Precision and lift at *k*, because the brief's use of the model is
  "identify the best products and promote them", which is a ranking with a budget.
- **Why?** Where ``[CLS]`` looks, and whether the learned price buckets recovered the
  inverted U that ``docs/EDA.md`` measured.

Every function here computes; nothing here draws. ``figures.py`` draws.
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
"""Two-sided 95%: the interval every observed rate in these tables carries."""


def wilson(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """A confidence interval that still behaves at 2.6% and n=40.

    The textbook normal interval goes below zero on a rate that low, which is where
    half of this dataset's phrases live. Wilson does not.
    """
    if total <= 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Scored:
    """One model's predictions on one set of rows, with the labels for those rows."""

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
    """Predicted BTR against observed BTR, by bin of the predicted score.

    Bins are equal-count rather than equal-width: the scores pile up near zero, so
    equal-width bins would put nine tenths of the rows in the first one and measure
    nothing. ``pd.qcut`` drops duplicate edges, so fewer than ``n_bins`` rows can come
    back -- that is a property of the predictions, and it is worth seeing.
    """
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
    """Expected calibration error: the row-weighted gap between the two columns."""
    weights = table["rows"] / table["rows"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())


DEFAULT_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
"""Promotion budgets, as a share of the catalogue shown."""


def ranking_gains(
    scored: Scored, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> pd.DataFrame:
    """Precision, recall and lift at each budget, sorting by the predicted score.

    Lift is precision@k divided by the base rate: how many times better than promoting
    at random. It is the number the business case is actually made of.
    """
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
    """Observed against predicted BTR for each level of a column, ranked by the gap.

    ``docs/EDA.md`` predicts where this goes wrong: the tier-B phrases buy at 2.6% and
    look, in text, exactly like the tier-A ones that buy at 64.7%. If the model is
    guessing anywhere, it is there, and this table says so with a number.
    """
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
        successes = int(group["actual"].sum())
        total = len(group)
        single_class = group["actual"].nunique() < 2
        records.append(
            {
                "level": level,
                "rows": total,
                "observed": successes / total,
                "predicted": float(group["predicted"].mean()),
                "gap": float(group["predicted"].mean()) - successes / total,
                "average_precision": float("nan")
                if single_class
                else float(
                    average_precision_score(group["actual"], group["predicted"])
                ),
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


def phrase_tokens(encoder, row) -> int:
    """The trailing parenthesis of a title, measured in the run's own tokens.

    Encoding the title and the title without its parenthesis and subtracting is exact
    for any tokenizer, brackets kept or removed, without assuming how it splits.
    """
    title = text_or_empty(getattr(row, "title", ""))
    if getattr(row, "popularity_phrase", NO_PHRASE) == NO_PHRASE or "(" not in title:
        return 0
    return len(encoder.encode(title)) - len(encoder.encode(title[: title.rindex("(")]))


def position_groups(encoder, frame: pd.DataFrame, indices) -> list[list[str]]:
    """Name every position of every row, so attention can be summed by meaning.

    The popularity phrase is the parenthesis at the end of the title, so its tokens are
    the last ones that field kept. Separating them from the other title tokens is the
    whole point: "attention on the title" would say nothing, "attention on the two
    words that decide the label" says everything.
    """
    rows = frame.iloc[list(indices)]
    named: list[list[str]] = []
    for row, spans in zip(rows.itertuples(index=False), encoder.layout(rows)):
        labels = [CLS_GROUP]
        for span in spans:
            field = [span.name] * span.kept
            if span.name == "title":
                phrase = min(phrase_tokens(encoder, row), span.kept)
                field[span.kept - phrase :] = [PHRASE_GROUP] * phrase
                field[: span.kept - phrase] = [TITLE_REST] * (span.kept - phrase)
            labels.extend([*field, SEP_GROUP])
        labels.extend([PADDING_GROUP] * (encoder.sequence_length - len(labels)))
        named.append(labels)
    return named


def cls_attention(
    model, encoder, frame: pd.DataFrame, indices, *, batch_size: int = 256
) -> pd.DataFrame:
    """How much of ``[CLS]``'s attention each group of positions receives.

    Two columns, and they answer different questions. ``mass`` is the share of the
    softmax that lands on the group, which a group of 35 tokens wins by being big.
    ``per_token`` divides it by the number of tokens, which is where two tokens
    carrying the whole label show up.
    """
    import torch

    text, tabular = encoder.transform(frame, indices)
    groups = position_groups(encoder, frame, indices)

    totals: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}
    n_rows = len(text)

    with torch.no_grad():
        model.eval()
        for start in range(0, n_rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, n_rows))
            weights = model.attention_of_cls(
                (text.select(rows), tabular.select(rows))
            )
            if weights.numel() == 0:
                return pd.DataFrame(columns=["layer", "group", "tokens", "mass", "per_token"])
            averaged = weights.mean(dim=2).cpu().numpy()
            for offset in range(averaged.shape[0]):
                names = groups[start + offset]
                for layer in range(averaged.shape[1]):
                    for position, name in enumerate(names):
                        if name == PADDING_GROUP:
                            continue
                        key = (layer, name)
                        totals[key] = totals.get(key, 0.0) + float(
                            averaged[offset, layer, position]
                        )
                        counts[key] = counts.get(key, 0) + 1

    records = [
        {
            "layer": layer,
            "group": name,
            "tokens": counts[(layer, name)] / n_rows,
            "mass": totals[(layer, name)] / n_rows,
            "per_token": totals[(layer, name)] / counts[(layer, name)],
        }
        for layer, name in sorted(totals, key=lambda key: (key[0], key[1]))
    ]
    return pd.DataFrame(records)


def price_bucket_recovery(
    model,
    encoder,
    frame: pd.DataFrame,
    indices,
    *,
    column: str = "price_position",
) -> pd.DataFrame:
    """The model's own price response: every row scored again as if its price sat in
    each bucket in turn, against the buy rate those buckets actually show."""
    import torch

    from src.model.encoding import PRICE_SLICE
    from src.model.training import predict

    if column not in encoder.spec.numeric_fields:
        raise ValueError(f"{column} is not one of the encoded numeric fields")

    rows = frame.iloc[list(indices)]
    raw = rows[column].to_numpy(dtype=np.float64)
    edges = encoder.bucket_edges(column)
    assigned = np.digitize(raw, edges)
    observed = rows["bought"].to_numpy().astype(float)

    text_batch, tab_batch = encoder.transform(frame, indices)
    as_is = predict(model, (text_batch, tab_batch))

    records = []
    for bucket in range(len(edges) + 1):
        members = assigned == bucket
        if not members.any():
            continue
        centre = float(raw[members].mean())
        pieces = torch.from_numpy(encoder.tabular_price_ratios(np.array([centre]))[0])
        x_tab = tab_batch.x_tab.clone()
        x_tab[:, PRICE_SLICE] = pieces.to(x_tab)
        counterfactual = (text_batch, replace(tab_batch, x_tab=x_tab))
        records.append(
            {
                "bucket": bucket,
                "centre": centre,
                "rows": int(members.sum()),
                "observed": float(observed[members].mean()),
                "counterfactual": float(predict(model, counterfactual).mean()),
                "as_is": float(as_is[members].mean()),
            }
        )
    return pd.DataFrame(records)


def text_truncation(encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """How much of each text field the token budget cut, field by field.

    The row that matters is ``title``: the popularity phrase sits at its end, so a
    single dropped token there is the strongest variable of the dataset going missing.
    """
    rows = frame.iloc[list(indices)]
    layouts = encoder.layout(rows)
    records = [
        {
            "field": name,
            "tokens": sum(layout[position].total for layout in layouts) / len(layouts),
            "rows_truncated": sum(
                1 for layout in layouts if layout[position].dropped
            ),
            "tokens_dropped": sum(layout[position].dropped for layout in layouts),
        }
        for position, name in enumerate(encoder.spec.text_fields)
    ]
    return pd.DataFrame(records)


def tower_norms(model, encoder, frame: pd.DataFrame, indices) -> pd.DataFrame:
    """Per-row length of each tower's output: whether the two branches are within an
    order of magnitude, and whether the length varies by row at all."""
    import torch

    from src.model.training import INFERENCE_BATCH

    text_batch, tab_batch = encoder.transform(frame, indices)
    device = next(model.parameters()).device
    model.eval()

    lengths: dict[str, list[np.ndarray]] = {"h_text": [], "h_tab": []}
    with torch.no_grad():
        for start in range(0, len(tab_batch), INFERENCE_BATCH):
            rows = torch.arange(start, min(start + INFERENCE_BATCH, len(tab_batch)))
            h_text = model.text_tower(text_batch.select(rows).to(device))
            h_tab = model.tabular_tower(tab_batch.select(rows).to(device).x_tab)
            lengths["h_text"].append(h_text.norm(dim=-1).cpu().numpy())
            lengths["h_tab"].append(h_tab.norm(dim=-1).cpu().numpy())
            dimensions = {"h_text": h_text.shape[-1], "h_tab": h_tab.shape[-1]}

    return pd.DataFrame(
        [
            {
                "tower": name,
                "dimensions": int(dimensions[name]),
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for name, values in (
                (name, np.concatenate(parts)) for name, parts in lengths.items()
            )
        ]
    )
