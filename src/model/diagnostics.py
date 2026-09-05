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


def position_groups(encoder, frame: pd.DataFrame, indices) -> list[list[str]]:
    """Name every position of every row, so attention can be summed by meaning.

    The popularity phrase is the parenthesis at the end of the title, so its tokens are
    the last tokens of the title field. Separating them from the other ~35 title tokens
    is the whole point: "attention on the title" would say nothing, "attention on the
    two words that decide the label" says everything.
    """
    spec = encoder.spec
    rows = frame.iloc[list(indices)]
    text_width = encoder.text_width

    from src.model.encoding import tokenize

    named: list[list[str]] = []
    for row in rows.itertuples(index=False):
        per_field = [
            (name, len(tokenize(getattr(row, name)))) for name in spec.text_fields
        ]
        labels: list[str] = []
        phrase = getattr(row, "popularity_phrase", NO_PHRASE)
        phrase_tokens = len(tokenize(phrase)) if phrase != NO_PHRASE else 0
        remaining = text_width
        for name, count in per_field:
            kept = min(count, remaining)
            field_labels = [name] * kept
            if name == "title":
                phrase_start = max(0, kept - phrase_tokens)
                field_labels[:phrase_start] = [TITLE_REST] * phrase_start
                if phrase_tokens:
                    field_labels[phrase_start:] = [PHRASE_GROUP] * (
                        kept - phrase_start
                    )
                else:
                    field_labels = [TITLE_REST] * kept
            labels.extend(field_labels)
            remaining -= kept

        labels.extend(["(padding)"] * remaining)
        named.append([CLS_GROUP, *labels])
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

    encoded = encoder.transform(frame, indices)
    groups = position_groups(encoder, frame, indices)

    totals: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}
    n_rows = len(encoded)

    with torch.no_grad():
        model.eval()
        for start in range(0, n_rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, n_rows))
            weights = model.attention_of_cls(encoded.select(rows))
            if weights.numel() == 0:
                return pd.DataFrame(columns=["layer", "group", "tokens", "mass", "per_token"])
            averaged = weights.mean(dim=2).cpu().numpy()
            for offset in range(averaged.shape[0]):
                names = groups[start + offset]
                for layer in range(averaged.shape[1]):
                    for position, name in enumerate(names):
                        if name == "(padding)":
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
    """Did the model learn the inverted U, or only learn to rank?

    For every bucket in turn, every row is re-encoded *as if* its price sat in that
    bucket -- the bucket index and the standardised value both moved, everything else
    held -- and scored. The resulting curve is the model's own price response, held
    against the buy rate the same buckets actually show. A model that only learned a
    monotone price effect produces a line here; ``docs/EDA.md`` says the truth is a
    hump.
    """
    import torch

    from src.model.training import predict

    if column not in encoder.spec.numeric_fields:
        raise ValueError(f"{column} is not one of the encoded numeric fields")
    position = encoder.spec.numeric_fields.index(column)

    rows = frame.iloc[list(indices)]
    raw = rows[column].to_numpy(dtype=np.float64)
    edges = encoder.bucket_edges(column)
    assigned = np.digitize(raw, edges)
    observed = rows["bought"].to_numpy().astype(float)

    encoded = encoder.transform(frame, indices)
    as_is = predict(model, encoded)

    records = []
    for bucket in range(len(edges) + 1):
        members = assigned == bucket
        if not members.any():
            continue
        centre = float(raw[members].mean())
        counterfactual = replace(
            encoded,
            numeric_values=encoded.numeric_values.clone(),
            numeric_buckets=encoded.numeric_buckets.clone(),
            numeric_ratios=encoded.numeric_ratios.clone(),
        )
        counterfactual.numeric_buckets[:, position] = bucket
        counterfactual.numeric_values[:, position] = float(
            encoder.standardise(column, np.array([centre]))[0]
        )
        counterfactual.numeric_ratios[:, position, :] = torch.from_numpy(
            encoder.piecewise_ratios(column, np.array([centre]))[0]
        ).to(encoded.numeric_ratios.device)
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


def bucket_embedding_axis(model, encoder, column: str = "price_position") -> pd.DataFrame:
    """The learned bucket vectors collapsed onto their first principal component.

    Ten free vectors in 64 dimensions cannot be read directly, but almost all of their
    variation lies on one axis, and the *shape* along it is what the slide claims. The
    sign is arbitrary and fixed by a convention (the largest loading is positive), so
    the shape is the claim, never the direction.
    """
    if model.numbers is None or model.numbers.buckets is None:
        raise ValueError("this configuration has no bucket table to read")
    position = encoder.spec.numeric_fields.index(column)
    n_buckets = model.numbers.n_buckets
    table = (
        model.numbers.buckets.weight.detach()
        .cpu()
        .numpy()[position * n_buckets : (position + 1) * n_buckets]
    )

    centred = table - table.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    axis = components[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return pd.DataFrame(
        {"bucket": np.arange(n_buckets), "component": centred @ axis}
    )
