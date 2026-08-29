"""How much a column separates, alone or with another one held fixed.

Two columns can each have a small p-value under :mod:`src.eda.noise` and still be the
same information said twice, so a column that passed there is read again inside
the buckets of one already kept. The reading is asymmetric on purpose: A absorbs
B when B stops separating inside A but A keeps separating inside B, and that is
what decides which of the two stays.
"""

from __future__ import annotations

import pandas as pd

from src.eda import noise
from src.eda.rates import rate_by_bucket, rate_by_level

BUCKETS = 5


def separation(
    frame: pd.DataFrame,
    column: str,
    *,
    buckets: int = BUCKETS,
    reps: int = noise.REPS,
    seed: int = 0,
) -> tuple[float, noise.PermutationResult] | None:
    """Separation of ``column`` and its row-wise empirical p-value.

    ``None`` when the slice has fewer than two observed groups: there is nothing
    to separate.
    """
    numeric = pd.api.types.is_numeric_dtype(frame[column])
    table = (
        rate_by_bucket(frame, column, buckets=buckets)
        if numeric
        else rate_by_level(frame, column)
    )
    if len(table) < 2:
        return None
    spread = float(table["rate"].max() - table["rate"].min()) * 100
    result = noise.row_test(
        table["rows"].to_numpy(),
        frame["bought"].to_numpy(),
        spread,
        reps=reps,
        seed=seed,
    )
    return spread, result


def _parts(frame: pd.DataFrame, held: str):
    if pd.api.types.is_numeric_dtype(frame[held]):
        keys = pd.qcut(frame[held], BUCKETS, labels=False, duplicates="drop")
        return frame.groupby(keys), lambda k: f"{int(k) + 1} de {BUCKETS}"
    return frame.groupby(frame[held].fillna("(nulo)")), str


def holding(frame: pd.DataFrame, candidate: str, held: str) -> pd.DataFrame:
    """Read ``candidate`` inside each bucket of ``held``, one row per bucket."""
    parts, label = _parts(frame, held)
    rows = []
    for key, part in parts:
        reading = separation(part, candidate)
        if reading is None:
            continue
        spread, result = reading
        rows.append(
            {
                f"tramo de {held}": label(key),
                "filas": len(part),
                f"separacion de {candidate}": round(spread, 1),
                "p95": round(result.percentile, 2),
                "p_value": round(result.p_value, 4),
                "significativa": result.significant,
            }
        )
    return pd.DataFrame(rows).set_index(f"tramo de {held}")


def absorption(frame: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """Both directions at once: each column read inside the other."""
    rows = []
    for candidate, held in ((second, first), (first, second)):
        table = holding(frame, candidate, held)
        rows.append(
            {
                "lectura": f"{candidate} fijando {held}",
                "tramos": len(table),
                "significativa en": int(table["significativa"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("lectura")
