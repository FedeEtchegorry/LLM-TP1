"""The paired-margin rule that decides whether a representation alternative replaces
the reference it is compared against.

Every comparison in this file is over exactly five per-fold AP values, because that is
the protocol's fold count (``configs.PROTOCOL.folds``). The margin is descriptive
dispersion, never inference: the five folds share training data, so their differences
are correlated and no test over folds would be valid. It is named **margen pareado**
and never "confidence interval", "significant" or a p-value.
"""

from __future__ import annotations

import numpy as np

IMPROVES = "improves"
LOSES = "loses"
TIE_BREAK = "tie-break"
INCONCLUSIVE = "inconclusive"

MOVES = (IMPROVES, TIE_BREAK)
"""The two outcomes that replace a reference with a candidate."""


def paired_margin(differences) -> tuple[float, float, float]:
    """Mean of the per-fold differences, plus and minus one standard deviation."""
    values = np.asarray(differences, dtype=float)
    if values.shape != (5,):
        raise ValueError("paired comparison requires exactly five folds")
    mean = float(values.mean())
    spread = float(values.std(ddof=1))
    return mean, mean - spread, mean + spread


def compare(incumbent_ap, candidate_ap) -> str:
    """``'improves'``, ``'loses'``, ``'tie-break'`` or ``'inconclusive'``.

    The margin straddles zero in the common case: that is the limit of what five
    folds resolve, and the default is to keep the incumbent. The only way past it is
    to win on both readings the numbers still allow -- a higher mean *and* a lower
    spread, a conjunction rather than a free pass, since a spread that happens to be
    smaller by chance would only pass on its own a quarter of the time.
    """
    incumbent_ap = np.asarray(incumbent_ap, dtype=float)
    candidate_ap = np.asarray(candidate_ap, dtype=float)
    _, low, high = paired_margin(candidate_ap - incumbent_ap)
    if low > 0.0:
        return IMPROVES
    if high < 0.0:
        return LOSES
    better = float(candidate_ap.mean()) > float(incumbent_ap.mean())
    steadier = float(candidate_ap.std(ddof=1)) < float(incumbent_ap.std(ddof=1))
    return TIE_BREAK if better and steadier else INCONCLUSIVE


def choose(reference: str, folds: dict[str, list[float]]) -> str:
    """The reference, unless some other key in ``folds`` improves on it or ties into
    it by the declared rule. Kept simple on purpose: this reconstruction only ever
    compares a reference against one alternative at a time (Task 3's eight cases),
    so ``folds`` has exactly two keys in every call this repository makes."""
    incumbent_ap = np.asarray(folds[reference], dtype=float)
    selected = reference
    for name, values in folds.items():
        if name == reference:
            continue
        candidate_ap = np.asarray(values, dtype=float)
        if compare(incumbent_ap, candidate_ap) in MOVES:
            selected = name
    return selected
