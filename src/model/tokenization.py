"""Whether the shared-token hypothesis explains part of the AP gap against L0.

    .venv/bin/python -m src.model.tokenization
    .venv/bin/python -m src.model.tokenization --fold 2

L0 gives every ``popularity_phrase`` its own free weight through a one-hot column.
The Transformer instead runs the phrase through ``encoding.tokenize`` -- whole-word
splitting, not subwords -- and every resulting word gets one shared row in the token
embedding table, fit only on training rows (``encoding.RowEncoder._fit_words``). If
two phrases as far apart in BTR as "Customer Favorite" and "Shopper Favorite" turn
out to share a token -- or both fall outside the training vocabulary and both land on
the same ``WORD_UNK`` row, which is the same coupling by a different route -- their
embeddings start out identical at the input layer, before any attention runs.

This measures whether that coupling exists. It does not train anything, load any
weights, or touch the test rows: the vocabulary is fit on one fold's training rows,
exactly as ``RowEncoder.fit`` would for a real run, and BTR is read off the
development rows (train + validation, across every fold) so a rare phrase is not
priced off a few hundred training rows alone.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from src.eda.loading import NO_PHRASE, load_dataset
from src.model.baseline import target_of
from src.model.configs import PROTOCOL
from src.model.encoding import EncodingSpec, RowEncoder, WORD_UNK, tokenize
from src.model.experiment import partition

PHRASE_COLUMN = "popularity_phrase"


def fit_vocabulary(frame: pd.DataFrame, train_indices) -> RowEncoder:
    """The real fitting path, not a reimplementation of it -- so this reflects
    exactly what a training run would see, never an approximation of it."""
    return RowEncoder(EncodingSpec()).fit(frame, train_indices)


def phrase_records(frame: pd.DataFrame, encoder: RowEncoder, rate_indices) -> dict:
    """Per distinct phrase: its tokens, their vocabulary ids, and its BTR.

    ``encoder._words`` is a private attribute; this reaches into it on purpose
    rather than duplicating ``_fit_words`` -- it is read-only introspection for a
    diagnostic, not a second implementation that could drift from the real one.
    """
    words = encoder._words  # noqa: SLF001 -- intentional, see docstring
    target = target_of(frame)
    rated = frame.iloc[list(rate_indices)]
    rated_target = target[list(rate_indices)]
    rated_phrase = rated[PHRASE_COLUMN].to_numpy()

    phrases = sorted(value for value in frame[PHRASE_COLUMN].unique() if value != NO_PHRASE)
    records = {}
    for phrase in phrases:
        tokens = tokenize(phrase)
        ids = [words.get(token, WORD_UNK) for token in tokens]
        mask = rated_phrase == phrase
        records[phrase] = {
            "tokens": tokens,
            "ids": ids,
            "n_rows": int(mask.sum()),
            "btr": float(rated_target[mask].mean()) if mask.any() else float("nan"),
        }
    return records


def _id_to_word(encoder: RowEncoder) -> dict:
    words = encoder._words  # noqa: SLF001
    return {token_id: word for word, token_id in words.items()}


def _label(token: str, token_id: int) -> str:
    return token if token_id != WORD_UNK else f"{token}[UNK]"


def phrase_frame(records: dict, encoder: RowEncoder) -> pd.DataFrame:
    """One row per phrase: its tokenization, and which of its tokens recur
    elsewhere in the popularity-phrase vocabulary -- shared by id, so two
    out-of-vocabulary words sharing ``WORD_UNK`` count as shared too."""
    id_to_word = _id_to_word(encoder)
    rows = []
    for phrase, record in records.items():
        shared_ids: set[int] = set()
        for other, other_record in records.items():
            if other == phrase:
                continue
            shared_ids |= set(record["ids"]) & set(other_record["ids"])
        shared_tokens = sorted(
            id_to_word.get(token_id, "[UNK]") if token_id != WORD_UNK else "[UNK]"
            for token_id in shared_ids
        )
        rows.append(
            {
                "phrase": phrase,
                "tokens": " ".join(
                    _label(token, token_id)
                    for token, token_id in zip(record["tokens"], record["ids"])
                ),
                "n_tokens": len(record["tokens"]),
                "shared_tokens": ", ".join(shared_tokens) or "-",
                "n_rows": record["n_rows"],
                "btr": record["btr"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("btr", ascending=False)
        .reset_index(drop=True)
    )


def pair_frame(records: dict, encoder: RowEncoder) -> pd.DataFrame:
    """One row per pair of phrases: how many tokens they share (by id) and how far
    apart their BTR is. Sorted by BTR gap descending -- the pairs where sharing a
    token would matter most sit at the top."""
    id_to_word = _id_to_word(encoder)
    rows = []
    for left, right in combinations(records, 2):
        left_ids, right_ids = set(records[left]["ids"]), set(records[right]["ids"])
        shared_ids = left_ids & right_ids
        shared_tokens = sorted(
            id_to_word.get(token_id, "[UNK]") if token_id != WORD_UNK else "[UNK]"
            for token_id in shared_ids
        )
        rows.append(
            {
                "phrase_a": left,
                "phrase_b": right,
                "n_shared": len(shared_ids),
                "shared_tokens": ", ".join(shared_tokens) or "-",
                "btr_a": records[left]["btr"],
                "btr_b": records[right]["btr"],
                "btr_gap": abs(records[left]["btr"] - records[right]["btr"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("btr_gap", ascending=False)
        .reset_index(drop=True)
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", type=int, default=0, help="which fold's training rows fit the vocabulary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_dataset(PROTOCOL.dataset)
    partitions = partition(frame)
    train_indices = partitions.folds[args.fold].train_indices

    encoder = fit_vocabulary(frame, train_indices)
    records = phrase_records(frame, encoder, partitions.development_indices)

    print(
        f"vocabulario ajustado sobre {len(train_indices)} filas de train "
        f"(fold {args.fold}), {len(encoder._words)} palabras distintas"  # noqa: SLF001
    )
    print(
        f"BTR calculado sobre {len(partitions.development_indices)} filas "
        "(train + validacion de todos los folds, nunca test)"
    )
    print(f"{len(records)} frases de popularidad distintas de {NO_PHRASE!r}")

    print("\n=== FRASES: TOKENIZACION Y BTR ===")
    table = phrase_frame(records, encoder)
    display = table.assign(btr=lambda d: (d["btr"] * 100).map("{:.1f}%".format))
    print(display.to_string(index=False))

    print("\n=== PARES: TOKENS COMPARTIDOS FRENTE A LA DIFERENCIA DE BTR ===")
    pairs = pair_frame(records, encoder)
    display = pairs.assign(
        btr_a=lambda d: (d["btr_a"] * 100).map("{:.1f}%".format),
        btr_b=lambda d: (d["btr_b"] * 100).map("{:.1f}%".format),
        btr_gap=lambda d: (d["btr_gap"] * 100).map("{:.1f} pp".format),
    )
    print(display.to_string(index=False))

    shared = pairs[pairs["n_shared"] > 0]
    if shared.empty:
        print("\nninguna frase comparte un token con otra: la hipotesis no aplica aqui")
    else:
        print(
            f"\n{len(shared)} de {len(pairs)} pares comparten al menos un token; "
            f"brecha de BTR promedio en esos pares: {shared['btr_gap'].mean() * 100:.1f} pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
