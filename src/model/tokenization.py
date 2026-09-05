"""How a string becomes a list of integers, on two independent axes.

+----------------+-------------------------------+------------------------------+
|                | ``keep_brackets = false``     | ``keep_brackets = true``     |
+================+===============================+==============================+
| ``whole-word`` | the first delivery, verbatim  | whole words **plus**         |
|                | -- all punctuation deleted    | punctuation as tokens        |
+----------------+-------------------------------+------------------------------+
| ``wordpiece``  | the control: brackets removed | the proposal: BERT's         |
|                | on purpose, rest untouched    | pipeline as it is            |
+----------------+-------------------------------+------------------------------+

"""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Self

PAD, UNK, CLS, SEP = 0, 1, 2, 3
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
N_SPECIAL = len(SPECIAL_TOKENS)
"""``[PAD]`` is id 0 so it can stay the token embedding's ``padding_idx``. There is no
``[MASK]``: no masked-language-modelling stage exists"""

CONTINUATION = "##"
DEFAULT_VOCABULARY_SIZE = 3_000
DEFAULT_MIN_FREQUENCY = 1
"""One occurrence is enough. The corpus has 422 distinct words and zero hapax, so a
higher threshold would only discard subword units we can afford to keep."""

BRACKETS = ("(", ")")

WHOLE_WORD = "whole-word"
WORDPIECE = "wordpiece"
FAMILIES: tuple[str, ...] = (WHOLE_WORD, WORDPIECE)

_WORDS_ONLY = re.compile(r"[a-z0-9]+")
_WORDS_AND_PUNCTUATION = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


class Tokenizer(Protocol):
    family: str
    keep_brackets: bool
    name: str

    def fit(self, texts: Iterable[str]) -> Self: ...

    def tokens(self, text: object) -> list[str]: ...

    def encode(self, text: object) -> list[int]: ...

    @property
    def vocabulary_size(self) -> int: ...

    @property
    def vocabulary(self) -> dict[str, int]: ...


class WholeWordTokenizer:
    """Whole words, with punctuation either deleted or kept as its own token.

    With ``keep_brackets=False`` this is byte-for-byte the first delivery's
    ``encoding.tokenize``: it is the baseline the reported v1 numbers were measured
    on, so it is kept identical rather than reimplemented. Unknown words all collapse
    onto a single ``[UNK]`` row, which is one of the reasons subwords are worth trying.
    """

    family = WHOLE_WORD

    def __init__(self, keep_brackets: bool = False) -> None:
        self.keep_brackets = keep_brackets
        self.name = f"{WHOLE_WORD}+punct" if keep_brackets else WHOLE_WORD
        self._pattern = _WORDS_AND_PUNCTUATION if keep_brackets else _WORDS_ONLY
        self._words: dict[str, int] = {}
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> Self:
        seen: set[str] = set()
        for text in texts:
            seen.update(self.tokens(text))
        self._words = {
            word: N_SPECIAL + position for position, word in enumerate(sorted(seen))
        }
        self._fitted = True
        return self

    def tokens(self, text: object) -> list[str]:
        return self._pattern.findall(str(text).lower())

    def encode(self, text: object) -> list[int]:
        self._require_fitted()
        return [self._words.get(token, UNK) for token in self.tokens(text)]

    @property
    def vocabulary_size(self) -> int:
        return N_SPECIAL + len(self._words)

    @property
    def vocabulary(self) -> dict[str, int]:
        return dict(zip(SPECIAL_TOKENS, range(N_SPECIAL))) | self._words

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"the {self.name} tokenizer was never fitted")


class WordPieceTokenizer:
    """BERT's tokenizer, with its vocabulary trained on our corpus rather than loaded.

    The four canonical pieces, in the order they run:

    ==============  =========================================================
    Normalizer      NFKC, strip accents, lowercase
    Pre-tokenizer   ``BertPreTokenizer``: splits on whitespace **and emits each
                    punctuation character as its own token**
    Model           WordPiece, greedy longest-match-first, ``##`` marking a
                    continuation (``steamable`` -> ``steam`` ``##able``)
    Trainer         ``WordPieceTrainer``, merging pairs by likelihood ratio
    ==============  =========================================================

    With ``keep_brackets=False`` the brackets are replaced by spaces before anything
    else runs, and every other punctuation mark is left where it is.

    """

    family = WORDPIECE

    def __init__(
        self,
        keep_brackets: bool = True,
        vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
    ) -> None:
        self.keep_brackets = keep_brackets
        self.name = WORDPIECE if keep_brackets else f"{WORDPIECE}-no-brackets"
        self.target_vocabulary_size = vocabulary_size
        self.min_frequency = min_frequency
        self._tokenizer = None

    def fit(self, texts: Iterable[str]) -> Self:
        models, normalizers, pre_tokenizers, trainers, HFTokenizer = _tokenizers()

        tokenizer = HFTokenizer(
            models.WordPiece(unk_token="[UNK]", continuing_subword_prefix=CONTINUATION)
        )
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), normalizers.StripAccents(), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.target_vocabulary_size,
            min_frequency=self.min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            continuing_subword_prefix=CONTINUATION,
            show_progress=False,
        )
        tokenizer.train_from_iterator((self._prepare(text) for text in texts), trainer)
        self._tokenizer = tokenizer
        return self

    def tokens(self, text: object) -> list[str]:
        return self._encoding(text).tokens

    def encode(self, text: object) -> list[int]:
        return self._encoding(text).ids

    @property
    def vocabulary_size(self) -> int:
        self._require_fitted()
        return self._tokenizer.get_vocab_size()

    @property
    def vocabulary(self) -> dict[str, int]:
        self._require_fitted()
        return self._tokenizer.get_vocab()

    def _prepare(self, text: object) -> str:
        prepared = str(text)
        if self.keep_brackets:
            return prepared
        for bracket in BRACKETS:
            prepared = prepared.replace(bracket, " ")
        return prepared

    def _encoding(self, text: object):
        self._require_fitted()
        return self._tokenizer.encode(self._prepare(text), add_special_tokens=False)

    def _require_fitted(self) -> None:
        if self._tokenizer is None:
            raise RuntimeError(f"the {self.name} tokenizer was never fitted")


def tokenizer_for(family: str, keep_brackets: bool, **options) -> Tokenizer:
    if family not in FAMILIES:
        raise ValueError(f"unknown tokenizer {family!r}; expected one of {FAMILIES}")
    if family == WHOLE_WORD:
        if options:
            raise ValueError("the whole-word tokenizer takes no options")
        return WholeWordTokenizer(keep_brackets=keep_brackets)
    return WordPieceTokenizer(keep_brackets=keep_brackets, **options)


def lengths(tokenizer: Tokenizer, texts: Iterable[str]) -> list[int]:
    return [len(tokenizer.encode(text)) for text in texts]


def _tokenizers():
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import models, normalizers, pre_tokenizers, trainers
    except ImportError as error:  # pragma: no cover -- environment, not logic
        raise ImportError(
            "WordPiece needs the 'tokenizers' package: pip install -r requirements.txt"
        ) from error
    return models, normalizers, pre_tokenizers, trainers, HFTokenizer
