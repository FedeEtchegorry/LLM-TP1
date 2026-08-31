"""The row model: heterogeneous embeddings, the encoder, and one logit out of ``[CLS]``.

This is where we leave the course notebook behind. It has one table of word embeddings
and predicts the next token at every position; we build a different vector for each
kind of position and predict a single probability for the row.

The numeric embedding is the decision ``docs/EDA.md`` §4 calls the most important in
the problem::

    e = value * w_col + b_col        # affine: keeps the ordering and the precision
      + bucket_table[bin(value)]     # free-form: represents the inverted U directly

``price_pct`` is ordered but its relation to buying is not monotonic, so the ordered
term alone cannot bend into the hump. The bucket table is the same column read a
second time as a category, and axis A measures what that second reading is worth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.attention import Block
from src.model.encoding import EncodedRows, RowEncoder

INIT_STD = 0.02
"""The notebook's initialisation, kept."""


PERIODIC_FREQUENCIES = 8
PERIODIC_SIGMA = 1.0


class NumericEmbedding(nn.Module):
    """One vector per numeric column, computed from the value rather than looked up.

    ``mode`` is axis A: ``affine`` keeps only the ordered term, ``buckets`` only the
    free-form one, ``affine+buckets`` sums them, and ``none`` leaves the position in
    the sequence but reads no value from it -- a column the model knows is there and
    nothing else. ``piecewise`` weights one vector per bucket by how far the value
    travelled through each, so it bends like ``buckets`` and stays ordered like
    ``affine``; ``periodic`` reads the value through ``sin`` and ``cos`` at learned
    frequencies instead.
    """

    def __init__(
        self, n_fields: int, n_buckets: int, d_model: int, mode: str
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.n_buckets = n_buckets
        self.affine = mode in ("affine", "affine+buckets")
        self.bucketed = mode in ("buckets", "affine+buckets")
        self.piecewise = mode == "piecewise"
        self.periodic = mode == "periodic"

        self.bias = nn.Parameter(torch.zeros(n_fields, d_model))
        self.missing = nn.Parameter(torch.randn(d_model) * INIT_STD)

        # Only what the mode uses is allocated, so the parameter count reports the
        # cost of the axis honestly: L3 and L4 differ by exactly the bucket table.
        self.weight = (
            nn.Parameter(torch.randn(n_fields, d_model) * INIT_STD)
            if self.affine
            else None
        )
        self.buckets = (
            nn.Embedding(n_fields * n_buckets, d_model) if self.bucketed else None
        )
        if self.buckets is not None:
            nn.init.normal_(self.buckets.weight, std=INIT_STD)

        self.pieces = (
            nn.Parameter(torch.randn(n_fields, n_buckets, d_model) * INIT_STD)
            if self.piecewise
            else None
        )

        if self.periodic:
            self.frequencies = nn.Parameter(
                torch.randn(n_fields, PERIODIC_FREQUENCIES) * PERIODIC_SIGMA
            )
            self.projection = nn.Parameter(
                torch.randn(n_fields, 2 * PERIODIC_FREQUENCIES, d_model) * INIT_STD
            )
        else:
            self.frequencies = None
            self.projection = None

    def forward(
        self,
        values: torch.Tensor,
        buckets: torch.Tensor,
        missing: torch.Tensor,
        ratios: torch.Tensor,
    ) -> torch.Tensor:
        """``(batch, n_fields)`` in, ``(batch, n_fields, d_model)`` out."""
        out = self.bias.unsqueeze(0).expand(values.shape[0], -1, -1).clone()
        if self.weight is not None:
            out = out + values.unsqueeze(-1) * self.weight
        if self.buckets is not None:
            offsets = torch.arange(self.n_fields, device=buckets.device) * self.n_buckets
            out = out + self.buckets(buckets + offsets)
        if self.pieces is not None:
            out = out + torch.einsum("bfk,fkd->bfd", ratios, self.pieces)
        if self.frequencies is not None:
            angles = 2.0 * math.pi * values.unsqueeze(-1) * self.frequencies
            waves = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            out = out + torch.einsum("bfk,fkd->bfd", waves, self.projection)
        return out + missing.unsqueeze(-1) * self.missing


class AttentionPooling(nn.Module):
    """A learned query that decides how much each position contributes."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * INIT_STD)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (x @ self.query) * x.shape[-1] ** -0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        return (torch.softmax(scores, dim=-1).unsqueeze(-1) * x).sum(dim=1)


def sinusoidal(length: int, d_model: int) -> torch.Tensor:
    """The fixed encoding from *Attention is All you Need*, for axis E."""
    position = torch.arange(length).unsqueeze(1).float()
    step = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * step)
    table[:, 1::2] = torch.cos(position * step)
    return table


class BtrTransformer(nn.Module):
    """Encoder over one row's heterogeneous sequence, ending in a single logit."""

    def __init__(self, encoder: RowEncoder, config, n_buckets: int) -> None:
        super().__init__()
        self.pooling = config.pooling
        d_model = config.d_model
        length = encoder.sequence_length + encoder.n_numeric

        self.tokens = nn.Embedding(encoder.vocabulary_size, d_model, padding_idx=0)
        self.fields = nn.Embedding(encoder.n_fields, d_model)
        self.numbers = (
            NumericEmbedding(
                encoder.n_numeric, n_buckets, d_model, config.numeric_embedding
            )
            if encoder.n_numeric
            else None
        )

        if config.positional == "learned":
            self.positions = nn.Embedding(length, d_model)
        elif config.positional == "sinusoidal":
            self.register_buffer("positions", sinusoidal(length, d_model))
        else:
            self.positions = None

        self.blocks = nn.ModuleList(
            Block(d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.attention_pool = (
            AttentionPooling(d_model) if config.pooling == "attention" else None
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def embed(self, batch: EncodedRows) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the sequence and the mask that says which positions are real."""
        x = self.tokens(batch.token_ids) + self.fields(batch.field_ids)
        mask = batch.padding_mask

        if self.numbers is not None:
            numeric = self.numbers(
                batch.numeric_values,
                batch.numeric_buckets,
                batch.numeric_missing,
                batch.numeric_ratios,
            )
            x = torch.cat([x, numeric], dim=1)
            mask = torch.cat(
                [mask, torch.ones(numeric.shape[:2], dtype=torch.bool, device=mask.device)],
                dim=1,
            )

        if isinstance(self.positions, nn.Embedding):
            x = x + self.positions(torch.arange(x.shape[1], device=x.device))
        elif self.positions is not None:
            x = x + self.positions[: x.shape[1]]
        return x, mask

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return x[:, 0]
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).float()
            return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.attention_pool(x, mask)

    def forward(self, batch: EncodedRows) -> torch.Tensor:
        """Logits, one per row."""
        x, mask = self.embed(batch)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(self.pool(x, mask))).squeeze(-1)

    def attention_of_cls(self, batch: EncodedRows) -> torch.Tensor:
        """Per-layer, per-head attention from ``[CLS]``, for the interpretability slide."""
        x, mask = self.embed(batch)
        collected = []
        for block in self.blocks:
            x, weights = block(x, mask, return_weights=True)
            collected.append(weights[:, :, 0, :])
        return torch.stack(collected, dim=1) if collected else torch.empty(0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
