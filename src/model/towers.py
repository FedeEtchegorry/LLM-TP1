"""Independent model towers for the late-fusion architecture.
tabular tower and fusion head belong to later tasks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.attention import Block
from src.model.encoding import TextBatch

INIT_STD = 0.02
N_TOKEN_TYPES = 3


def _init_weights(module: nn.Module) -> None:
    """Initialize Transformer linear and embedding layers consistently."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


class FirstTokenPooling(nn.Module):
    """Use the contextualized first position as the row representation."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


class MeanPooling(nn.Module):
    """Average real text positions and exclude padding."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


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
    """The fixed encoding from *Attention is All you Need*."""
    position = torch.arange(length).unsqueeze(1).float()
    step = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * step)
    table[:, 1::2] = torch.cos(position * step)
    return table


class TextTower(nn.Module):
    """Embed and contextualize a ``TextBatch``, then pool it to ``(B, d_model)``."""

    def __init__(self, vocabulary_size: int, sequence_length: int, config) -> None:
        super().__init__()
        d_model = config.d_model

        self.tokens = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.segments = nn.Embedding(N_TOKEN_TYPES, d_model)

        if config.positional == "learned":
            self.positions = nn.Embedding(sequence_length, d_model)
        elif config.positional == "sinusoidal":
            self.register_buffer("positions", sinusoidal(sequence_length, d_model))
        elif config.positional == "none":
            self.positions = None
        else:
            raise ValueError(f"unknown positional encoding: {config.positional}")

        self.blocks = nn.ModuleList(
            Block(d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.pooler = self._pooler(config.pooling, d_model)
        self.apply(_init_weights)

    @staticmethod
    def _pooler(name: str, d_model: int) -> nn.Module:
        if name == "cls":
            return FirstTokenPooling()
        if name == "mean":
            return MeanPooling()
        if name == "attention":
            return AttentionPooling(d_model)
        raise ValueError(f"unknown pooling: {name}")

    def embed(self, batch: TextBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the text sequence and its real-position mask."""
        mask = batch.attention_mask.bool()
        x = self.tokens(batch.input_ids) + self.segments(batch.token_type_ids)

        if isinstance(self.positions, nn.Embedding):
            position_ids = torch.arange(x.shape[1], device=x.device)
            x = x + self.positions(position_ids)
        elif self.positions is not None:
            x = x + self.positions[: x.shape[1]]
        return x, mask

    def forward(self, batch: TextBatch) -> torch.Tensor:
        x, mask = self.embed(batch)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(self.pooler(x, mask))
