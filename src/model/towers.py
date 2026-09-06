"""Independent towers and fusion head for the late-fusion architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.attention import Block
from src.model.encoding import TextBatch

INIT_STD = 0.02
N_TOKEN_TYPES = 3

LINEAR, MLP = "linear", "mlp"
ARCHITECTURES: tuple[str, ...] = (LINEAR, MLP)


def _init_weights(module: nn.Module) -> None:
    """Zeroes the ``[PAD]`` row, which ``padding_idx`` then freezes for the whole run."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


class WeightedPooling(nn.Module):

    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.weights(x, mask)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class FirstTokenPooling(WeightedPooling):
    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Position 0 is [CLS] and is never padding. 
        weights = x.new_zeros(mask.shape)
        weights[:, 0] = 1.0
        return weights


class MeanPooling(WeightedPooling):
    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=x.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)


class AttentionPooling(WeightedPooling):
    """A query that is a parameter, not derived from the sequence."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * INIT_STD)

    def weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (x @ self.query) * x.shape[-1] ** -0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)


def _stack(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    architecture: str,
) -> nn.Sequential:
    """Without the hidden layer the output can only add its inputs."""
    if architecture == LINEAR:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    if architecture == MLP:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    raise ValueError(f"unknown architecture {architecture!r}; expected {ARCHITECTURES}")


def sinusoidal(length: int, d_model: int) -> torch.Tensor:
    """Sines and cosines at geometrically spaced frequencies."""
    position = torch.arange(length).unsqueeze(1).float()
    step = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * step)
    table[:, 1::2] = torch.cos(position * step)
    return table


class LearnedPositions(nn.Module):
    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.table = nn.Embedding(length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table(torch.arange(x.shape[1], device=x.device))


class SinusoidalPositions(nn.Module):
    """Scaled to ``INIT_STD``: raw, the table outweighs the token it labels 35 to 1."""

    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.register_buffer("table", sinusoidal(length, d_model) * INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[: x.shape[1]]


class TextTower(nn.Module):
    def __init__(self, vocabulary_size: int, sequence_length: int, config) -> None:
        super().__init__()
        d_model = config.d_model

        self.tokens = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.segments = nn.Embedding(N_TOKEN_TYPES, d_model)

        self.positions = self._positions(
            config.positional, sequence_length, d_model
        )
        self.embedding_norm = (
            nn.LayerNorm(d_model) if config.embedding_norm else nn.Identity()
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            Block(d_model, config.n_heads, config.dropout, config.ffn_multiplier)
            for _ in range(config.n_layers)
        )
        self.pooler = self._pooler(config.pooling, d_model)
        self.projection = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
            if getattr(config, "pooler_projection", True)
            else nn.Identity()
        )
        self.apply(_init_weights)

    @staticmethod
    def _positions(name: str, length: int, d_model: int) -> nn.Module | None:
        if name == "learned":
            return LearnedPositions(length, d_model)
        if name == "sinusoidal":
            return SinusoidalPositions(length, d_model)
        if name == "none":
            return None
        raise ValueError(f"unknown positional encoding: {name}")

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
        mask = batch.attention_mask.bool()
        x = self.tokens(batch.input_ids) + self.segments(batch.token_type_ids)
        if self.positions is not None:
            x = x + self.positions(x)
        return self.embedding_dropout(self.embedding_norm(x)), mask

    def _contextualize(self, batch: TextBatch) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.embed(batch)
        for block in self.blocks:
            x = block(x, mask)
        return x, mask

    def forward_with_pooling_weights(
        self, batch: TextBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self._contextualize(batch)
        weights = self.pooler.weights(x, mask)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return self.projection(pooled), weights

    def pooling_weights(self, batch: TextBatch) -> torch.Tensor:
        x, mask = self._contextualize(batch)
        return self.pooler.weights(x, mask)

    def forward(self, batch: TextBatch) -> torch.Tensor:
        pooled, _ = self.forward_with_pooling_weights(batch)
        return pooled

    def attention_of_cls(self, batch: TextBatch) -> torch.Tensor:
        """Attention from ``[CLS]``: ``(rows, layers, heads, positions)``."""
        x, mask = self.embed(batch)
        collected = []
        for block in self.blocks:
            x, weights = block(x, mask, return_weights=True)
            collected.append(weights[:, :, 0, :])
        return torch.stack(collected, dim=1) if collected else torch.empty(0)


class TabularTower(nn.Module):
    """On PyTorch's default init: ``INIT_STD`` would reach the fusion far quieter."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.1,
        architecture: str = MLP,
    ) -> None:
        super().__init__()
        self.layers = _stack(input_dim, hidden_dim, output_dim, dropout, architecture)

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        return self.layers(x_tab)


class FusionHead(nn.Module):
    """The concatenation only stacks the two vectors; this is what mixes them."""

    def __init__(
        self,
        text_dim: int,
        tabular_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        architecture: str = MLP,
    ) -> None:
        super().__init__()
        self.layers = _stack(
            text_dim + tabular_dim, hidden_dim, 1, dropout, architecture
        )

    def forward(self, h_text: torch.Tensor, h_tab: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((h_text, h_tab), dim=-1)).squeeze(-1)
