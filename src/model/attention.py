"""Self-attention written by hand, from the course notebook, made encoder-only.

Kept from ``clase2_step_by_step_transformer.ipynb``: the class names, the shapes and
the pre-LN block, so the filiation with the class material stays visible.

Changed, each change a design decision:

- **No causal mask.** The notebook is decoder-only and hides the future because it
  generates text left to right. A row is classified whole, so the price token has to
  be able to look at the popularity phrase and the other way around.
- **A padding mask instead.** The notebook's windows are a fixed 32 tokens; our rows
  carry 30 to 49 text tokens, so the padded tail must not be attended to.
- **The scaling is applied once.** ``Head.forward`` in the notebook divides by
  ``sqrt(d_k)`` twice -- ``* k.shape[-1]**-0.5`` and then ``/ self.head_size**0.5``,
  where ``k.shape[-1]`` *is* ``head_size``. The net scaling is ``1/d_k`` instead of
  ``1/sqrt(d_k)``, which flattens attention towards a uniform average; at
  ``head_size=16`` the scores come out four times too small.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

NEGATIVE_INFINITY = float("-inf")


class Head(nn.Module):
    """One head of self-attention: every position attends to every real position."""

    def __init__(self, head_size: int, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x`` is (batch, positions, n_embd); ``padding_mask`` is True on real ones."""
        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        if padding_mask is not None:
            weights = weights.masked_fill(
                ~padding_mask.unsqueeze(1), NEGATIVE_INFINITY
            )
        weights = F.softmax(weights, dim=-1)

        out = self.dropout(weights) @ self.value(x)
        return (out, weights) if return_weights else out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated and projected back to ``n_embd``."""

    def __init__(
        self, num_heads: int, head_size: int, n_embd: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(head_size, n_embd, dropout) for _ in range(num_heads)
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_weights:
            joined = torch.cat([head(x, padding_mask) for head in self.heads], dim=-1)
            return self.dropout(self.proj(joined))

        outputs, weights = zip(
            *(head(x, padding_mask, return_weights=True) for head in self.heads)
        )
        joined = torch.cat(outputs, dim=-1)
        return self.dropout(self.proj(joined)), torch.stack(weights, dim=1)


class FeedFoward(nn.Module):
    """The position-wise MLP: out to four times ``n_embd`` and back."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN block: normalise, sublayer, add back onto the residual stream."""

    def __init__(
        self, n_embd: int, n_head: int, dropout: float
    ) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError(f"n_embd={n_embd} is not divisible by n_head={n_head}")
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            attended, weights = self.sa(
                self.ln1(x), padding_mask, return_weights=True
            )
        else:
            attended, weights = self.sa(self.ln1(x), padding_mask), None
        x = x + attended
        x = x + self.ffwd(self.ln2(x))
        return (x, weights) if return_weights else x
