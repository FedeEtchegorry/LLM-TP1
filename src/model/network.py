"""The late-fusion BTR model, composed from independent text and tabular towers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.encoding import TABULAR_WIDTH, RowEncoder, TabBatch, TextBatch
from src.model.towers import FusionHead, TabularTower, TextTower

TowerBatch = tuple[TextBatch, TabBatch]


class BtrTransformer(nn.Module):
    """Compose the two modality-specific towers and the late-fusion head."""

    def __init__(self, encoder: RowEncoder, config) -> None:
        super().__init__()
        self.text_tower = TextTower(
            vocabulary_size=encoder.vocabulary_size,
            sequence_length=encoder.sequence_length,
            config=config,
        )
        self.tabular_tower = TabularTower(
            input_dim=TABULAR_WIDTH,
            output_dim=config.tabular_dim,
            dropout=config.dropout,
            architecture=config.tab_tower,
        )
        self.fusion_head = FusionHead(
            text_dim=config.d_model,
            tabular_dim=config.tabular_dim,
            dropout=config.dropout,
            architecture=config.fusion_head,
        )

    def forward(self, batch: TowerBatch) -> torch.Tensor:
        text_batch, tabular_batch = batch
        h_text = self.text_tower(text_batch)
        h_tab = self.tabular_tower(tabular_batch.x_tab)
        return self.fusion_head(h_text, h_tab)

    def attention_of_cls(self, batch: TowerBatch) -> torch.Tensor:
        """Only the text tower has a sequence to attend over; the tabular half of the
        batch is not read."""
        text_batch, _ = batch
        return self.text_tower.attention_of_cls(text_batch)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
