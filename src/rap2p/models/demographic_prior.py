"""Demographic prior router: r_ij^d = R_d([h_i^d; e_j]).

`h_i^d` comes from a frozen sentence embedding of the respondent's formatted
demographic text (see data.py:format_demographics), projected by a small
trainable MLP. Conditioning additionally on the target item embedding `e_j`
lets the *same* demographic profile push the gate differently depending on
which question is being asked (e.g. income matters more for a taxation
question than for a national-pride question).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DemographicPriorRouter(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_modules: int,
        rank_blocks: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.rank_blocks = rank_blocks
        self.demographic_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.item_projection = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_layers * num_modules * rank_blocks),
        )

    def forward(self, demographic_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        demo = self.demographic_encoder(demographic_embeddings)
        item = self.item_projection(item_embeddings)
        contribution = self.output_head(torch.cat([demo, item], dim=-1))
        batch = demographic_embeddings.shape[0]
        return contribution.view(batch, self.num_layers, self.num_modules, self.rank_blocks)
