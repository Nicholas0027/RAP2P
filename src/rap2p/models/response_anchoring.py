"""Target-aware response anchoring: the paper's central mechanism.

For a target question q_j and K known (question, answer) pairs, each known
answer is represented as

    r_ik = MLP([e_k ; E_y(y_ik)])

and weighted by an attention score that mixes learned semantic similarity with
a precomputed, leakage-safe item-item correlation prior C_jk (see
item_graph.py):

    alpha_ijk = softmax_k[ (W_q e_j)^T (W_k e_k) / sqrt(d) + gamma * C_jk ]
    h_{i->j}^H = sum_k alpha_ijk r_ik

Unlike a static profile encoder (P2PStaticModel), this recomputes the summary
for *every* target question, so the same K answers are weighted differently
depending on what is being predicted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResponseAnchoringEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_modules: int,
        rank_blocks: int,
        max_options: int,
        answer_embed_dim: int = 16,
        dropout: float = 0.05,
        correlation_gamma_init: float = 1.0,
        learnable_gamma: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.rank_blocks = rank_blocks
        self.hidden_dim = hidden_dim

        self.answer_embedding = nn.Embedding(max_options, answer_embed_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(embedding_dim + answer_embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.query_projection = nn.Linear(embedding_dim, hidden_dim)
        self.key_projection = nn.Linear(embedding_dim, hidden_dim)
        # Always a Parameter so it follows .to(device) / state_dict; a fixed
        # (non-learnable) gamma is expressed via requires_grad=False rather
        # than a plain tensor attribute, which would not migrate devices.
        self.gamma = nn.Parameter(torch.tensor(float(correlation_gamma_init)), requires_grad=learnable_gamma)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_layers * num_modules * rank_blocks),
        )

    def forward(
        self,
        target_item_embeddings: torch.Tensor,
        history_item_embeddings: torch.Tensor,
        history_answer_index: torch.Tensor,
        history_mask: torch.Tensor,
        correlation_bias: torch.Tensor,
        history_keep: torch.Tensor,
        correlation_keep: torch.Tensor,
        use_correlation_graph: bool = True,
    ) -> torch.Tensor:
        batch = target_item_embeddings.shape[0]
        values = self.value_mlp(
            torch.cat([history_item_embeddings, self.answer_embedding(history_answer_index)], dim=-1)
        )  # (batch, k, hidden)
        query = self.query_projection(target_item_embeddings).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_projection(history_item_embeddings)  # (batch, k, hidden)
        scores = (query * keys).sum(-1) / math.sqrt(self.hidden_dim)  # (batch, k)

        if use_correlation_graph:
            corr = correlation_bias * correlation_keep.to(correlation_bias.dtype).unsqueeze(-1)
            scores = scores + self.gamma * corr

        empty = ~history_mask.any(dim=1)
        safe_mask = history_mask.clone()
        if empty.any():
            safe_mask[empty, 0] = True
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, k, 1)
        summary = (alpha * values).sum(dim=1)  # (batch, hidden)
        summary = summary * (~empty).to(summary.dtype).unsqueeze(-1)
        summary = summary * history_keep.to(summary.dtype).unsqueeze(-1)

        contribution = self.output_head(summary)
        return contribution.view(batch, self.num_layers, self.num_modules, self.rank_blocks)
