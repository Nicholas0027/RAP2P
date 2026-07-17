"""P2P-Static: a capacity-matched control for RAP2P's central claim (RQ2).

Same shared rank-block LoRA basis, same router hidden size, same number of
gate values -- but the gate comes from one **static, non-target-conditioned**
respondent profile (demographics + an unweighted average of the K known
answers), computed once and reused for every target question. This isolates
exactly the mechanism RAP2P adds: target-aware, correlation-graph-biased
attention over the same K known answers, recomputed per target question.

Not a reproduction of the official P2P hypernetwork stack (different training
data format, different profile encoder, generates full adapter weights rather
than mixing weights over a shared basis) -- report results as "P2P-style
static-profile control," see README.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import decoder_layers, last_token_logits, restricted_logits
from .gating import patch_rank_block_lora


class P2PStaticModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        hidden_dim: int = 128,
        rank_blocks: int = 4,
        block_rank: int = 4,
        alpha: int = 32,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        target_last_n_layers: int = 8,
        max_options: int = 11,
        answer_embed_dim: int = 16,
        dropout: float = 0.05,
        use_demographics: bool = True,
        use_history: bool = True,
    ):
        super().__init__()
        if getattr(backbone, "is_gradient_checkpointing", False):
            raise ValueError(
                "Load the backbone with gradient_checkpointing=False for P2PStaticModel — "
                "see the matching guard in RAP2PModel for the full rationale (near-zero "
                "memory benefit on a frozen backbone; closure-captured gate makes gradient "
                "flow depend on checkpoint implementation details)."
            )
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        layers = decoder_layers(backbone)
        self.adapter_refs = patch_rank_block_lora(
            list(layers), target_modules, target_last_n_layers, rank_blocks, block_rank, alpha
        )
        self.num_layers = len(self.adapter_refs)
        self.num_modules = len(target_modules)
        self.rank_blocks = rank_blocks
        object.__setattr__(self, "_adapter_flat", [a for modules in self.adapter_refs for a in modules])

        self.bias = nn.Parameter(torch.zeros(self.num_layers, self.num_modules, rank_blocks))
        self.answer_embedding = nn.Embedding(max_options, answer_embed_dim)
        self.demographic_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.history_value = nn.Sequential(
            nn.Linear(embedding_dim + answer_embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.profile_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_layers * self.num_modules * rank_blocks),
        )
        self.use_demographics = use_demographics
        self.use_history = use_history

    def _apply_gates(self, gate: torch.Tensor) -> None:
        flat_index = 0
        for layer_index in range(self.num_layers):
            for module_index in range(self.num_modules):
                self._adapter_flat[flat_index].set_gate(gate[:, layer_index, module_index, :])
                flat_index += 1

    def clear_gates(self) -> None:
        for adapter in self._adapter_flat:
            adapter.clear_gate()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        demographic_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,  # unused: gate is not target-conditioned, by design
        history_item_embeddings: torch.Tensor,
        history_answer_index: torch.Tensor,
        history_mask: torch.Tensor,
        correlation_bias: torch.Tensor,  # unused: no correlation graph in the static control
        label_token_ids: torch.Tensor,
        option_mask: torch.Tensor,
        k: int,
        **_unused_modality_dropout_kwargs,
    ) -> dict[str, torch.Tensor]:
        del item_embeddings, correlation_bias
        batch = demographic_embeddings.shape[0]

        demo = self.demographic_encoder(demographic_embeddings) if self.use_demographics else torch.zeros(
            batch, self.demographic_encoder[0].out_features, device=demographic_embeddings.device
        )

        if self.use_history:
            values = self.history_value(
                torch.cat([history_item_embeddings, self.answer_embedding(history_answer_index)], dim=-1)
            )
            mask = history_mask.to(values.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            history_summary = (values * mask).sum(dim=1) / denom
        else:
            history_summary = torch.zeros(batch, demo.shape[-1], device=demographic_embeddings.device)

        contribution = self.profile_head(torch.cat([demo, history_summary], dim=-1))
        gate_logits = self.bias.unsqueeze(0) + contribution.view(batch, self.num_layers, self.num_modules, self.rank_blocks)
        gate = torch.softmax(gate_logits, dim=-1)

        self._apply_gates(gate)
        vocabulary_logits = last_token_logits(self.backbone, input_ids, attention_mask)
        label_logits = restricted_logits(vocabulary_logits, label_token_ids, option_mask)
        # Gates deliberately persist through backward — see the matching
        # comment in rap2p_model.py: gradient checkpointing recomputes layer
        # forwards during loss.backward() and must observe the same gate.

        return {"label_logits": label_logits, "gate": gate, "mean_gate_share": gate.mean(dim=(0, 1, 2))}
