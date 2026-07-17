"""RAP2P: demographic prior + target-aware response anchoring -> prior-residual
gate -> rank-block-gated LoRA (see gating.py for the adapter, response_anchoring.py
for the response-anchoring mechanism, demographic_prior.py for the prior branch).

    g_ij^(l,m) = softmax[ b^(l,m) + r_ij^d,(l,m) + rho_K * r_ij^H,(l,m) ]
    rho_K = K / (K + tau)

Ablation surface (see README "Baseline naming honesty" and paper draft
Limitations for which of these are "free" via modality dropout at eval time
vs. require a separately retrained checkpoint):

  - use_demographics=False        : zero the prior contribution (also toggled per-example
                                     at train time via `demographics_keep`; free at eval time)
  - use_history=False              : zero the response-anchoring contribution (ditto; free at eval time)
  - use_correlation_graph=False     : disable the correlation-graph bias term entirely.
                                      Set at *construction* time for the true RAP2P-noGraph
                                      ablation (gamma fixed at 0, non-trainable); the stochastic
                                      `correlation_keep` dropout mask gives a second, "free" version
                                      of this ablation on the main checkpoint for cross-checking.
  - uniform_gate=True               : ignore the learned gate entirely, mix the four rank
                                       blocks uniformly (tests whether the router does anything).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import decoder_layers, last_token_logits, restricted_logits
from .demographic_prior import DemographicPriorRouter
from .gating import patch_rank_block_lora
from .response_anchoring import ResponseAnchoringEncoder


class RAP2PModel(nn.Module):
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
        evidence_tau: float = 2.0,
        use_demographics: bool = True,
        use_history: bool = True,
        use_correlation_graph: bool = True,
        learnable_gamma: bool = True,
        uniform_gate: bool = False,
    ):
        super().__init__()
        if getattr(backbone, "is_gradient_checkpointing", False):
            raise ValueError(
                "Load the backbone with gradient_checkpointing=False for RAP2PModel "
                "(workflows.build_model_and_collator does this automatically). Checkpointing "
                "buys almost nothing here — the frozen backbone builds no autograd graph "
                "before the first patched layer, so only the last target_last_n_layers "
                "retain activations anyway — and the per-example gate reaches the "
                "checkpointed region as a closure-captured non-leaf tensor, which makes "
                "correct gradient flow depend on non-reentrant-checkpoint implementation "
                "details (verified to work on the pinned torch version, see "
                "test_gradients_survive_checkpointing_with_persistent_gates, but not a "
                "contract worth relying on in production runs)."
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
        self.demographic_prior = DemographicPriorRouter(
            embedding_dim, hidden_dim, self.num_layers, self.num_modules, rank_blocks, dropout
        )
        self.response_anchoring = ResponseAnchoringEncoder(
            embedding_dim, hidden_dim, self.num_layers, self.num_modules, rank_blocks,
            max_options, answer_embed_dim, dropout, correlation_gamma_init=1.0, learnable_gamma=learnable_gamma,
        )
        self.evidence_tau = float(evidence_tau)
        self.use_demographics = use_demographics
        self.use_history = use_history
        self.use_correlation_graph = use_correlation_graph
        self.uniform_gate = uniform_gate

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
        item_embeddings: torch.Tensor,
        history_item_embeddings: torch.Tensor,
        history_answer_index: torch.Tensor,
        history_mask: torch.Tensor,
        correlation_bias: torch.Tensor,
        label_token_ids: torch.Tensor,
        option_mask: torch.Tensor,
        k: int,  # nominal batch K; kept for the shared model_forward interface, but rho uses history_mask row sums (below)
        demographics_keep: torch.Tensor | None = None,
        history_keep: torch.Tensor | None = None,
        correlation_keep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = demographic_embeddings.shape[0]
        device = demographic_embeddings.device
        if demographics_keep is None:
            demographics_keep = torch.ones(batch, dtype=torch.bool, device=device)
        if history_keep is None:
            history_keep = torch.ones(batch, dtype=torch.bool, device=device)
        if correlation_keep is None:
            correlation_keep = torch.ones(batch, dtype=torch.bool, device=device)
        if not self.use_demographics:
            demographics_keep = torch.zeros_like(demographics_keep)
        if not self.use_history:
            history_keep = torch.zeros_like(history_keep)
        if not self.use_correlation_graph:
            correlation_keep = torch.zeros_like(correlation_keep)

        prior_contribution = self.demographic_prior(demographic_embeddings, item_embeddings)
        prior_contribution = prior_contribution * demographics_keep.to(prior_contribution.dtype).view(batch, 1, 1, 1)

        history_contribution = self.response_anchoring(
            item_embeddings, history_item_embeddings, history_answer_index, history_mask,
            correlation_bias, history_keep, correlation_keep, use_correlation_graph=self.use_correlation_graph,
        )
        # Evidence weight from each example's ACTUAL number of known answers
        # (history_mask row sums), not the batch's nominal `k` — history_rows
        # can return fewer than k pairs for a sparsely-answering respondent,
        # and rho must reflect the evidence the model actually received.
        effective_k = history_mask.sum(dim=1).to(history_contribution.dtype)
        rho_k = (effective_k / (effective_k + self.evidence_tau)).view(batch, 1, 1, 1)

        gate_logits = self.bias.unsqueeze(0) + prior_contribution + rho_k * history_contribution
        if self.uniform_gate:
            gate = torch.full_like(gate_logits, 1.0 / self.rank_blocks)
        else:
            gate = torch.softmax(gate_logits, dim=-1)

        self._apply_gates(gate)
        vocabulary_logits = last_token_logits(self.backbone, input_ids, attention_mask)
        label_logits = restricted_logits(vocabulary_logits, label_token_ids, option_mask)
        # Do NOT clear gates here. Gradient checkpointing re-executes each
        # wrapped decoder layer's forward during loss.backward() to rebuild the
        # graph; the recompute reads the adapter's `_gate` attribute via
        # closure, so clearing it before backward would make the recompute take
        # the no-adapter branch — either crashing the recompute-consistency
        # check or silently severing every trainable parameter's gradient.
        # Gates persist until the next forward overwrites them (each forward
        # always calls _apply_gates first, so stale state cannot influence a
        # later batch); call clear_gates() explicitly if adapter-free backbone
        # output is ever needed.

        return {
            "label_logits": label_logits,
            "gate": gate,
            "mean_gate_share": gate.mean(dim=(0, 1, 2)),
            "rho_k": rho_k,
        }
