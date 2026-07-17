"""The RAP2P adapter layer: a rank-R LoRA split into `rank_blocks` blocks of
`block_rank`, mixed per (respondent, target-question) by a softmax gate over
blocks. This is the shared "survey LoRA basis" -- every respondent uses the
same B_b, A_b atoms; only the mixture weights are personalized.

ΔW_{i,j} = sum_b g_{i,j,b} * B_b A_b,  g_{i,j} = softmax(...) in R^{rank_blocks}

Implemented without materializing a full-rank per-example matrix: A is applied
once (concatenated across blocks), the block-wise gate scales the low-rank
intermediate, and B is applied once. This is exactly the block mixture above
because matrix multiplication is linear in the (B_b A_b) terms.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRankBlockLoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank_blocks: int, block_rank: int, alpha: int):
        super().__init__()
        if not hasattr(base, "weight"):
            raise TypeError(f"Expected a linear-like module, got {type(base)}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        out_features, in_features = base.weight.shape
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank_blocks = int(rank_blocks)
        self.block_rank = int(block_rank)
        self.total_rank = self.rank_blocks * self.block_rank
        self.scaling = float(alpha) / max(1, self.total_rank)

        # Named lora_a/lora_b (not the shorter "a"/"b") so training.py's optimizer
        # grouping (`.lora_a`/`.lora_b` suffix match) cannot collide with any other
        # module's attribute names.
        self.lora_a = nn.Parameter(torch.empty(self.total_rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.total_rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self._gate: torch.Tensor | None = None  # (batch, rank_blocks), softmax already applied

    def set_gate(self, gate: torch.Tensor) -> None:
        if gate.shape[-1] != self.rank_blocks:
            raise ValueError(f"Expected gate with {self.rank_blocks} blocks, got shape {tuple(gate.shape)}")
        self._gate = gate

    def clear_gate(self) -> None:
        self._gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self._gate is None:
            return output
        gate = self._gate.to(dtype=x.dtype)  # (batch, rank_blocks)
        low = F.linear(x, self.lora_a.to(dtype=x.dtype))  # (batch, seq, total_rank)
        batch, seq, total_rank = low.shape
        low = low.view(batch, seq, self.rank_blocks, self.block_rank)
        low = low * gate.view(batch, 1, self.rank_blocks, 1)
        low = low.reshape(batch, seq, total_rank)
        dynamic = F.linear(low, self.lora_b.to(dtype=x.dtype)) * self.scaling
        return output + dynamic


def patch_rank_block_lora(
    decoder_layers: list[nn.Module],
    target_modules: tuple[str, ...],
    target_last_n_layers: int,
    rank_blocks: int,
    block_rank: int,
    alpha: int,
) -> list[list[DynamicRankBlockLoRALinear]]:
    """Patch q_proj/v_proj of the last N decoder layers in place; return the
    per-(layer, module) adapter references in a fixed, deterministic order
    matching how the router will index into `set_gate`.
    """
    start = max(0, len(decoder_layers) - int(target_last_n_layers))
    adapter_refs: list[list[DynamicRankBlockLoRALinear]] = []
    for layer in decoder_layers[start:]:
        modules: list[DynamicRankBlockLoRALinear] = []
        for name in target_modules:
            original = getattr(layer.self_attn, name)
            wrapped = DynamicRankBlockLoRALinear(original, rank_blocks, block_rank, alpha)
            setattr(layer.self_attn, name, wrapped)
            modules.append(wrapped)
        adapter_refs.append(modules)
    return adapter_refs
