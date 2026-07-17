from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rap2p.models.gating import DynamicRankBlockLoRALinear, patch_rank_block_lora  # noqa: E402


def test_gate_mixture_matches_explicit_per_block_sum():
    """Delta_W(x) must equal sum_b g_b * (B_b (A_b x)) * scaling, computed the
    slow, explicit way -- this is the algebraic identity the whole adapter
    layer's efficiency (patch_rank_block_lora) rests on.
    """
    torch.manual_seed(0)
    in_features, out_features = 6, 5
    rank_blocks, block_rank = 4, 3
    base = nn.Linear(in_features, out_features, bias=False)
    layer = DynamicRankBlockLoRALinear(base, rank_blocks, block_rank, alpha=8)
    # Give lora_b nonzero values -- at init it's zero (standard LoRA init) so
    # the identity would trivially hold at 0 regardless of a bug.
    with torch.no_grad():
        layer.lora_b.normal_(std=0.1)

    batch, seq = 3, 2
    x = torch.randn(batch, seq, in_features)
    gate = torch.softmax(torch.randn(batch, rank_blocks), dim=-1)
    layer.set_gate(gate)
    output = layer(x)

    # Explicit per-block computation.
    base_output = base(x)
    expected_dynamic = torch.zeros(batch, seq, out_features)
    for b in range(rank_blocks):
        a_b = layer.lora_a[b * block_rank : (b + 1) * block_rank]  # (block_rank, in_features)
        b_b = layer.lora_b[:, b * block_rank : (b + 1) * block_rank]  # (out_features, block_rank)
        low = torch.einsum("bsi,ri->bsr", x, a_b)
        block_output = torch.einsum("bsr,or->bso", low, b_b)
        expected_dynamic += gate[:, b].view(batch, 1, 1) * block_output
    expected = base_output + expected_dynamic * layer.scaling

    assert torch.allclose(output, expected, atol=1e-5)


def test_clear_gate_falls_back_to_frozen_base_output():
    base = nn.Linear(4, 4, bias=False)
    layer = DynamicRankBlockLoRALinear(base, rank_blocks=2, block_rank=2, alpha=4)
    x = torch.randn(2, 3, 4)
    assert torch.allclose(layer(x), base(x))
    layer.set_gate(torch.ones(2, 2) / 2)
    layer.clear_gate()
    assert torch.allclose(layer(x), base(x))


def test_base_parameters_are_frozen():
    base = nn.Linear(4, 4)
    layer = DynamicRankBlockLoRALinear(base, rank_blocks=2, block_rank=2, alpha=4)
    assert not any(p.requires_grad for p in layer.base.parameters())
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad


def test_set_gate_rejects_wrong_block_count():
    base = nn.Linear(4, 4)
    layer = DynamicRankBlockLoRALinear(base, rank_blocks=4, block_rank=2, alpha=4)
    with pytest.raises(ValueError):
        layer.set_gate(torch.ones(2, 3))  # 3 != rank_blocks(4)


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, hidden, bias=False)


def test_patch_rank_block_lora_only_touches_last_n_layers():
    hidden = 8
    layers = [_TinyLayer(hidden) for _ in range(6)]
    adapter_refs = patch_rank_block_lora(layers, ("q_proj", "v_proj"), target_last_n_layers=2, rank_blocks=2, block_rank=2, alpha=4)
    assert len(adapter_refs) == 2
    for layer in layers[:4]:
        assert isinstance(layer.self_attn.q_proj, nn.Linear)
    for layer in layers[4:]:
        assert isinstance(layer.self_attn.q_proj, DynamicRankBlockLoRALinear)
        assert isinstance(layer.self_attn.v_proj, DynamicRankBlockLoRALinear)
