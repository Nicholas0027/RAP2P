"""Test the Stage-1 warm start: mapping a Global-QLoRA (PEFT) checkpoint's
lora_A/lora_B into the rank-block basis, with the uniform-gate compensation
(lora_b scaled by rank_blocks) that makes the initial ΔW reproduce the
Stage-1 adapter under a freshly-initialized (≈uniform) gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fakes import EMBED_DIM, HIDDEN, N_LAYERS, N_OPTIONS, VOCAB, FakeCausalLM  # noqa: E402
from rap2p.models.rap2p_model import RAP2PModel  # noqa: E402
from rap2p.workflows import initialize_basis_from_peft_checkpoint  # noqa: E402

RANK_BLOCKS, BLOCK_RANK = 2, 2
TOTAL_RANK = RANK_BLOCKS * BLOCK_RANK


def _fake_peft_checkpoint(tmpdir: Path, n_layers: int, hidden: int) -> tuple[Path, dict]:
    state = {}
    for layer in range(n_layers):
        for module in ("q_proj", "v_proj"):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            state[f"{prefix}.lora_A.default.weight"] = torch.randn(TOTAL_RANK, hidden)
            state[f"{prefix}.lora_B.default.weight"] = torch.randn(hidden, TOTAL_RANK)
    path = tmpdir / "best_nll.pt"
    torch.save({"model": state, "metadata": {}}, path)
    return path, state


def test_warm_start_copies_last_n_layers_with_gate_compensation():
    torch.manual_seed(0)
    model = RAP2PModel(
        FakeCausalLM(VOCAB, HIDDEN, N_LAYERS),
        embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=RANK_BLOCKS, block_rank=BLOCK_RANK, alpha=4,
        target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS,
        answer_embed_dim=4, dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path, state = _fake_peft_checkpoint(Path(tmpdir), N_LAYERS, HIDDEN)
        n_initialized = initialize_basis_from_peft_checkpoint(model, path, RANK_BLOCKS)

    assert n_initialized == 2 * 2  # last 2 layers x (q_proj, v_proj)
    offset = N_LAYERS - model.num_layers
    for local_index, modules in enumerate(model.adapter_refs):
        for module_index, adapter in enumerate(modules):
            module = ("q_proj", "v_proj")[module_index]
            prefix = f"base_model.model.model.layers.{offset + local_index}.self_attn.{module}"
            assert torch.allclose(adapter.lora_a, state[f"{prefix}.lora_A.default.weight"])
            assert torch.allclose(adapter.lora_b, state[f"{prefix}.lora_B.default.weight"] * RANK_BLOCKS)


def test_warm_start_uniform_gate_reproduces_stage1_delta():
    """With a uniform gate (1/B per block), the compensated basis must produce
    exactly the Stage-1 adapter's ΔW = scaling * B @ A on any input."""
    torch.manual_seed(1)
    model = RAP2PModel(
        FakeCausalLM(VOCAB, HIDDEN, N_LAYERS),
        embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=RANK_BLOCKS, block_rank=BLOCK_RANK, alpha=4,
        target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS,
        answer_embed_dim=4, dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path, state = _fake_peft_checkpoint(Path(tmpdir), N_LAYERS, HIDDEN)
        initialize_basis_from_peft_checkpoint(model, path, RANK_BLOCKS)

    adapter = model.adapter_refs[0][0]
    offset = N_LAYERS - model.num_layers
    prefix = f"base_model.model.model.layers.{offset}.self_attn.q_proj"
    a = state[f"{prefix}.lora_A.default.weight"]
    b = state[f"{prefix}.lora_B.default.weight"]

    x = torch.randn(2, 3, HIDDEN)
    uniform_gate = torch.full((2, RANK_BLOCKS), 1.0 / RANK_BLOCKS)
    adapter.set_gate(uniform_gate)
    output = adapter(x)
    stage1_delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * adapter.scaling
    expected = adapter.base(x) + stage1_delta
    assert torch.allclose(output, expected, atol=1e-5)


def test_warm_start_raises_on_rank_mismatch():
    torch.manual_seed(2)
    model = RAP2PModel(
        FakeCausalLM(VOCAB, HIDDEN, N_LAYERS),
        embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=4, block_rank=2, alpha=4,  # total rank 8 != checkpoint's 4
        target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS,
        answer_embed_dim=4, dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _fake_peft_checkpoint(Path(tmpdir), N_LAYERS, HIDDEN)
        with pytest.raises(ValueError, match="Shape mismatch"):
            initialize_basis_from_peft_checkpoint(model, path, 4)
