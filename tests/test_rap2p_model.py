"""End-to-end forward-pass tests for RAP2PModel against a tiny stand-in
backbone (no real HF download needed) -- these exercise the actual mechanism
(prior-residual gate, evidence weighting rho_K, ablation flags) rather than
just checking tensor shapes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rap2p.models.rap2p_model import RAP2PModel  # noqa: E402
from rap2p.models.p2p_static import P2PStaticModel  # noqa: E402
from fakes import EMBED_DIM, HIDDEN, N_LAYERS, N_OPTIONS, VOCAB, FakeCausalLM  # noqa: E402


def _build_rap2p(**overrides) -> RAP2PModel:
    torch.manual_seed(0)
    backbone = FakeCausalLM(VOCAB, HIDDEN, N_LAYERS)
    kwargs = dict(
        embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=2, block_rank=2, alpha=4,
        target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS,
        answer_embed_dim=4, dropout=0.0, evidence_tau=2.0,
    )
    kwargs.update(overrides)
    return RAP2PModel(backbone, **kwargs)


def _dummy_batch(batch: int = 3, seq: int = 5, max_k: int = 4):
    return dict(
        input_ids=torch.randint(0, VOCAB, (batch, seq)),
        attention_mask=torch.ones(batch, seq, dtype=torch.long),
        demographic_embeddings=torch.randn(batch, EMBED_DIM),
        item_embeddings=torch.randn(batch, EMBED_DIM),
        history_item_embeddings=torch.randn(batch, max_k, EMBED_DIM),
        history_answer_index=torch.randint(0, N_OPTIONS, (batch, max_k)),
        history_mask=torch.ones(batch, max_k, dtype=torch.bool),
        correlation_bias=torch.randn(batch, max_k) * 0.5,
        label_token_ids=torch.arange(N_OPTIONS),
        option_mask=torch.ones(batch, N_OPTIONS, dtype=torch.bool),
    )


def test_forward_shapes():
    model = _build_rap2p()
    batch_inputs = _dummy_batch()
    output = model(**batch_inputs, k=5)
    assert output["label_logits"].shape == (3, N_OPTIONS)
    assert output["gate"].shape == (3, model.num_layers, model.num_modules, model.rank_blocks)
    assert torch.allclose(output["gate"].sum(dim=-1), torch.ones(3, model.num_layers, model.num_modules), atol=1e-5)


def test_empty_history_mask_zeroes_out_history_contribution():
    """rho is derived from history_mask row sums (the evidence the model
    actually received), so an all-padding history must contribute nothing to
    the gate no matter what garbage sits in the padded embedding slots."""
    model = _build_rap2p()
    batch_inputs = _dummy_batch()
    batch_inputs["history_mask"] = torch.zeros_like(batch_inputs["history_mask"])
    out_empty = model(**batch_inputs, k=0)
    batch_inputs2 = dict(batch_inputs)
    batch_inputs2["history_item_embeddings"] = torch.randn_like(batch_inputs["history_item_embeddings"]) * 50
    out_empty_different_padding = model(**batch_inputs2, k=0)
    assert torch.allclose(out_empty["gate"], out_empty_different_padding["gate"], atol=1e-5)
    assert torch.allclose(out_empty["rho_k"], torch.zeros_like(out_empty["rho_k"]))


def test_rho_uses_actual_history_count_not_nominal_k():
    """A respondent with fewer answered items than the batch's nominal K gets
    a smaller evidence weight -- rho comes from history_mask, not from k."""
    model = _build_rap2p()
    batch_inputs = _dummy_batch(batch=2, max_k=4)
    batch_inputs["history_mask"] = torch.tensor([[True, True, True, True], [True, False, False, False]])
    output = model(**batch_inputs, k=8)  # nominal k deliberately wrong
    rho = output["rho_k"].view(-1)
    assert rho[0].item() == pytest.approx(4 / (4 + 2))
    assert rho[1].item() == pytest.approx(1 / (1 + 2))


def test_demographics_keep_false_changes_gate():
    model = _build_rap2p()
    batch_inputs = _dummy_batch()
    keep_true = torch.ones(3, dtype=torch.bool)
    keep_false = torch.zeros(3, dtype=torch.bool)
    out_with = model(**batch_inputs, k=3, demographics_keep=keep_true)
    out_without = model(**batch_inputs, k=3, demographics_keep=keep_false)
    assert not torch.allclose(out_with["gate"], out_without["gate"])


def test_use_demographics_false_forces_keep_regardless_of_mask():
    model = _build_rap2p(use_demographics=False)
    batch_inputs = _dummy_batch()
    keep_true = torch.ones(3, dtype=torch.bool)
    out_a = model(**batch_inputs, k=3, demographics_keep=keep_true)
    out_b = model(**batch_inputs, k=3, demographics_keep=torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(out_a["gate"], out_b["gate"], atol=1e-5)


def test_uniform_gate_ignores_learned_router_entirely():
    model = _build_rap2p(uniform_gate=True)
    batch_inputs = _dummy_batch()
    out_a = model(**batch_inputs, k=5)
    batch_inputs["demographic_embeddings"] = torch.randn_like(batch_inputs["demographic_embeddings"]) * 100
    out_b = model(**batch_inputs, k=5)
    expected = torch.full_like(out_a["gate"], 1.0 / model.rank_blocks)
    assert torch.allclose(out_a["gate"], expected, atol=1e-6)
    assert torch.allclose(out_a["gate"], out_b["gate"], atol=1e-6)


def test_use_correlation_graph_false_ignores_correlation_bias():
    model = _build_rap2p(use_correlation_graph=False, learnable_gamma=False)
    batch_inputs = _dummy_batch()
    out_a = model(**batch_inputs, k=5)
    batch_inputs["correlation_bias"] = torch.randn_like(batch_inputs["correlation_bias"]) * 1000
    out_b = model(**batch_inputs, k=5)
    assert torch.allclose(out_a["gate"], out_b["gate"], atol=1e-5)


def test_gates_persist_after_forward_for_gradient_checkpoint_recompute():
    """Gates must NOT be cleared inside forward: gradient checkpointing
    re-executes decoder-layer forwards during loss.backward(), and the
    recompute reads the adapter's _gate via closure. Stale state cannot leak
    into a later batch because every forward overwrites all gates first."""
    model = _build_rap2p()
    batch_inputs = _dummy_batch()
    out_first = model(**batch_inputs, k=5)
    for adapter in model._adapter_flat:
        assert adapter._gate is not None  # persists for backward-time recompute
    # A second forward with different inputs overwrites the gates -- outputs
    # must be determined by the new inputs alone, not the previous batch.
    batch_inputs2 = _dummy_batch()
    batch_inputs2["demographic_embeddings"] = torch.randn_like(batch_inputs2["demographic_embeddings"]) * 3
    out_second = model(**batch_inputs2, k=5)
    assert not torch.allclose(out_first["gate"], out_second["gate"])


def _max_abs_grad(parameters):
    grads = [p.grad.abs().max().item() for p in parameters if p.grad is not None]
    return max(grads) if grads else 0.0


def _randomize_lora_b(model) -> None:
    """At standard LoRA init lora_b is all-zero, so the gate has exactly zero
    influence on the output and the router's gradient is legitimately zero at
    step 0 (it starts flowing after the first optimizer step, or immediately
    with the Global-QLoRA warm start). Gradient-flow tests must therefore give
    lora_b nonzero values first, or they'd test the init artifact instead of
    the gradient path."""
    with torch.no_grad():
        for adapter in model._adapter_flat:
            adapter.lora_b.normal_(std=0.1)


def test_gradients_flow_to_basis_and_router_without_checkpointing():
    """The production configuration for rank-block models (checkpointing OFF,
    enforced by workflows.build_model_and_collator + the constructor guard):
    a plain backward pass must reach both the LoRA basis and every router
    branch."""
    model = _build_rap2p()
    _randomize_lora_b(model)
    batch_inputs = _dummy_batch()
    output = model(**batch_inputs, k=5)
    loss = output["label_logits"].float().logsumexp(dim=-1).mean()
    loss.backward()

    basis_params = [a.lora_a for a in model._adapter_flat] + [a.lora_b for a in model._adapter_flat]
    router_params = list(model.demographic_prior.parameters()) + list(model.response_anchoring.parameters()) + [model.bias]
    assert _max_abs_grad(basis_params) > 0, "LoRA basis received no gradient"
    assert _max_abs_grad(router_params) > 0, "router/prior/anchoring received no gradient"


def test_gradients_survive_checkpointing_with_persistent_gates():
    """Regression test for the original P0 bug: forward used to call
    clear_gates() before returning, so non-reentrant checkpoint recompute
    (which re-executes each layer's forward during loss.backward()) found
    _gate=None and took the no-adapter branch — inconsistent recompute,
    severed gradients. With gates persisting through backward, both the
    basis leaves AND the router (reached through the non-leaf gate tensor
    captured via closure) receive gradient under torch's non-reentrant
    checkpoint. Production still disables checkpointing for rank-block models
    (workflows.build_model_and_collator) because the frozen backbone makes it
    near-free anyway — this test simply pins the fixed behavior so a future
    reintroduction of clear-in-forward fails loudly."""
    from torch.utils.checkpoint import checkpoint

    model = _build_rap2p()
    _randomize_lora_b(model)  # rule out the zero-init artifact; see helper docstring
    for layer in model.backbone.model.layers:
        original_forward = layer.forward

        def checkpointed_forward(x, _original=original_forward):
            return checkpoint(_original, x, use_reentrant=False)

        layer.forward = checkpointed_forward

    batch_inputs = _dummy_batch()
    output = model(**batch_inputs, k=5)
    loss = output["label_logits"].float().logsumexp(dim=-1).mean()
    loss.backward()

    basis_params = [a.lora_a for a in model._adapter_flat] + [a.lora_b for a in model._adapter_flat]
    router_params = list(model.demographic_prior.parameters()) + list(model.response_anchoring.parameters()) + [model.bias]
    assert _max_abs_grad(basis_params) > 0
    assert _max_abs_grad(router_params) > 0


def test_constructor_rejects_checkpointing_enabled_backbone():
    backbone = FakeCausalLM(VOCAB, HIDDEN, N_LAYERS)
    backbone.is_gradient_checkpointing = True
    with pytest.raises(ValueError, match="gradient_checkpointing=False"):
        RAP2PModel(
            backbone, embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=2, block_rank=2, alpha=4,
            target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS,
            answer_embed_dim=4, dropout=0.0,
        )


def test_gate_responds_to_each_enabled_signal_at_k_greater_than_zero():
    """Positive control for the paper's central mechanism: with everything
    enabled and K>0, the gate must actually move when the target item, the
    history content, or the correlation bias changes."""
    model = _build_rap2p()
    base_inputs = _dummy_batch()
    base_gate = model(**base_inputs, k=5)["gate"]

    changed_item = dict(base_inputs)
    changed_item["item_embeddings"] = torch.randn_like(base_inputs["item_embeddings"]) * 3
    assert not torch.allclose(base_gate, model(**changed_item, k=5)["gate"])

    changed_history = dict(base_inputs)
    changed_history["history_answer_index"] = (base_inputs["history_answer_index"] + 2) % N_OPTIONS
    assert not torch.allclose(base_gate, model(**changed_history, k=5)["gate"])

    changed_correlation = dict(base_inputs)
    # Perturbation must be NON-uniform across the K history slots: softmax
    # attention is invariant to adding the same constant to every score, so a
    # uniform shift correctly leaves the gate unchanged.
    ramp = torch.linspace(0.0, 5.0, base_inputs["correlation_bias"].shape[1]).unsqueeze(0)
    changed_correlation["correlation_bias"] = base_inputs["correlation_bias"] + ramp
    assert not torch.allclose(base_gate, model(**changed_correlation, k=5)["gate"])


def test_p2p_static_ignores_item_embeddings_by_construction():
    torch.manual_seed(0)
    backbone = FakeCausalLM(VOCAB, HIDDEN, N_LAYERS)
    model = P2PStaticModel(
        backbone, embedding_dim=EMBED_DIM, hidden_dim=12, rank_blocks=2, block_rank=2, alpha=4,
        target_modules=("q_proj", "v_proj"), target_last_n_layers=2, max_options=N_OPTIONS, answer_embed_dim=4, dropout=0.0,
    )
    batch_inputs = _dummy_batch()
    out_a = model(**batch_inputs, k=5)
    batch_inputs["item_embeddings"] = torch.randn_like(batch_inputs["item_embeddings"]) * 100
    out_b = model(**batch_inputs, k=5)
    assert torch.allclose(out_a["gate"], out_b["gate"], atol=1e-5)
