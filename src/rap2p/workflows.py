"""High-level orchestration used by scripts/*.py. Keeps run-name -> (model
class, ablation flags, collator kind) mapping in one place so scripts stay thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .batching import BatchSpec, ModalityDropout, PanelBatchIterator, SurveyCollator
from .config import ensure_artifact_dirs
from .data import PanelStore, make_synthetic_panels
from .embeddings import EmbeddingStore, SyntheticEmbeddingStore
from .inference import predict
from .item_graph import ItemGraph, compute_item_graph
from .models import P2PStaticModel, RAP2PModel, build_shared_lora, load_backbone_and_tokenizer
from .training import resume_training_state, train_model
from .utils import seed_everything

# run_name -> collator kind (prompt/embedding policy, see prompting.py docstring)
RUN_TO_COLLATOR_KIND = {
    "global_qlora": "global_qlora",
    "context_qlora": "context_qlora",
    "p2p_static": "p2p_static",
    "rap2p": "rap2p",
    "rap2p_no_graph": "rap2p",
    "rap2p_no_history_retrained": "rap2p",
}


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_experiment_data(config: Mapping[str, Any], smoke: bool, with_embeddings: bool):
    if smoke:
        responses, orders = make_synthetic_panels(seed=int(config["seed"]))
        store = PanelStore(responses, orders)
        if not with_embeddings:
            return store, None, None
        embeddings = SyntheticEmbeddingStore(dimension=32)
        train_only = responses[responses["split"].eq("train") & ~responses["is_unseen_item"]]
        item_graph = ItemGraph(compute_item_graph(train_only, shrinkage_lambda=5, min_n_jk=1))
        return store, embeddings, item_graph
    store = PanelStore.from_dir(config["paths"]["processed"])
    embeddings = EmbeddingStore(config["paths"]["embeddings"]) if with_embeddings else None
    item_graph = ItemGraph.from_dir(config["paths"]["item_graph"]) if with_embeddings else None
    return store, embeddings, item_graph


def initialize_basis_from_peft_checkpoint(model, checkpoint_path: str | Path, rank_blocks: int) -> int:
    """Stage-1 warm start: copy a trained Global-QLoRA (PEFT) checkpoint's
    lora_A/lora_B weights into the rank-block basis of a RAP2PModel /
    P2PStaticModel, so Stage-2 training starts from the population survey
    adapter instead of random init.

    PEFT names its weights `...layers.{i}.self_attn.{q,v}_proj.lora_A.default.weight`
    (shape (R, in)) and `...lora_B.default.weight` (shape (out, R)) — the same
    shapes as our concatenated `lora_a`/`lora_b`, so a direct copy splits
    cleanly into blocks along the rank dimension. One correction is needed:
    at init the softmax gate is ~uniform (1/B per block), so ΔW would start at
    (1/B)·BA instead of BA; scaling the copied lora_b by `rank_blocks` makes
    the initial ΔW reproduce the Stage-1 adapter exactly under a uniform gate.

    PEFT applies LoRA to every matching layer; the model only patches the last
    N, so the copy maps PEFT layer (total - N + j) -> adapter_refs[j].
    Returns the number of (layer, module) locations initialized.
    """
    import re

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload["model"]
    pattern = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.\w+\.weight$")
    by_location: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        match = pattern.search(name)
        if match is None:
            continue
        layer_index, module_name, which = int(match.group(1)), match.group(2), match.group(3)
        by_location.setdefault((layer_index, module_name), {})[which] = tensor
    if not by_location:
        raise ValueError(f"No PEFT lora_A/lora_B weights found in {checkpoint_path}")

    total_layers = max(layer for layer, _ in by_location) + 1
    offset = total_layers - model.num_layers
    module_order = ("q_proj", "v_proj")
    initialized = 0
    for local_index, modules in enumerate(model.adapter_refs):
        for module_index, adapter in enumerate(modules):
            source = by_location.get((offset + local_index, module_order[module_index]))
            if source is None or "A" not in source or "B" not in source:
                raise ValueError(f"Checkpoint missing lora weights for layer {offset + local_index} {module_order[module_index]}")
            if source["A"].shape != adapter.lora_a.shape or source["B"].shape != adapter.lora_b.shape:
                raise ValueError(
                    f"Shape mismatch at layer {offset + local_index} {module_order[module_index]}: "
                    f"checkpoint A{tuple(source['A'].shape)}/B{tuple(source['B'].shape)} vs "
                    f"basis a{tuple(adapter.lora_a.shape)}/b{tuple(adapter.lora_b.shape)} — "
                    "population_rank must equal rank_blocks * block_rank."
                )
            with torch.no_grad():
                adapter.lora_a.copy_(source["A"].to(adapter.lora_a.dtype))
                adapter.lora_b.copy_(source["B"].to(adapter.lora_b.dtype) * rank_blocks)
            initialized += 1
    return initialized


def _rap2p_ablation_flags(run_name: str) -> dict[str, bool]:
    if run_name == "rap2p_no_graph":
        return {"use_correlation_graph": False, "learnable_gamma": False}
    if run_name == "rap2p_no_history_retrained":
        return {"use_history": False}
    return {}


def build_model_and_collator(config: Mapping[str, Any], run_name: str, smoke: bool = False, ablation: str | None = None):
    model_config = config["model"]
    kind = RUN_TO_COLLATOR_KIND[run_name]
    with_embeddings = kind in {"p2p_static", "rap2p"}

    model_name = model_config["smoke_backbone"] if smoke else model_config["backbone"]
    dtype = "float32" if smoke or not torch.cuda.is_available() else model_config["dtype"]
    quantization = None if smoke else model_config["quantization"]
    # Gradient checkpointing is ONLY for the PEFT runs (global/context QLoRA),
    # where LoRA spans every layer and checkpointing genuinely saves memory.
    # The rank-block models train without it: their frozen backbone builds no
    # autograd graph before the first patched layer (only the last
    # `target_last_n_layers` layers retain activations, so the saving would be
    # near-zero), and their closure-captured gate tensor makes gradient flow
    # under checkpointing an implementation detail of non-reentrant checkpoint
    # (works on the pinned torch version, but not a contract to lean on —
    # see the constructor guard in RAP2PModel).
    use_checkpointing = (
        not smoke and config["training"]["gradient_checkpointing"] and kind in {"global_qlora", "context_qlora"}
    )
    base, tokenizer = load_backbone_and_tokenizer(
        model_name, dtype=dtype, quantization=quantization, device_map=None,
        gradient_checkpointing=use_checkpointing,
    )

    store, embeddings, item_graph = load_experiment_data(config, smoke, with_embeddings)
    embedding_dim = embeddings.dimension if embeddings is not None else 384  # smoke-mode placeholder dim

    max_options = int(model_config["max_answer_options"])
    common_kwargs = dict(
        embedding_dim=embedding_dim,
        hidden_dim=model_config["router_hidden"],
        rank_blocks=model_config["rank_blocks"],
        block_rank=model_config["block_rank"],
        alpha=int(model_config.get("alpha", 32)),
        target_modules=tuple(model_config["target_modules"]),
        target_last_n_layers=model_config["target_last_n_layers"],
        max_options=max_options,
        dropout=model_config["dropout"],
    )

    if kind in {"global_qlora", "context_qlora"}:
        model = build_shared_lora(
            base, model_config["population_rank"], int(model_config.get("alpha", 32)),
            model_config["dropout"], model_config["target_modules"],
        )
    elif kind == "p2p_static":
        model = P2PStaticModel(base, **common_kwargs)
    elif run_name == "rap2p":
        flags = {"uniform_gate": True} if ablation == "uniform_gate" else {}
        model = RAP2PModel(base, evidence_tau=2.0, **{**common_kwargs, **flags})
    elif run_name in ("rap2p_no_graph", "rap2p_no_history_retrained"):
        model = RAP2PModel(base, evidence_tau=2.0, **{**common_kwargs, **_rap2p_ablation_flags(run_name)})
    else:
        raise ValueError(f"Unknown run_name: {run_name}")

    collator = SurveyCollator(
        tokenizer, store, embeddings, item_graph, model_config["max_length"], max_options, kind=kind,
    )
    return model, tokenizer, store, embeddings, item_graph, collator


def run_training_job(
    config: Mapping[str, Any],
    run_name: str,
    seed: int,
    smoke: bool = False,
    max_optimizer_steps: int | None = None,
    time_budget_minutes: float | None = None,
) -> list[dict[str, Any]]:
    ensure_artifact_dirs(config)
    seed_everything(seed)
    model, _, store, _, _, collator = build_model_and_collator(config, run_name, smoke)
    data_cfg = config["data"]
    training_cfg = config["training"]
    run_cfg = config["runs"][run_name]
    kind = RUN_TO_COLLATOR_KIND[run_name]

    # Stage-1 warm start for rank-block-basis models (skipped in smoke mode,
    # where no Global-QLoRA checkpoint exists). The source run must have been
    # trained first — README's run order lists global_qlora before these runs.
    init_from = run_cfg.get("init_basis_from")
    if init_from and not smoke and kind in {"p2p_static", "rap2p"}:
        checkpoints_dir = Path(config["paths"]["checkpoints"])
        candidate = checkpoints_dir / f"{init_from}_seed{seed}"
        if not (candidate / "best_nll.pt").exists() and not (candidate / "last.pt").exists():
            fallback_seed = int(config["runs"][init_from]["seeds"][0])
            candidate = checkpoints_dir / f"{init_from}_seed{fallback_seed}"
        source = candidate / "best_nll.pt"
        if not source.exists():
            source = candidate / "last.pt"
        if not source.exists():
            raise FileNotFoundError(
                f"init_basis_from={init_from!r} requires a trained checkpoint under {candidate} — "
                f"train `{init_from}` first (see README run order), or remove init_basis_from from the config."
            )
        n_locations = initialize_basis_from_peft_checkpoint(model, source, int(config["model"]["rank_blocks"]))
        print(f"[warm-start] initialized {n_locations} (layer, module) basis locations from {source}")

    train_spec = BatchSpec(
        split="train", k_values=list(data_cfg["k_values"]), calibration_seed=int(data_cfg["calibration_seed"]),
        option_seed=0, panels_per_batch=int(training_cfg["micro_batch"]), targets_per_panel=4,
        random_option_permutation=True, item_pool="seen",
    )
    validation_spec = BatchSpec(
        split="validation", k_values=[5], calibration_seed=int(data_cfg["calibration_seed"]), option_seed=0,
        panels_per_batch=1, targets_per_panel=4, random_option_permutation=False, item_pool="seen", shuffle=False,
    )
    train_iterator = PanelBatchIterator(store, train_spec, seed)
    validation_iterator = PanelBatchIterator(store, validation_spec, seed + 1)

    _md_enabled = bool(data_cfg.get("enable_modality_dropout", True))
    modality_dropout = (
        ModalityDropout(**data_cfg["modality_dropout"])
        if (run_name == "rap2p" and _md_enabled)
        else None
    )
    steps_range = run_cfg["steps"]
    steps = int(max_optimizer_steps or (steps_range[1] if not smoke else 20))

    output_dir = Path(config["paths"]["checkpoints"]) / f"{run_name}_seed{seed}"
    return train_model(
        model=model, kind=kind, train_iterator=train_iterator, validation_iterator=validation_iterator,
        collator=collator, device=choose_device(), output_dir=output_dir, max_optimizer_steps=steps,
        gradient_accumulation=int(training_cfg["gradient_accumulation"]), lr=float(run_cfg["lr"]),
        lora_lr=float(run_cfg["lora_lr"]) if "lora_lr" in run_cfg else None,
        weight_decay=float(training_cfg["weight_decay"]), warmup_fraction=float(training_cfg["warmup_fraction"]),
        validation_every=int(training_cfg["validation_every"]),
        time_budget_minutes=float(time_budget_minutes or run_cfg["time_budget_minutes"]),
        stop_margin_minutes=float(training_cfg["stop_margin_minutes"]),
        ordinal_weight=float(training_cfg["ordinal_loss_weight"]),
        balance_weight=float(training_cfg["balance_loss_weight"]),
        router_collapse_threshold=float(training_cfg["router_collapse_threshold"]),
        grad_clip=float(training_cfg["grad_clip"]), modality_dropout=modality_dropout,
        metadata={"run_name": run_name, "seed": seed},
    )


def load_best_weights(model, checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "best_nll.pt"
    if not path.exists():
        path = checkpoint_dir / "last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    return path


def run_prediction_job(
    config: Mapping[str, Any],
    run_name: str,
    method_name: str,
    output_name: str,
    checkpoint_run: str | None = None,
    k_values: list[int] | None = None,
    split: str = "test",
    item_pool: str = "seen",
    domains: list[str] | None = None,
    smoke: bool = False,
    model_ablation: str | None = None,
    option_seed: int = 0,
    respondent_sample: int | None = None,
    **ablation_kwargs: Any,
):
    """`model_ablation` selects a construction-time model variant (currently
    only "uniform_gate", see build_model_and_collator); `**ablation_kwargs` are
    the forward-time "free" ablation flags (force_demographics_off, etc., see
    inference.predict) applied to whatever model gets built. `option_seed`
    varies the option-label permutation (the permutation-robustness axis);
    `respondent_sample` deterministically subsamples panels (used to keep the
    5-permutation robustness pass affordable).
    """
    model, _, store, _, _, collator = build_model_and_collator(config, run_name, smoke, ablation=model_ablation)
    if checkpoint_run is not None:
        load_best_weights(model, Path(config["paths"]["checkpoints"]) / checkpoint_run)
    data_cfg = config["data"]
    kind = RUN_TO_COLLATOR_KIND[run_name]
    return predict(
        model=model, kind=kind, method_name=method_name, store=store, collator=collator, split=split,
        k_values=k_values or list(data_cfg["k_values"]), calibration_seed=int(data_cfg["calibration_seed"]),
        option_seed=int(option_seed), device=choose_device(),
        output_path=Path(config["paths"]["predictions"]) / output_name,
        batch_size=int(config["evaluation"]["batch_size"]), domains=domains, item_pool=item_pool,
        respondent_sample=respondent_sample,
        **ablation_kwargs,
    )
