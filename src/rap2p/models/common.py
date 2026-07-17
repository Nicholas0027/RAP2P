from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_dtype(name: str):
    aliases = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"Unknown dtype {name}")
    return aliases[name]


def load_backbone_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    quantization: str | None = None,
    device_map: str | Mapping[str, Any] | None = None,
    gradient_checkpointing: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype(dtype), "trust_remote_code": True, "low_cpu_mem_usage": True}
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype(dtype),
            bnb_4bit_use_double_quant=True,
        )
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def build_shared_lora(base_model, rank: int, alpha: int, dropout: float, target_modules: list[str]):
    """Used for Global QLoRA and Context QLoRA: one ordinary (non-block-gated)
    LoRA shared by every respondent -- the personalization channel here is
    whatever the prompt text contains, not the adapter.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=int(rank), lora_alpha=int(alpha),
        lora_dropout=float(dropout), target_modules=list(target_modules), bias="none",
    )
    model = get_peft_model(base_model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def decoder_layers(model) -> nn.ModuleList:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise AttributeError("Could not locate decoder layers; inspect the backbone architecture")


def last_token_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    return output.logits[:, -1, :]


def restricted_logits(vocabulary_logits: torch.Tensor, label_token_ids: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    logits = vocabulary_logits.index_select(-1, label_token_ids)
    return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


def semantic_probabilities_torch(label_logits: torch.Tensor, permutations: list[list[int]]) -> torch.Tensor:
    label_probabilities = F.softmax(label_logits.float(), dim=-1)
    semantic = torch.zeros_like(label_probabilities)
    for row_index, permutation in enumerate(permutations):
        for label_index, semantic_index in enumerate(permutation):
            semantic[row_index, semantic_index] = label_probabilities[row_index, label_index]
    return semantic


def choice_loss(label_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(label_logits.float(), targets)


def ordinal_loss(label_logits: torch.Tensor, targets: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
    """Expected normalized absolute distance: sum_c p(c) * |c - y| / (n_options - 1).

    RAP2P's design writes this unnormalized (sum_c p(c)|c - y|); we normalize by
    (n_options - 1) so items with different scale lengths (4-point vs 5-point
    Likert) contribute comparably instead of the loss implicitly up-weighting
    longer scales. Set `normalize=False` to recover the literal spec.
    """
    probabilities = F.softmax(label_logits.float(), dim=-1)
    max_options = probabilities.shape[-1]
    positions = torch.arange(max_options, device=probabilities.device, dtype=probabilities.dtype)
    distance = (positions.unsqueeze(0) - targets.unsqueeze(1).float()).abs()
    denom = (n_options.to(probabilities.dtype) - 1).clamp_min(1)
    per_example = (probabilities * distance).sum(-1) / denom
    return per_example.mean()


def router_balance_loss(mean_gate_share: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Penalize a rank-block collapsing onto a single block; only engaged if
    `router_collapse_threshold` is tripped (see training.py)."""
    target = torch.full_like(mean_gate_share, 1.0 / n_blocks)
    return (mean_gate_share - target).square().mean()


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in trainable_names}


def save_trainable_checkpoint(model: nn.Module, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": trainable_state_dict(model), "metadata": dict(metadata or {})}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_trainable_checkpoint(model: nn.Module, path: str | Path, strict: bool = False) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    payload["missing_keys"] = missing
    return payload
