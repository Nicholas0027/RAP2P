from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .batching import BatchSpec, ModalityDropout, PanelBatchIterator, SurveyCollator, move_batch_to_device
from .models.common import (
    choice_loss,
    last_token_logits,
    ordinal_loss,
    restricted_logits,
    router_balance_loss,
    trainable_state_dict,
)
from .utils import TimeBudget, write_json

PLAIN_LM_KINDS = {"global_qlora", "context_qlora"}
RAP2P_MODEL_INPUT_KEYS = {
    "input_ids", "attention_mask", "demographic_embeddings", "item_embeddings",
    "history_item_embeddings", "history_answer_index", "history_mask", "correlation_bias",
    "label_token_ids", "option_mask", "k", "demographics_keep", "history_keep", "correlation_keep",
}


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_training_state(model, optimizer, scheduler, step: int, history: list[dict[str, Any]], path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
    _atomic_torch_save(
        {
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": int(step),
            "history": history,
            "metadata": dict(metadata or {}),
        },
        Path(path),
    )


def resume_training_state(model, optimizer, scheduler, path: str | Path) -> tuple[int, list[dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        return 0, []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload["step"]), list(payload.get("history", []))


def build_optimizer(model, lr: float, lora_lr: float | None, weight_decay: float):
    """LoRA-basis parameters (named `*.lora_a` / `*.lora_b`, see gating.py) get
    their own, much smaller learning rate than the router/encoder parameters --
    the basis should move slowly since it is shared by every respondent.
    """
    router_params, basis_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_basis = name.endswith(".lora_a") or name.endswith(".lora_b")
        (basis_params if is_basis and lora_lr is not None else router_params).append(parameter)
    groups = []
    if router_params:
        groups.append({"params": router_params, "lr": lr})
    if basis_params:
        groups.append({"params": basis_params, "lr": lora_lr})
    if not groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def model_forward(model, batch: Mapping[str, Any], kind: str) -> dict[str, torch.Tensor]:
    if kind in PLAIN_LM_KINDS:
        vocabulary_logits = last_token_logits(model, batch["input_ids"], batch["attention_mask"])
        return {"label_logits": restricted_logits(vocabulary_logits, batch["label_token_ids"], batch["option_mask"])}
    inputs = {key: batch[key] for key in RAP2P_MODEL_INPUT_KEYS if key in batch}
    return model(**inputs)


@torch.no_grad()
def quick_validation(model, kind: str, iterator: PanelBatchIterator, collator: SurveyCollator, device: torch.device, max_batches: int = 20) -> dict[str, float]:
    model.eval()
    losses = []
    for batch_index, (records, k, option_seed) in enumerate(iterator):
        if batch_index >= max_batches:
            break
        batch = collator(records, k, iterator.spec.calibration_seed, option_seed, iterator.spec.random_option_permutation)
        batch = move_batch_to_device(batch, device)
        with _autocast(device):
            output = model_forward(model, batch, kind)
            losses.append(float(choice_loss(output["label_logits"], batch["targets"]).item()))
    model.train()
    return {"validation_nll": float(sum(losses) / max(1, len(losses)))}


def train_model(
    model,
    kind: str,
    train_iterator: PanelBatchIterator,
    validation_iterator: PanelBatchIterator,
    collator: SurveyCollator,
    device: torch.device,
    output_dir: str | Path,
    max_optimizer_steps: int,
    gradient_accumulation: int,
    lr: float,
    lora_lr: float | None,
    weight_decay: float,
    warmup_fraction: float,
    validation_every: int,
    time_budget_minutes: float,
    stop_margin_minutes: float,
    ordinal_weight: float,
    balance_weight: float,
    router_collapse_threshold: float,
    grad_clip: float,
    modality_dropout: ModalityDropout | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best_nll.pt"
    model.to(device)
    optimizer = build_optimizer(model, lr, lora_lr, weight_decay)
    scheduler = build_scheduler(optimizer, int(max_optimizer_steps * warmup_fraction), max_optimizer_steps)
    step, history = resume_training_state(model, optimizer, scheduler, last_checkpoint)
    effective_margin = min(float(stop_margin_minutes), max(1.0, float(time_budget_minutes) * 0.2))
    budget = TimeBudget(time_budget_minutes, effective_margin)
    best_nll = min((row.get("validation_nll", float("inf")) for row in history), default=float("inf"))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    running = {"choice": 0.0, "ordinal": 0.0, "balance": 0.0}
    # Once tripped, the collapse guard stays engaged across time-budget
    # restarts — recover the flag from the resumed history rather than
    # silently dropping it on every process start.
    router_collapse_engaged = any(record.get("router_collapse_engaged", False) for record in history)

    while step < max_optimizer_steps and not budget.should_stop():
        for records, k, option_seed in train_iterator:
            batch = collator(
                records, k, train_iterator.spec.calibration_seed, option_seed,
                train_iterator.spec.random_option_permutation,
                modality_dropout=modality_dropout, dropout_seed=step,
            )
            batch = move_batch_to_device(batch, device)
            with _autocast(device):
                output = model_forward(model, batch, kind)
                choice = choice_loss(output["label_logits"], batch["targets"])
                ordinal = ordinal_loss(output["label_logits"], batch["targets"], batch["n_options"])
                loss = choice + ordinal_weight * ordinal
                balance = choice.new_zeros(())
                if "mean_gate_share" in output and balance_weight > 0:
                    if float(output["mean_gate_share"].max().item()) > router_collapse_threshold:
                        router_collapse_engaged = True
                    if router_collapse_engaged:
                        balance = router_balance_loss(output["mean_gate_share"], output["mean_gate_share"].shape[-1])
                        loss = loss + balance_weight * balance
                loss = loss / gradient_accumulation
            loss.backward()
            accumulated += 1
            running["choice"] += float(choice.detach().item())
            running["ordinal"] += float(ordinal.detach().item())
            running["balance"] += float(balance.detach().item())

            if accumulated % gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % validation_every == 0 or step == max_optimizer_steps:
                metrics = quick_validation(model, kind, validation_iterator, collator, device)
                record = {
                    "step": step, "elapsed_minutes": budget.elapsed_minutes, "lr": scheduler.get_last_lr()[0],
                    "router_collapse_engaged": router_collapse_engaged,
                    **{key: value / max(1, accumulated) for key, value in running.items()},
                    **metrics,
                }
                history.append(record)
                write_json(output_dir / "history.json", history)
                save_training_state(model, optimizer, scheduler, step, history, last_checkpoint, metadata)
                if metrics["validation_nll"] < best_nll:
                    best_nll = metrics["validation_nll"]
                    save_training_state(model, optimizer, scheduler, step, history, best_checkpoint, metadata)
                running = {"choice": 0.0, "ordinal": 0.0, "balance": 0.0}
                accumulated = 0

            if step >= max_optimizer_steps or budget.should_stop():
                break

    save_training_state(model, optimizer, scheduler, step, history, last_checkpoint, metadata)
    return history
