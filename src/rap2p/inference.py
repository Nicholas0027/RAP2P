from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .batching import SurveyCollator, iter_prediction_records, move_batch_to_device
from .data import CANONICAL_DEMOGRAPHIC_COLUMNS, PanelStore
from .models.common import semantic_probabilities_torch
from .training import model_forward
from .utils import stable_int


def subsample_panels(frame: pd.DataFrame, n_panels: int | None, tag: str = "eval_subsample") -> pd.DataFrame:
    """Deterministic respondent-level subsample: same tag + panel population
    always selects the same panels, independent of option_seed/K, so
    permutation-robustness replicates cover identical respondents."""
    if n_panels is None:
        return frame
    panels = sorted(frame["panel_id"].unique())
    if len(panels) <= n_panels:
        return frame
    ranked = sorted(panels, key=lambda pid: stable_int(tag, pid))
    keep = set(ranked[:n_panels])
    return frame[frame["panel_id"].isin(keep)]


@torch.no_grad()
def predict(
    model,
    kind: str,
    method_name: str,
    store: PanelStore,
    collator: SurveyCollator,
    split: str,
    k_values: Iterable[int],
    calibration_seed: int,
    option_seed: int,
    device: torch.device,
    output_path: str | Path,
    batch_size: int = 16,
    domains: list[str] | None = None,
    item_pool: str = "seen",
    respondent_sample: int | None = None,
    random_permutation: bool = True,
    force_demographics_off: bool = False,
    force_history_off: bool = False,
    force_correlation_off: bool = False,
    override_item_embeddings: torch.Tensor | None = None,
) -> pd.DataFrame:
    """`force_*_off` implement the "free" ablations on a single RAP2P checkpoint
    (see rap2p_model.py docstring): they hard-zero the corresponding
    keep-mask for every example in this prediction pass, regardless of what the
    stochastic training-time modality dropout would have sampled.
    `override_item_embeddings`, if given, replaces the target item's embedding
    for every example -- used for the "target-independent" ablation, where the
    caller passes a precomputed per-domain mean item embedding instead of e_j.
    """
    model.to(device)
    model.eval()
    predictions: list[dict[str, Any]] = []
    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, domains, item_pool=item_pool)
        targets = subsample_panels(targets, respondent_sample)
        for records in iter_prediction_records(targets, batch_size):
            batch = collator(records, int(k), calibration_seed, option_seed, random_permutation)
            batch = move_batch_to_device(batch, device)
            if force_demographics_off and "demographics_keep" in batch:
                batch["demographics_keep"] = torch.zeros_like(batch["demographics_keep"])
            if force_history_off and "history_keep" in batch:
                batch["history_keep"] = torch.zeros_like(batch["history_keep"])
            if force_correlation_off and "correlation_keep" in batch:
                batch["correlation_keep"] = torch.zeros_like(batch["correlation_keep"])
            if override_item_embeddings is not None and "item_embeddings" in batch:
                batch["item_embeddings"] = override_item_embeddings.to(batch["item_embeddings"].device).expand(
                    batch["item_embeddings"].shape[0], -1
                )
            output = model_forward(model, batch, kind)
            semantic = semantic_probabilities_torch(output["label_logits"], batch["permutations"])
            for index, record in enumerate(records):
                n_options = int(record["n_options"])
                probabilities = semantic[index, :n_options].float().cpu().numpy()
                probabilities = probabilities / probabilities.sum()
                target = int(record["answer_index"])
                predictions.append(
                    {
                        "method": method_name,
                        "row_id": record["row_id"],
                        "panel_id": record["panel_id"],
                        "domain": record["domain"],
                        "question_id": record["question_id"],
                        "question_key": record["question_key"],
                        "k": int(k),
                        "split": split,
                        "item_pool": item_pool,
                        "calibration_seed": int(calibration_seed),
                        "option_seed": int(option_seed),
                        "answer_index": target,
                        "n_options": n_options,
                        "survey_weight": float(record["survey_weight"]),
                        "probabilities_json": json.dumps(probabilities.tolist()),
                        "predicted_index": int(probabilities.argmax()),
                        "nll": float(-np.log(max(probabilities[target], 1e-12))),
                        "brier": float(np.square(probabilities - np.eye(n_options)[target]).sum()),
                        "normalized_ordinal_error": float(abs(int(probabilities.argmax()) - target) / max(1, n_options - 1)),
                        "correct": int(probabilities.argmax() == target),
                        **{column: record[column] for column in CANONICAL_DEMOGRAPHIC_COLUMNS},
                    }
                )
    output = pd.DataFrame(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


def domain_mean_item_embedding(embeddings, items_frame: pd.DataFrame, domain: str) -> "torch.Tensor":
    """Precompute a domain-average item embedding for the target-independent
    ablation (RAP2P conditioned on "some average question" instead of e_j)."""
    keys = items_frame.loc[items_frame["domain"].eq(domain), "question_key"].tolist()
    vectors = embeddings.items.batch(keys)
    return torch.from_numpy(vectors.mean(axis=0, keepdims=True).astype(np.float32))
