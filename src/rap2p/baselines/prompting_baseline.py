"""Local-8B and strong-API sparse-ICL prompting baselines.

Both put demographics + K known answers directly in the prompt (see
prompting.build_prompt(..., include_history_in_prompt=True)) -- the same
information RAP2P receives, just delivered as tokens instead of adapter
routing. This is Context QLoRA's zero-shot-adapter cousin: no training at all,
frozen backbone (Local-8B) or an external API (Strong-API).

The API baseline cannot return calibrated option probabilities (no logprob
access for most hosted chat APIs), so its predicted label is treated as a hard
argmax with a small fixed smoothing mass -- NLL/Brier for this method are
therefore an approximation, not a calibration measurement. Report accuracy and
ordinal MAE as primary for the API row; flag NLL/Brier as approximate in the
table footnote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..data import CANONICAL_DEMOGRAPHIC_COLUMNS, PanelStore, demographic_record_fields
from ..prompting import OPTION_LABELS, build_prompt, deterministic_permutation, semantic_probabilities
from ..utils import stable_int


def _hard_label_probabilities(predicted_index: int, n_options: int, epsilon: float = 1e-3) -> np.ndarray:
    probabilities = np.full(n_options, epsilon / max(1, n_options - 1))
    probabilities[predicted_index] = 1.0 - epsilon
    return probabilities / probabilities.sum()


def predict_local_prompt(
    store: PanelStore,
    model,
    tokenizer,
    method_name: str,
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    device,
    max_length: int = 512,
    batch_size: int = 8,
    split: str = "test",
    item_pool: str = "seen",
    include_history: bool = True,
    option_seed: int | None = None,
    respondent_sample: int | None = None,
) -> pd.DataFrame:
    """`option_seed=None` keeps options in semantic order (main-table pass);
    an integer applies the same deterministic per-row label permutation used
    by inference.predict, with probabilities mapped back to semantic order —
    required for the permutation-robustness table to include this baseline.
    """
    import torch

    from ..inference import subsample_panels
    from ..models.common import last_token_logits, restricted_logits
    from ..prompting import option_token_ids

    label_token_ids = torch.tensor(option_token_ids(tokenizer, len(OPTION_LABELS)), device=device)
    model.to(device).eval()
    records: list[dict[str, Any]] = []

    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, item_pool=item_pool)
        targets = subsample_panels(targets, respondent_sample)
        rows = targets.to_dict("records")
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts, semantic_targets, n_options_list, permutations = [], [], [], []
            for row in chunk:
                history = store.history_rows(row["panel_id"], row["question_id"], int(k), calibration_seed)
                permutation = (
                    deterministic_permutation(int(row["n_options"]), option_seed, row["row_id"])
                    if option_seed is not None
                    else None
                )
                prompt, _, semantic_order = build_prompt(row, history.to_dict("records"), permutation, include_history)
                prompts.append(prompt)
                semantic_targets.append(int(row["answer_index"]))
                n_options_list.append(int(row["n_options"]))
                permutations.append(semantic_order)
            tokens = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                vocabulary_logits = last_token_logits(model, tokens["input_ids"], tokens["attention_mask"])
            n_options_tensor = torch.tensor(n_options_list, device=device)
            option_mask = torch.arange(len(label_token_ids), device=device).unsqueeze(0) < n_options_tensor.unsqueeze(1)
            label_logits = restricted_logits(vocabulary_logits, label_token_ids, option_mask)
            probabilities_batch = torch.softmax(label_logits.float(), dim=-1).cpu().numpy()
            for row, target, n_options, permutation, probabilities in zip(
                chunk, semantic_targets, n_options_list, permutations, probabilities_batch
            ):
                label_probability = probabilities[:n_options]
                probability = semantic_probabilities(label_probability, permutation)
                probability = probability / probability.sum()
                predicted = int(probability.argmax())
                records.append(
                    {
                        "method": method_name, "row_id": row["row_id"], "panel_id": row["panel_id"], "domain": row["domain"],
                        "question_id": row["question_id"], "question_key": row["question_key"], "k": int(k), "split": split,
                        "item_pool": item_pool, "calibration_seed": int(calibration_seed),
                        "option_seed": -1 if option_seed is None else int(option_seed),
                        "answer_index": target, "n_options": n_options, "survey_weight": float(row["survey_weight"]),
                        "probabilities_json": json.dumps(probability.tolist()), "predicted_index": predicted,
                        "nll": float(-np.log(max(probability[target], 1e-12))),
                        "brier": float(np.square(probability - np.eye(n_options)[target]).sum()),
                        "normalized_ordinal_error": float(abs(predicted - target) / max(1, n_options - 1)),
                        "correct": int(predicted == target),
                        **{col: row[col] for col in CANONICAL_DEMOGRAPHIC_COLUMNS},
                    }
                )
    output = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


def parse_label_response(text: str, n_options: int) -> tuple[int, bool]:
    """Extract the answered option letter from an API text response.

    Returns (semantic-safe label index, parse_failed). Only *standalone*
    single-letter tokens that are VALID labels for this question (index <
    n_options) count as candidates — a stray capital inside a preamble word
    can't match because of the word boundaries, and a standalone "I" (index 8)
    is only accepted when the question actually has 9+ options. When several
    candidates appear ("Answer: B. Agree" -> B; "A or B? B" -> B), the LAST one
    wins, since answers conventionally end with the chosen letter. No valid
    candidate -> (0, True): callers must record parse_failed rather than treat
    the fallback as a real prediction.
    """
    valid = set(OPTION_LABELS[:n_options])
    candidates = [token.upper() for token in re.findall(r"\b([A-Ka-k])\b", text) if token.upper() in valid]
    if not candidates:
        return 0, True
    return OPTION_LABELS.index(candidates[-1]), False


def predict_api_prompt(
    store: PanelStore,
    call_fn: Callable[[str], str],
    method_name: str,
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    respondent_sample: int = 500,
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    """`call_fn(prompt) -> raw text response` should call the API at
    temperature 0 requesting a single option letter; see scripts/run_api_baseline.py
    for concrete Anthropic/OpenAI client constructors.
    """
    all_panels = sorted(store.responses[store.responses["split"].eq(split)]["panel_id"].unique())
    ranked = sorted(all_panels, key=lambda pid: stable_int("api_subsample", pid))
    sampled_panels = set(ranked[:respondent_sample])

    records: list[dict[str, Any]] = []
    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, item_pool=item_pool)
        targets = targets[targets["panel_id"].isin(sampled_panels)]
        for row in targets.itertuples(index=False):
            history = store.history_rows(row.panel_id, row.question_id, int(k), calibration_seed)
            prompt, target, _ = build_prompt(row._asdict(), history.to_dict("records"), None, True)
            response_text = call_fn(prompt)
            n_options = int(row.n_options)
            predicted, parse_failed = parse_label_response(response_text, n_options)
            probability = _hard_label_probabilities(predicted, n_options)
            records.append(
                {
                    "method": method_name, "row_id": row.row_id, "panel_id": row.panel_id, "domain": row.domain,
                    "question_id": row.question_id, "question_key": row.question_key, "k": int(k), "split": split,
                    "item_pool": item_pool, "calibration_seed": int(calibration_seed), "option_seed": 0,
                    "answer_index": target, "n_options": n_options, "survey_weight": float(row.survey_weight),
                    "probabilities_json": json.dumps(probability.tolist()), "predicted_index": predicted,
                    "nll": float(-np.log(max(probability[target], 1e-12))),
                    "brier": float(np.square(probability - np.eye(n_options)[target]).sum()),
                    "normalized_ordinal_error": float(abs(predicted - target) / max(1, n_options - 1)),
                    "correct": int(predicted == target),
                    "approximate_probabilities": True,
                    "parse_failed": bool(parse_failed),
                    "raw_response": response_text[:200],
                    **demographic_record_fields(row),
                }
            )
    output = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


def make_anthropic_call_fn(model: str = "claude-sonnet-5", max_tokens: int = 16) -> Callable[[str], str]:
    import anthropic

    client = anthropic.Anthropic()

    def call_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    return call_fn


def make_openai_call_fn(model: str = "gpt-4o", max_tokens: int = 16) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI()

    def call_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return call_fn
