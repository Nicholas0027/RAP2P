"""Prompt construction and option-label permutation utilities.

Central design decision (see README / paper draft RQ2): RAP2P, RAP2P-noGraph,
RAP2P-noHistory-retrained, and P2P-Static all receive the *same* text prompt as
Global QLoRA -- demographics + target question, **no history text**. Only
Context QLoRA's prompt includes the respondent's K known answers. This keeps
"information available via the text channel" identical across every method
except the one baseline whose entire point is to test in-context personalization,
so any gap between Context QLoRA and RAP2P can only come from how the shared
information is used (prompt tokens vs. adapter routing), not from RAP2P quietly
getting extra tokens Context QLoRA doesn't.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import stable_int

OPTION_LABELS = list("ABCDEFGHIJK")


def truncate_text(text: str, characters: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= characters else text[: characters - 1].rstrip() + "…"


def history_text(history_rows: Sequence[Mapping[str, Any]]) -> str:
    if not history_rows:
        return "No previous answers are available."
    lines = []
    for index, row in enumerate(history_rows, start=1):
        lines.append(f"{index}. {truncate_text(row['question'], 180)} -> {truncate_text(row['answer_text'], 90)}")
    return "\n".join(lines)


def deterministic_permutation(n_options: int, seed: int, row_id: str) -> np.ndarray:
    rng = np.random.default_rng(stable_int(seed, row_id))
    return rng.permutation(n_options)


def build_prompt(
    row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    permutation: Sequence[int] | None,
    include_history_in_prompt: bool,
) -> tuple[str, int, list[int]]:
    options = json.loads(row["options_json"]) if isinstance(row["options_json"], str) else list(row["options_json"])
    n_options = len(options)
    if n_options > len(OPTION_LABELS):
        raise ValueError(f"{n_options} options exceed supported labels: {row['row_id']}")
    if permutation is None:
        permutation = list(range(n_options))
    permutation = [int(value) for value in permutation]
    if sorted(permutation) != list(range(n_options)):
        raise ValueError("permutation must contain every semantic option exactly once")

    option_lines = [
        f"{OPTION_LABELS[label_index]}. {truncate_text(options[semantic_index], 180)}"
        for label_index, semantic_index in enumerate(permutation)
    ]
    answer_index = int(row["answer_index"])
    correct_label = permutation.index(answer_index)

    sections = [
        "Predict this respondent's survey answer. Return exactly one option letter.",
        "",
        f"Respondent profile:\n{truncate_text(row['demographic_text'], 600)}",
    ]
    if include_history_in_prompt:
        sections.append(f"Previous answers:\n{history_text(history_rows)}")
    sections.append(f"Target question:\n{truncate_text(row['question'], 600)}\n" + "\n".join(option_lines) + "\n\nAnswer:")
    prompt = "\n\n".join(sections)
    return prompt, correct_label, permutation


def option_token_ids(tokenizer, max_options: int) -> list[int]:
    token_ids: list[int] = []
    for label in OPTION_LABELS[:max_options]:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Option label {label!r} is not one token for {tokenizer.name_or_path}: {encoded}. "
                "Choose a tokenizer-specific label set before training."
            )
        token_ids.append(int(encoded[0]))
    return token_ids


def semantic_probabilities(label_probabilities: np.ndarray, label_to_semantic: Sequence[int]) -> np.ndarray:
    semantic = np.zeros_like(label_probabilities)
    for label_index, semantic_index in enumerate(label_to_semantic):
        semantic[int(semantic_index)] = label_probabilities[label_index]
    return semantic
