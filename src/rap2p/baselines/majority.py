"""Majority baseline: per-question most frequent training-set option. No
demographics, no history -- the non-personalized floor every other method
must clear."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data import PanelStore, demographic_record_fields


def predict_majority(
    store: PanelStore,
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    # Fit on train respondents AND seen items only: an unseen item's majority
    # answer computed from train respondents would leak exactly the signal the
    # OOD-Item axis holds out. Unseen-item targets fall back to the global
    # (position-level) majority index across all seen items.
    train = store.responses[store.responses["split"].eq("train") & ~store.responses["is_unseen_item"]]
    majority_index: dict[str, int] = {}
    for question_key, group in train.groupby("question_key"):
        counts = np.bincount(group["answer_index"].astype(int), minlength=int(group["n_options"].max()))
        majority_index[question_key] = int(counts.argmax())
    global_counts = np.bincount(train["answer_index"].astype(int), minlength=int(train["n_options"].max()))
    global_majority = int(global_counts.argmax())

    records: list[dict[str, Any]] = []
    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, item_pool=item_pool)
        for row in targets.itertuples(index=False):
            n_options = int(row.n_options)
            predicted = majority_index.get(row.question_key, global_majority)
            predicted = min(predicted, n_options - 1)
            probabilities = np.full(n_options, 1e-3)
            probabilities[predicted] = 1.0 - 1e-3 * (n_options - 1)
            target = int(row.answer_index)
            records.append(
                {
                    "method": "majority", "row_id": row.row_id, "panel_id": row.panel_id, "domain": row.domain,
                    "question_id": row.question_id, "question_key": row.question_key, "k": int(k), "split": split,
                    "item_pool": item_pool, "calibration_seed": int(calibration_seed), "option_seed": 0,
                    "answer_index": target, "n_options": n_options, "survey_weight": float(row.survey_weight),
                    "probabilities_json": json.dumps(probabilities.tolist()), "predicted_index": predicted,
                    "nll": float(-np.log(max(probabilities[target], 1e-12))),
                    "brier": float(np.square(probabilities - np.eye(n_options)[target]).sum()),
                    "normalized_ordinal_error": float(abs(predicted - target) / max(1, n_options - 1)),
                    "correct": int(predicted == target),
                    **demographic_record_fields(row),
                }
            )
    output = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output
