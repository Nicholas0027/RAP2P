"""Smoothed demographic-group frequency baseline (hierarchical backoff): a
strong, cheap group-level prior. Does not use K known answers -- the point of
comparison for RQ1 is whether *individual* answers add anything beyond this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data import PanelStore, demographic_record_fields


def _counts_to_probability(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


class FrequencyBaseline:
    def __init__(self, alpha: float = 1.0, minimum_group_n: int = 25, demographic: bool = True):
        self.alpha = alpha
        self.minimum_group_n = minimum_group_n
        self.demographic = demographic
        self.tables: dict[tuple[str, ...], dict[tuple[Any, ...], tuple[np.ndarray, int]]] = {}
        self.levels = [
            ("question_key", "country", "sex", "age_bin"),
            ("question_key", "country", "sex"),
            ("question_key", "country"),
            ("question_key",),
        ]

    def fit(self, frame: pd.DataFrame) -> "FrequencyBaseline":
        # Train respondents AND seen items only: fitting an unseen item's
        # answer-frequency table from train respondents would leak the very
        # distribution the OOD-Item axis holds out. Unseen-item targets miss
        # every backoff level and fall through to predict_one's uniform floor.
        train = frame[frame["split"].eq("train") & ~frame["is_unseen_item"]]
        for level in self.levels:
            table: dict[tuple[Any, ...], tuple[np.ndarray, int]] = {}
            for key, group in train.groupby(list(level), dropna=False, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                n_options = int(group["n_options"].max())
                counts = np.zeros(n_options, dtype=np.float64)
                for answer, weight in zip(group["answer_index"], group["survey_weight"]):
                    counts[int(answer)] += float(weight)
                table[key] = (counts, len(group))
            self.tables[level] = table
        return self

    def predict_one(self, row: Any) -> np.ndarray:
        levels = self.levels if self.demographic else [self.levels[-1]]
        for level in levels:
            key = tuple(getattr(row, column) for column in level)
            result = self.tables[level].get(key)
            if result is None:
                continue
            counts, n = result
            if len(level) > 1 and n < self.minimum_group_n:
                continue
            return _counts_to_probability(counts[: int(row.n_options)], self.alpha)
        return np.ones(int(row.n_options), dtype=np.float64) / int(row.n_options)


def predict_frequency_baseline(
    store: PanelStore,
    method: str,
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    demographic: bool,
    alpha: float = 1.0,
    minimum_group_n: int = 25,
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    baseline = FrequencyBaseline(alpha, minimum_group_n, demographic).fit(store.responses)
    records: list[dict[str, Any]] = []
    for k in k_values:
        targets = store.target_rows(split, int(k), calibration_seed, item_pool=item_pool)
        for row in targets.itertuples(index=False):
            probabilities = baseline.predict_one(row)
            target = int(row.answer_index)
            predicted = int(probabilities.argmax())
            records.append(
                {
                    "method": method, "row_id": row.row_id, "panel_id": row.panel_id, "domain": row.domain,
                    "question_id": row.question_id, "question_key": row.question_key, "k": int(k), "split": split,
                    "item_pool": item_pool, "calibration_seed": int(calibration_seed), "option_seed": 0,
                    "answer_index": target, "n_options": int(row.n_options), "survey_weight": float(row.survey_weight),
                    "probabilities_json": json.dumps(probabilities.tolist()), "predicted_index": predicted,
                    "nll": float(-np.log(max(probabilities[target], 1e-12))),
                    "brier": float(np.square(probabilities - np.eye(len(probabilities))[target]).sum()),
                    "normalized_ordinal_error": float(abs(predicted - target) / max(1, len(probabilities) - 1)),
                    "correct": int(predicted == target),
                    **demographic_record_fields(row),
                }
            )
    output = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output
