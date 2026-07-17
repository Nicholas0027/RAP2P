"""Leakage-safe, shrinkage-corrected item-item Spearman correlation graph C_jk.

This is the "psychometric graph" bias used by RAP2P's target-aware response
anchoring (see models/response_anchoring.py): when a respondent's target
question is q_j, a known historical answer to q_k is up-weighted in proportion
to how strongly q_j and q_k co-vary across *training* respondents, shrunk
towards zero when few respondents answered both.

Leakage rule (load-bearing, do not relax): C_jk must be estimated only from
`responses[split == "train"]`. Because `ood_intersection` respondents and
validation/test respondents are never labeled "train", filtering on split is
sufficient -- no separate exclusion list is required. `compute_item_graph`
asserts this at call sites via `assert_train_only`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def assert_train_only(frame: pd.DataFrame) -> None:
    splits = set(frame["split"].unique()) if "split" in frame.columns else set()
    if splits - {"train"}:
        raise ValueError(
            f"Item graph must be estimated on split == 'train' only, got splits={splits}. "
            "This is a leakage guard, not a style preference."
        )
    if "is_unseen_item" in frame.columns and bool(frame["is_unseen_item"].any()):
        raise ValueError(
            "Item graph must exclude is_unseen_item rows: including them would estimate "
            "C_jk for (unseen item, seen item) pairs from train respondents' real answers "
            "to the unseen item, leaking exactly the signal the OOD-Item axis holds out. "
            "Filter with `frame[frame.split.eq('train') & ~frame.is_unseen_item]`."
        )


def compute_item_graph(
    train_responses: pd.DataFrame,
    shrinkage_lambda: float,
    min_n_jk: int,
) -> dict[str, pd.DataFrame]:
    """Return {domain: symmetric DataFrame of C_jk indexed/columned by question_key}."""
    assert_train_only(train_responses)
    graphs: dict[str, pd.DataFrame] = {}
    for domain, group in train_responses.groupby("domain"):
        wide = group.pivot_table(index="panel_id", columns="question_key", values="normalized_answer")
        items = sorted(wide.columns)
        wide = wide[items]
        rho = wide.corr(method="spearman", min_periods=2)
        counts = wide.notna().astype(int)
        n_jk = counts.T @ counts  # co-answer counts per item pair
        shrink = n_jk / (n_jk + shrinkage_lambda)
        c = (shrink * rho).fillna(0.0)
        c = c.where(n_jk >= min_n_jk, 0.0)
        np.fill_diagonal(c.values, 1.0)
        graphs[domain] = c
    return graphs


def save_item_graph(graphs: Mapping[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {domain: matrix.to_dict(orient="split") for domain, matrix in graphs.items()}
    with (output_dir / "item_graph.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class ItemGraph:
    """Fast (domain, question_key, question_key) -> C_jk lookup, loaded once."""

    def __init__(self, graphs: Mapping[str, pd.DataFrame]):
        self._lookup: dict[str, dict[tuple[str, str], float]] = {}
        for domain, matrix in graphs.items():
            table: dict[tuple[str, str], float] = {}
            for row_key in matrix.index:
                for col_key in matrix.columns:
                    table[(row_key, col_key)] = float(matrix.at[row_key, col_key])
            self._lookup[domain] = table

    @classmethod
    def from_dir(cls, directory: str | Path) -> "ItemGraph":
        directory = Path(directory)
        payload: dict[str, Any] = json.loads((directory / "item_graph.json").read_text(encoding="utf-8"))
        graphs = {
            domain: pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
            for domain, data in payload.items()
        }
        return cls(graphs)

    def get(self, domain: str, target_key: str, history_key: str) -> float:
        return self._lookup.get(domain, {}).get((target_key, history_key), 0.0)

    def batch(self, domain: str, target_key: str, history_keys: list[str]) -> np.ndarray:
        table = self._lookup.get(domain, {})
        return np.asarray([table.get((target_key, key), 0.0) for key in history_keys], dtype=np.float32)
