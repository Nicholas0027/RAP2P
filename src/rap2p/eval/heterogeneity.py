"""Matched-Profile Heterogeneity: the sharpest test of whether RAP2P
distinguishes two people the demographic prior alone cannot -- two respondents
with (near-)identical demographics whose *known* answers diverge should also
get divergent predictions on the *held-out* target items.

Both D_true and D_pred are computed on exactly the same set of target rows
(the intersection of what both respondents in a pair were evaluated on at a
given method/k/split), so the comparison is apples-to-apples.

Pair discovery uses leave-one-attribute-out bucketing (a small locality-sensitive
hashing trick) rather than all-pairs comparison, since a test split can have
thousands of respondents per domain and full O(n^2) comparison would not
finish in a reasonable evaluation budget. Each bucket is deterministically
capped (`max_bucket_size`) to bound worst-case pair counts.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils import stable_int

DEFAULT_MATCH_ATTRIBUTES = ("age_bin", "education", "income_quintile", "urbanicity")


def _hamming(a: tuple, b: tuple) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def find_matched_pairs(
    panels: pd.DataFrame,
    attributes: Iterable[str] = DEFAULT_MATCH_ATTRIBUTES,
    max_attribute_distance: int = 1,
    max_bucket_size: int = 200,
    seed: int = 1701,
) -> pd.DataFrame:
    """`panels` must have one row per panel_id with `domain` and the given
    attribute columns. Returns a DataFrame of (domain, panel_id_a, panel_id_b).
    """
    attributes = list(attributes)
    pairs: list[dict[str, str]] = []
    for domain, group in panels.groupby("domain"):
        records = list(zip(group["panel_id"], group[attributes].itertuples(index=False, name=None)))
        seen_pairs: set[tuple[str, str]] = set()

        bucket_keys = [attributes] if max_attribute_distance == 0 else [
            [a for i, a in enumerate(attributes) if i != drop] for drop in range(len(attributes))
        ]
        for keys in bucket_keys:
            buckets: dict[tuple, list[tuple[str, tuple]]] = {}
            for panel_id, values in records:
                bucket_key = tuple(values[attributes.index(k)] for k in keys)
                buckets.setdefault(bucket_key, []).append((panel_id, values))
            for bucket_key, members in buckets.items():
                if len(members) < 2:
                    continue
                if len(members) > max_bucket_size:
                    # Rank by a hash that includes the bucket key, so an
                    # oversized bucket subsamples independently of every other
                    # bucket — a panel_id-only hash would deterministically
                    # keep the same low-hash respondents in EVERY oversized
                    # bucket and exclude high-hash respondents from the whole
                    # analysis.
                    members = sorted(
                        members,
                        key=lambda item: stable_int(seed, "heterogeneity_bucket", str(bucket_key), item[0]),
                    )[:max_bucket_size]
                for (panel_a, values_a), (panel_b, values_b) in combinations(members, 2):
                    if panel_a == panel_b:
                        continue
                    key = tuple(sorted((panel_a, panel_b)))
                    if key in seen_pairs:
                        continue
                    if _hamming(values_a, values_b) <= max_attribute_distance:
                        seen_pairs.add(key)
                        pairs.append({"domain": domain, "panel_id_a": key[0], "panel_id_b": key[1]})
    return pd.DataFrame(pairs)


def filter_pairs_by_answer_divergence(
    pairs: pd.DataFrame,
    responses: pd.DataFrame,
    selection_items: int = 5,
    min_divergence: float = 0.35,
    min_shared_items: int = 3,
    seed: int = 1701,
) -> pd.DataFrame:
    """Keep only matched pairs whose *known* answers actually diverge — the
    paper's stated definition (near-identical demographics, divergent
    answers), which demographic matching alone does not enforce.

    Selection variable and evaluation targets must stay disjoint, or the
    filter would inflate D_true by construction. For each pair, the
    "selection items" are the `selection_items` lowest-stable-hash items
    answered by BOTH respondents (a canonical, calibration-order-independent
    choice); divergence = mean |normalized answer difference| over them. Pairs
    with fewer than `min_shared_items` commonly-answered items, or divergence
    below `min_divergence`, are dropped. The retained selection items are
    recorded per pair (`selection_items_json`) so matched_profile_heterogeneity
    can exclude them from the target comparison.
    """
    import json

    answers: dict[str, dict[str, float]] = {}
    for row in responses.itertuples(index=False):
        answers.setdefault(row.panel_id, {})[row.question_key] = float(row.answer_index) / max(1, int(row.n_options) - 1)

    kept: list[dict] = []
    for row in pairs.itertuples(index=False):
        a = answers.get(row.panel_id_a, {})
        b = answers.get(row.panel_id_b, {})
        common = sorted(set(a) & set(b), key=lambda qk: stable_int(seed, "divergence_selection", qk))
        selected = common[:selection_items]
        if len(selected) < min_shared_items:
            continue
        divergence = float(np.mean([abs(a[qk] - b[qk]) for qk in selected]))
        if divergence < min_divergence:
            continue
        kept.append(
            {
                "domain": row.domain, "panel_id_a": row.panel_id_a, "panel_id_b": row.panel_id_b,
                "history_divergence": divergence, "selection_items_json": json.dumps(selected),
            }
        )
    return pd.DataFrame(kept)


def _probability_array(value: str) -> np.ndarray:
    import json

    return np.asarray(json.loads(value), dtype=np.float64)


def matched_profile_heterogeneity(
    predictions: pd.DataFrame,
    matched_pairs: pd.DataFrame,
    min_shared_targets: int = 3,
    split: str = "test",
) -> pd.DataFrame:
    """For each (method, k) and each matched pair, compare D_true (ground-truth
    normalized-answer divergence) against D_pred (expected-prediction
    divergence) over the shared held-out target items. Reports Spearman
    rho_hetero and the Diversity Recovery Ratio DRR = E[D_pred] / E[D_true].

    If `matched_pairs` carries a `selection_items_json` column (produced by
    filter_pairs_by_answer_divergence), those items are excluded from the
    target comparison so the selection variable and the evaluated outcome stay
    disjoint.
    """
    import json

    predictions = predictions[predictions["split"].eq(split)].copy()
    predictions["expected_score"] = predictions.apply(
        lambda row: float((_probability_array(row["probabilities_json"]) * np.arange(row["n_options"])).sum() / max(1, row["n_options"] - 1)),
        axis=1,
    )
    predictions["human_score"] = predictions["answer_index"] / np.maximum(predictions["n_options"] - 1, 1)
    has_selection = "selection_items_json" in matched_pairs.columns

    records = []
    for (method, k), group in predictions.groupby(["method", "k"]):
        by_panel = {
            panel_id: sub.set_index("question_key")[["expected_score", "human_score"]]
            for panel_id, sub in group.groupby("panel_id")
        }
        d_true, d_pred = [], []
        for row in matched_pairs.itertuples(index=False):
            a = by_panel.get(row.panel_id_a)
            b = by_panel.get(row.panel_id_b)
            if a is None or b is None:
                continue
            shared = a.index.intersection(b.index)
            if has_selection:
                shared = shared.difference(json.loads(row.selection_items_json))
            if len(shared) < min_shared_targets:
                continue
            d_true.append(float((a.loc[shared, "human_score"] - b.loc[shared, "human_score"]).abs().mean()))
            d_pred.append(float((a.loc[shared, "expected_score"] - b.loc[shared, "expected_score"]).abs().mean()))
        if len(d_true) < 5:
            records.append({"method": method, "k": int(k), "n_pairs": len(d_true), "rho_heterogeneity": float("nan"), "drr": float("nan")})
            continue
        d_true_arr, d_pred_arr = np.asarray(d_true), np.asarray(d_pred)
        rho = float(pd.Series(d_pred_arr).corr(pd.Series(d_true_arr), method="spearman"))
        drr = float(d_pred_arr.mean() / d_true_arr.mean()) if d_true_arr.mean() > 0 else float("nan")
        records.append({"method": method, "k": int(k), "n_pairs": len(d_true), "rho_heterogeneity": rho, "drr": drr})
    return pd.DataFrame(records)
