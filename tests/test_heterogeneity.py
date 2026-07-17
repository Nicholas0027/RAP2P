from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rap2p.eval.heterogeneity import (
    filter_pairs_by_answer_divergence,
    find_matched_pairs,
    matched_profile_heterogeneity,
)


def _panels(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "panel_id": f"p{i}", "domain": "d",
                "age_bin": ["18-29", "30-44"][i % 2],
                "education": ["Primary", "Secondary"][i % 2],
                "income_quintile": f"Q{1 + i % 5}",
                "urbanicity": ["Urban", "Rural"][i % 2],
            }
        )
    return pd.DataFrame(rows)


def test_find_matched_pairs_exact_match_only_pairs_identical_profiles():
    panels = _panels()
    pairs = find_matched_pairs(panels, max_attribute_distance=0, seed=0)
    for row in pairs.itertuples(index=False):
        a = panels.set_index("panel_id").loc[row.panel_id_a]
        b = panels.set_index("panel_id").loc[row.panel_id_b]
        assert (a[["age_bin", "education", "income_quintile", "urbanicity"]] == b[["age_bin", "education", "income_quintile", "urbanicity"]]).all()


def test_find_matched_pairs_distance_one_allows_a_single_mismatch():
    panels = _panels()
    exact = find_matched_pairs(panels, max_attribute_distance=0, seed=0)
    relaxed = find_matched_pairs(panels, max_attribute_distance=1, seed=0)
    assert len(relaxed) >= len(exact)
    attrs = ["age_bin", "education", "income_quintile", "urbanicity"]
    lookup = panels.set_index("panel_id")
    for row in relaxed.itertuples(index=False):
        a, b = lookup.loc[row.panel_id_a, attrs], lookup.loc[row.panel_id_b, attrs]
        assert int((a != b).sum()) <= 1


def test_find_matched_pairs_never_pairs_a_panel_with_itself():
    panels = _panels()
    pairs = find_matched_pairs(panels, max_attribute_distance=1, seed=0)
    assert not (pairs["panel_id_a"] == pairs["panel_id_b"]).any()


def _prediction_row(method: str, k: int, panel_id: str, question_key: str, answer_index: int, n_options: int, probability: list[float]) -> dict:
    return {
        "method": method, "k": k, "split": "test", "panel_id": panel_id, "question_key": question_key,
        "answer_index": answer_index, "n_options": n_options, "probabilities_json": json.dumps(probability),
    }


def test_matched_profile_heterogeneity_rewards_a_model_that_tracks_true_divergence():
    # 5 matched pairs (minimum for matched_profile_heterogeneity to report a
    # non-NaN rho/DRR rather than treating the sample as too small to trust).
    pair_ids = [(f"p{2*i}", f"p{2*i+1}") for i in range(5)]
    pairs = pd.DataFrame([{"domain": "d", "panel_id_a": a, "panel_id_b": b} for a, b in pair_ids])
    rows = []
    for a, b in pair_ids:
        for question in ["d::v0", "d::v1", "d::v2", "d::v3"]:
            # Ground truth: `a` answers high on every item, `b` answers low -- a large true divergence.
            rows.append(_prediction_row("good_model", 5, a, question, 4, 5, [0.0, 0.0, 0.0, 0.0, 1.0]))
            rows.append(_prediction_row("good_model", 5, b, question, 0, 5, [1.0, 0.0, 0.0, 0.0, 0.0]))
            # A collapsed model predicts the same (uninformative) middle answer for everyone.
            rows.append(_prediction_row("collapsed_model", 5, a, question, 4, 5, [0.0, 0.0, 1.0, 0.0, 0.0]))
            rows.append(_prediction_row("collapsed_model", 5, b, question, 0, 5, [0.0, 0.0, 1.0, 0.0, 0.0]))
    predictions = pd.DataFrame(rows)

    result = matched_profile_heterogeneity(predictions, pairs, min_shared_targets=2)
    good = result[result["method"].eq("good_model")].iloc[0]
    collapsed = result[result["method"].eq("collapsed_model")].iloc[0]
    assert good["n_pairs"] == 5
    assert good["drr"] > collapsed["drr"]
    assert collapsed["drr"] == 0.0  # collapsed model predicts zero divergence between every pair


def test_matched_profile_heterogeneity_returns_nan_below_minimum_pair_count():
    pairs = pd.DataFrame([{"domain": "d", "panel_id_a": "p0", "panel_id_b": "p1"}])
    rows = [
        _prediction_row("some_model", 0, "p0", "d::v0", 2, 5, [0.2, 0.2, 0.2, 0.2, 0.2]),
        _prediction_row("some_model", 0, "p1", "d::v0", 2, 5, [0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    result = matched_profile_heterogeneity(pd.DataFrame(rows), pairs, min_shared_targets=2)
    assert result.iloc[0]["n_pairs"] == 0
    assert np.isnan(result.iloc[0]["rho_heterogeneity"])


def _responses_for_pair(divergent: bool) -> pd.DataFrame:
    """p0/p1 answer 8 shared items; divergent pairs answer at opposite scale
    ends, convergent pairs answer identically."""
    rows = []
    for i in range(8):
        rows.append({"panel_id": "p0", "question_key": f"d::v{i}", "answer_index": 4, "n_options": 5})
        rows.append({"panel_id": "p1", "question_key": f"d::v{i}", "answer_index": 0 if divergent else 4, "n_options": 5})
    return pd.DataFrame(rows)


def test_divergence_filter_keeps_divergent_pairs_and_drops_convergent_ones():
    pairs = pd.DataFrame([{"domain": "d", "panel_id_a": "p0", "panel_id_b": "p1"}])
    kept = filter_pairs_by_answer_divergence(
        pairs, _responses_for_pair(divergent=True), selection_items=3, min_divergence=0.35, min_shared_items=3
    )
    assert len(kept) == 1
    assert kept.iloc[0]["history_divergence"] == 1.0
    dropped = filter_pairs_by_answer_divergence(
        pairs, _responses_for_pair(divergent=False), selection_items=3, min_divergence=0.35, min_shared_items=3
    )
    assert dropped.empty


def test_divergence_filter_drops_pairs_with_too_few_shared_items():
    pairs = pd.DataFrame([{"domain": "d", "panel_id_a": "p0", "panel_id_b": "p1"}])
    responses = pd.DataFrame(
        [
            {"panel_id": "p0", "question_key": "d::v0", "answer_index": 4, "n_options": 5},
            {"panel_id": "p1", "question_key": "d::v1", "answer_index": 0, "n_options": 5},  # no overlap at all
        ]
    )
    kept = filter_pairs_by_answer_divergence(pairs, responses, selection_items=3, min_divergence=0.0, min_shared_items=1)
    assert kept.empty


def test_selection_items_are_excluded_from_target_comparison():
    """The items used to select a pair (recorded in selection_items_json) must
    not also be counted as evaluation targets -- otherwise the filter inflates
    D_true by construction."""
    import json

    pairs = pd.DataFrame(
        [
            {
                "domain": "d", "panel_id_a": "p0", "panel_id_b": "p1",
                "history_divergence": 1.0,
                "selection_items_json": json.dumps(["d::v0", "d::v1"]),
            }
        ]
    )
    rows = []
    # Selection items d::v0/v1: maximally divergent truth. Target items
    # d::v2..v4: identical truth. If selection items leaked into targets,
    # D_true would be > 0; correctly excluded, D_true == 0.
    for question in ["d::v0", "d::v1"]:
        rows.append(_prediction_row("m", 5, "p0", question, 4, 5, [0.0, 0.0, 0.0, 0.0, 1.0]))
        rows.append(_prediction_row("m", 5, "p1", question, 0, 5, [1.0, 0.0, 0.0, 0.0, 0.0]))
    for question in ["d::v2", "d::v3", "d::v4"]:
        rows.append(_prediction_row("m", 5, "p0", question, 2, 5, [0.0, 0.0, 1.0, 0.0, 0.0]))
        rows.append(_prediction_row("m", 5, "p1", question, 2, 5, [0.0, 0.0, 1.0, 0.0, 0.0]))
    # Build 5 such pairs so the metric reports non-NaN aggregates.
    all_pairs = pd.concat([pairs.assign(panel_id_a=f"p{2*i}", panel_id_b=f"p{2*i+1}") for i in range(5)], ignore_index=True)
    all_rows = []
    for i in range(5):
        for row in rows:
            clone = dict(row)
            clone["panel_id"] = f"p{2*i}" if row["panel_id"] == "p0" else f"p{2*i+1}"
            all_rows.append(clone)
    result = matched_profile_heterogeneity(pd.DataFrame(all_rows), all_pairs, min_shared_targets=2)
    record = result.iloc[0]
    assert record["n_pairs"] == 5
    # Targets are the identical-truth items only -> mean true divergence is 0,
    # so DRR hits its division guard and reports NaN. Had the maximally
    # divergent selection items leaked into the targets, D_true would be
    # positive and DRR would be a (spuriously meaningful) finite number.
    assert np.isnan(record["drr"])
