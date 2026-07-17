"""Tests for the two evaluation-side blocking bugs found in review: the
permutation-robustness pipeline (option_seed must actually change the
permutation, and the consistency metric must see multiple seeds) and the API
label parser (must not grab a stray letter from a preamble)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rap2p.baselines.prompting_baseline import parse_label_response
from rap2p.eval.permutation import permutation_consistency
from rap2p.prompting import deterministic_permutation


def test_deterministic_permutation_varies_with_seed_and_is_reproducible():
    a0 = deterministic_permutation(5, 0, "row1")
    a0_again = deterministic_permutation(5, 0, "row1")
    a1 = deterministic_permutation(5, 1, "row1")
    assert (a0 == a0_again).all()
    seeds = [deterministic_permutation(5, s, "row1").tolist() for s in range(5)]
    assert len({tuple(p) for p in seeds}) > 1, "different option_seeds must yield different permutations"


def test_permutation_consistency_counts_semantic_stability_across_seeds():
    rows = []
    for option_seed in range(3):
        # row_stable: same semantic answer under every permutation.
        rows.append({"method": "m", "k": 5, "row_id": "row_stable", "option_seed": option_seed, "predicted_index": 2})
        # row_flippy: answer changes with the permutation.
        rows.append({"method": "m", "k": 5, "row_id": "row_flippy", "option_seed": option_seed, "predicted_index": option_seed})
    table = permutation_consistency(pd.DataFrame(rows))
    assert len(table) == 1
    record = table.iloc[0]
    assert record["n_permutations"] == 3
    assert record["permutation_consistency"] == 0.5  # 1 of 2 rows stable


def test_permutation_consistency_returns_empty_for_single_seed():
    rows = [{"method": "m", "k": 5, "row_id": f"r{i}", "option_seed": 0, "predicted_index": 1} for i in range(4)]
    table = permutation_consistency(pd.DataFrame(rows))
    assert table.empty


def test_parse_label_response_clean_answers():
    assert parse_label_response("B", 5) == (1, False)
    assert parse_label_response(" C.", 5) == (2, False)
    assert parse_label_response("(D)", 5) == (3, False)
    assert parse_label_response("b", 5) == (1, False)


def test_parse_label_response_ignores_preamble_letters():
    # "I" is a valid label only for 9+ option questions; with 5 options it
    # must not be extracted from a conversational preamble.
    assert parse_label_response("I think the answer is B", 5) == (1, False)
    assert parse_label_response("The answer is C because...", 5) == (2, False)
    # Multiple standalone candidates: the LAST one wins.
    assert parse_label_response("A or B? Definitely B", 5) == (1, False)


def test_parse_label_response_flags_unparseable():
    index, failed = parse_label_response("no idea", 5)
    assert failed is True and index == 0
    index, failed = parse_label_response("", 5)
    assert failed is True
    # A letter outside the valid range for this question is not a candidate.
    index, failed = parse_label_response("G", 5)
    assert failed is True


def test_parse_label_response_accepts_high_letters_when_valid():
    assert parse_label_response("I", 10) == (8, False)
    assert parse_label_response("The answer is J", 10) == (9, False)


def test_subsample_panels_is_deterministic_and_respects_size():
    from rap2p.inference import subsample_panels

    frame = pd.DataFrame({"panel_id": [f"p{i}" for i in range(50)] * 2, "value": np.arange(100)})
    small = subsample_panels(frame, 10)
    small_again = subsample_panels(frame, 10)
    assert small["panel_id"].nunique() == 10
    assert sorted(small["panel_id"].unique()) == sorted(small_again["panel_id"].unique())
    assert subsample_panels(frame, None) is frame
    assert subsample_panels(frame, 100)["panel_id"].nunique() == 50  # no-op when sample >= population
