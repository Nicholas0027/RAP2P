"""Regression tests for the unseen-item leakage family: every 'train-only'
statistic must also exclude is_unseen_item rows, or the OOD-Item axis leaks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rap2p.data import PanelStore, add_question_statistics, make_synthetic_panels
from rap2p.item_graph import compute_item_graph


def _panels_with_unseen():
    responses, orders = make_synthetic_panels(n_panels=40, seed=0)
    assert responses["is_unseen_item"].any(), "synthetic data must include unseen items for these tests"
    return responses, orders


def test_item_graph_rejects_unseen_items():
    responses, _ = _panels_with_unseen()
    train = responses[responses["split"].eq("train")]
    with pytest.raises(ValueError, match="unseen"):
        compute_item_graph(train, shrinkage_lambda=5, min_n_jk=1)  # forgot ~is_unseen_item
    train_seen = train[~train["is_unseen_item"]]
    graphs = compute_item_graph(train_seen, shrinkage_lambda=5, min_n_jk=1)
    unseen_keys = set(responses.loc[responses["is_unseen_item"], "question_key"])
    for matrix in graphs.values():
        assert not (set(matrix.index) & unseen_keys)
        assert not (set(matrix.columns) & unseen_keys)


def test_majority_baseline_never_fits_unseen_items():
    responses, orders = _panels_with_unseen()
    store = PanelStore(responses, orders)
    from rap2p.baselines.majority import predict_majority

    with tempfile.TemporaryDirectory() as tmpdir:
        preds = predict_majority(store, [0], 0, Path(tmpdir) / "m.parquet", split="test", item_pool="unseen")
    # Every unseen-item prediction must be the GLOBAL majority fallback -- if a
    # per-item majority had been fit on unseen items, at least one prediction
    # would differ from the global mode wherever the item's own mode differs.
    train_seen = responses[responses["split"].eq("train") & ~responses["is_unseen_item"]]
    global_majority = int(np.bincount(train_seen["answer_index"].astype(int)).argmax())
    assert (preds["predicted_index"] == min(global_majority, int(preds["n_options"].min()) - 1)).all()


def test_frequency_baseline_falls_back_to_uniform_for_unseen_items():
    responses, orders = _panels_with_unseen()
    store = PanelStore(responses, orders)
    from rap2p.baselines.demographic_frequency import predict_frequency_baseline

    with tempfile.TemporaryDirectory() as tmpdir:
        preds = predict_frequency_baseline(
            store, "demo_freq", [0], 0, Path(tmpdir) / "f.parquet", demographic=True,
            split="test", item_pool="unseen",
        )
    import json

    for value, n_options in zip(preds["probabilities_json"], preds["n_options"]):
        probabilities = np.asarray(json.loads(value))
        assert np.allclose(probabilities, 1.0 / int(n_options)), "unseen-item prediction should be the uniform floor"


def test_question_statistics_use_neutral_defaults_for_unseen_items():
    responses, _ = _panels_with_unseen()
    stats = add_question_statistics(responses)
    unseen = stats[stats["is_unseen_item"]]
    assert (unseen["question_mean"] == 0.5).all()
    assert (unseen["question_std"] == 0.25).all()
    seen_train_items = set(stats.loc[stats["split"].eq("train") & ~stats["is_unseen_item"], "question_key"])
    seen = stats[stats["question_key"].isin(seen_train_items)]
    assert seen["question_mean"].notna().all()
