from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rap2p.item_graph import ItemGraph, assert_train_only, compute_item_graph, save_item_graph


def _synthetic_responses(n_panels: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for panel_index in range(n_panels):
        latent = rng.normal()
        split = "train" if panel_index < 50 else "test"
        # v0 and v1 are strongly correlated (share the same latent driver);
        # v2 is pure noise, uncorrelated with either.
        v0 = latent + rng.normal(scale=0.2)
        v1 = latent + rng.normal(scale=0.2)
        v2 = rng.normal()
        for question_key, value in (("d::v0", v0), ("d::v1", v1), ("d::v2", v2)):
            rows.append(
                {
                    "panel_id": f"p{panel_index}", "domain": "d", "question_key": question_key,
                    "normalized_answer": value, "split": split,
                }
            )
    return pd.DataFrame(rows)


def test_assert_train_only_rejects_mixed_splits():
    frame = _synthetic_responses()
    with pytest.raises(ValueError):
        compute_item_graph(frame, shrinkage_lambda=10, min_n_jk=5)  # includes "test" rows
    with pytest.raises(ValueError):
        assert_train_only(frame)


def test_correlation_graph_recovers_known_structure():
    frame = _synthetic_responses()
    train_only = frame[frame["split"].eq("train")]
    graphs = compute_item_graph(train_only, shrinkage_lambda=5, min_n_jk=5)
    matrix = graphs["d"]
    assert matrix.at["d::v0", "d::v1"] > matrix.at["d::v0", "d::v2"]
    assert matrix.at["d::v0", "d::v1"] > 0.3
    assert abs(matrix.at["d::v0", "d::v2"]) < 0.3
    assert matrix.at["d::v0", "d::v0"] == pytest.approx(1.0)


def test_shrinkage_pulls_low_count_pairs_toward_zero():
    frame = _synthetic_responses(n_panels=60)
    train_only = frame[frame["split"].eq("train")]
    graphs_min1 = compute_item_graph(train_only, shrinkage_lambda=5, min_n_jk=1)
    graphs_min1000 = compute_item_graph(train_only, shrinkage_lambda=5, min_n_jk=1000)
    # With an unreachable min_n_jk threshold every off-diagonal entry must be zeroed.
    matrix = graphs_min1000["d"]
    off_diagonal = matrix.to_numpy()[~np.eye(matrix.shape[0], dtype=bool)]
    assert (off_diagonal == 0).all()
    assert graphs_min1["d"].at["d::v0", "d::v1"] != 0


def test_item_graph_round_trip_and_lookup(tmp_path):
    frame = _synthetic_responses()
    train_only = frame[frame["split"].eq("train")]
    graphs = compute_item_graph(train_only, shrinkage_lambda=5, min_n_jk=5)
    save_item_graph(graphs, tmp_path)
    loaded = ItemGraph.from_dir(tmp_path)
    assert loaded.get("d", "d::v0", "d::v1") == pytest.approx(graphs["d"].at["d::v0", "d::v1"])
    batch = loaded.batch("d", "d::v0", ["d::v1", "d::v2", "unknown_key"])
    assert batch.shape == (3,)
    assert batch[2] == 0.0  # unknown history item -> no signal, not an error
