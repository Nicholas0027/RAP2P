from __future__ import annotations

import pandas as pd
import pytest

from rap2p.eval.bootstrap import bootstrap_table, holm_correction, paired_bootstrap


def test_holm_correction_matches_hand_worked_example():
    # p = [0.01, 0.04, 0.03, 0.20] at original indices [0, 1, 2, 3], alpha = 0.05.
    # Sorted ascending: 0.01, 0.03, 0.04, 0.20 against step-down thresholds
    # 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05.
    # 0.01 <= 0.0125 -> reject; 0.03 > 0.0167 -> step-down stops here, so
    # everything from 0.03 onward (0.03, 0.04, 0.20) is *not* rejected, even
    # though 0.03 alone would clear a later, looser threshold.
    p_values = [0.01, 0.04, 0.03, 0.20]
    results = holm_correction(p_values, alpha=0.05)
    assert results[0]["reject_null"] is True   # p=0.01, first sorted position
    assert results[1]["reject_null"] is False  # p=0.04
    assert results[2]["reject_null"] is False  # p=0.03 -- step-down already stopped
    assert results[3]["reject_null"] is False  # p=0.20


def test_holm_correction_preserves_input_order():
    p_values = [0.5, 0.001, 0.3]
    results = holm_correction(p_values)
    assert len(results) == len(p_values)
    assert results[1]["reject_null"] is True
    assert results[0]["p_value"] == 0.5
    assert results[2]["p_value"] == 0.3


def _toy_predictions() -> pd.DataFrame:
    rows = []
    for panel_index in range(40):
        # method_b is systematically more accurate (80%) than method_a (20%).
        rows.append({"method": "method_a", "domain": "d", "panel_id": f"p{panel_index}", "split": "test", "k": 5, "correct": 1 if panel_index % 5 == 0 else 0})
        rows.append({"method": "method_b", "domain": "d", "panel_id": f"p{panel_index}", "split": "test", "k": 5, "correct": 1 if panel_index % 5 else 0})
    return pd.DataFrame(rows)


def test_paired_bootstrap_detects_a_real_difference():
    predictions = _toy_predictions()
    result = paired_bootstrap(predictions, "method_b", "method_a", metric="accuracy", k=5, replicates=500, seed=0)
    assert result["n_panels"] == 40
    assert result["difference"] > 0  # method_b is the more-accurate one in this fixture
    assert result["ci_low"] > 0 or result["p_two_sided"] < 0.5


def test_paired_bootstrap_raises_on_no_overlap():
    predictions = _toy_predictions()
    disjoint = pd.concat(
        [
            predictions[predictions["method"].eq("method_a") & predictions["panel_id"].isin([f"p{i}" for i in range(20)])],
            predictions[predictions["method"].eq("method_b") & predictions["panel_id"].isin([f"p{i}" for i in range(20, 40)])],
        ]
    )
    with pytest.raises(ValueError):
        paired_bootstrap(disjoint, "method_b", "method_a", k=5)


def test_bootstrap_table_runs_over_multiple_k():
    predictions = pd.concat(
        [_toy_predictions().assign(k=k) for k in (0, 5)],
        ignore_index=True,
    )
    table = bootstrap_table(predictions, [("method_b", "method_a")], k_values=(0, 5), replicates=200, seed=1)
    assert len(table) == 2
    assert "holm_reject_null" in table.columns
