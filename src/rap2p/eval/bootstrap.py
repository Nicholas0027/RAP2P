"""Respondent-level paired bootstrap with Holm correction across the family of
method-vs-Context-QLoRA (and method-vs-P2P-Static) comparisons that back every
"significantly better" claim in the paper draft.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def paired_bootstrap(
    predictions: pd.DataFrame,
    method: str,
    reference: str,
    metric: str = "accuracy",
    k: int | None = None,
    split: str = "test",
    replicates: int = 2000,
    seed: int = 1701,
) -> dict[str, float]:
    frame = predictions[predictions["split"].eq(split)]
    if k is not None:
        frame = frame[frame["k"].eq(k)]
    if "item_pool" in frame.columns and frame["item_pool"].nunique() > 1:
        raise ValueError(
            "paired_bootstrap received predictions spanning multiple item_pools "
            f"({sorted(frame['item_pool'].unique())}); filter to a single item_pool first "
            "or the seen/unseen populations get silently blended."
        )
    value_col = "correct" if metric == "accuracy" else metric
    panel = frame.groupby(["method", "domain", "panel_id"], as_index=False)[value_col].mean()
    left = panel[panel["method"].eq(method)].rename(columns={value_col: "candidate"})
    right = panel[panel["method"].eq(reference)].rename(columns={value_col: "reference"})
    paired = left.merge(right[["domain", "panel_id", "reference"]], on=["domain", "panel_id"])
    if paired.empty:
        raise ValueError(f"No overlapping (domain, panel_id) rows between {method!r} and {reference!r}")
    paired["difference"] = paired["candidate"] - paired["reference"]

    observed = float(paired.groupby("domain")["difference"].mean().mean())
    rng = np.random.default_rng(seed)
    domain_groups = [group["difference"].to_numpy() for _, group in paired.groupby("domain")]
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        draws[replicate] = np.mean([rng.choice(values, size=len(values), replace=True).mean() for values in domain_groups])

    return {
        "method": method, "reference": reference, "metric": metric, "k": -1 if k is None else int(k),
        "difference": observed, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
        "n_panels": int(paired["panel_id"].nunique()),
    }


def holm_correction(p_values: Iterable[float], alpha: float = 0.05) -> list[dict[str, float]]:
    """Holm-Bonferroni step-down correction. Returns, in the *original* input
    order, whether each comparison rejects the null at the family-wise alpha.
    """
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    m = len(indexed)
    results: list[dict[str, float] | None] = [None] * m
    still_rejecting = True
    for rank, (original_index, p_value) in enumerate(indexed):
        threshold = alpha / (m - rank)
        reject = still_rejecting and p_value <= threshold
        still_rejecting = still_rejecting and reject
        results[original_index] = {"p_value": p_value, "holm_threshold": threshold, "reject_null": reject}
    return results  # type: ignore[return-value]


def bootstrap_table(
    predictions: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    metric: str = "accuracy",
    k_values: Iterable[int] = (0, 1, 3, 5, 8),
    replicates: int = 2000,
    seed: int = 1701,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for method, reference in comparisons:
        for k in k_values:
            rows.append(paired_bootstrap(predictions, method, reference, metric, k, replicates=replicates, seed=seed))
    table = pd.DataFrame(rows)
    holm = holm_correction(table["p_two_sided"].tolist(), alpha=alpha)
    table["holm_reject_null"] = [r["reject_null"] for r in holm]
    table["holm_threshold"] = [r["holm_threshold"] for r in holm]
    return table
