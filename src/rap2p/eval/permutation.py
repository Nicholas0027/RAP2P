"""Option-order robustness (see Questioning the Survey Responses of LLMs,
NeurIPS 2024): re-run prediction with several fixed `option_seed` values on the
same 500-respondent subsample and check whether the *semantic* predicted
answer (predictions are already stored in semantic, not label, order -- see
models/common.py:semantic_probabilities_torch) stays the same.
"""

from __future__ import annotations

import pandas as pd


def permutation_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    """`predictions` must contain multiple `option_seed` values for the same
    (method, k, row_id) triples — the concatenation of the perm_*.parquet
    files that evaluate_all.py's permutation loop writes (one prediction pass
    per option_seed over the same respondent subsample; the option_seed is
    threaded into the collator's per-row deterministic label permutation).
    """
    records = []
    for (method, k), group in predictions.groupby(["method", "k"]):
        n_seeds = group["option_seed"].nunique()
        if n_seeds < 2:
            continue
        per_row = group.groupby("row_id")["predicted_index"].nunique()
        consistent = (per_row == 1).mean()
        records.append({"method": method, "k": int(k), "n_permutations": int(n_seeds), "permutation_consistency": float(consistent)})
    return pd.DataFrame(records)
