"""Table 1 (accuracy/MAE/NLL/AUAC) and Table 2 (Group JS, CorrErr, CorrSim)
metrics. Structural metrics use each prediction's *expected* normalized score
E[y] = sum_c p(c) * c / (n_options - 1) rather than Monte Carlo sampled panels
-- simpler, deterministic, and sufficient for the point this table needs to
make (does the model's probability mass covary across items the way humans
do), at the cost of not testing full-distribution panel coherence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

METRIC_COLUMNS = ["correct", "normalized_ordinal_error", "nll", "brier"]
SEED_SUFFIX = re.compile(r"_seed\d+$")


def strip_seed_suffix(method: str) -> str:
    return SEED_SUFFIX.sub("", method)


def load_predictions(
    directory: str | Path,
    patterns: Iterable[str] = ("*.parquet",),
    exclude_prefixes: tuple[str, ...] = ("perm_",),
) -> pd.DataFrame:
    """Permutation-robustness files (perm_*) are excluded by default: they
    cover a respondent subsample at non-primary option seeds, and mixing them
    into the primary tables would double-count those respondents. Load them
    explicitly with patterns=("perm_*.parquet",), exclude_prefixes=()."""
    directory = Path(directory)
    paths = sorted({path for pattern in patterns for path in directory.glob(pattern)})
    paths = [p for p in paths if not any(p.name.startswith(prefix) for prefix in exclude_prefixes)]
    if not paths:
        raise FileNotFoundError(f"No prediction parquet files found in {directory}")
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    key = ["method", "row_id", "k", "split", "item_pool", "calibration_seed", "option_seed"]
    return combined.sort_values(key).drop_duplicates(key, keep="last").reset_index(drop=True)


def respondent_macro_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    panel = predictions.groupby(["method", "k", "split", "item_pool", "domain", "panel_id"], as_index=False)[METRIC_COLUMNS].mean()
    domain = panel.groupby(["method", "k", "split", "item_pool", "domain"], as_index=False)[METRIC_COLUMNS].mean()
    aggregate = domain.groupby(["method", "k", "split", "item_pool"], as_index=False)[METRIC_COLUMNS].mean()
    aggregate["domain"] = "macro"
    combined = pd.concat([domain, aggregate], ignore_index=True)
    return combined.rename(columns={"correct": "accuracy", "normalized_ordinal_error": "mae"})


def delta_k(macro_metrics: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Delta_K = metric(K) - metric(K=0), per method/domain/split."""
    records = []
    for (method, domain, split), group in macro_metrics.groupby(["method", "domain", "split"]):
        group = group.sort_values("k")
        zero = group.loc[group["k"].eq(0), metric]
        if zero.empty:
            continue
        baseline = float(zero.iloc[0])
        for row in group.itertuples(index=False):
            records.append({"method": method, "domain": domain, "split": split, "k": int(row.k), "delta": getattr(row, metric) - baseline})
    return pd.DataFrame(records)


def auac(macro_metrics: pd.DataFrame, metric: str = "accuracy", k_values: tuple[int, ...] = (0, 1, 3, 5, 8)) -> pd.DataFrame:
    """Area under the metric-vs-K curve, averaged over the given K grid."""
    records = []
    for (method, domain, split), group in macro_metrics.groupby(["method", "domain", "split"]):
        subset = group[group["k"].isin(k_values)]
        if subset.empty:
            continue
        records.append({"method": method, "domain": domain, "split": split, "metric": metric, "auac": float(subset[metric].mean())})
    return pd.DataFrame(records)


def _probability_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


def group_js_metrics(predictions: pd.DataFrame, group_columns: tuple[str, ...] = ("country", "age_bin")) -> pd.DataFrame:
    """Marginal-distribution Jensen-Shannon divergence between human and
    predicted answer distributions, aggregated within each demographic group.
    """
    records = []
    group_cols = [c for c in group_columns if c in predictions.columns]
    key_cols = ["method", "k", "split", "domain", "question_key", *group_cols]
    for keys, group in predictions.groupby(key_cols, dropna=False, sort=False):
        n_options = int(group["n_options"].max())
        weights = group["survey_weight"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        human = np.bincount(group["answer_index"].astype(int), weights=weights, minlength=n_options)
        predicted = np.zeros(n_options, dtype=float)
        for weight, value in zip(weights, group["probabilities_json"]):
            predicted += weight * _probability_array(value)
        human = np.clip(human, 1e-12, None); human /= human.sum()
        predicted = np.clip(predicted, 1e-12, None); predicted /= predicted.sum()
        record = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        record["js"] = float(jensenshannon(human, predicted, base=2.0) ** 2)
        records.append(record)
    question = pd.DataFrame(records)
    domain = question.groupby(["method", "k", "split", "domain"], as_index=False)["js"].mean()
    aggregate = domain.groupby(["method", "k", "split"], as_index=False)["js"].mean()
    aggregate["domain"] = "macro"
    return pd.concat([domain, aggregate], ignore_index=True)


def _expected_scores(predictions: pd.DataFrame) -> pd.Series:
    def expected(row) -> float:
        probability = _probability_array(row["probabilities_json"])
        positions = np.arange(len(probability))
        return float((probability * positions).sum() / max(1, len(probability) - 1))

    return predictions.apply(expected, axis=1)


def correlation_structure_metrics(predictions: pd.DataFrame, minimum_pair_n: int = 20) -> pd.DataFrame:
    """CorrErr (RMSE between human and predicted upper-triangle item-item
    correlations) and CorrSim (Pearson correlation between the two upper-triangle
    vectors -- do the models agree on *which* items covary, not just by how much).
    """
    predictions = predictions.copy()
    predictions["expected_score"] = _expected_scores(predictions)
    predictions["human_score"] = predictions["answer_index"] / np.maximum(predictions["n_options"] - 1, 1)

    records = []
    for (method, k, split, domain), group in predictions.groupby(["method", "k", "split", "domain"], sort=False):
        human_wide = group.pivot_table(index="panel_id", columns="question_key", values="human_score")
        predicted_wide = group.pivot_table(index="panel_id", columns="question_key", values="expected_score")
        human_corr = human_wide.corr(min_periods=minimum_pair_n)
        predicted_corr = predicted_wide.corr(min_periods=minimum_pair_n)
        mask = np.triu(np.ones(human_corr.shape, dtype=bool), k=1)
        valid = mask & np.isfinite(human_corr.to_numpy()) & np.isfinite(predicted_corr.to_numpy())
        if not valid.any():
            records.append({"method": method, "k": int(k), "split": split, "domain": domain, "corr_err": float("nan"), "corr_sim": float("nan")})
            continue
        human_values = human_corr.to_numpy()[valid]
        predicted_values = predicted_corr.to_numpy()[valid]
        corr_err = float(np.sqrt(np.mean(np.square(human_values - predicted_values))))
        corr_sim = float(np.corrcoef(human_values, predicted_values)[0, 1]) if len(human_values) > 1 else float("nan")
        records.append({"method": method, "k": int(k), "split": split, "domain": domain, "corr_err": corr_err, "corr_sim": corr_sim})
    domain_table = pd.DataFrame(records)
    aggregate = domain_table.groupby(["method", "k", "split"], as_index=False)[["corr_err", "corr_sim"]].mean()
    aggregate["domain"] = "macro"
    return pd.concat([domain_table, aggregate], ignore_index=True)


def worst_group_accuracy(predictions: pd.DataFrame, group_columns: tuple[str, ...] = ("country", "age_bin", "education", "income_quintile"), min_group_n: int = 30) -> pd.DataFrame:
    group_cols = [c for c in group_columns if c in predictions.columns]
    if not group_cols:
        return pd.DataFrame(columns=["method", "k", "split", "worst_group_accuracy", "mean_group_accuracy", "n_groups"])
    records = []
    for (method, k, split), group in predictions.groupby(["method", "k", "split"]):
        panel_acc = group.groupby(["panel_id", *group_cols])["correct"].mean().reset_index()
        cell_acc = panel_acc.groupby(group_cols).agg(accuracy=("correct", "mean"), n=("correct", "size")).reset_index()
        cell_acc = cell_acc[cell_acc["n"] >= min_group_n]
        if cell_acc.empty:
            continue
        records.append(
            {
                "method": method, "k": int(k), "split": split,
                "worst_group_accuracy": float(cell_acc["accuracy"].min()),
                "mean_group_accuracy": float(cell_acc["accuracy"].mean()),
                "n_groups": int(len(cell_acc)),
            }
        )
    return pd.DataFrame(records)


def aggregate_seeds(macro: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed method labels (e.g. rap2p_seed1701/rap2p_seed7) into
    one row per base method with mean and std across seeds — the numbers the
    paper's tables actually report. Methods without a seed suffix pass
    through with std 0/NaN untouched.
    """
    frame = macro.copy()
    frame["base_method"] = frame["method"].map(strip_seed_suffix)
    value_columns = [c for c in ("accuracy", "mae", "nll", "brier") if c in frame.columns]
    grouped = frame.groupby(["base_method", "k", "split", "item_pool", "domain"], as_index=False)
    means = grouped[value_columns].mean()
    stds = grouped[value_columns].std().rename(columns={c: f"{c}_std" for c in value_columns})
    counts = grouped["method"].nunique().rename(columns={"method": "n_seeds"})
    out = means.merge(stds, on=["base_method", "k", "split", "item_pool", "domain"])
    out = out.merge(counts, on=["base_method", "k", "split", "item_pool", "domain"])
    return out.rename(columns={"base_method": "method"})


def build_main_table(
    macro_seed_mean: pd.DataFrame,
    k_values: tuple[int, ...] = (0, 1, 3, 5, 8),
    reference_k: int = 5,
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    """Assemble the paper's Table 1 layout: one row per method, accuracy at
    each K, AUAC (unweighted mean accuracy over the K grid), and MAE/NLL at
    the reference K — directly from the seed-aggregated macro table.
    """
    frame = macro_seed_mean[
        macro_seed_mean["split"].eq(split)
        & macro_seed_mean["item_pool"].eq(item_pool)
        & macro_seed_mean["domain"].eq("macro")
        & macro_seed_mean["k"].isin(k_values)
    ]
    rows = []
    for method, group in frame.groupby("method"):
        by_k = group.set_index("k")
        row: dict[str, object] = {"method": method}
        for k in k_values:
            row[f"acc_k{k}"] = float(by_k.at[k, "accuracy"]) if k in by_k.index else float("nan")
        available = [row[f"acc_k{k}"] for k in k_values if not np.isnan(row[f"acc_k{k}"])]
        row["auac"] = float(np.mean(available)) if available else float("nan")
        row[f"mae_k{reference_k}"] = float(by_k.at[reference_k, "mae"]) if reference_k in by_k.index else float("nan")
        row[f"nll_k{reference_k}"] = float(by_k.at[reference_k, "nll"]) if reference_k in by_k.index else float("nan")
        row["n_seeds"] = int(group["n_seeds"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auac", ascending=False).reset_index(drop=True)


def compute_metric_tables(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    macro = respondent_macro_metrics(predictions)
    macro_seed_mean = aggregate_seeds(macro)
    tables = {
        "respondent_macro": macro,
        "respondent_macro_seed_mean": macro_seed_mean,
        "table1_main": build_main_table(macro_seed_mean),
        "delta_k_accuracy": delta_k(macro, "accuracy"),
        "auac_accuracy": auac(macro, "accuracy"),
        "auac_nll": auac(macro, "nll"),
        "group_js": group_js_metrics(predictions),
        "correlation_structure": correlation_structure_metrics(predictions),
        "worst_group_accuracy": worst_group_accuracy(predictions),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
    return tables
