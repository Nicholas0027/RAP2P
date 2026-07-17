#!/usr/bin/env python3
"""Run every prediction pass needed for Tables 1-3 + the heterogeneity table +
permutation robustness, then compute all metric/bootstrap tables.

This does NOT call the strong-API baseline (see scripts/run_api_baseline.py --
kept separate because it needs API keys and has its own cost/rate-limit budget).

Usage:
    python scripts/evaluate_all.py --config configs/mvp.yaml
    python scripts/evaluate_all.py --config configs/mvp.yaml --skip-permutation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rap2p.baselines.majority import predict_majority  # noqa: E402
from rap2p.baselines.demographic_frequency import predict_frequency_baseline  # noqa: E402
from rap2p.baselines.mirt import fit_mirt, load_mirt, predict_mirt  # noqa: E402
from rap2p.baselines.prompting_baseline import predict_local_prompt  # noqa: E402
from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.data import PanelStore  # noqa: E402
from rap2p.embeddings import EmbeddingStore  # noqa: E402
from rap2p.eval.bootstrap import bootstrap_table  # noqa: E402
from rap2p.eval.heterogeneity import (  # noqa: E402
    DEFAULT_MATCH_ATTRIBUTES,
    filter_pairs_by_answer_divergence,
    find_matched_pairs,
    matched_profile_heterogeneity,
)
from rap2p.eval.metrics import compute_metric_tables, load_predictions  # noqa: E402
from rap2p.eval.permutation import permutation_consistency  # noqa: E402
from rap2p.inference import domain_mean_item_embedding  # noqa: E402
from rap2p.models.common import load_backbone_and_tokenizer  # noqa: E402
from rap2p.workflows import choose_device, run_prediction_job  # noqa: E402

# (run_name, method_label, seed) for every trained checkpoint expected by the
# compute plan in README "Compute budget". Predictions for each seed are kept
# separate (method label includes the seed) so bootstrap_table/aggregation can
# report seed variance explicitly rather than silently averaging it away.
TRAINED_RUNS = [
    ("global_qlora", "global_qlora", 1701), ("global_qlora", "global_qlora", 7),
    ("context_qlora", "context_qlora", 1701), ("context_qlora", "context_qlora", 7),
    ("p2p_static", "p2p_static", 1701), ("p2p_static", "p2p_static", 7),
    ("rap2p", "rap2p", 1701), ("rap2p", "rap2p", 7), ("rap2p", "rap2p", 42),
    ("rap2p_no_graph", "rap2p_no_graph", 1701),
    ("rap2p_no_history_retrained", "rap2p_no_history_retrained", 1701),
]
PRIMARY_RAP2P_SEED = 1701  # checkpoint used for the "free" ablations + heterogeneity + permutation


def run_trained_predictions(config, k_values, split, item_pool) -> None:
    for run_name, method_label, seed in TRAINED_RUNS:
        checkpoint_run = f"{run_name}_seed{seed}"
        method = f"{method_label}_seed{seed}"
        output_name = f"{method}__{split}__{item_pool}.parquet"
        print(f"[predict] {method} split={split} item_pool={item_pool}")
        run_prediction_job(
            config, run_name, method, output_name, checkpoint_run=checkpoint_run,
            k_values=k_values, split=split, item_pool=item_pool,
        )


def run_free_ablations(config, k_values, split="test", item_pool="seen") -> None:
    checkpoint_run = f"rap2p_seed{PRIMARY_RAP2P_SEED}"
    ablations = {
        "rap2p_no_demographics_free": dict(force_demographics_off=True),
        "rap2p_no_history_free": dict(force_history_off=True),
        "rap2p_no_graph_free": dict(force_correlation_off=True),
    }
    for method, kwargs in ablations.items():
        print(f"[predict] {method} (free ablation on primary RAP2P checkpoint)")
        run_prediction_job(
            config, "rap2p", method, f"{method}__{split}__{item_pool}.parquet",
            checkpoint_run=checkpoint_run, k_values=k_values, split=split, item_pool=item_pool, **kwargs,
        )

    print("[predict] rap2p_uniform_gate_free (ignores learned router entirely)")
    run_prediction_job(
        config, "rap2p", "rap2p_uniform_gate_free", f"rap2p_uniform_gate_free__{split}__{item_pool}.parquet",
        checkpoint_run=checkpoint_run, k_values=k_values, split=split, item_pool=item_pool, model_ablation="uniform_gate",
    )

    embeddings = EmbeddingStore(config["paths"]["embeddings"])
    items = pd.read_parquet(Path(config["paths"]["processed"]) / "items.parquet")
    for domain in config["data"]["domains"]:
        mean_embedding = domain_mean_item_embedding(embeddings, items, domain)
        method = f"rap2p_target_independent_free_{domain.replace(' ', '_')}"
        print(f"[predict] {method}")
        run_prediction_job(
            config, "rap2p", method, f"{method}__{split}__{item_pool}.parquet",
            checkpoint_run=checkpoint_run, k_values=k_values, split=split, item_pool=item_pool,
            domains=[domain], override_item_embeddings=mean_embedding,
        )


def run_non_llm_baselines(config, k_values, split, item_pool) -> None:
    store = PanelStore.from_dir(config["paths"]["processed"])
    predictions_dir = Path(config["paths"]["predictions"])

    print("[predict] majority")
    predict_majority(store, k_values, config["data"]["calibration_seed"], predictions_dir / f"majority__{split}__{item_pool}.parquet", split=split, item_pool=item_pool)

    print("[predict] demographic_frequency")
    predict_frequency_baseline(
        store, "demographic_frequency", k_values, config["data"]["calibration_seed"],
        predictions_dir / f"demographic_frequency__{split}__{item_pool}.parquet",
        demographic=True, split=split, item_pool=item_pool,
    )
    print("[predict] question_frequency (no demographics)")
    predict_frequency_baseline(
        store, "question_frequency", k_values, config["data"]["calibration_seed"],
        predictions_dir / f"question_frequency__{split}__{item_pool}.parquet",
        demographic=False, split=split, item_pool=item_pool,
    )

    if item_pool == "unseen":
        print("[skip] mirt has no parameters for unseen items (see baselines/mirt.py)")
        return
    mirt_dir = Path(config["paths"]["checkpoints"]) / "mirt"
    responses = store.responses
    if not (mirt_dir / "metadata.json").exists():
        print("[fit] mirt")
        fit_mirt(responses, mirt_dir)
    model, encoder, metadata = load_mirt(mirt_dir)
    print("[predict] mirt")
    predict_mirt(
        responses, store.history_rows, store.target_rows, model, encoder, metadata,
        k_values, config["data"]["calibration_seed"], predictions_dir / f"mirt__{split}__{item_pool}.parquet",
        split=split, item_pool=item_pool,
    )


def run_local_prompt_baselines(config, k_values, split, item_pool) -> None:
    store = PanelStore.from_dir(config["paths"]["processed"])
    model_config = config["model"]
    device = choose_device()
    base, tokenizer = load_backbone_and_tokenizer(
        model_config["backbone"], dtype=model_config["dtype"], quantization=model_config["quantization"], device_map=None,
    )
    predictions_dir = Path(config["paths"]["predictions"])
    print("[predict] local_persona (demographics only, no history)")
    predict_local_prompt(
        store, base, tokenizer, "local_persona", k_values, config["data"]["calibration_seed"],
        predictions_dir / f"local_persona__{split}__{item_pool}.parquet", device,
        max_length=model_config["max_length"], split=split, item_pool=item_pool, include_history=False,
    )
    print("[predict] local_sparse_icl (demographics + K known answers in prompt)")
    predict_local_prompt(
        store, base, tokenizer, "local_sparse_icl", k_values, config["data"]["calibration_seed"],
        predictions_dir / f"local_sparse_icl__{split}__{item_pool}.parquet", device,
        max_length=model_config["max_length"], split=split, item_pool=item_pool, include_history=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-local-prompt", action="store_true", help="Local-8B prompting is a full extra inference pass; skip to save time on a quick pass.")
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    k_values = list(config["data"]["k_values"])

    # --- ID-respondent test split (Table 1, Table 2) ---
    run_non_llm_baselines(config, k_values, split="test", item_pool="seen")
    if not args.skip_local_prompt:
        run_local_prompt_baselines(config, k_values, split="test", item_pool="seen")
    run_trained_predictions(config, k_values, split="test", item_pool="seen")
    run_free_ablations(config, k_values, split="test", item_pool="seen")

    # --- OOD-intersection split (Table 3) ---
    run_non_llm_baselines(config, k_values, split="ood_intersection", item_pool="seen")
    run_trained_predictions(config, k_values, split="ood_intersection", item_pool="seen")

    # --- OOD-item axis (Table 3; MIRT/question_frequency-with-history are N/A here) ---
    run_non_llm_baselines(config, k_values, split="test", item_pool="unseen")
    run_trained_predictions(config, k_values, split="test", item_pool="unseen")

    # --- Permutation robustness (Table S2): re-run the SAME 500-respondent
    # subsample at several option_seed values. option_seed is threaded through
    # to the collator's per-row deterministic permutation, so each pass really
    # does present the options in a different order; the subsample keeps this
    # affordable (5 permutations x full test would be five extra full passes).
    if not args.skip_permutation:
        perm_cfg = config["evaluation"]["permutation"]
        n_perm = int(perm_cfg["n_permutations"])
        perm_sample = int(perm_cfg["respondent_sample"])
        for option_seed in range(n_perm):
            print(f"[permutation] option_seed={option_seed}")
            run_prediction_job(
                config, "rap2p", f"rap2p_seed{PRIMARY_RAP2P_SEED}", f"perm_rap2p_seed{option_seed}.parquet",
                checkpoint_run=f"rap2p_seed{PRIMARY_RAP2P_SEED}", k_values=k_values, split="test", item_pool="seen",
                option_seed=option_seed, respondent_sample=perm_sample,
            )
            run_prediction_job(
                config, "context_qlora", "context_qlora_seed1701", f"perm_context_qlora_seed{option_seed}.parquet",
                checkpoint_run="context_qlora_seed1701", k_values=k_values, split="test", item_pool="seen",
                option_seed=option_seed, respondent_sample=perm_sample,
            )
        if not args.skip_local_prompt:
            store = PanelStore.from_dir(config["paths"]["processed"])
            model_config = config["model"]
            base, tokenizer = load_backbone_and_tokenizer(
                model_config["backbone"], dtype=model_config["dtype"], quantization=model_config["quantization"], device_map=None,
            )
            for option_seed in range(n_perm):
                print(f"[permutation] local_sparse_icl option_seed={option_seed}")
                predict_local_prompt(
                    store, base, tokenizer, "local_sparse_icl", k_values, config["data"]["calibration_seed"],
                    Path(config["paths"]["predictions"]) / f"perm_local_sparse_icl_seed{option_seed}.parquet",
                    choose_device(), max_length=model_config["max_length"], split="test", item_pool="seen",
                    include_history=True, option_seed=option_seed, respondent_sample=perm_sample,
                )

    # --- Aggregate metrics: load_predictions excludes perm_* files by default,
    # so the subsampled permutation replicates never pollute the primary tables ---
    predictions = load_predictions(config["paths"]["predictions"])
    tables = compute_metric_tables(predictions, config["paths"]["metrics"])
    for name, table in tables.items():
        print(f"[metrics] {name}: {len(table)} rows")

    eval_cfg = config["evaluation"]
    comparisons = [
        (f"rap2p_seed{PRIMARY_RAP2P_SEED}", f"context_qlora_seed{PRIMARY_RAP2P_SEED}"),
        (f"rap2p_seed{PRIMARY_RAP2P_SEED}", f"p2p_static_seed{PRIMARY_RAP2P_SEED}"),
        (f"rap2p_seed{PRIMARY_RAP2P_SEED}", "mirt"),
    ]
    id_test = predictions[predictions["split"].eq("test") & predictions["item_pool"].eq("seen")]
    bootstrap = bootstrap_table(
        id_test, comparisons, metric="accuracy", k_values=list(config["data"]["k_values"]),
        replicates=int(eval_cfg["bootstrap_resamples"]), alpha=float(eval_cfg["holm_alpha"]),
    )
    bootstrap.to_csv(Path(config["paths"]["metrics"]) / "bootstrap_accuracy.csv", index=False)
    print(f"[bootstrap] wrote {len(bootstrap)} rows")

    panels = pd.read_parquet(Path(config["paths"]["processed"]) / "panels.parquet")
    responses = pd.read_parquet(Path(config["paths"]["processed"]) / "responses.parquet")
    matched = config["data"]["matched_pair"]
    raw_pairs = find_matched_pairs(
        panels[panels["split"].eq("test")], DEFAULT_MATCH_ATTRIBUTES,
        int(matched["max_attribute_distance"]), seed=int(config["seed"]),
    )
    pairs = filter_pairs_by_answer_divergence(
        raw_pairs, responses[responses["split"].eq("test")],
        selection_items=int(matched["selection_items"]),
        min_divergence=float(matched["min_answer_divergence"]),
        min_shared_items=int(matched["min_shared_history_items"]),
        seed=int(config["seed"]),
    )
    heterogeneity = matched_profile_heterogeneity(id_test, pairs, min_shared_targets=int(matched["min_shared_targets"]))
    heterogeneity.to_csv(Path(config["paths"]["metrics"]) / "heterogeneity.csv", index=False)
    print(f"[heterogeneity] wrote {len(heterogeneity)} rows from {len(pairs)} divergence-filtered pairs (of {len(raw_pairs)} demographic matches)")

    if not args.skip_permutation:
        perm_predictions = load_predictions(
            config["paths"]["predictions"], patterns=("perm_*.parquet",), exclude_prefixes=()
        )
        perm_table = permutation_consistency(perm_predictions)
        perm_table.to_csv(Path(config["paths"]["metrics"]) / "permutation_consistency.csv", index=False)
        print(f"[permutation] wrote {len(perm_table)} rows")


if __name__ == "__main__":
    main()
