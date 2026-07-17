#!/usr/bin/env python3
"""Day-1 gate: confirm enough matched demographic-profile pairs exist per
domain to make the heterogeneity metric (Table 2) statistically meaningful
*before* spending any GPU time. If a domain comes in under the threshold,
relax `data.matched_pair` in the config (fewer attributes, larger
max_attribute_distance) rather than discovering this after training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.eval.heterogeneity import (  # noqa: E402
    DEFAULT_MATCH_ATTRIBUTES,
    filter_pairs_by_answer_divergence,
    find_matched_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--minimum-pairs", type=int, default=200)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    processed = Path(config["paths"]["processed"])
    panels = pd.read_parquet(processed / "panels.parquet")
    responses = pd.read_parquet(processed / "responses.parquet")
    test_panels = panels[panels["split"].eq("test")]

    matched = config["data"]["matched_pair"]
    raw_pairs = find_matched_pairs(
        test_panels,
        attributes=DEFAULT_MATCH_ATTRIBUTES,
        max_attribute_distance=int(matched["max_attribute_distance"]),
        seed=int(config["seed"]),
    )
    # The gate must count pairs surviving the SAME divergence filter the real
    # evaluation applies -- raw demographic matches alone overstate what the
    # heterogeneity metric will actually have to work with.
    pairs = filter_pairs_by_answer_divergence(
        raw_pairs, responses[responses["split"].eq("test")],
        selection_items=int(matched["selection_items"]),
        min_divergence=float(matched["min_answer_divergence"]),
        min_shared_items=int(matched["min_shared_history_items"]),
        seed=int(config["seed"]),
    )
    raw_counts = raw_pairs.groupby("domain").size() if not raw_pairs.empty else pd.Series(dtype=int)
    counts = pairs.groupby("domain").size() if not pairs.empty else pd.Series(dtype=int)
    ok = True
    for domain in config["data"]["domains"]:
        n_raw = int(raw_counts.get(domain, 0))
        n = int(counts.get(domain, 0))
        status = "OK" if n >= args.minimum_pairs else "BELOW THRESHOLD"
        if n < args.minimum_pairs:
            ok = False
        print(f"{domain}: {n} divergence-filtered pairs of {n_raw} demographic matches ({status}, need >= {args.minimum_pairs})")

    if not ok:
        print(
            "\nSome domains are below threshold. Before spending GPU time, relax "
            "data.matched_pair (lower min_answer_divergence, raise max_attribute_distance, "
            "or drop an attribute in eval/heterogeneity.py:DEFAULT_MATCH_ATTRIBUTES), "
            "then re-run this check."
        )
        raise SystemExit(1)
    print("\nOK: all domains have enough matched pairs for the heterogeneity metric.")


if __name__ == "__main__":
    main()
