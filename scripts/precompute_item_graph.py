#!/usr/bin/env python3
"""Compute and cache the leakage-safe, shrinkage-corrected item-item
correlation graph C_jk from *train-split-only* responses (see item_graph.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.item_graph import compute_item_graph, save_item_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    responses = pd.read_parquet(Path(config["paths"]["processed"]) / "responses.parquet")
    # Train respondents only AND seen items only -- unseen items must not get
    # C_jk entries (leakage guard, enforced again inside compute_item_graph).
    train_only = responses[responses["split"].eq("train") & ~responses["is_unseen_item"]]

    graphs = compute_item_graph(
        train_only,
        shrinkage_lambda=float(config["data"]["correlation_shrinkage_lambda"]),
        min_n_jk=int(config["data"]["correlation_min_n_jk"]),
    )
    save_item_graph(graphs, config["paths"]["item_graph"])
    for domain, matrix in graphs.items():
        print(f"{domain}: {matrix.shape[0]} items, mean |C_jk| off-diagonal = {matrix.to_numpy().mean():.4f}")


if __name__ == "__main__":
    main()
