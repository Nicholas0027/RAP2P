#!/usr/bin/env python3
"""Day-1 gate: re-verify the leakage guards on disk (not just at prepare-time).
Exits non-zero if anything fails, so it is safe to wire into a pre-training check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.data import validate_item_holdout_leakage, validate_no_leakage  # noqa: E402
import pandas as pd  # noqa: E402


def check_item_graph_artifact(config) -> dict:
    """If the precomputed item graph exists, assert no unseen item ever got a
    C_jk row/column -- the on-disk counterpart of item_graph.assert_train_only."""
    graph_path = Path(config["paths"]["item_graph"]) / "item_graph.json"
    if not graph_path.exists():
        return {"item_graph_checked": False, "reason": "not yet precomputed"}
    items = pd.read_parquet(Path(config["paths"]["processed"]) / "items.parquet")
    unseen_keys = set(items.loc[items["is_unseen_item"], "question_key"])
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    offenders = []
    for domain, data in payload.items():
        offenders.extend(key for key in data["index"] if key in unseen_keys)
        offenders.extend(key for key in data["columns"] if key in unseen_keys)
    report = {"item_graph_checked": True, "unseen_items_in_graph": len(offenders)}
    if offenders:
        raise AssertionError(f"Item graph contains unseen items (leakage): {sorted(set(offenders))[:5]}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    processed = Path(config["paths"]["processed"])
    responses = pd.read_parquet(processed / "responses.parquet")
    orders = pd.read_parquet(processed / "calibration_orders.parquet")

    split_report = validate_no_leakage(responses)
    item_report = validate_item_holdout_leakage(responses, orders)
    graph_report = check_item_graph_artifact(config)
    print(json.dumps({"split_leakage": split_report, "item_holdout_leakage": item_report, "item_graph": graph_report}, indent=2, default=str))
    print("OK: no leakage detected.")


if __name__ == "__main__":
    main()
