#!/usr/bin/env python3
"""Strong-API sparse-ICL baseline on a fixed 500-respondent subsample (see
evaluation.api in the config). Kept separate from evaluate_all.py: needs API
keys, has its own cost/rate-limit budget, and its probabilities are a hard-label
approximation (see baselines/prompting_baseline.py docstring).

Requires `pip install -e .[api]` and ANTHROPIC_API_KEY or OPENAI_API_KEY set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rap2p.baselines.prompting_baseline import (  # noqa: E402
    make_anthropic_call_fn,
    make_openai_call_fn,
    predict_api_prompt,
)
from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.data import PanelStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    store = PanelStore.from_dir(config["paths"]["processed"])
    api_cfg = config["evaluation"]["api"]

    if args.provider == "anthropic":
        call_fn = make_anthropic_call_fn(args.model or "claude-sonnet-5")
        method_name = "api_sparse_icl_anthropic"
    else:
        call_fn = make_openai_call_fn(args.model or "gpt-4o")
        method_name = "api_sparse_icl_openai"

    output_path = Path(config["paths"]["predictions"]) / f"{method_name}__test__seen.parquet"
    predict_api_prompt(
        store, call_fn, method_name, list(api_cfg["k_values"]), config["data"]["calibration_seed"],
        output_path, respondent_sample=int(api_cfg["respondent_sample"]), split="test", item_pool="seen",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
