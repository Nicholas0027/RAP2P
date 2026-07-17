#!/usr/bin/env python3
"""Train one run/seed. See configs/mvp.yaml `runs.*.seeds` for which seeds are
expected for each run in the main compute plan (README "Compute budget" table).

Examples:
    python scripts/train.py --config configs/mvp.yaml --run global_qlora   --seed 1701
    python scripts/train.py --config configs/mvp.yaml --run rap2p         --seed 42
    RAP2P_SMOKE=1 python scripts/train.py --config configs/mvp.yaml --run rap2p --seed 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.workflows import RUN_TO_COLLATOR_KIND, run_training_job  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, choices=sorted(RUN_TO_COLLATOR_KIND) + ["rap2p_no_graph", "rap2p_no_history_retrained"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Zero-GPU plumbing check; never a reported result.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    args = parser.parse_args()

    smoke = args.smoke or bool(int(os.environ.get("RAP2P_SMOKE", "0")))
    config = load_and_prepare(args.config)
    history = run_training_job(
        config, args.run, args.seed, smoke=smoke,
        max_optimizer_steps=args.max_steps, time_budget_minutes=args.time_budget_minutes,
    )
    print(json.dumps(history[-1] if history else {}, indent=2, default=str))
    print(f"Wrote checkpoints to {Path(config['paths']['checkpoints']) / f'{args.run}_seed{args.seed}'}")


if __name__ == "__main__":
    main()
