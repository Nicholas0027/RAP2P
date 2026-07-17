#!/usr/bin/env python3
"""Cache frozen demographic/item sentence embeddings so no embedding model
needs to be loaded during GPU training (see embeddings.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.embeddings import cache_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    responses = pd.read_parquet(Path(config["paths"]["processed"]) / "responses.parquet")
    counts = cache_embeddings(responses, config["paths"]["embeddings"], config["model"]["embedding_model"], device=args.device)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
