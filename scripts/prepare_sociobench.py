#!/usr/bin/env python3
"""Parse SocioBench, curate items/countries, assign the three splits (ID
respondent / OOD intersection / OOD item), and write everything to
`paths.processed`. This single script performs the whole data-prep pipeline
described in data.py -- there is no separate "build splits" step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rap2p.config import load_and_prepare  # noqa: E402
from rap2p.data import prepare_sociobench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    audit = prepare_sociobench(config)
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
