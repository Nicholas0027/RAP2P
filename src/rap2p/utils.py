from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)
    # Note: setting PYTHONHASHSEED at runtime does NOT affect the already-running
    # interpreter's hash(); it only covers subprocesses. Nothing load-bearing in
    # this codebase uses builtin hash()/set-iteration order (everything uses
    # stable_int's sha256), so this is defense-in-depth for subprocesses only.
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(*parts: Any) -> int:
    """Deterministic hash used for reproducible sub-sampling (Python's hash() is salted per-process)."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def stable_int(*parts: Any, modulo: int | None = None) -> int:
    value = stable_hash(*parts)
    return value if modulo is None else value % modulo


def deterministic_rng(*parts: Any) -> random.Random:
    return random.Random(stable_hash(*parts))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    temporary.replace(path)


class TimeBudget:
    """Wall-clock stop signal so a training job checkpoints before a Colab/cluster time limit."""

    def __init__(self, budget_minutes: float, margin_minutes: float):
        self.start = time.monotonic()
        self.budget_seconds = max(0.0, float(budget_minutes)) * 60.0
        self.margin_seconds = max(0.0, float(margin_minutes)) * 60.0

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.start) / 60.0

    def should_stop(self) -> bool:
        elapsed = time.monotonic() - self.start
        return elapsed >= (self.budget_seconds - self.margin_seconds)


class RunningStats:
    """Tiny streaming mean/variance tracker, used where pulling in numpy is overkill."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5
