"""Frozen sentence-embedding cache for demographics, items, and answer-pairs.

Embeddings are precomputed once with a frozen encoder (`model.embedding_model`)
and memory-mapped at train/eval time so no embedding model needs to be loaded
during GPU training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import stable_int


class EmbeddingTable:
    def __init__(self, array_path: str | Path, index_path: str | Path):
        self.array = np.load(array_path, mmap_mode="r")
        index = pd.read_parquet(index_path)
        self.lookup = dict(zip(index["key"].astype(str), index["position"].astype(int)))
        self.dimension = int(self.array.shape[1])

    def __getitem__(self, key: str) -> np.ndarray:
        return np.asarray(self.array[self.lookup[str(key)]], dtype=np.float32)

    def batch(self, keys: Iterable[str]) -> np.ndarray:
        positions = [self.lookup[str(key)] for key in keys]
        return np.asarray(self.array[positions], dtype=np.float32)


class EmbeddingStore:
    """Only two tables are needed: demographics (per panel) and items (per
    question_key). A historical answer's representation is built inside the
    model from `items[question_key]` plus a small *trainable* answer-position
    embedding (see models/response_anchoring.py) rather than from a separately
    cached "question+answer" text embedding -- this keeps the attention
    query/key space (target item vs. history item) consistent and lets the
    model learn how much an answer's position should shift the representation,
    instead of baking that into a frozen text encoder.
    """

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.demographics = EmbeddingTable(directory / "demographics.npy", directory / "demographics_index.parquet")
        self.items = EmbeddingTable(directory / "items.npy", directory / "items_index.parquet")
        dimensions = {self.demographics.dimension, self.items.dimension}
        if len(dimensions) != 1:
            raise ValueError(f"Embedding dimensions differ across tables: {dimensions}")
        self.dimension = dimensions.pop()


class DeterministicSyntheticTable:
    """Duck-types EmbeddingTable's `.batch()` for the zero-GPU smoke path (see
    workflows.load_experiment_data): every key gets a fixed, key-seeded random
    vector, so the same key always returns the same "embedding" without ever
    touching a real encoder. Never used to produce a reported result.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def batch(self, keys: Iterable[str]) -> np.ndarray:
        return np.stack(
            [np.random.default_rng(stable_int("smoke_embedding", str(key))).standard_normal(self.dimension).astype(np.float32) for key in keys]
        )


class SyntheticEmbeddingStore:
    def __init__(self, dimension: int = 32):
        self.dimension = dimension
        self.demographics = DeterministicSyntheticTable(dimension)
        self.items = DeterministicSyntheticTable(dimension)


def _encode_table(model, keys: list[str], texts: list[str], output: Path, name: str, batch_size: int) -> None:
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    np.save(output / f"{name}.npy", embeddings)
    pd.DataFrame({"key": keys, "position": np.arange(len(keys), dtype=np.int64)}).to_parquet(
        output / f"{name}_index.parquet", index=False
    )


def cache_embeddings(
    responses: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
    batch_size: int = 128,
    device: str | None = None,
) -> dict[str, int]:
    from sentence_transformers import SentenceTransformer

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    demographics = responses[["panel_id", "demographic_text"]].drop_duplicates("panel_id").sort_values("panel_id")
    items = responses[["question_key", "question"]].drop_duplicates("question_key").sort_values("question_key")

    _encode_table(model, demographics["panel_id"].tolist(), demographics["demographic_text"].tolist(), output, "demographics", batch_size)
    _encode_table(model, items["question_key"].tolist(), items["question"].tolist(), output, "items", batch_size)
    return {"demographics": len(demographics), "items": len(items)}
