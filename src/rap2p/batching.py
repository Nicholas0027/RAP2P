from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .data import PanelStore
from .embeddings import EmbeddingStore
from .item_graph import ItemGraph
from .prompting import build_prompt, deterministic_permutation, option_token_ids
from .utils import stable_int

# Methods whose text prompt never includes history (see prompting.py docstring).
TEXT_METHODS_WITH_HISTORY_IN_PROMPT = {"context_qlora"}
EMBEDDING_METHODS = {"p2p_static", "rap2p"}


@dataclass
class BatchSpec:
    split: str
    k_values: list[int]
    calibration_seed: int
    option_seed: int
    panels_per_batch: int = 4
    targets_per_panel: int = 4
    random_option_permutation: bool = True
    domains: list[str] | None = None
    item_pool: str = "seen"
    shuffle: bool = True
    fixed_k: int | None = None  # override for evaluation passes that fix a single K


@dataclass
class ModalityDropout:
    demographics: float = 0.0
    history: float = 0.0
    correlation: float = 0.0

    @classmethod
    def none(cls) -> "ModalityDropout":
        return cls(0.0, 0.0, 0.0)


class PanelBatchIterator:
    """Yield grouped item rows so every microbatch contains within-panel target pairs."""

    def __init__(self, store: PanelStore, spec: BatchSpec, seed: int):
        self.store = store
        self.spec = spec
        frame = store.responses[store.responses["split"].eq(spec.split)].copy()
        if spec.domains:
            frame = frame[frame["domain"].isin(spec.domains)]
        if spec.item_pool == "seen":
            frame = frame[~frame["is_unseen_item"]]
        elif spec.item_pool == "unseen":
            frame = frame[frame["is_unseen_item"]]
        elif spec.item_pool != "all":
            raise ValueError(f"Unknown item_pool: {spec.item_pool}")
        self.frame = frame
        self.panel_ids = frame["panel_id"].drop_duplicates().tolist()
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[list[dict[str, Any]], int, int]]:
        epoch_key = self.epoch if self.spec.shuffle else 0
        rng = np.random.default_rng(stable_int(self.seed, epoch_key))
        panel_ids = np.asarray(self.panel_ids, dtype=object)
        if self.spec.shuffle:
            rng.shuffle(panel_ids)
            self.epoch += 1
        for start in range(0, len(panel_ids), self.spec.panels_per_batch):
            selected_panels = panel_ids[start : start + self.spec.panels_per_batch]
            records: list[dict[str, Any]] = []
            k = int(self.spec.fixed_k) if self.spec.fixed_k is not None else int(rng.choice(self.spec.k_values))
            batch_option_seed = int(rng.integers(0, 2**31 - 1)) if self.spec.random_option_permutation else self.spec.option_seed
            for panel_id in selected_panels:
                panel = self.frame[self.frame["panel_id"].eq(panel_id)]
                if panel.empty:
                    continue
                indices = rng.choice(panel.index, size=min(self.spec.targets_per_panel, len(panel)), replace=False)
                records.extend(panel.loc[indices].to_dict("records"))
            if records:
                yield records, k, batch_option_seed


class SurveyCollator:
    """Builds model-ready batches. `kind` selects the text-prompt policy and
    whether embedding tensors (demographic/item/history/correlation) are attached.
    """

    def __init__(
        self,
        tokenizer,
        store: PanelStore,
        embeddings: EmbeddingStore | None,
        item_graph: ItemGraph | None,
        max_length: int,
        max_options: int,
        kind: str,
    ):
        self.tokenizer = tokenizer
        self.store = store
        self.embeddings = embeddings
        self.item_graph = item_graph
        self.max_length = int(max_length)
        self.kind = kind
        self.label_token_ids = option_token_ids(tokenizer, max_options)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(
        self,
        records: list[Mapping[str, Any]],
        k: int,
        calibration_seed: int,
        option_seed: int,
        random_permutation: bool,
        modality_dropout: ModalityDropout | None = None,
        dropout_seed: int = 0,
    ) -> dict[str, Any]:
        import torch

        include_history_in_prompt = self.kind in TEXT_METHODS_WITH_HISTORY_IN_PROMPT
        prompts: list[str] = []
        targets: list[int] = []
        permutations: list[list[int]] = []
        histories: list[pd.DataFrame] = []
        for record in records:
            history = self.store.history_rows(record["panel_id"], record["question_id"], k, calibration_seed)
            histories.append(history)
            permutation = (
                deterministic_permutation(int(record["n_options"]), option_seed, record["row_id"])
                if random_permutation
                else np.arange(int(record["n_options"]))
            )
            prompt, target, semantic_order = build_prompt(
                record, history.to_dict("records"), permutation, include_history_in_prompt
            )
            prompts.append(prompt)
            targets.append(target)
            permutations.append(semantic_order)

        tokens = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=True
        )
        n_options = torch.tensor([int(record["n_options"]) for record in records], dtype=torch.long)
        option_mask = torch.arange(len(self.label_token_ids)).unsqueeze(0) < n_options.unsqueeze(1)
        batch: dict[str, Any] = {
            **tokens,
            "targets": torch.tensor(targets, dtype=torch.long),
            "n_options": n_options,
            "option_mask": option_mask,
            "label_token_ids": torch.tensor(self.label_token_ids, dtype=torch.long),
            "permutations": permutations,
            "records": [dict(record) for record in records],
            "k": int(k),
            "true_normalized": torch.tensor([float(record["normalized_answer"]) for record in records]),
            "question_mean": torch.tensor([float(record["question_mean"]) for record in records]),
            "question_std": torch.tensor([float(record["question_std"]) for record in records]),
        }

        if self.kind in EMBEDDING_METHODS:
            if self.embeddings is None:
                raise ValueError(f"kind={self.kind!r} requires an EmbeddingStore")
            dimension = self.embeddings.dimension
            max_k = max((len(history) for history in histories), default=0)
            max_k = max(1, max_k)
            history_item_embeddings = np.zeros((len(records), max_k, dimension), dtype=np.float32)
            history_answer_index = np.zeros((len(records), max_k), dtype=np.int64)
            history_n_options = np.ones((len(records), max_k), dtype=np.int64)
            history_mask = np.zeros((len(records), max_k), dtype=bool)
            correlation_bias = np.zeros((len(records), max_k), dtype=np.float32)
            for index, (record, history) in enumerate(zip(records, histories)):
                keys = history["question_key"].astype(str).tolist()
                if keys:
                    history_item_embeddings[index, : len(keys)] = self.embeddings.items.batch(keys)
                    history_answer_index[index, : len(keys)] = history["answer_index"].to_numpy()
                    history_n_options[index, : len(keys)] = history["n_options"].to_numpy()
                    history_mask[index, : len(keys)] = True
                    if self.item_graph is not None:
                        correlation_bias[index, : len(keys)] = self.item_graph.batch(
                            record["domain"], record["question_key"], keys
                        )
            batch.update(
                demographic_embeddings=torch.from_numpy(
                    self.embeddings.demographics.batch(record["panel_id"] for record in records)
                ),
                item_embeddings=torch.from_numpy(
                    self.embeddings.items.batch(record["question_key"] for record in records)
                ),
                history_item_embeddings=torch.from_numpy(history_item_embeddings),
                history_answer_index=torch.from_numpy(history_answer_index),
                history_n_options=torch.from_numpy(history_n_options),
                history_mask=torch.from_numpy(history_mask),
                correlation_bias=torch.from_numpy(correlation_bias),
            )
            n = len(records)
            if modality_dropout is None or self.kind != "rap2p":
                batch["demographics_keep"] = torch.ones(n, dtype=torch.bool)
                batch["history_keep"] = torch.ones(n, dtype=torch.bool)
                batch["correlation_keep"] = torch.ones(n, dtype=torch.bool)
            else:
                rng = np.random.default_rng(stable_int(dropout_seed, "modality_dropout"))
                batch["demographics_keep"] = torch.from_numpy(rng.random(n) >= modality_dropout.demographics)
                batch["history_keep"] = torch.from_numpy(rng.random(n) >= modality_dropout.history)
                batch["correlation_keep"] = torch.from_numpy(rng.random(n) >= modality_dropout.correlation)
        return batch


def move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def iter_prediction_records(frame: pd.DataFrame, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    records = frame.to_dict("records")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
