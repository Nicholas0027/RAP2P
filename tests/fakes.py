"""Shared tiny stand-in backbone for model tests: exposes `.model.layers` and
`self_attn.q_proj/v_proj` (what models/common.py:decoder_layers and
patch_rank_block_lora expect) plus a `.forward` returning `.logits`, matching
models/common.py:last_token_logits — no HF download needed."""

from __future__ import annotations

import torch
import torch.nn as nn

VOCAB, HIDDEN, N_LAYERS, EMBED_DIM, N_OPTIONS = 20, 8, 3, 16, 5


class FakeAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.v_proj(x)


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = FakeAttention(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(x)


class FakeInnerModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FakeOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB, hidden: int = HIDDEN, n_layers: int = N_LAYERS):
        super().__init__()
        self.model = FakeInnerModel(hidden, n_layers)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False, logits_to_keep: int | None = None):
        del attention_mask, use_cache, logits_to_keep
        return FakeOutput(self.lm_head(self.model(self.embed(input_ids))))
