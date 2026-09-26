"""Fixed GPT-style Transformer language model for CS 489 Assignment 1, Part 2.

DO NOT MODIFY THIS FILE. It is part of the fixed measuring instrument for
Part 2. The TAs restore the pristine starter copies before re-running any
entry point, and local edits to the fixed files may be treated as an
academic-integrity violation.

The TAs rebuild your submitted checkpoint with:

    model = load_checkpoint("artifacts/part2/config.json", "artifacts/part2/model.pt")

so your submitted config must contain the "vocab_size" and "model" sections
described in the starter README (see part2_config.example.json).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIG_KEYS = ("context_length", "d_model", "n_layer", "n_head", "d_ff", "dropout")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """A small GPT-style decoder-only language model with tied embeddings."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_layer: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied input/output embeddings

        self.apply(self._init_weights)
        # GPT-2 style scaled init for the residual projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config: dict) -> "TransformerLM":
        """Build a model from a Part-2 config dict (see part2_config.example.json)."""
        try:
            model_cfg = dict(config["model"])
            vocab_size = int(config["vocab_size"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                'config must contain "vocab_size" (int) and a "model" section'
            ) from e
        unknown = set(model_cfg) - set(MODEL_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(vocab_size=vocab_size, **model_cfg)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.config["context_length"]:
            raise ValueError(
                f"sequence length {T} exceeds context_length {self.config['context_length']}"
            )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def load_config(config_path: str | Path) -> dict:
    """Read a Part-2 config file (JSON, or YAML if pyyaml is installed)."""
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "reading a YAML config requires pyyaml; add it to your environment"
            ) from e
        return yaml.safe_load(text)
    return json.loads(text)


def load_checkpoint(
    config_path: str | Path,
    model_path: str | Path,
    device: torch.device | str | None = None,
) -> TransformerLM:
    """Rebuild a model from a Part-2 config file and load a model.pt checkpoint."""
    model = TransformerLM.from_config(load_config(config_path))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    model.eval()
    return model
