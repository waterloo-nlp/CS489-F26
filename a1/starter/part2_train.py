"""Fixed training script for the CS 489 Assignment 1 Part-2 Transformer.

DO NOT MODIFY THIS FILE. It is part of the fixed measuring instrument for
Part 2. The TAs restore the pristine starter copies before re-running any
entry point, and local edits to the fixed files may be treated as an
academic-integrity violation.

The trainer consumes token-id streams (uint32 binary files produced by your
own tokenizer, see data.py) instead of raw text, so that it works unchanged
with any tokenization algorithm. Every hyperparameter below is fixed; only
the vocabulary size varies across Part-2 runs.

Usage:
    uv run python part2_train.py --train-ids TRAIN_BIN --dev-ids DEV_BIN \
        --vocab-size V --output OUT_DIR [--dev-text DEV_TXT]

Outputs under OUT_DIR:
    model.pt         final model state_dict (loads via part2_transformer_lm.load_checkpoint)
    run_config.json  the full run configuration; extend it with your tokenizer
                     section and submit it as artifacts/part2/config.json

When --dev-text (the cleaned dev text) is given, the final report includes
dev bits-per-character. You can also import and call train() from your own
part2.py; see the starter README for the intended workflow.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import count_chars, read_ids
from part2_transformer_lm import TransformerLM

# ---------------------------------------------------------------------------
# Fixed configuration. Only --vocab-size varies across Part-2 runs.
# ---------------------------------------------------------------------------
FIXED_MODEL_CONFIG = {
    "context_length": 256,
    "d_model": 384,
    "n_layer": 6,
    "n_head": 6,
    "d_ff": 1536,
    "dropout": 0.0,
}
FIXED_TRAINING_CONFIG = {
    "batch_size": 64,
    "max_steps": 2000,
    "warmup_steps": 100,
    "learning_rate": 6e-4,
    "min_learning_rate": 6e-5,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_interval": 200,
    "eval_max_tokens": 262_144,
    "seed": 489,
}
_EVAL_BATCH_SIZE = 8


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _native_bf16_supported() -> bool:
    """True only for GPUs with native bf16 support (Ampere+), not emulated bf16."""
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older PyTorch without the including_emulation argument
        return torch.cuda.is_bf16_supported()


class _Amp:
    """Autocast/GradScaler setup: bf16 where natively supported (Ampere+), fp16
    with a GradScaler on older GPUs (e.g. the Colab T4), fp32 on CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"
        self.dtype = None
        self.scaler = None
        if self.enabled:
            self.dtype = torch.bfloat16 if _native_bf16_supported() else torch.float16
            if self.dtype == torch.float16:
                try:
                    self.scaler = torch.amp.GradScaler("cuda")
                except AttributeError:  # older PyTorch
                    self.scaler = torch.cuda.amp.GradScaler()

    def autocast(self):
        if self.enabled:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()


def get_lr(step: int) -> float:
    """Linear warmup followed by cosine decay to min_learning_rate."""
    cfg = FIXED_TRAINING_CONFIG
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg["min_learning_rate"] + (cfg["learning_rate"] - cfg["min_learning_rate"]) * cosine


@torch.inference_mode()
def evaluate_token_nll(
    model: TransformerLM,
    ids: np.ndarray,
    context_length: int,
    batch_size: int,
    device: torch.device,
    max_tokens: int | None = None,
    amp: _Amp | None = None,
) -> tuple[float, int]:
    """Total token-level NLL (in nats) over a token-id stream.

    Slides a window of size context_length + 1 with stride context_length and
    scores every token after the first, including the trailing partial window.
    Every scored token is predicted from its true preceding context (up to
    context_length tokens back). The first token has no preceding context
    and is not scored.

    Returns (total_nll_nats, n_scored_tokens).
    """
    if len(ids) < 2:
        raise ValueError("token-id streams must contain at least two tokens to score")
    was_training = model.training
    model.eval()
    amp = amp or _Amp(device)
    full_starts = np.arange(0, len(ids) - context_length, context_length)
    starts = full_starts
    if max_tokens is not None:
        starts = starts[: max(1, max_tokens // context_length)]
    total_nll = 0.0
    n_scored = 0
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        windows = np.stack([ids[s : s + context_length + 1] for s in batch_starts])
        windows = windows.astype(np.int64)
        x = torch.from_numpy(windows[:, :-1]).to(device)
        y = torch.from_numpy(windows[:, 1:]).to(device)
        with amp.autocast():
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
        total_nll += loss.item()
        n_scored += y.numel()
    if len(starts) == len(full_starts) and (max_tokens is None or n_scored < max_tokens):
        tail_start = len(full_starts) * context_length
        if tail_start + 1 < len(ids):
            window = ids[tail_start:]
            x = torch.from_numpy(window[:-1].astype(np.int64)[None, :]).to(device)
            y = torch.from_numpy(window[1:].astype(np.int64)[None, :]).to(device)
            with amp.autocast():
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="sum",
                )
            total_nll += loss.item()
            n_scored += y.numel()
    if was_training:
        model.train()
    return total_nll, n_scored


def train(
    train_ids_path: str | Path,
    dev_ids_path: str | Path,
    vocab_size: int,
    output_dir: str | Path,
    dev_text_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train the fixed Transformer on a token-id stream and write the Part-2 artifacts."""
    device = device or pick_device()
    cfg_m = FIXED_MODEL_CONFIG
    cfg_t = FIXED_TRAINING_CONFIG
    torch.manual_seed(cfg_t["seed"])
    rng = np.random.default_rng(cfg_t["seed"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_ids = read_ids(train_ids_path)
    dev_ids = read_ids(dev_ids_path)
    ctx = cfg_m["context_length"]
    if len(train_ids) < ctx + 1 or len(dev_ids) < ctx + 1:
        raise ValueError(f"token-id streams must contain at least {ctx + 1} tokens")

    model = TransformerLM(vocab_size=vocab_size, **cfg_m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params / 1e6:.1f}M (vocab_size={vocab_size})")

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg_t["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg_t["learning_rate"],
        betas=(0.9, 0.95),
    )
    amp = _Amp(device)

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = rng.integers(0, len(train_ids) - ctx - 1, size=cfg_t["batch_size"])
        windows = np.stack([train_ids[s : s + ctx + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(windows[:, :-1]).to(device)
        y = torch.from_numpy(windows[:, 1:]).to(device)
        return x, y

    print(f"training on {device}" + (f" in {amp.dtype}" if amp.enabled else ""))
    model.train()
    t0 = time.time()
    tokens_seen = 0
    for step in range(cfg_t["max_steps"]):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sample_batch()
        with amp.autocast():
            logits, loss = model(x, y)
        del logits
        if amp.scaler is not None:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
            amp.scaler.step(optimizer)
            amp.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += x.numel()
        if step == 0 or (step + 1) % 100 == 0:
            rate = tokens_seen / (time.time() - t0)
            print(
                f"step {step + 1}/{cfg_t['max_steps']}  "
                f"train loss {loss.item():.4f}  lr {lr:.2e}  {rate:.0f} tok/s"
            )
        if (step + 1) % cfg_t["eval_interval"] == 0 or (step + 1) == cfg_t["max_steps"]:
            nll, n_tok = evaluate_token_nll(
                model, dev_ids, ctx, _EVAL_BATCH_SIZE, device,
                max_tokens=cfg_t["eval_max_tokens"], amp=amp,
            )
            mean = nll / n_tok
            print(f"  dev prefix ({n_tok} tokens): nll/token {mean:.4f}  ppl {math.exp(mean):.2f}")

    nll, n_tok = evaluate_token_nll(model, dev_ids, ctx, _EVAL_BATCH_SIZE, device, amp=amp)
    mean_nll = nll / n_tok
    metrics = {
        "dev_tokens_scored": n_tok,
        "dev_nll_per_token": mean_nll,
        "dev_token_ppl": math.exp(mean_nll),
    }
    print(f"final dev: nll/token {mean_nll:.4f}  ppl {math.exp(mean_nll):.2f}  ({n_tok} tokens scored)")

    if dev_text_path is not None:
        n_chars = count_chars(dev_text_path)
        bpc = nll / (n_chars * math.log(2))
        metrics["dev_chars"] = n_chars
        metrics["dev_bpc"] = bpc
        print(f"final dev bpc: {bpc:.4f}  ({n_chars} chars)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    run_config = {
        "vocab_size": vocab_size,
        "model": dict(cfg_m),
        "training": dict(cfg_t),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    if device.type == "cuda":
        print(
            f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved"
        )
    print(f"wrote {out / 'model.pt'} and {out / 'run_config.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed trainer for the CS 489 A1 Part-2 Transformer.")
    parser.add_argument("--train-ids", required=True, help="uint32 binary token-id file for training")
    parser.add_argument("--dev-ids", required=True, help="uint32 binary token-id file for evaluation")
    parser.add_argument("--vocab-size", type=int, required=True, help="tokenizer vocabulary size V")
    parser.add_argument("--output", required=True, help="output directory for model.pt and run_config.json")
    parser.add_argument(
        "--dev-text",
        default=None,
        help="cleaned dev text file; enables bits-per-character reporting",
    )
    args = parser.parse_args()
    train(
        args.train_ids,
        args.dev_ids,
        args.vocab_size,
        args.output,
        dev_text_path=args.dev_text,
    )


if __name__ == "__main__":
    main()
