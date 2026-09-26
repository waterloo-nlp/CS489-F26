"""Fixed bits-per-character evaluator for CS 489 Assignment 1, Part 2.

DO NOT MODIFY THIS FILE. Every bpc number you report (baseline, sweep, and
final) must come from this evaluator so that your dev numbers are computed
exactly the way the TAs compute test numbers.

Usage:
    uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
        --model artifacts/part2/model.pt --ids DEV_BIN --text DEV_TXT

Definitions (identical for dev and test):
- ids: uint32 binary token-id file produced by your tokenizer for the text.
- N_chars: len(text) in Unicode code points (see data.count_chars).
- Scoring: a window of size context_length + 1 slides over the id stream
  with stride context_length; every token after the first is scored, including
  the trailing partial window. Every scored token is predicted from its true
  preceding context. The first token has no preceding context and is not scored.
- bpc = (sum of token-level NLL in nats) / (N_chars * ln 2).
"""

from __future__ import annotations

import argparse
import json
import math

from data import count_chars, read_ids
from part2_train import _Amp, evaluate_token_nll, pick_device
from part2_transformer_lm import load_checkpoint


def compute_bpc(
    config_path: str,
    model_path: str,
    ids_path: str,
    text_path: str,
    batch_size: int = 64,
) -> dict:
    device = pick_device()
    model = load_checkpoint(config_path, model_path, device=device)
    ids = read_ids(ids_path)
    context_length = model.config["context_length"]
    nll, n_tokens = evaluate_token_nll(
        model, ids, context_length, batch_size, device, amp=_Amp(device)
    )
    n_chars = count_chars(text_path)
    bpc = nll / (n_chars * math.log(2))
    return {
        "tokens_scored": n_tokens,
        "chars": n_chars,
        "nll_per_token": nll / n_tokens,
        "token_ppl": math.exp(nll / n_tokens),
        "bpc": bpc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed bits-per-character evaluator (CS 489 A1 Part 2).")
    parser.add_argument("--config", required=True, help="Part-2 config file (JSON or YAML)")
    parser.add_argument("--model", required=True, help="model.pt checkpoint")
    parser.add_argument("--ids", required=True, help="uint32 binary token-id file for the text")
    parser.add_argument("--text", required=True, help="the cleaned text the ids encode")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = compute_bpc(args.config, args.model, args.ids, args.text, batch_size=args.batch_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
