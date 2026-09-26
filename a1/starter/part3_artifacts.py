"""Artifact save/load helpers for CS 489 Assignment 1.

DO NOT MODIFY THIS FILE. It defines the submission format of the Part-3
n-gram model artifact (model.json.gz). The TAs load your submitted model
with load_ngram_model, so the format below is the single source of truth.

Part-3 n-gram model format (a JSON object, gzip-compressed):

    {
      "format": "cs489-a1-ngram-v1",
      "n": 4,                     # model order (int, n >= 1)
      "alpha": 0.01,              # add-alpha smoothing parameter (float >= 0)
      "vocab_size": 50001,        # emittable vocabulary size (int); see below
      "bos": 50001,               # optional: BOS token id prepended to every line
      "counts": {                 # count tables for every order 1..n
        "1": {"123": 4567, ...},
        "2": {"12 123": 890, ...},
        ...
      }
    }

- counts[str(k)][key] is the integer number of times the k-gram appears in
  the training data, for every k in 1..n.
- A k-gram key is the token ids joined by single spaces, in order; build it
  with ngram_key([t1, ..., tk]) and parse it with parse_ngram_key(key).
- Line-based convention: text is processed line by line. Each line is
  tokenized independently with your Part-1 BPE tokenizer, n-grams and
  contexts never cross line boundaries, and the end-of-sequence token EOS
  (id 50,000, one past the submitted BPE vocabulary) is appended to every
  line and scored like any other token. If the model records "bos", that
  token (id 50,001) is prepended to every line as context but is never
  scored itself. vocab_size must count every token the model can emit:
  50,000 BPE tokens plus EOS (50,001 in total), plus BOS if used (50,002).
- Scoring semantics (what the TA's scorer implements): for a token w with
  preceding context ctx of length m within the line (m = 0 at the very
  start of a line when no BOS is used, capped at n - 1), the add-alpha
  probability is

      P(w | ctx) = (C(ctx + [w]) + alpha) / (C(ctx) + alpha * V)

  where C(ctx + [w]) = counts[str(m + 1)].get(key, 0), and the denominator
  count C(ctx) is counts[str(m)].get(key, 0) for m >= 1; for the empty
  context (m = 0), C(()) is the total number of training tokens, i.e. the
  sum of all unigram counts in counts["1"].
- The probability of a line is the product over its scored positions t of
  P(w_t | the up-to-(n-1) preceding tokens within the line), with no
  backoff. Corpus perplexity aggregates the token-level NLL over all lines
  (EOS tokens included) in the usual way.

Note that full high-order count tables can grow large (a dense n = 5 model
at V = 50,000 can exceed 100 MB gzipped). Keep the submission size in mind
when you select your best configuration in Part 3.3.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

NGRAM_FORMAT = "cs489-a1-ngram-v1"


def save_json_gz(obj, path: str | Path) -> None:
    """Serialize obj as JSON, gzip-compressed, to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json_gz(path: str | Path):
    """Load an object written by save_json_gz."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def ngram_key(token_ids) -> str:
    """Encode a token-id sequence as a count-table key (space-joined ids)."""
    return " ".join(str(int(t)) for t in token_ids)


def parse_ngram_key(key: str) -> tuple[int, ...]:
    """Inverse of ngram_key; the empty string maps to the empty tuple."""
    if not key:
        return ()
    return tuple(int(t) for t in key.split())


def _validate_ngram_model(model: dict) -> None:
    if not isinstance(model, dict):
        raise ValueError("n-gram model must be a dict")
    n = model.get("n")
    if not isinstance(n, int) or n < 1:
        raise ValueError('model["n"] must be an int >= 1')
    alpha = model.get("alpha")
    if not isinstance(alpha, (int, float)) or alpha < 0:
        raise ValueError('model["alpha"] must be a number >= 0')
    vocab_size = model.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError('model["vocab_size"] must be an int >= 1')
    counts = model.get("counts")
    if not isinstance(counts, dict):
        raise ValueError('model["counts"] must be a dict')
    for k in range(1, n + 1):
        table = counts.get(str(k))
        if not isinstance(table, dict):
            raise ValueError(f'model["counts"] must contain a table for order {k}')
    bos = model.get("bos")
    if bos is not None and (not isinstance(bos, int) or bos < 0):
        raise ValueError('model["bos"], if present, must be an int >= 0')


def save_ngram_model(model: dict, path: str | Path) -> None:
    """Validate and save a Part-3 n-gram model to model.json.gz.

    model must contain the keys "n", "alpha", "vocab_size", and "counts"
    (with one count table per order 1..n), as specified in the module
    docstring; "bos" is optional. The "format" tag is added automatically.
    """
    _validate_ngram_model(model)
    payload = {
        "format": NGRAM_FORMAT,
        "n": model["n"],
        "alpha": float(model["alpha"]),
        "vocab_size": int(model["vocab_size"]),
        "counts": model["counts"],
    }
    if model.get("bos") is not None:
        payload["bos"] = int(model["bos"])
    save_json_gz(payload, path)


def load_ngram_model(path: str | Path) -> dict:
    """Load and validate a model.json.gz written by save_ngram_model."""
    payload = load_json_gz(path)
    if not isinstance(payload, dict) or payload.get("format") != NGRAM_FORMAT:
        raise ValueError(
            f"{path} is not a {NGRAM_FORMAT} model; use save_ngram_model to write it"
        )
    _validate_ngram_model(payload)
    return payload
