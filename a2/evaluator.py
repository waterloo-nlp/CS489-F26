"""Framework-neutral, NumPy-only evaluator for the fixed CS 489 A2 byte protocol."""

import math
from numbers import Integral

import numpy as np


BLOCK_LENGTH = 256
BOS_ID = 256
PAD_ID = 257
VOCAB_SIZE = 258


def _batches(documents, batch_size):
    """Yield batches of independent, document-local byte blocks."""
    batch = []
    for document in documents:
        if not isinstance(document, str):
            raise TypeError("Each document must be a complete string")
        encoded = document.encode("utf-8")
        for start in range(0, len(encoded), BLOCK_LENGTH):
            batch.append(encoded[start : start + BLOCK_LENGTH])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _batch_nll(predict_logits, blocks):
    """Construct one batch and sum its real-byte NLLs in float64."""
    shape = (len(blocks), max(map(len, blocks)))
    input_ids = np.full(shape, PAD_ID, dtype=np.int64)
    attention_mask = np.zeros(shape, dtype=np.int64)
    targets = np.zeros(shape, dtype=np.int64)
    for row, block in enumerate(blocks):
        byte_ids = np.frombuffer(block, dtype=np.uint8)
        length = len(block)
        input_ids[row, 0] = BOS_ID
        input_ids[row, 1:length] = byte_ids[:-1]
        attention_mask[row, :length] = 1
        targets[row, :length] = byte_ids

    real_positions = attention_mask.astype(bool)
    target_ids = targets[real_positions]
    logits = predict_logits(input_ids, attention_mask)
    if (
        not isinstance(logits, np.ndarray)
        or np.ma.isMaskedArray(logits)
        or not np.issubdtype(logits.dtype, np.floating)
    ):
        raise TypeError("Expected an unmasked NumPy floating-point array of logits")
    if logits.shape != (*shape, VOCAB_SIZE):
        raise ValueError("Expected logits of shape (batch, length, 258)")

    # Boolean indexing copies real positions, excluding padding before arithmetic.
    scores = logits[real_positions].astype(
        np.result_type(logits.dtype, np.float64), copy=False
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scores -= scores.max(axis=1, keepdims=True)
        log_partition = np.log(np.exp(scores).sum(axis=1))
        # Use the shifted target to avoid cancellation with a large common offset.
        losses = log_partition - scores[np.arange(target_ids.size), target_ids]
        nll = losses.sum(dtype=np.float64)
    return float(nll)


def evaluate(predict_logits, documents, *, batch_size=64) -> dict:
    """Return byte-weighted metrics for an iterable of complete document strings.

    Encode each document as unchanged UTF-8 and split into independent blocks
    of at most 256 bytes, keeping short tails and skipping empty documents.
    ``batch_size`` must be a positive integer; an empty-byte corpus is rejected.

    ``predict_logits(input_ids, attention_mask)`` receives CPU NumPy int64
    arrays of shape (B, T), T <= 256, and returns an unmasked NumPy floating
    array of raw logits with shape (B, T, 258).
    Inputs are [BOS=256] + block[:-1], right-padded with PAD=257; the mask is 1
    for real positions (including BOS), 0 for padding.
    Position t predicts block[t], without another shift or an EOS target.
    All 258 classes are normalized together, and padding positions are ignored.
    Nonfinite logits are allowed if the resulting loss is finite.
    The callback handles device conversion, causal prediction, disabled
    gradients/dropout, and resetting state and positions to zero per block.

    Return a dict with nll (summed NLL in nats), num_bytes (target count),
    loss (nll / num_bytes), and ppl (exp(loss)).
    Softmax uses at least float64 precision; NLL accumulates in float64.
    Nonfinite metrics and perplexity overflow raise ValueError.
    """
    if not callable(predict_logits):
        raise TypeError("predict_logits must be callable")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be a positive integer, not bool")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(documents, (str, bytes, bytearray)):
        raise TypeError("documents must be an iterable of complete strings")

    nll = 0.0
    num_bytes = 0
    for blocks in _batches(documents, batch_size):
        nll += _batch_nll(predict_logits, blocks)
        num_bytes += sum(map(len, blocks))
        if not math.isfinite(nll):
            raise ValueError("Non-finite NLL or loss")
    if num_bytes == 0:
        raise ValueError("Cannot evaluate a corpus with no UTF-8 bytes")
    loss = nll / num_bytes
    try:
        ppl = math.exp(loss)
    except OverflowError as error:
        raise ValueError("Perplexity overflow") from error
    return {"nll": nll, "num_bytes": num_bytes, "loss": loss, "ppl": ppl}
