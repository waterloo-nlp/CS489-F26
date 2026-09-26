"""Part-2 tokenizer interface (you must implement load_tokenizer yourself).

The TAs score your Part-2 checkpoint roughly as follows:

    import part2_tokenizer
    tokenizer = part2_tokenizer.load_tokenizer("artifacts/part2/tokenizer/")
    ids = tokenizer.encode(test_text)

Contract:
- load_tokenizer(path) reloads everything it needs from the given directory
  (your submitted artifacts/part2/tokenizer/ folder) and returns a tokenizer
  object.
- tokenizer.encode(text) maps a cleaned text string to a list of token ids,
  each in {0, ..., V - 1} for your submitted vocabulary size V.
- Encoding must be lossless because bits-per-character divides by the full
  character count of the text. Decoding the resulting token IDs must
  reproduce the input text exactly, including line breaks, repeated
  whitespace, and unseen characters: decode(encode(text)) == text.
"""

from __future__ import annotations


def load_tokenizer(path: str):
    """Load your Part-2 tokenizer from the given directory.

    Replace this stub with your own implementation; see the module docstring
    for the exact contract.
    """
    raise NotImplementedError(
        "part2_tokenizer.load_tokenizer is a starter stub: implement it so it "
        "reloads your tokenizer from artifacts/part2/tokenizer/."
    )
