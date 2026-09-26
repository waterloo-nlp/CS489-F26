"""Data loading helpers for CS 489 Assignment 1.

DO NOT MODIFY THIS FILE. The TAs restore the pristine starter copies before
re-running any entry point.

Token-id streams are stored as uint32 binary files, which is why vocabulary
sizes must satisfy V <= 2**32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def iter_split_files(input_dir: str | Path, split: str = "train") -> list[Path]:
    """Return every {split}.txt file under input_dir, sorted by relative path.

    This is the canonical concatenation order used throughout the assignment:
    sorted order of relative paths, for example ar/train.txt, de/train.txt,
    fi/train.txt, hu/train.txt, ru/train.txt, vi/train.txt, zh/train.txt.
    """
    root = Path(input_dir)
    files = [p for p in root.rglob(f"{split}.txt") if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def load_split(input_dir: str | Path, split: str = "train") -> str:
    """Concatenate every {split}.txt file under input_dir into one string.

    Files are read in the canonical sorted order (see iter_split_files) and
    concatenated verbatim. Every released data file ends with a newline, so
    lines never merge across file boundaries.
    """
    return "".join(
        path.read_text(encoding="utf-8") for path in iter_split_files(input_dir, split)
    )


def write_ids(path: str | Path, ids) -> None:
    """Write a token-id sequence to a uint32 binary file."""
    arr = np.asarray(ids)
    if arr.size == 0:
        raise ValueError("refusing to write an empty token-id sequence")
    if arr.min() < 0 or arr.max() > 2**32 - 1:
        raise ValueError("token ids must lie in [0, 2**32 - 1] for uint32 storage")
    arr.astype(np.uint32).tofile(path)


def read_ids(path: str | Path) -> np.ndarray:
    """Read a uint32 binary token-id file written by write_ids."""
    return np.fromfile(path, dtype=np.uint32)


def count_chars(path: str | Path) -> int:
    """Number of characters (Unicode code points) in a UTF-8 text file.

    This is the N_chars denominator used by bits-per-character everywhere in
    this assignment.
    """
    return len(Path(path).read_text(encoding="utf-8"))
