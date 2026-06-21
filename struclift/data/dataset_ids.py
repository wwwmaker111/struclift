"""
Deterministic ids for dataset JSONL so values fit nn.Embedding vocabularies.

Replaces process-random Python hash() with zlib.crc32 (stable across runs).
"""

from __future__ import annotations

import zlib


def stable_embedding_id(label: str, vocab_size: int, *, pad_id: int = 0) -> int:
    """
    Map a string label to integer in ``[1, vocab_size - 1]``.

    Index ``pad_id`` (default 0) is reserved for padding / empty sequences.
    For ``vocab_size <= 1``, returns ``pad_id``.
    """
    if vocab_size <= 1:
        return pad_id
    raw = label.encode("utf-8", errors="replace")
    h = zlib.crc32(raw) & 0xFFFFFFFF
    return 1 + (h % (vocab_size - 1))
