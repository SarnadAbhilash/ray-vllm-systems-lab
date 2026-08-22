from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence


def cache_namespace(adapter_id: str | None, cache_salt: str | None) -> bytes:
    """Stable namespace for metadata that intentionally partitions cached blocks."""

    return json.dumps(
        {"adapter_id": adapter_id, "cache_salt": cache_salt},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def build_hash_chain(
    token_ids: Sequence[int],
    block_size: int,
    *,
    adapter_id: str | None = None,
    cache_salt: str | None = None,
) -> tuple[bytes, ...]:
    """Build a parent-linked full-block hash chain without importing vLLM internals.

    This deliberately models vLLM's identity semantics, not its exact hash function.
    Partial tail blocks are excluded because they are not reusable full blocks.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    namespace = cache_namespace(adapter_id, cache_salt)
    parent = b"\x00" * 32
    chain: list[bytes] = []
    full_length = len(token_ids) - (len(token_ids) % block_size)
    for offset in range(0, full_length, block_size):
        block = token_ids[offset : offset + block_size]
        packed = struct.pack(f">{len(block)}q", *block)
        parent = hashlib.sha256(parent + namespace + packed).digest()
        chain.append(parent)
    return tuple(chain)

