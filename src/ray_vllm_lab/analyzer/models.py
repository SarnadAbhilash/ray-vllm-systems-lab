from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestRecord:
    """A normalized prompt or OpenAI-compatible chat request."""

    request_id: str
    prompt: str | None = None
    messages: tuple[dict[str, Any], ...] = ()
    adapter_id: str | None = None
    cache_salt: str | None = None
    source_line: int = 0


@dataclass(frozen=True)
class TokenizedRequest:
    """The token sequence and cache namespace that define prefix identity."""

    request_id: str
    token_ids: tuple[int, ...]
    adapter_id: str | None = None
    cache_salt: str | None = None


@dataclass(frozen=True)
class SharedPrefixGroup:
    depth_blocks: int
    shared_tokens: int
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class Divergence:
    after_blocks: int
    after_tokens: int
    request_ids: tuple[str, ...]
    branch_count: int


@dataclass(frozen=True)
class BlockSizeResult:
    block_size: int
    total_prompt_tokens: int
    full_block_tokens: int
    tail_tokens: int
    estimated_reusable_tokens: int
    estimated_hit_ratio: float
    unique_blocks: int
    total_blocks: int
    shared_prefix_groups: tuple[SharedPrefixGroup, ...] = ()
    divergences: tuple[Divergence, ...] = ()


@dataclass(frozen=True)
class AnalysisReport:
    schema_version: str
    model: str
    tokenizer_revision: str | None
    request_count: int
    block_sizes: tuple[int, ...]
    results: tuple[BlockSizeResult, ...]
    assumptions: tuple[str, ...] = field(default_factory=tuple)

