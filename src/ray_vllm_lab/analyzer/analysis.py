from __future__ import annotations

from collections import defaultdict

from .hashing import build_hash_chain
from .models import (
    AnalysisReport,
    BlockSizeResult,
    Divergence,
    SharedPrefixGroup,
    TokenizedRequest,
)


def _prefix_structure(
    chains: list[tuple[bytes, ...]],
    requests: list[TokenizedRequest],
    block_size: int,
    limit: int,
) -> tuple[tuple[SharedPrefixGroup, ...], tuple[Divergence, ...]]:
    prefix_members: dict[tuple[bytes, ...], list[str]] = defaultdict(list)
    children: dict[tuple[bytes, ...], dict[bytes, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for request, chain in zip(requests, chains, strict=True):
        for depth in range(1, len(chain) + 1):
            prefix = chain[:depth]
            prefix_members[prefix].append(request.request_id)
            if depth < len(chain):
                children[prefix][chain[depth]].add(request.request_id)

    candidates = [
        SharedPrefixGroup(
            depth_blocks=len(prefix),
            shared_tokens=len(prefix) * block_size,
            request_ids=tuple(dict.fromkeys(ids)),
        )
        for prefix, ids in prefix_members.items()
        if len(set(ids)) >= 2
    ]
    candidates.sort(
        key=lambda group: (group.shared_tokens * len(group.request_ids), group.shared_tokens),
        reverse=True,
    )

    divergences = []
    for prefix, branches in children.items():
        if len(branches) < 2:
            continue
        member_ids = tuple(dict.fromkeys(prefix_members[prefix]))
        divergences.append(
            Divergence(
                after_blocks=len(prefix),
                after_tokens=len(prefix) * block_size,
                request_ids=member_ids,
                branch_count=len(branches),
            )
        )
    divergences.sort(
        key=lambda item: (item.after_tokens * len(item.request_ids), item.branch_count),
        reverse=True,
    )
    return tuple(candidates[:limit]), tuple(divergences[:limit])


def _analyze_block_size(
    requests: list[TokenizedRequest], block_size: int, top_groups: int
) -> BlockSizeResult:
    chains = [
        build_hash_chain(
            request.token_ids,
            block_size,
            adapter_id=request.adapter_id,
            cache_salt=request.cache_salt,
        )
        for request in requests
    ]
    seen: set[bytes] = set()
    hit_blocks = 0
    for chain in chains:
        hit_blocks += sum(block_hash in seen for block_hash in chain)
        seen.update(chain)

    total_prompt_tokens = sum(len(request.token_ids) for request in requests)
    total_blocks = sum(len(chain) for chain in chains)
    full_block_tokens = total_blocks * block_size
    reusable_tokens = hit_blocks * block_size
    groups, divergences = _prefix_structure(chains, requests, block_size, top_groups)
    return BlockSizeResult(
        block_size=block_size,
        total_prompt_tokens=total_prompt_tokens,
        full_block_tokens=full_block_tokens,
        tail_tokens=total_prompt_tokens - full_block_tokens,
        estimated_reusable_tokens=reusable_tokens,
        estimated_hit_ratio=(reusable_tokens / full_block_tokens if full_block_tokens else 0.0),
        unique_blocks=len(seen),
        total_blocks=total_blocks,
        shared_prefix_groups=groups,
        divergences=divergences,
    )


def analyze_requests(
    requests: list[TokenizedRequest],
    *,
    model: str,
    block_sizes: tuple[int, ...] = (8, 16, 32),
    tokenizer_revision: str | None = None,
    top_groups: int = 10,
) -> AnalysisReport:
    if not requests:
        raise ValueError("at least one request is required")
    if not block_sizes or any(size <= 0 for size in block_sizes):
        raise ValueError("block sizes must be positive")
    normalized_sizes = tuple(dict.fromkeys(block_sizes))
    return AnalysisReport(
        schema_version="1.0",
        model=model,
        tokenizer_revision=tokenizer_revision,
        request_count=len(requests),
        block_sizes=normalized_sizes,
        results=tuple(
            _analyze_block_size(requests, size, top_groups) for size in normalized_sizes
        ),
        assumptions=(
            "Requests are evaluated in JSONL arrival order with an initially empty cache.",
            "The estimate assumes infinite cache capacity and no eviction, preemption, "
            "or KV offload.",
            "Only complete blocks are reusable; partial tail blocks are excluded.",
            "The standalone SHA-256 chain models block identity but does not import or claim "
            "exact vLLM internals.",
            "Adapter IDs and cache salts partition cache identity when present.",
        ),
    )
