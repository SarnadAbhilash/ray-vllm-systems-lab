from __future__ import annotations

import math
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_requests(
    requests: list[dict[str, Any]],
    *,
    duration_seconds: float,
    block_size: int,
    resource_hour_cost_usd: float,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("at least one request measurement is required")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    completed = [row for row in requests if row.get("error") is None]
    if not completed:
        raise ValueError("no requests completed successfully")

    latencies = [float(row["latency_seconds"]) for row in completed]
    ttfts = [float(row["ttft_seconds"]) for row in completed]
    tpots = [float(row["tpot_seconds"]) for row in completed]
    input_tokens = sum(int(row["input_tokens"]) for row in completed)
    output_tokens = sum(int(row["output_tokens"]) for row in completed)
    cached_tokens = sum(int(row["cached_tokens"]) for row in completed)
    cacheable_tokens = sum(
        (int(row["input_tokens"]) // block_size) * block_size for row in completed
    )
    total_tokens = input_tokens + output_tokens
    estimated_cost = resource_hour_cost_usd * duration_seconds / 3600

    return {
        "request_count": len(requests),
        "completed_requests": len(completed),
        "failed_requests": len(requests) - len(completed),
        "duration_seconds": duration_seconds,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
        "ttft_p50_seconds": percentile(ttfts, 0.50),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "ttft_p99_seconds": percentile(ttfts, 0.99),
        "tpot_p50_seconds": percentile(tpots, 0.50),
        "tpot_p95_seconds": percentile(tpots, 0.95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tokens_per_second": input_tokens / duration_seconds,
        "output_tokens_per_second": output_tokens / duration_seconds,
        "total_tokens_per_second": total_tokens / duration_seconds,
        "observed_cached_tokens": cached_tokens,
        "observed_cacheable_tokens": cacheable_tokens,
        "observed_prefix_cache_hit_ratio": (
            cached_tokens / cacheable_tokens if cacheable_tokens else 0.0
        ),
        "estimated_resource_cost_usd": estimated_cost,
        "estimated_cost_per_million_total_tokens_usd": (
            estimated_cost * 1_000_000 / total_tokens if total_tokens else 0.0
        ),
        "estimated_cost_per_million_output_tokens_usd": (
            estimated_cost * 1_000_000 / output_tokens if output_tokens else 0.0
        ),
    }


def gpu_summary(
    samples: list[dict[str, Any]], start_unix: float, end_unix: float
) -> dict[str, float | int]:
    selected = [
        sample
        for sample in samples
        if start_unix <= float(sample["sampled_at_unix_seconds"]) <= end_unix
    ]
    utilization = [float(sample["utilization_percent"]) for sample in selected]
    active = [value for value in utilization if value > 0]
    return {
        "sample_count": len(selected),
        "mean_utilization_percent": (
            sum(utilization) / len(utilization) if utilization else 0.0
        ),
        "mean_active_utilization_percent": sum(active) / len(active) if active else 0.0,
        "peak_memory_used_gib": (
            max(float(sample["memory_used_mib"]) for sample in selected) / 1024
            if selected
            else 0.0
        ),
    }
