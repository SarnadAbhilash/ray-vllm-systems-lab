from __future__ import annotations

import pytest

from ray_vllm_lab.serving.metrics import aggregate_requests, gpu_summary, percentile


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)


def test_aggregate_requests_includes_cache_throughput_and_cost() -> None:
    rows = [
        {
            "latency_seconds": 1.0,
            "ttft_seconds": 0.2,
            "tpot_seconds": 0.1,
            "input_tokens": 35,
            "output_tokens": 9,
            "cached_tokens": 16,
            "error": None,
        },
        {
            "latency_seconds": 2.0,
            "ttft_seconds": 0.4,
            "tpot_seconds": 0.2,
            "input_tokens": 33,
            "output_tokens": 11,
            "cached_tokens": 32,
            "error": None,
        },
    ]

    result = aggregate_requests(
        rows,
        duration_seconds=2.0,
        block_size=16,
        resource_hour_cost_usd=1.8,
    )

    assert result["latency_p50_seconds"] == pytest.approx(1.5)
    assert result["input_tokens_per_second"] == pytest.approx(34)
    assert result["output_tokens_per_second"] == pytest.approx(10)
    assert result["observed_cacheable_tokens"] == 64
    assert result["observed_prefix_cache_hit_ratio"] == pytest.approx(0.75)
    assert result["estimated_resource_cost_usd"] == pytest.approx(0.001)
    assert result["estimated_cost_per_million_total_tokens_usd"] == pytest.approx(
        0.001 * 1_000_000 / 88
    )


def test_gpu_summary_selects_measurement_window() -> None:
    samples = [
        {"sampled_at_unix_seconds": 1.0, "utilization_percent": 99, "memory_used_mib": 1},
        {
            "sampled_at_unix_seconds": 2.0,
            "utilization_percent": 40,
            "memory_used_mib": 1024,
        },
        {
            "sampled_at_unix_seconds": 3.0,
            "utilization_percent": 80,
            "memory_used_mib": 2048,
        },
    ]

    result = gpu_summary(samples, 1.5, 3.0)

    assert result["sample_count"] == 2
    assert result["mean_utilization_percent"] == pytest.approx(60)
    assert result["peak_memory_used_gib"] == pytest.approx(2)
