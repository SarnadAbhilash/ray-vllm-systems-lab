import pytest

from ray_vllm_lab.training.metrics import build_scaling_summary, validate_run_metrics
from ray_vllm_lab.training.runner import GpuSampler


def run_metrics(throughput: float, memory: float = 10.0) -> dict[str, float]:
    return {
        "training_tokens_per_second": throughput,
        "peak_gpu_memory_gib": memory,
        "eval_loss_before": 2.0,
        "eval_loss_after": 1.5,
        "eval_perplexity_before": 7.4,
        "eval_perplexity_after": 4.5,
        "eval_token_accuracy_before": 0.2,
        "eval_token_accuracy_after": 0.3,
    }


def test_builds_scaling_and_recovery_summary() -> None:
    recovery = {
        **run_metrics(150),
        "checkpoint_recovery_seconds": 42.0,
        "checkpoint_restore_seconds": 7.0,
        "replayed_optimizer_steps": 2,
    }
    summary = build_scaling_summary(run_metrics(100), run_metrics(180, 8.0), recovery)

    assert summary["two_gpu_speedup"] == 1.8
    assert summary["two_gpu_scaling_efficiency"] == 0.9
    assert summary["checkpoint_recovery_seconds"] == 42.0
    assert summary["replayed_optimizer_steps"] == 2


def test_rejects_missing_or_nonpositive_measurements() -> None:
    with pytest.raises(ValueError, match="peak_gpu_memory_gib"):
        validate_run_metrics({"training_tokens_per_second": 1})
    with pytest.raises(ValueError, match="throughput"):
        validate_run_metrics(run_metrics(0))


def test_gpu_sampler_summary_uses_all_and_active_samples() -> None:
    sampler = GpuSampler()
    sampler.samples = [
        {"utilization_percent": 0.0, "memory_used_mib": 1024.0},
        {"utilization_percent": 50.0, "memory_used_mib": 2048.0},
        {"utilization_percent": 100.0, "memory_used_mib": 3072.0},
    ]

    summary = sampler.summary()

    assert summary["mean_utilization_percent"] == 50.0
    assert summary["mean_active_utilization_percent"] == 75.0
    assert summary["p95_utilization_percent"] == 100.0
    assert summary["peak_memory_used_gib"] == 3.0
