from __future__ import annotations

import math
from typing import Any

REQUIRED_RUN_METRICS = (
    "training_tokens_per_second",
    "peak_gpu_memory_gib",
    "eval_loss_before",
    "eval_loss_after",
    "eval_perplexity_before",
    "eval_perplexity_after",
    "eval_token_accuracy_before",
    "eval_token_accuracy_after",
)


def _finite_number(run: dict[str, Any], name: str) -> float:
    if name not in run:
        raise ValueError(f"run is missing {name}")
    value = float(run[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_run_metrics(run: dict[str, Any]) -> None:
    for name in REQUIRED_RUN_METRICS:
        _finite_number(run, name)
    if _finite_number(run, "training_tokens_per_second") <= 0:
        raise ValueError("training throughput must be positive")
    if _finite_number(run, "peak_gpu_memory_gib") <= 0:
        raise ValueError("peak GPU memory must be positive")


def build_scaling_summary(
    one_gpu: dict[str, Any],
    two_gpu: dict[str, Any],
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_run_metrics(one_gpu)
    validate_run_metrics(two_gpu)
    throughput_1 = _finite_number(one_gpu, "training_tokens_per_second")
    throughput_2 = _finite_number(two_gpu, "training_tokens_per_second")
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "one_gpu_tokens_per_second": throughput_1,
        "two_gpu_tokens_per_second": throughput_2,
        "two_gpu_speedup": throughput_2 / throughput_1,
        "two_gpu_scaling_efficiency": throughput_2 / (2 * throughput_1),
        "one_gpu_peak_memory_gib": _finite_number(one_gpu, "peak_gpu_memory_gib"),
        "two_gpu_peak_memory_gib_per_device": _finite_number(
            two_gpu, "peak_gpu_memory_gib"
        ),
        "quality": {
            "eval_loss_before": _finite_number(two_gpu, "eval_loss_before"),
            "eval_loss_after": _finite_number(two_gpu, "eval_loss_after"),
            "eval_perplexity_before": _finite_number(two_gpu, "eval_perplexity_before"),
            "eval_perplexity_after": _finite_number(two_gpu, "eval_perplexity_after"),
            "eval_token_accuracy_before": _finite_number(
                two_gpu, "eval_token_accuracy_before"
            ),
            "eval_token_accuracy_after": _finite_number(two_gpu, "eval_token_accuracy_after"),
        },
    }
    if recovery is not None:
        recovery_seconds = _finite_number(recovery, "checkpoint_recovery_seconds")
        if recovery_seconds <= 0:
            raise ValueError("checkpoint recovery time must be positive")
        summary["checkpoint_recovery_seconds"] = recovery_seconds
        summary["checkpoint_restore_seconds"] = _finite_number(
            recovery, "checkpoint_restore_seconds"
        )
        summary["replayed_optimizer_steps"] = int(recovery["replayed_optimizer_steps"])
    return summary
