from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _percent_change(before: float, after: float) -> float:
    return (after / before - 1) * 100 if before else 0.0


def _paired_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    prefix_off = {
        condition["condition_id"]: condition
        for condition in payload["prefix_off"]["conditions"]
    }
    prefix_on = {
        condition["condition_id"]: condition
        for condition in payload["prefix_on"]["conditions"]
    }
    if prefix_off.keys() != prefix_on.keys():
        raise ValueError("cache-on and cache-off condition sets do not match")

    rows = []
    for condition_id, off in prefix_off.items():
        on = prefix_on[condition_id]
        off_metrics = off["metrics"]
        on_metrics = on["metrics"]
        rows.append(
            {
                "condition_id": condition_id,
                "target": on["target"],
                "prompt_shape": on["prompt_shape"],
                "concurrency": on["concurrency"],
                "request_count_per_mode": on_metrics["request_count"],
                "predicted_cache_hit_ratio": on_metrics[
                    "predicted_prefix_cache_hit_ratio"
                ],
                "observed_cache_hit_ratio": on_metrics[
                    "observed_prefix_cache_hit_ratio"
                ],
                "prediction_absolute_error": on_metrics[
                    "prediction_absolute_error"
                ],
                "cache_off_ttft_p50_ms": off_metrics["ttft_p50_seconds"] * 1000,
                "cache_on_ttft_p50_ms": on_metrics["ttft_p50_seconds"] * 1000,
                "ttft_change_percent": _percent_change(
                    off_metrics["ttft_p50_seconds"],
                    on_metrics["ttft_p50_seconds"],
                ),
                "cache_off_tpot_p50_ms": off_metrics["tpot_p50_seconds"] * 1000,
                "cache_on_tpot_p50_ms": on_metrics["tpot_p50_seconds"] * 1000,
                "cache_off_latency_p50_ms": (
                    off_metrics["latency_p50_seconds"] * 1000
                ),
                "cache_on_latency_p50_ms": on_metrics["latency_p50_seconds"] * 1000,
                "cache_off_latency_p95_ms": (
                    off_metrics["latency_p95_seconds"] * 1000
                ),
                "cache_on_latency_p95_ms": on_metrics["latency_p95_seconds"] * 1000,
                "cache_off_latency_p99_ms": (
                    off_metrics["latency_p99_seconds"] * 1000
                ),
                "cache_on_latency_p99_ms": on_metrics["latency_p99_seconds"] * 1000,
                "cache_off_input_tokens_per_second": off_metrics[
                    "input_tokens_per_second"
                ],
                "cache_on_input_tokens_per_second": on_metrics[
                    "input_tokens_per_second"
                ],
                "cache_off_output_tokens_per_second": off_metrics[
                    "output_tokens_per_second"
                ],
                "cache_on_output_tokens_per_second": on_metrics[
                    "output_tokens_per_second"
                ],
                "output_throughput_change_percent": _percent_change(
                    off_metrics["output_tokens_per_second"],
                    on_metrics["output_tokens_per_second"],
                ),
                "cache_off_peak_gpu_memory_gib": off["gpu_telemetry"][
                    "peak_memory_used_gib"
                ],
                "cache_on_peak_gpu_memory_gib": on["gpu_telemetry"][
                    "peak_memory_used_gib"
                ],
                "cache_off_cost_per_million_output_tokens_usd": off_metrics[
                    "estimated_cost_per_million_output_tokens_usd"
                ],
                "cache_on_cost_per_million_output_tokens_usd": on_metrics[
                    "estimated_cost_per_million_output_tokens_usd"
                ],
            }
        )
    return rows


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _paired_rows(payload)
    all_conditions = [
        condition
        for mode in ("prefix_off", "prefix_on")
        for condition in payload[mode]["conditions"]
    ]
    best = max(rows, key=lambda row: row["output_throughput_change_percent"])
    worst = min(rows, key=lambda row: row["output_throughput_change_percent"])
    prediction_errors = [row["prediction_absolute_error"] for row in rows]
    on_costs = [
        row["cache_on_cost_per_million_output_tokens_usd"] for row in rows
    ]
    summary = {
        "schema_version": "1.0",
        "experiment_id": payload["experiment_id"],
        "measured_condition_count": len(all_conditions),
        "paired_condition_count": len(rows),
        "total_measured_requests": sum(
            condition["metrics"]["request_count"] for condition in all_conditions
        ),
        "failed_requests": sum(
            condition["metrics"]["failed_requests"] for condition in all_conditions
        ),
        "maximum_observed_active_requests": max(
            payload[mode]["continuous_batching"][
                "maximum_observed_active_requests"
            ]
            for mode in ("prefix_off", "prefix_on")
        ),
        "prefix_cache_prediction": {
            "mean_absolute_error": statistics.mean(prediction_errors),
            "maximum_absolute_error": max(prediction_errors),
            "exact_condition_count": sum(error < 1e-12 for error in prediction_errors),
            "condition_count": len(prediction_errors),
        },
        "median_ttft_change_percent": statistics.median(
            row["ttft_change_percent"] for row in rows
        ),
        "median_output_throughput_change_percent": statistics.median(
            row["output_throughput_change_percent"] for row in rows
        ),
        "largest_output_throughput_gain": {
            "condition_id": best["condition_id"],
            "change_percent": best["output_throughput_change_percent"],
        },
        "largest_output_throughput_regression": {
            "condition_id": worst["condition_id"],
            "change_percent": worst["output_throughput_change_percent"],
        },
        "cache_on_cost_per_million_output_tokens_usd": {
            "minimum": min(on_costs),
            "median": statistics.median(on_costs),
            "maximum": max(on_costs),
            "scope": "steady-state condition time; startup and warm-up excluded",
        },
        "modes": {
            mode: {
                "startup_seconds": payload[mode]["startup_seconds"],
                "benchmark_seconds": payload[mode]["benchmark_seconds"],
                "peak_gpu_memory_gib": payload[mode]["gpu_telemetry"]["summary"][
                    "peak_memory_used_gib"
                ],
                "mean_gpu_utilization_percent": payload[mode]["gpu_telemetry"][
                    "summary"
                ]["mean_utilization_percent"],
                "maximum_observed_active_requests": payload[mode][
                    "continuous_batching"
                ]["maximum_observed_active_requests"],
            }
            for mode in ("prefix_off", "prefix_on")
        },
        "hardware": payload["prefix_on"]["hardware"],
        "requested_resource_hour_cost_usd": payload["prefix_on"]["config"][
            "modal_gpu_hour_cost_usd"
        ]
        + payload["prefix_on"]["config"]["reserved_cpus"]
        * payload["prefix_on"]["config"]["modal_cpu_hour_cost_usd"]
        + payload["prefix_on"]["config"]["reserved_memory_gib"]
        * payload["prefix_on"]["config"]["modal_memory_gib_hour_cost_usd"],
        "conditions": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, default=Path("results/phase3/summary.json"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/phase3/conditions.csv"),
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = build_summary(payload)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["conditions"][0]))
        writer.writeheader()
        writer.writerows(summary["conditions"])
    compact = {key: value for key, value in summary.items() if key != "conditions"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
