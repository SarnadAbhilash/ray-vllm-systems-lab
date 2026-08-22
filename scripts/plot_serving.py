from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

COLORS = {"prefix_off": "#64748b", "prefix_on": "#4f46e5"}
LABELS = {"prefix_off": "Prefix cache off", "prefix_on": "Prefix cache on"}


def _median_series(
    conditions: list[dict[str, Any]], target: str, metric: str
) -> tuple[list[int], list[float]]:
    concurrency = sorted({int(row["concurrency"]) for row in conditions})
    values = []
    for level in concurrency:
        selected = [
            float(row["metrics"][metric])
            for row in conditions
            if row["target"] == target and row["concurrency"] == level
        ]
        values.append(statistics.median(selected))
    return concurrency, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("charts/phase3"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for axis, target in zip(axes, ("base", "adapter"), strict=True):
        for mode in ("prefix_off", "prefix_on"):
            x, y = _median_series(
                payload[mode]["conditions"],
                target,
                "output_tokens_per_second",
            )
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axis.set_title("Base model" if target == "base" else "LoRA adapter")
        axis.set_xlabel("Concurrent requests")
        axis.set_xticks(x)
    axes[0].set_ylabel("Median output tokens / second\n(across prompt shapes)")
    axes[1].legend(frameon=False)
    fig.suptitle("vLLM continuous-batching throughput on one NVIDIA L4")
    fig.tight_layout()
    fig.savefig(args.output_dir / "serving_throughput.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for axis, target in zip(axes, ("base", "adapter"), strict=True):
        for mode in ("prefix_off", "prefix_on"):
            x, y = _median_series(
                payload[mode]["conditions"],
                target,
                "latency_p95_seconds",
            )
            axis.plot(
                x,
                [value * 1000 for value in y],
                marker="o",
                linewidth=2,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axis.set_title("Base model" if target == "base" else "LoRA adapter")
        axis.set_xlabel("Concurrent requests")
        axis.set_xticks(x)
    axes[0].set_ylabel("Median p95 request latency (ms)\n(across prompt shapes)")
    axes[1].legend(frameon=False)
    fig.suptitle("Tail latency under increasing concurrency")
    fig.tight_layout()
    fig.savefig(args.output_dir / "serving_p95_latency.png", dpi=180)
    plt.close(fig)

    on_conditions = payload["prefix_on"]["conditions"]
    markers = {"short": "o", "long": "s", "agentic": "^"}
    fig, axis = plt.subplots(figsize=(6.2, 5.3))
    for shape, marker in markers.items():
        selected = [row for row in on_conditions if row["prompt_shape"] == shape]
        axis.scatter(
            [row["metrics"]["predicted_prefix_cache_hit_ratio"] * 100 for row in selected],
            [row["metrics"]["observed_prefix_cache_hit_ratio"] * 100 for row in selected],
            marker=marker,
            s=62,
            alpha=0.8,
            label=shape.capitalize(),
        )
    limits = (70, 97)
    axis.plot(limits, limits, linestyle="--", color="#0f172a", label="Exact prediction")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Offline predicted reusable full-block tokens (%)")
    axis.set_ylabel("Observed vLLM cached full-block tokens (%)")
    axis.set_title("Prefix-cache prediction versus observation")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cache_prediction.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
