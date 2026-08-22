from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("charts/phase2"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = payload["summary"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    labels = ["1 × L4", "2 × L4"]
    throughput = [
        summary["one_gpu_tokens_per_second"],
        summary["two_gpu_tokens_per_second"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar([0, 1], throughput, color=["#475569", "#4f46e5"], width=0.55)
    axis.bar_label(bars, fmt="%.0f tok/s", padding=4)
    axis.set_xticks([0, 1], labels)
    axis.set_ylabel("Training input tokens / second")
    axis.set_title(
        "FSDP scaling: "
        f"{summary['two_gpu_speedup']:.2f}× speedup, "
        f"{summary['two_gpu_scaling_efficiency']:.0%} efficiency"
    )
    axis.set_ylim(0, max(throughput) * 1.22)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_scaling.png", dpi=180)
    plt.close(fig)

    quality = summary["quality"]
    before = [quality["eval_perplexity_before"], quality["eval_token_accuracy_before"] * 100]
    after = [quality["eval_perplexity_after"], quality["eval_token_accuracy_after"] * 100]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    for axis, metric, values in zip(
        axes,
        ["Held-out perplexity (lower is better)", "Token accuracy (%, higher is better)"],
        [(before[0], after[0]), (before[1], after[1])],
        strict=True,
    ):
        bars = axis.bar([0, 1], values, color=["#94a3b8", "#0f766e"], width=0.55)
        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.set_xticks([0, 1], ["Base", "LoRA"])
        axis.set_title(metric, fontsize=10)
        axis.set_ylim(0, max(values) * 1.25)
    fig.suptitle("Quality before and after Dolly instruction tuning")
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_quality.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
