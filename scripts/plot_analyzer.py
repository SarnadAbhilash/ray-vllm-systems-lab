from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("charts"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    results = report["results"]
    block_sizes = [str(item["block_size"]) for item in results]
    positions = list(range(len(block_sizes)))
    hit_ratios = [100 * item["estimated_hit_ratio"] for item in results]
    reusable = [item["estimated_reusable_tokens"] for item in results]
    non_reusable = [
        item["full_block_tokens"] - item["estimated_reusable_tokens"] for item in results
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(positions, hit_ratios, color="#4f46e5", width=0.55)
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_xticks(positions, block_sizes)
    axis.set_ylim(0, max(hit_ratios + [10]) * 1.2)
    axis.set_xlabel("KV-cache block size (tokens)")
    axis.set_ylabel("Predicted reusable full-block tokens")
    axis.set_title("Offline cacheability changes with block granularity")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cacheability_by_block_size.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(positions, reusable, label="Reusable", color="#0f766e", width=0.55)
    axis.bar(
        positions,
        non_reusable,
        bottom=reusable,
        label="First-compute",
        color="#cbd5e1",
        width=0.55,
    )
    axis.set_xticks(positions, block_sizes)
    axis.set_xlabel("KV-cache block size (tokens)")
    axis.set_ylabel("Full-block prompt tokens")
    axis.set_title("Reusable versus first-compute prompt work")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "token_accounting.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
