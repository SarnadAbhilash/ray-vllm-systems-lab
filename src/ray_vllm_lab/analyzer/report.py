from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .models import AnalysisReport


def report_to_dict(report: AnalysisReport) -> dict[str, Any]:
    return dataclasses.asdict(report)


def write_json(report: AnalysisReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")


def render_markdown(report: AnalysisReport) -> str:
    lines = [
        "# Offline prefix-cache analysis",
        "",
        f"Model/tokenizer: `{report.model}`  ",
        f"Requests: {report.request_count}",
        "",
        "| Block size | Prompt tokens | Full-block tokens | Reusable tokens | "
        "Estimated hit ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in report.results:
        lines.append(
            f"| {result.block_size} | {result.total_prompt_tokens:,} | "
            f"{result.full_block_tokens:,} | {result.estimated_reusable_tokens:,} | "
            f"{result.estimated_hit_ratio:.1%} |"
        )
    lines.extend(["", "## Strongest shared prefixes", ""])
    for result in report.results:
        lines.append(f"### {result.block_size}-token blocks")
        lines.append("")
        if not result.shared_prefix_groups:
            lines.append("No reusable full-block prefixes found.")
        for group in result.shared_prefix_groups[:5]:
            ids = ", ".join(f"`{item}`" for item in group.request_ids)
            lines.append(f"- {group.shared_tokens} tokens shared by {ids}")
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {assumption}" for assumption in report.assumptions)
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(report: AnalysisReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
