from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib import request

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if match:
            name = match.group("name")
            totals[name] = totals.get(name, 0.0) + float(match.group("value"))
    return totals


def load_metrics(source: str) -> dict[str, float]:
    if source.startswith(("http://", "https://")):
        with request.urlopen(source, timeout=10) as response:  # noqa: S310
            return parse_prometheus(response.read().decode())
    return parse_prometheus(Path(source).read_text(encoding="utf-8"))


def compare_report(
    report_path: str | Path, metrics: dict[str, float], block_size: int
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    selected = next(
        (item for item in report["results"] if item["block_size"] == block_size),
        None,
    )
    if selected is None:
        raise ValueError(f"report does not contain block size {block_size}")
    queries = metrics.get("vllm:prefix_cache_queries", 0.0)
    hits = metrics.get("vllm:prefix_cache_hits", 0.0)
    observed_ratio = hits / queries if queries else 0.0
    predicted_ratio = float(selected["estimated_hit_ratio"])
    return {
        "schema_version": "1.0",
        "block_size": block_size,
        "predicted_hit_ratio": predicted_ratio,
        "observed_hit_ratio": observed_ratio,
        "absolute_error": abs(predicted_ratio - observed_ratio),
        "observed_query_tokens": queries,
        "observed_hit_tokens": hits,
        "note": (
            "A gap is expected when finite capacity, scheduling, interleaving, or eviction matters."
        ),
    }
