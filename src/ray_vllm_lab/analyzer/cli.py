from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .analysis import analyze_requests
from .io import load_jsonl
from .metrics import compare_report, load_metrics
from .report import render_markdown, write_json, write_markdown
from .tokenize import HuggingFaceConversationTokenizer, tokenize_records


def _block_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("block sizes must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("block sizes must be positive")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prefix-cache-lab",
        description="Analyze prefix-cache reuse before spending GPU time.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="tokenize and analyze a JSONL workload")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    analyze.add_argument("--revision")
    analyze.add_argument("--block-sizes", type=_block_sizes, default=(8, 16, 32))
    analyze.add_argument("--output", type=Path, default=Path("results/prefix-cache-report.json"))
    analyze.add_argument("--markdown", type=Path)
    analyze.add_argument("--trust-remote-code", action="store_true")

    compare = subparsers.add_parser("compare", help="compare prediction with vLLM metrics")
    compare.add_argument("report", type=Path)
    compare.add_argument("--metrics", required=True, help="Prometheus file or /metrics URL")
    compare.add_argument("--block-size", type=int, default=16)
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        records = load_jsonl(args.input)
        tokenizer = HuggingFaceConversationTokenizer(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        report = analyze_requests(
            tokenize_records(records, tokenizer),
            model=args.model,
            tokenizer_revision=args.revision,
            block_sizes=args.block_sizes,
        )
        write_json(report, args.output)
        if args.markdown:
            write_markdown(report, args.markdown)
        print(render_markdown(report), end="")
        return 0

    comparison = compare_report(args.report, load_metrics(args.metrics), args.block_size)
    rendered = json.dumps(comparison, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
