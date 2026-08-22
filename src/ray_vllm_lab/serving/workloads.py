from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ray_vllm_lab.analyzer.io import load_jsonl


@dataclass(frozen=True)
class RenderedPrompt:
    request_id: str
    prompt: str
    token_ids: tuple[int, ...]


SHORT_QUESTIONS = (
    "Name one practical benefit of request batching.",
    "What does time to first token measure?",
    "Define a model checkpoint in one sentence.",
    "Why cap request concurrency?",
    "What is a GPU memory watermark?",
    "When is a cache entry reusable?",
    "What does p99 latency describe?",
    "Why pin a model revision?",
)

LONG_TOPICS = (
    "distributed preprocessing",
    "checkpoint recovery",
    "continuous batching",
    "prefix cache identity",
    "tail latency",
    "GPU memory planning",
    "adapter multiplexing",
    "load-test methodology",
)


def _long_context(topic: str, index: int) -> str:
    paragraph = (
        f"Section {index} studies {topic}. The system receives a bounded stream of requests, "
        "records arrival and completion times, separates initialization from steady state, and "
        "preserves raw "
        "measurements for later inspection. Every conclusion must identify its workload, hardware, "
        "software versions, and measurement window. Capacity limits, queueing, cache residency, "
        "and scheduler decisions can change the result, so the experiment varies one declared "
        "dimension at a time and treats unexpected behavior as evidence. "
    )
    return paragraph * 6


def build_message_workloads(agentic_jsonl: str | Path) -> dict[str, list[list[dict[str, str]]]]:
    short = [[{"role": "user", "content": question}] for question in SHORT_QUESTIONS]
    long = [
        [
            {
                "role": "system",
                "content": "Use only the supplied experiment note and answer concisely.",
            },
            {
                "role": "user",
                "content": (
                    f"Experiment note:\n{_long_context(topic, index)}\n"
                    "Question: identify the two most important controls."
                ),
            },
        ]
        for index, topic in enumerate(LONG_TOPICS, start=1)
    ]
    records = load_jsonl(agentic_jsonl)
    agentic = [list(record.messages) for record in records if record.messages]
    if not agentic:
        raise ValueError("agentic workload contains no chat requests")
    return {"short": short, "long": long, "agentic": agentic}


def render_workloads(
    workloads: dict[str, list[list[dict[str, Any]]]], tokenizer: Any
) -> dict[str, list[RenderedPrompt]]:
    rendered: dict[str, list[RenderedPrompt]] = {}
    for shape, conversations in workloads.items():
        rows = []
        for index, messages in enumerate(conversations):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(token_ids, Mapping):
                token_ids = token_ids["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            rows.append(
                RenderedPrompt(
                    request_id=f"{shape}-{index:02d}",
                    prompt=str(prompt),
                    token_ids=tuple(int(token_id) for token_id in token_ids),
                )
            )
        rendered[shape] = rows
    return rendered


def expand_prompts(prompts: list[RenderedPrompt], count: int) -> list[RenderedPrompt]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if count <= 0:
        raise ValueError("count must be positive")
    expanded = []
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        expanded.append(
            RenderedPrompt(
                request_id=f"{prompt.request_id}-r{index:04d}",
                prompt=prompt.prompt,
                token_ids=prompt.token_ids,
            )
        )
    return expanded
