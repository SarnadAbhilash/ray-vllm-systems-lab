from __future__ import annotations

import json
from pathlib import Path

from ray_vllm_lab.serving.workloads import (
    RenderedPrompt,
    build_message_workloads,
    expand_prompts,
    render_workloads,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert add_generation_prompt
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if tokenize:
            return list(range(len(rendered.split())))
        return rendered + "|assistant:"


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | dict[str, list[list[int]]]:
        result = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if tokenize:
            return {"input_ids": [result]}
        return result


def test_build_render_and_expand_workloads(tmp_path: Path) -> None:
    source = tmp_path / "requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "agent-1",
                "messages": [
                    {"role": "system", "content": "shared policy"},
                    {"role": "user", "content": "question"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    workloads = build_message_workloads(source)
    rendered = render_workloads(workloads, FakeTokenizer())
    expanded = expand_prompts(rendered["agentic"], 3)

    assert set(rendered) == {"short", "long", "agentic"}
    assert len(rendered["short"]) == 8
    assert len(rendered["long"]) == 8
    assert len(rendered["agentic"]) == 1
    assert [prompt.request_id for prompt in expanded] == [
        "agentic-00-r0000",
        "agentic-00-r0001",
        "agentic-00-r0002",
    ]
    assert all(isinstance(prompt, RenderedPrompt) for prompt in expanded)
    assert expanded[0].token_ids == expanded[2].token_ids


def test_render_accepts_transformers_five_mapping_shape() -> None:
    rendered = render_workloads(
        {"short": [[{"role": "user", "content": "two words"}]]},
        MappingTokenizer(),
    )

    assert rendered["short"][0].token_ids == (0, 1)
