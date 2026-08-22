from typing import Any

from ray_vllm_lab.analyzer.io import iter_jsonl
from ray_vllm_lab.analyzer.tokenize import tokenize_records


class ToyTokenizer:
    def encode_prompt(self, prompt: str) -> list[int]:
        return [len(word) for word in prompt.split()]

    def encode_messages(self, messages: tuple[dict[str, Any], ...]) -> list[int]:
        rendered = " ".join(f"{item['role']} {item['content']}" for item in messages)
        return self.encode_prompt(rendered)


def test_tokenizes_normalized_conversations() -> None:
    records = iter_jsonl(
        [{"id": "r1", "messages": [{"role": "user", "content": "hello cache"}]}]
    )
    tokenized = tokenize_records(records, ToyTokenizer())

    assert tokenized[0].request_id == "r1"
    assert tokenized[0].token_ids == (4, 5, 5)

