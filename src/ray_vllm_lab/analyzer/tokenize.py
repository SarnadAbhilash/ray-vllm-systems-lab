from __future__ import annotations

from typing import Any, Protocol

from .models import RequestRecord, TokenizedRequest


class ConversationTokenizer(Protocol):
    def encode_prompt(self, prompt: str) -> list[int]: ...

    def encode_messages(self, messages: tuple[dict[str, Any], ...]) -> list[int]: ...


class HuggingFaceConversationTokenizer:
    """Loads only tokenizer assets; no model weights or GPU are required."""

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def encode_prompt(self, prompt: str) -> list[int]:
        return list(self.tokenizer.encode(prompt, add_special_tokens=True))

    def encode_messages(self, messages: tuple[dict[str, Any], ...]) -> list[int]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("the selected tokenizer does not provide a chat template")
        token_ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(token_ids)


def tokenize_records(
    records: list[RequestRecord], tokenizer: ConversationTokenizer
) -> list[TokenizedRequest]:
    tokenized: list[TokenizedRequest] = []
    for record in records:
        if record.prompt is not None:
            token_ids = tokenizer.encode_prompt(record.prompt)
        else:
            token_ids = tokenizer.encode_messages(record.messages)
        tokenized.append(
            TokenizedRequest(
                request_id=record.request_id,
                token_ids=tuple(int(token_id) for token_id in token_ids),
                adapter_id=record.adapter_id,
                cache_salt=record.cache_salt,
            )
        )
    return tokenized

