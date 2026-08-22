from typing import Any

import numpy as np

from ray_vllm_lab.training.data import TokenizeDollyBatch


class ToyChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        tokens = [1, 2, 3]
        if len(messages) == 2:
            tokens.extend([4, 5, 6])
        elif add_generation_prompt:
            tokens.append(4)
        return tokens


def test_tokenize_dolly_batch_masks_prompt_and_pads() -> None:
    actor = TokenizeDollyBatch.__new__(TokenizeDollyBatch)
    actor.model_id = "toy"
    actor.model_revision = "revision"
    actor.max_length = 8
    actor.tokenizer = ToyChatTokenizer()
    batch: dict[str, Any] = {
        "instruction": np.asarray(["Explain caching"]),
        "context": np.asarray([""]),
        "response": np.asarray(["Reuse prior blocks"]),
    }

    result = actor(batch)

    assert result["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 0, 0]]
    assert result["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert result["labels"].tolist() == [[-100, -100, -100, -100, 5, 6, -100, -100]]
    assert result["token_count"].tolist() == [6]
    assert result["answer_token_count"].tolist() == [2]
