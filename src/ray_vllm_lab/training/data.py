from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import TrainingConfig


@dataclass
class TokenizeDollyBatch:
    """Stateful Ray Data actor that loads one tokenizer per worker."""

    model_id: str
    model_revision: str
    max_length: int

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        token_counts: list[int] = []
        answer_token_counts: list[int] = []

        for instruction, context, response in zip(
            batch["instruction"],
            batch["context"],
            batch["response"],
            strict=True,
        ):
            user_content = str(instruction).strip()
            if str(context).strip():
                user_content += f"\n\nContext:\n{str(context).strip()}"
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": str(response).strip()},
            ]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            full_ids = list(full_ids[: self.max_length])
            prompt_length = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            attention = [1] * len(full_ids)
            padding = self.max_length - len(full_ids)
            input_ids = full_ids + [self.tokenizer.pad_token_id] * padding
            attention += [0] * padding
            labels += [-100] * padding

            input_rows.append(input_ids)
            attention_rows.append(attention)
            label_rows.append(labels)
            token_counts.append(sum(attention))
            answer_token_counts.append(sum(label != -100 for label in labels))

        return {
            "input_ids": np.asarray(input_rows, dtype=np.int64),
            "attention_mask": np.asarray(attention_rows, dtype=np.int64),
            "labels": np.asarray(label_rows, dtype=np.int64),
            "token_count": np.asarray(token_counts, dtype=np.int64),
            "answer_token_count": np.asarray(answer_token_counts, dtype=np.int64),
        }


def build_ray_datasets(config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, split, tokenize, and materialize the fixed Phase-2 corpus with Ray Data."""

    import ray
    from datasets import load_dataset

    started = time.perf_counter()
    total_examples = config.train_examples + config.eval_examples
    source = load_dataset(
        config.dataset_id,
        split="train",
        revision=config.dataset_revision,
    )
    selected = source.shuffle(seed=config.seed).select(range(total_examples))
    train_hf = selected.select(range(config.train_examples))
    eval_hf = selected.select(range(config.train_examples, total_examples))

    constructor_kwargs = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
    }

    def tokenize(dataset: Any) -> Any:
        return ray.data.from_huggingface(dataset).repartition(4).map_batches(
            TokenizeDollyBatch,
            fn_constructor_kwargs=constructor_kwargs,
            batch_size=32,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=2),
            num_cpus=1,
        )

    train_dataset = tokenize(train_hf).materialize()
    eval_dataset = tokenize(eval_hf).materialize()
    duration = time.perf_counter() - started
    input_tokens = int(train_dataset.sum("token_count")) + int(
        eval_dataset.sum("token_count")
    )
    stats = {
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "train_examples": train_dataset.count(),
        "eval_examples": eval_dataset.count(),
        "input_tokens": input_tokens,
        "preprocessing_seconds": duration,
        "preprocessing_examples_per_second": total_examples / duration,
        "preprocessing_tokens_per_second": input_tokens / duration,
        "ray_data_stats": train_dataset.stats(),
    }
    return {"train": train_dataset, "eval": eval_dataset}, stats
