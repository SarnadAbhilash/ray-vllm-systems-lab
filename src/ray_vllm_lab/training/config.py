from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen experiment contract shared by one- and two-GPU runs."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    dataset_id: str = "databricks/databricks-dolly-15k"
    dataset_revision: str = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    seed: int = 20260822
    train_examples: int = 128
    eval_examples: int = 32
    max_length: int = 192
    global_batch_size: int = 8
    micro_batch_size: int = 4
    max_steps: int = 12
    warmup_steps: int = 2
    checkpoint_step: int = 6
    failure_after_step: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def validate(self, num_workers: int) -> None:
        if num_workers not in (1, 2):
            raise ValueError("Phase 2 supports exactly one or two workers")
        if self.global_batch_size % (self.micro_batch_size * num_workers):
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size * num_workers"
            )
        if self.max_steps <= self.checkpoint_step:
            raise ValueError("checkpoint_step must occur before max_steps")
        if not self.checkpoint_step < self.failure_after_step < self.max_steps:
            raise ValueError("failure_after_step must be between checkpoint_step and max_steps")
        required_examples = self.max_steps * self.global_batch_size
        if self.train_examples < required_examples:
            raise ValueError(
                f"train_examples must be at least {required_examples} for the fixed step budget"
            )
        if self.max_length <= 0 or self.micro_batch_size <= 0:
            raise ValueError("sequence and batch sizes must be positive")

    def gradient_accumulation_steps(self, num_workers: int) -> int:
        self.validate(num_workers)
        return self.global_batch_size // (self.micro_batch_size * num_workers)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result


def config_from_dict(values: dict[str, Any]) -> TrainingConfig:
    normalized = dict(values)
    if "target_modules" in normalized:
        normalized["target_modules"] = tuple(normalized["target_modules"])
    return TrainingConfig(**normalized)
