from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ServingConfig:
    """Frozen serving and load-test contract for Phase 3."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    base_model_name: str = "qwen-base"
    adapter_name: str = "dolly-lora"
    adapter_path: str = "/artifacts/phase2/adapters/phase2-20260822-final-2gpu"
    agentic_jsonl: str = "/root/data/serving/agentic_requests.jsonl"
    max_model_len: int = 1024
    block_size: int = 16
    gpu_memory_utilization: float = 0.80
    max_lora_rank: int = 8
    max_tokens: int = 16
    min_tokens: int = 8
    seed: int = 20260822
    concurrency_levels: tuple[int, ...] = (1, 8, 32, 64)
    prompt_shapes: tuple[str, ...] = ("short", "long", "agentic")
    model_targets: tuple[str, ...] = ("base", "adapter")
    warmup_requests_per_target: int = 2
    modal_gpu_hour_cost_usd: float = 0.80
    modal_cpu_hour_cost_usd: float = 0.04730
    modal_memory_gib_hour_cost_usd: float = 0.00800
    reserved_cpus: int = 8
    reserved_memory_gib: int = 32

    def validate(self) -> None:
        if self.block_size not in (8, 16, 32):
            raise ValueError("block_size must be 8, 16, or 32")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if self.min_tokens <= 0 or self.max_tokens < self.min_tokens:
            raise ValueError("token limits must be positive and ordered")
        if not self.concurrency_levels or any(level <= 0 for level in self.concurrency_levels):
            raise ValueError("concurrency levels must be positive")
        if set(self.prompt_shapes) != {"short", "long", "agentic"}:
            raise ValueError("prompt_shapes must contain short, long, and agentic")
        if set(self.model_targets) != {"base", "adapter"}:
            raise ValueError("model_targets must contain base and adapter")

    def request_count(self, concurrency: int, *, smoke: bool = False) -> int:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        return max(concurrency, 8 if smoke else 32)

    @property
    def requested_resource_hour_cost_usd(self) -> float:
        return (
            self.modal_gpu_hour_cost_usd
            + self.reserved_cpus * self.modal_cpu_hour_cost_usd
            + self.reserved_memory_gib * self.modal_memory_gib_hour_cost_usd
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("concurrency_levels", "prompt_shapes", "model_targets"):
            result[name] = list(result[name])
        return result


def config_from_dict(values: dict[str, Any]) -> ServingConfig:
    normalized = dict(values)
    for name in ("concurrency_levels", "prompt_shapes", "model_targets"):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    return ServingConfig(**normalized)
