from __future__ import annotations

import json
import math
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .config import config_from_dict


def _distributed_sum(value: float, device: Any) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _distributed_max(value: float, device: Any) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name in {"input_ids", "attention_mask", "labels"}
    }


def _evaluate(model: Any, dataset: Any, batch_size: int, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    local_loss_sum = 0.0
    local_answer_tokens = 0
    local_correct = 0
    # FSDP lazily creates parameter views on first forward. ``inference_mode``
    # would make those views unusable for the later training backward pass.
    with torch.no_grad():
        for batch in dataset.iter_torch_batches(batch_size=batch_size, prefetch_batches=1):
            model_inputs = _move_batch(batch, device)
            outputs = model(**model_inputs)
            labels = model_inputs["labels"][:, 1:]
            active = labels.ne(-100)
            answer_tokens = int(active.sum().item())
            local_loss_sum += float(outputs.loss.item()) * answer_tokens
            predictions = outputs.logits[:, :-1].argmax(dim=-1)
            local_correct += int(((predictions == labels) & active).sum().item())
            local_answer_tokens += answer_tokens

    answer_tokens = _distributed_sum(local_answer_tokens, device)
    loss_sum = _distributed_sum(local_loss_sum, device)
    correct = _distributed_sum(local_correct, device)
    mean_loss = loss_sum / max(answer_tokens, 1.0)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": correct / max(answer_tokens, 1.0),
        "answer_tokens": answer_tokens,
    }


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    trainer_state: dict[str, Any],
    directory: str,
    rank: int,
    is_fsdp: bool,
) -> None:
    import torch
    import torch.distributed as dist
    if is_fsdp:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_config,
            optim_config,
        ):
            full_model_state = model.state_dict()
            full_optimizer_state = FSDP.optim_state_dict(model, optimizer)
        if rank == 0:
            peft_model = model.module
            # ``save_pretrained`` performs the PEFT adapter filtering itself.
            # Passing an already-filtered adapter state would filter twice and
            # produce a syntactically valid but empty safetensors file.
            peft_model.save_pretrained(
                directory,
                state_dict=full_model_state,
                safe_serialization=True,
            )
            torch.save(full_optimizer_state, Path(directory) / "optimizer.pt")
    elif rank == 0:
        model.save_pretrained(directory, safe_serialization=True)
        torch.save(optimizer.state_dict(), Path(directory) / "optimizer.pt")

    if rank == 0:
        (Path(directory) / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()


def _load_optimizer_state(
    model: Any,
    optimizer: Any,
    full_optimizer_state: dict[str, Any] | None,
    is_fsdp: bool,
) -> None:
    if full_optimizer_state is None and not is_fsdp:
        return
    if is_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        sharded_state = FSDP.scatter_full_optim_state_dict(
            full_optimizer_state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded_state)
    else:
        optimizer.load_state_dict(full_optimizer_state)


def _failure_marker(path: str, rank: int, step: int, device: Any) -> bool:
    import torch
    import torch.distributed as dist

    should_fail = 0
    marker = Path(path)
    if rank == 0 and not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"failed_at_unix_seconds": time.time(), "step": step}, handle)
                handle.write("\n")
            should_fail = 1
    signal = torch.tensor(should_fail, dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.broadcast(signal, src=0)
    return bool(signal.item())


def train_loop_per_worker(loop_config: dict[str, Any]) -> None:
    """Ray Train worker loop for LoRA SFT with 2-GPU FSDP and durable recovery."""

    import ray.train
    import ray.train.torch
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    config = config_from_dict(loop_config["training_config"])
    context = ray.train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    gradient_accumulation = config.gradient_accumulation_steps(world_size)
    device = ray.train.torch.get_device()
    if isinstance(device, list):
        device = device[0]
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    checkpoint = ray.train.get_checkpoint()
    restore_started = time.perf_counter()
    trainer_state: dict[str, Any] = {
        "completed_steps": 0,
        "measured_tokens": 0.0,
        "compute_seconds": 0.0,
        "eval_before": None,
    }
    loaded_optimizer_state = None

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False

    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
            trainer_state = json.loads(
                (Path(checkpoint_dir) / "trainer_state.json").read_text(encoding="utf-8")
            )
            if rank == 0:
                loaded_optimizer_state = torch.load(
                    Path(checkpoint_dir) / "optimizer.pt",
                    map_location="cpu",
                    weights_only=False,
                )
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.target_modules),
                bias="none",
            ),
        )

    parallel_strategy_kwargs: dict[str, Any] = {
        "use_orig_params": True,
        "device_id": device,
        "sync_module_states": world_size > 1,
        "limit_all_gathers": True,
    }
    if world_size > 1:
        from peft.utils.other import fsdp_auto_wrap_policy

        parallel_strategy_kwargs["auto_wrap_policy"] = fsdp_auto_wrap_policy(model)
    model = ray.train.torch.prepare_model(
        model,
        parallel_strategy="fsdp",
        parallel_strategy_kwargs=parallel_strategy_kwargs,
    )
    is_fsdp = world_size > 1
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if checkpoint is not None:
        _load_optimizer_state(model, optimizer, loaded_optimizer_state, is_fsdp)
    restore_seconds = time.perf_counter() - restore_started if checkpoint is not None else 0.0

    train_dataset = ray.train.get_dataset_shard("train")
    eval_dataset = ray.train.get_dataset_shard("eval")
    if train_dataset is None or eval_dataset is None:
        raise RuntimeError("Ray Train did not provide the required dataset shards")

    if trainer_state["eval_before"] is None:
        trainer_state["eval_before"] = _evaluate(
            model, eval_dataset, config.micro_batch_size, device
        )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    completed_steps = int(trainer_state["completed_steps"])
    skip_microbatches = completed_steps * gradient_accumulation
    accumulated_microbatches = 0
    local_tokens = 0.0
    recovery_seconds = 0.0
    resumed = checkpoint is not None
    marker_path = loop_config["failure_marker_path"]
    marker_data = None
    if resumed and Path(marker_path).exists():
        marker_data = json.loads(Path(marker_path).read_text(encoding="utf-8"))

    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        train_dataset.iter_torch_batches(
            batch_size=config.micro_batch_size,
            prefetch_batches=2,
        )
    ):
        if batch_index < skip_microbatches:
            continue
        if completed_steps >= config.max_steps:
            break

        is_accumulation_boundary = accumulated_microbatches + 1 == gradient_accumulation
        sync_context = (
            nullcontext()
            if is_accumulation_boundary or not hasattr(model, "no_sync")
            else model.no_sync()
        )
        step_started = time.perf_counter()
        model_inputs = _move_batch(batch, device)
        with sync_context:
            outputs = model(**model_inputs)
            (outputs.loss / gradient_accumulation).backward()
        local_tokens += float(model_inputs["attention_mask"].sum().item())
        accumulated_microbatches += 1

        if not is_accumulation_boundary:
            trainer_state["compute_seconds"] += time.perf_counter() - step_started
            continue

        if is_fsdp:
            model.clip_grad_norm_(config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trainer_state["compute_seconds"] += time.perf_counter() - step_started
        completed_steps += 1
        accumulated_microbatches = 0

        global_tokens = _distributed_sum(local_tokens, device)
        trainer_state["measured_tokens"] += (
            global_tokens if completed_steps > config.warmup_steps else 0.0
        )
        local_tokens = 0.0
        if completed_steps == config.warmup_steps:
            trainer_state["compute_seconds"] = 0.0
        trainer_state["completed_steps"] = completed_steps

        if resumed and marker_data is not None and recovery_seconds == 0.0:
            recovery_seconds = time.time() - float(marker_data["failed_at_unix_seconds"])

        should_checkpoint = completed_steps in {config.checkpoint_step, config.max_steps}
        if should_checkpoint:
            eval_after = (
                _evaluate(model, eval_dataset, config.micro_batch_size, device)
                if completed_steps == config.max_steps
                else None
            )
            model.train()
            peak_memory = _distributed_max(
                torch.cuda.max_memory_allocated(device) / (1024**3),
                device,
            )
            compute_seconds = _distributed_max(
                float(trainer_state["compute_seconds"]),
                device,
            )
            measured_tokens = _distributed_sum(
                float(trainer_state["measured_tokens"]) / world_size,
                device,
            )
            trainer_state["compute_seconds"] = compute_seconds
            trainer_state["measured_tokens"] = measured_tokens
            metrics = {
                "completed_steps": completed_steps,
                "world_size": world_size,
                "global_batch_size": config.global_batch_size,
                "gradient_accumulation_steps": gradient_accumulation,
                "training_tokens": measured_tokens,
                "training_compute_seconds": compute_seconds,
                "training_tokens_per_second": measured_tokens / max(compute_seconds, 1e-9),
                "peak_gpu_memory_gib": peak_memory,
                "eval_loss_before": trainer_state["eval_before"]["loss"],
                "eval_perplexity_before": trainer_state["eval_before"]["perplexity"],
                "eval_token_accuracy_before": trainer_state["eval_before"]["token_accuracy"],
                "checkpoint_restore_seconds": restore_seconds,
                "checkpoint_recovery_seconds": recovery_seconds,
                "resumed_from_checkpoint": resumed,
                "replayed_optimizer_steps": (
                    int(marker_data["step"]) - config.checkpoint_step
                    if marker_data is not None
                    else 0
                ),
                "parallel_strategy": "fsdp" if is_fsdp else "single_gpu",
            }
            if eval_after is not None:
                metrics.update(
                    {
                        "eval_loss_after": eval_after["loss"],
                        "eval_perplexity_after": eval_after["perplexity"],
                        "eval_token_accuracy_after": eval_after["token_accuracy"],
                    }
                )
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                _save_checkpoint(
                    model,
                    optimizer,
                    trainer_state,
                    checkpoint_dir,
                    rank,
                    is_fsdp,
                )
                ray_checkpoint = (
                    ray.train.Checkpoint.from_directory(checkpoint_dir) if rank == 0 else None
                )
                ray.train.report(metrics, checkpoint=ray_checkpoint)

        if (
            loop_config["inject_failure"]
            and completed_steps == config.failure_after_step
            and _failure_marker(marker_path, rank, completed_steps, device)
        ):
            raise RuntimeError(
                f"intentional Phase-2 recovery probe after optimizer step {completed_steps}"
            )

    if completed_steps < config.max_steps:
        raise RuntimeError(
            f"dataset exhausted at optimizer step {completed_steps}; expected {config.max_steps}"
        )
    if dist.is_initialized():
        dist.barrier()
