from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from ray_vllm_lab.analyzer import TokenizedRequest, analyze_requests
from ray_vllm_lab.training.runner import GpuSampler

from .client import run_condition, wait_until_healthy, warm_up
from .config import ServingConfig
from .deployment import build_serve_application
from .metrics import aggregate_requests, gpu_summary
from .workloads import build_message_workloads, expand_prompts, render_workloads


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    devices = []
    for line in result.stdout.splitlines():
        index, name, memory_mib, driver = (part.strip() for part in line.split(",", 3))
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return devices


def _prediction(
    prompts: list[Any],
    *,
    config: ServingConfig,
    target: str,
) -> dict[str, Any]:
    adapter_id = config.adapter_name if target == "adapter" else None
    requests = [
        TokenizedRequest(
            request_id=prompt.request_id,
            token_ids=prompt.token_ids,
            adapter_id=adapter_id,
        )
        for prompt in prompts
    ]
    report = analyze_requests(
        requests,
        model=config.model_id,
        tokenizer_revision=config.model_revision,
        block_sizes=(config.block_size,),
    )
    result = report.results[0]
    return {
        "estimated_reusable_tokens": result.estimated_reusable_tokens,
        "full_block_tokens": result.full_block_tokens,
        "predicted_prefix_cache_hit_ratio": result.estimated_hit_ratio,
        "assumptions": list(report.assumptions),
    }


def _validate_prompt_lengths(
    rendered: dict[str, list[Any]], config: ServingConfig
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for shape, prompts in rendered.items():
        lengths = [len(prompt.token_ids) for prompt in prompts]
        longest = max(lengths)
        if longest + config.max_tokens > config.max_model_len:
            raise ValueError(
                f"{shape} prompt length {longest} plus output limit exceeds max_model_len"
            )
        stats[shape] = {
            "request_templates": len(lengths),
            "minimum_input_tokens": min(lengths),
            "mean_input_tokens": sum(lengths) / len(lengths),
            "maximum_input_tokens": longest,
        }
    return stats


def run_serving_experiment(
    *,
    run_name: str,
    prefix_caching: bool,
    smoke: bool = False,
    artifact_root: str = "/artifacts",
) -> dict[str, Any]:
    """Run one isolated prefix-cache mode and persist raw request-level results."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    import ray
    import torch
    import vllm
    from ray import serve
    from transformers import AutoTokenizer

    config = ServingConfig()
    config.validate()
    experiment_started = time.time()
    startup_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    workloads = build_message_workloads(config.agentic_jsonl)
    rendered = render_workloads(workloads, tokenizer)
    workload_stats = _validate_prompt_lengths(rendered, config)

    concurrency_levels = (1, 8) if smoke else config.concurrency_levels
    prompt_shapes = ("agentic",) if smoke else config.prompt_shapes
    base_url = "http://127.0.0.1:8000"
    ray.init(
        num_gpus=1,
        num_cpus=config.reserved_cpus,
        include_dashboard=False,
        log_to_driver=True,
    )
    sampler = GpuSampler()
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 8000})
        application = build_serve_application(
            config.as_dict(),
            prefix_caching=prefix_caching,
        )
        serve.run(application, name="vllm-benchmark", route_prefix="/")
        health_wait_seconds = asyncio.run(wait_until_healthy(base_url))
        startup_seconds = time.perf_counter() - startup_started

        for target in config.model_targets:
            model_name = (
                config.base_model_name if target == "base" else config.adapter_name
            )
            asyncio.run(warm_up(base_url, rendered["short"][0], model_name, config))

        conditions = []
        raw_requests: dict[str, list[dict[str, Any]]] = {}
        sampler.start()
        benchmark_started = time.perf_counter()
        try:
            for target in config.model_targets:
                model_name = (
                    config.base_model_name if target == "base" else config.adapter_name
                )
                for shape in prompt_shapes:
                    for concurrency in concurrency_levels:
                        count = config.request_count(concurrency, smoke=smoke)
                        prompts = expand_prompts(rendered[shape], count)
                        prediction = _prediction(prompts, config=config, target=target)
                        condition_id = f"{target}-{shape}-c{concurrency}"
                        result = asyncio.run(
                            run_condition(
                                base_url=base_url,
                                prompts=prompts,
                                model=model_name,
                                concurrency=concurrency,
                                config=config,
                            )
                        )
                        aggregate = aggregate_requests(
                            result["requests"],
                            duration_seconds=result["duration_seconds"],
                            block_size=config.block_size,
                            resource_hour_cost_usd=(
                                config.requested_resource_hour_cost_usd
                            ),
                        )
                        aggregate.update(prediction)
                        aggregate["prediction_absolute_error"] = abs(
                            prediction["predicted_prefix_cache_hit_ratio"]
                            - aggregate["observed_prefix_cache_hit_ratio"]
                        )
                        conditions.append(
                            {
                                "condition_id": condition_id,
                                "target": target,
                                "prompt_shape": shape,
                                "concurrency": concurrency,
                                "prefix_caching": prefix_caching,
                                "serve_stats": result["serve_stats"],
                                "measurement_window": {
                                    "start_unix_seconds": result["start_unix_seconds"],
                                    "end_unix_seconds": result["end_unix_seconds"],
                                },
                                "metrics": aggregate,
                            }
                        )
                        raw_requests[condition_id] = result["requests"]
        finally:
            sampler.stop()
        benchmark_seconds = time.perf_counter() - benchmark_started

        for condition in conditions:
            window = condition["measurement_window"]
            condition["gpu_telemetry"] = gpu_summary(
                sampler.samples,
                float(window["start_unix_seconds"]),
                float(window["end_unix_seconds"]),
            )

        manifest = {
            "schema_version": "1.0",
            "run_name": run_name,
            "status": "complete",
            "prefix_caching": prefix_caching,
            "smoke": smoke,
            "started_at_unix_seconds": experiment_started,
            "finished_at_unix_seconds": time.time(),
            "startup_seconds": startup_seconds,
            "health_wait_seconds": health_wait_seconds,
            "benchmark_seconds": benchmark_seconds,
            "config": config.as_dict(),
            "workloads": workload_stats,
            "conditions": conditions,
            "raw_requests": raw_requests,
            "continuous_batching": {
                "maximum_observed_active_requests": max(
                    int(condition["serve_stats"]["max_active_requests"])
                    for condition in conditions
                ),
                "evidence": (
                    "Serve replica active-request high-water mark observed while vLLM "
                    "scheduled streamed requests through one engine."
                ),
            },
            "gpu_telemetry": {
                "scope": "measured load-test conditions; model startup and warm-up excluded",
                "summary": sampler.summary(),
                "samples": sampler.samples,
            },
            "hardware": {
                "gpus": _gpu_inventory(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "ray": str(ray.__version__),
                "vllm": str(vllm.__version__),
            },
        }
        run_path = Path(artifact_root) / "phase3" / "runs" / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return json.loads(json.dumps(manifest))
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
        serve.shutdown()
        ray.shutdown()
