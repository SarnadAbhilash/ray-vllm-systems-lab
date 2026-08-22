from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .config import ServingConfig
from .workloads import RenderedPrompt


async def wait_until_healthy(base_url: str, timeout_seconds: float = 600) -> float:
    import httpx

    started = time.perf_counter()
    deadline = started + timeout_seconds
    async with httpx.AsyncClient(timeout=10) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"serving endpoint was not healthy after {timeout_seconds} seconds")


async def _stream_completion(
    client: Any,
    *,
    base_url: str,
    prompt: RenderedPrompt,
    model: str,
    config: ServingConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    done: dict[str, Any] | None = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json={
                "request_id": prompt.request_id,
                "model": model,
                "prompt": prompt.prompt,
                "max_tokens": config.max_tokens,
                "min_tokens": config.min_tokens,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "delta" and first_token_at is None:
                    first_token_at = time.perf_counter()
                elif event["type"] == "done":
                    done = event
        finished = time.perf_counter()
        if first_token_at is None or done is None:
            raise RuntimeError("stream ended without token and completion events")
        output_tokens = int(done["output_tokens"])
        latency = finished - started
        ttft = first_token_at - started
        return {
            "request_id": prompt.request_id,
            "latency_seconds": latency,
            "ttft_seconds": ttft,
            "tpot_seconds": (
                (latency - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0
            ),
            "input_tokens": int(done["input_tokens"]),
            "output_tokens": output_tokens,
            "cached_tokens": int(done["cached_tokens"]),
            "finish_reason": done.get("finish_reason"),
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - preserve failures in the raw benchmark
        return {
            "request_id": prompt.request_id,
            "latency_seconds": time.perf_counter() - started,
            "ttft_seconds": 0.0,
            "tpot_seconds": 0.0,
            "input_tokens": len(prompt.token_ids),
            "output_tokens": 0,
            "cached_tokens": 0,
            "finish_reason": None,
            "error": f"{type(error).__name__}: {error}",
        }


async def warm_up(
    base_url: str,
    prompt: RenderedPrompt,
    model: str,
    config: ServingConfig,
) -> None:
    import httpx

    timeout = httpx.Timeout(300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(config.warmup_requests_per_target):
            result = await _stream_completion(
                client,
                base_url=base_url,
                prompt=prompt,
                model=model,
                config=config,
            )
            if result["error"] is not None:
                raise RuntimeError(f"warm-up failed: {result['error']}")


async def run_condition(
    *,
    base_url: str,
    prompts: list[RenderedPrompt],
    model: str,
    concurrency: int,
    config: ServingConfig,
) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        reset = await client.post(f"{base_url}/admin/reset-prefix-cache")
        reset.raise_for_status()
        semaphore = asyncio.Semaphore(concurrency)
        ready = asyncio.Event()

        async def invoke(prompt: RenderedPrompt) -> dict[str, Any]:
            await ready.wait()
            async with semaphore:
                return await _stream_completion(
                    client,
                    base_url=base_url,
                    prompt=prompt,
                    model=model,
                    config=config,
                )

        tasks = [asyncio.create_task(invoke(prompt)) for prompt in prompts]
        start_unix = time.time()
        started = time.perf_counter()
        ready.set()
        rows = await asyncio.gather(*tasks)
        duration = time.perf_counter() - started
        end_unix = time.time()
        stats_response = await client.get(f"{base_url}/admin/stats")
        stats_response.raise_for_status()
    return {
        "requests": rows,
        "duration_seconds": duration,
        "start_unix_seconds": start_unix,
        "end_unix_seconds": end_unix,
        "serve_stats": stats_response.json(),
    }
