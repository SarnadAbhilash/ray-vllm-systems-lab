from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from .config import config_from_dict


def build_serve_application(config_values: dict[str, Any], *, prefix_caching: bool) -> Any:
    """Build a one-replica Ray Serve application backed by vLLM AsyncLLM."""

    from ray import serve
    from starlette.responses import JSONResponse, StreamingResponse
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    config = config_from_dict(config_values)
    config.validate()
    @serve.deployment(
        ray_actor_options={"num_cpus": 2, "num_gpus": 1},
        max_ongoing_requests=128,
    )
    class VLLMService:
        def __init__(self) -> None:
            engine_args = AsyncEngineArgs(
                model=config.model_id,
                revision=config.model_revision,
                tokenizer_revision=config.model_revision,
                dtype="bfloat16",
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_prefix_caching=prefix_caching,
                block_size=config.block_size,
                enable_lora=True,
                max_lora_rank=config.max_lora_rank,
                max_loras=1,
                enforce_eager=True,
                seed=config.seed,
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
            self.adapter = LoRARequest(
                config.adapter_name,
                1,
                config.adapter_path,
            )
            self.active_requests = 0
            self.max_active_requests = 0
            self.completed_requests = 0

        async def __call__(self, request: Any) -> Any:
            path = request.url.path
            method = request.method
            if method == "GET" and path == "/health":
                return JSONResponse(
                    {
                        "status": "ok",
                        "prefix_caching": prefix_caching,
                        "active_requests": self.active_requests,
                    }
                )
            if method == "GET" and path == "/v1/models":
                return JSONResponse(
                    {
                        "data": [
                            {"id": config.base_model_name, "type": "base"},
                            {"id": config.adapter_name, "type": "lora"},
                        ]
                    }
                )
            if method == "POST" and path == "/v1/completions":
                return await self.completions(await request.json())
            if method == "POST" and path == "/admin/reset-prefix-cache":
                return await self.reset_prefix_cache()
            if method == "GET" and path == "/admin/stats":
                return JSONResponse(self.stats())
            return JSONResponse({"detail": "not found"}, status_code=404)

        async def completions(self, body: dict[str, Any]) -> Any:
            prompt = body.get("prompt")
            model = body.get("model")
            if not isinstance(prompt, str) or not prompt:
                return JSONResponse(
                    {"detail": "prompt must be non-empty text"},
                    status_code=400,
                )
            if model not in (config.base_model_name, config.adapter_name):
                return JSONResponse(
                    {"detail": f"unknown model: {model}"},
                    status_code=404,
                )

            max_tokens = int(body.get("max_tokens", config.max_tokens))
            min_tokens = int(body.get("min_tokens", config.min_tokens))
            if min_tokens <= 0 or max_tokens < min_tokens:
                return JSONResponse(
                    {"detail": "invalid output token limits"},
                    status_code=400,
                )
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                seed=config.seed,
            )
            lora_request = self.adapter if model == config.adapter_name else None

            async def stream() -> AsyncIterator[str]:
                request_id = str(body.get("request_id") or uuid4())
                emitted_tokens = 0
                prior_text = ""
                final_output: Any | None = None
                self.active_requests += 1
                self.max_active_requests = max(
                    self.max_active_requests,
                    self.active_requests,
                )
                try:
                    async for output in self.engine.generate(
                        prompt,
                        sampling,
                        request_id,
                        lora_request=lora_request,
                    ):
                        final_output = output
                        if not output.outputs:
                            continue
                        completion = output.outputs[0]
                        token_count = len(completion.token_ids)
                        new_tokens = token_count - emitted_tokens
                        if new_tokens <= 0:
                            continue
                        delta_text = completion.text[len(prior_text) :]
                        emitted_tokens = token_count
                        prior_text = completion.text
                        yield json.dumps(
                            {
                                "type": "delta",
                                "new_tokens": new_tokens,
                                "text": delta_text,
                            },
                            separators=(",", ":"),
                        ) + "\n"

                    if final_output is None or not final_output.outputs:
                        raise RuntimeError("vLLM completed without a request output")
                    completion = final_output.outputs[0]
                    yield json.dumps(
                        {
                            "type": "done",
                            "request_id": request_id,
                            "input_tokens": len(final_output.prompt_token_ids),
                            "output_tokens": len(completion.token_ids),
                            "cached_tokens": int(final_output.num_cached_tokens or 0),
                            "finish_reason": completion.finish_reason,
                        },
                        separators=(",", ":"),
                    ) + "\n"
                    self.completed_requests += 1
                finally:
                    self.active_requests -= 1

            return StreamingResponse(stream(), media_type="application/x-ndjson")

        async def reset_prefix_cache(self) -> Any:
            if self.active_requests:
                return JSONResponse(
                    {"detail": "requests are still active"},
                    status_code=409,
                )
            reset = await self.engine.reset_prefix_cache()
            self.max_active_requests = 0
            return JSONResponse(
                {"reset": bool(reset), "prefix_caching": prefix_caching}
            )

        def stats(self) -> dict[str, int]:
            return {
                "active_requests": self.active_requests,
                "max_active_requests": self.max_active_requests,
                "completed_requests": self.completed_requests,
            }

    return VLLMService.bind()
