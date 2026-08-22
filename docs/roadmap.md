# Delivery roadmap

## Phase 1 — repository foundation and offline analyzer (complete)

Acceptance criteria:

- Standalone JSONL analyzer requires no model weights or GPU.
- Exact chat-template tokenization supports plain, chat, and OpenAI Batch request shapes.
- Arrival-order reuse, cache namespaces, full-block accounting, block-size comparison, shared groups,
  and divergence reporting have deterministic tests.
- JSON/Markdown reports, two charts, and vLLM Prometheus comparison are reproducible.
- CI passes from a clean checkout.

## Phase 2 — Ray Data + Ray Train/FSDP LoRA

Deliverables:

- Frozen instruction dataset manifest and Ray Data preprocessing pipeline.
- Ray Train V2 `TorchTrainer` entrypoint with PyTorch FSDP and PEFT LoRA.
- Modal image, secrets, persistent Volume, and 1/2-GPU launch commands.
- Checkpoint save/load with an intentional one-time failure and timed recovery.
- Training throughput, GPU memory, scaling efficiency, recovery time, and fixed base/adapter quality.
- Raw run manifests and charts generated from artifacts, not hand-entered values.

## Phase 3 — vLLM + Ray Serve inference lab

Deliverables:

- Base and LoRA adapter served through a Ray Serve deployment wrapping vLLM.
- Streaming load generator for concurrency 1/8/32/64 and short/long/repeated-prefix workloads.
- Prefix caching on/off, continuous batching evidence, optional KV-cache quantization if verified.
- TTFT, TPOT/inter-token latency, p50/p95/p99, input/output throughput, GPU memory, cache hit ratio,
  and cost per million tokens.
- Offline prediction versus observed vLLM metrics with the gap investigated.

## Phase 4 — portfolio polish and upstream contribution

Deliverables:

- Final 90-second README path, two or three final charts, and consolidated results table.
- Three-minute demonstration video linked from the repository.
- A substantive Ray or vLLM issue/PR focused on documentation, testing, observability, or benchmarking.
- Maintainer permission before any attempt to turn the standalone analyzer into upstream vLLM code.

