# End-to-end operational walkthrough

This walkthrough is completed alongside the serving phase. It provides a compact validation path for
the entire system without replacing the exact reproduction commands or raw artifacts.

## 1. Architecture and inputs

Start with the README architecture, then inspect the pinned model and dataset revisions. Confirm that
Ray Data produces materialized training and evaluation shards with prompt tokens masked from loss.

## 2. Training and recovery

Run the same configuration on one and two GPUs. Inspect the persistent LoRA checkpoint, adapter tensor
inventory, throughput, memory, and utilization. Run the controlled failure probe and confirm that Ray
restores from the last durable checkpoint, reports replayed optimizer steps, and completes evaluation.

## 3. Serving and load behavior

Deploy the base model and named LoRA adapter through Ray Serve and vLLM. Query both paths, then run the
fixed concurrency and prompt-shape matrix. Inspect TTFT, TPOT, request latency percentiles, token
throughput, GPU memory, and continuous-batching behavior.

## 4. Prefix-cache validation

Run the offline analyzer on the agentic JSONL workload. Compare block sizes, full-block alignment, and
cache namespace fragmentation. Capture isolated vLLM prefix-cache counters and compare observed token
hit ratio with the offline prediction.

## 5. Reproducibility checks

Finish by verifying the source commit, dependency versions, hardware manifest, raw result hashes,
tests, and chart regeneration. Any failed experiment that changes the implementation or interpretation
belongs in the relevant phase report.
