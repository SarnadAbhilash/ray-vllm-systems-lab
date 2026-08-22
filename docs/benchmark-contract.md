# Benchmark contract

This document defines the scorecard before GPU runs so results cannot be selected after the fact.

## Training matrix

| Dimension | Values |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA, fixed rank/alpha/dropout across runs |
| Workers / GPUs | 1 and 2, one GPU per Ray Train worker |
| Parallelism | PyTorch FSDP with PEFT-aware auto-wrapping and `use_orig_params=True` |
| Data | Same committed train/eval split and maximum sequence length |
| Optimizer work | Same global batch size and number of optimizer steps |
| Repetitions | One warm-up plus three measured runs where budget permits |
| Recovery | Inject exactly one failure after a durable checkpoint |

Training metrics:

- **Training tokens/second:** non-padding input tokens processed divided by measured training wall time,
  excluding model/tokenizer download and Ray startup.
- **Peak GPU memory:** maximum of `torch.cuda.max_memory_allocated()` across workers; `nvidia-smi` is
  captured separately for whole-process context.
- **Two-GPU scaling efficiency:** `throughput_2gpu / (2 * throughput_1gpu)`.
- **Checkpoint recovery time:** wall time from injected failure detection until the first successful
  post-restore optimizer step.
- **Quality:** exact-match and token-level F1 on one frozen evaluation file, measured with the same
  deterministic generation settings for base and adapter.

## Serving matrix

| Dimension | Values |
|---|---|
| Model target | Base and LoRA adapter |
| Prefix caching | Off and on |
| Concurrency | 1, 8, 32, 64 |
| Prompt shape | Short, long, repeated-prefix agentic |
| Generation | Fixed deterministic output cap per prompt class |
| KV cache | Default dtype; optional FP8 only when hardware and model support are verified |

Serving metrics:

- **TTFT:** client send time to first streamed token.
- **Inter-token latency / TPOT:** mean time between streamed output tokens after the first token.
- **Request latency:** p50, p95, and p99 over completed non-warm-up requests.
- **Throughput:** input and output tokens divided by steady-state wall time.
- **Prefix-cache hit ratio:** increase in `vllm:prefix_cache_hits` divided by the increase in
  `vllm:prefix_cache_queries` over the isolated run.
- **GPU memory:** peak device memory during steady state, with model load separated from KV growth.
- **Estimated cost per million tokens:** measured GPU-seconds multiplied by the recorded provider
  price at run time, divided by total tokens, with input/output counts shown separately.

## Required provenance per run

Every raw result must include the Git commit, UTC timestamp, image/dependency lock hash, GPU type and
count, model/tokenizer revisions, CLI arguments, random seed, warm-up policy, request count, failures,
and path to raw logs. Failed runs remain in the artifact index.
