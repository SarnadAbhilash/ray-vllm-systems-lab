# Ray + vLLM Train-to-Serve Systems Lab

[![CI](https://github.com/SarnadAbhilash/ray-vllm-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/SarnadAbhilash/ray-vllm-systems-lab/actions/workflows/ci.yml)

Training and serving LLMs are usually demonstrated as separate notebooks, which hides the hard systems questions at their boundary: whether preprocessing feeds GPUs fast enough, whether distributed checkpoints actually recover, whether an adapter can move into a production engine, and whether agentic prompt structure makes KV-cache reuse predictable. This repository builds one measured lifecycle—from Ray Data through Ray Train/FSDP and LoRA checkpoints into vLLM on Ray Serve—then tests it under controlled load. Its original component is a no-GPU prefix-cache workload analyzer that estimates reusable full KV blocks from real JSONL conversations before an inference experiment spends GPU time.

> **Current milestone:** Phases 1 and 2 are complete. The offline analyzer, Ray Data pipeline,
> 1/2-GPU LoRA training, FSDP checkpoints, injected-failure recovery, raw telemetry, and charts are
> reproducible. vLLM/Ray Serve load measurements remain pending rather than simulated.

## Architecture

```mermaid
flowchart LR
    A["Instruction JSONL"] --> B["Ray Data<br/>parse, template, tokenize"]
    B --> C["Ray Train + PyTorch FSDP<br/>1 vs 2 GPU LoRA fine-tuning"]
    C --> D["Versioned checkpoint<br/>LoRA adapter + metrics"]
    D --> E["vLLM engine<br/>base + adapter"]
    E --> F["Ray Serve<br/>autoscaling API"]
    F --> G["Load generator<br/>c=1, 8, 32, 64"]
    G --> H["Prometheus + GPU metrics<br/>TTFT, ITL, latency, throughput"]
    A --> I["Offline prefix-cache analyzer<br/>tokenize + full-block hash chains"]
    I --> J["Predicted cacheability<br/>block sizes 8, 16, 32"]
    H --> K["Predicted vs observed<br/>prefix-cache hit ratio"]
    J --> K
```

## Results snapshot

Phase 2 used the same pinned Qwen 0.5B model, Dolly split, global batch, and 12-step budget on Modal
NVIDIA L4 GPUs. This is one measured run per condition; it is a bottleneck probe, not a confidence
interval.

| Training measurement | 1 × L4 | 2 × L4 FSDP |
|---|---:|---:|
| Input tokens/second | 2,734.6 | 1,256.5 |
| Peak allocated memory/device | 3.557 GiB | 3.371 GiB |
| Mean GPU utilization, full fit | 6.2% | 14.6% |
| Speedup / scaling efficiency | 1.000× / — | 0.459× / 22.97% |

The controlled worker failure resumed from the step-6 checkpoint in **21.78 seconds**, including a
**2.52-second restore path**, and replayed two optimizer steps. Held-out perplexity improved from
**8.6308 to 7.5921** and teacher-forced token accuracy from **53.52% to 54.85%**.

![One- versus two-GPU training throughput](charts/phase2/training_scaling.png)

![Held-out quality before and after LoRA](charts/phase2/training_quality.png)

The detailed interpretation, raw-artifact links, and failure analysis are in
[docs/phase-2-report.md](docs/phase-2-report.md). Phase 1's checked-in 10-request agentic sample
produced these offline upper-bound cache estimates:

| Measurement | 8-token blocks | 16-token blocks | 32-token blocks |
|---|---:|---:|---:|
| Prompt tokens | 809 | 809 | 809 |
| Full-block tokens | 776 | 720 | 640 |
| Reusable full-block tokens | 360 | 336 | 288 |
| Predicted hit ratio | 46.4% | 46.7% | 45.0% |

![Cacheability by block size](charts/cacheability_by_block_size.png)

![Reusable versus first-compute tokens](charts/token_accounting.png)

The full train-to-serve scorecard and measurement rules live in
[docs/benchmark-contract.md](docs/benchmark-contract.md). No unmeasured serving number is presented
as a result.

## Reproduce Phase 1

Prerequisites: `uv`, Python 3.10+, and internet access for the tokenizer on the first run. Model
weights and a GPU are not required.

```bash
git clone https://github.com/SarnadAbhilash/ray-vllm-systems-lab.git
cd ray-vllm-systems-lab
make install
make verify
```

Run the analyzer on another plain-prompt, OpenAI chat, or OpenAI Batch-style JSONL file:

```bash
uv run prefix-cache-lab analyze requests.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --block-sizes 8,16,32 \
  --output results/my-workload.json \
  --markdown results/my-workload.md
```

Compare a 16-token-block prediction with counters captured from a vLLM `/metrics` endpoint:

```bash
uv run prefix-cache-lab compare results/my-workload.json \
  --metrics http://localhost:8000/metrics \
  --block-size 16 \
  --output results/predicted-vs-observed.json
```

The analyzer intentionally does not import vLLM engine internals. It creates a parent-linked SHA-256
chain over complete token blocks and cache-partitioning metadata, models an initially empty cache in
JSONL arrival order, and assumes infinite capacity. That preserves the important identity and block
granularity semantics while keeping this a standalone prototype. Scheduler interleaving, eviction,
preemption, memory pressure, and KV offload can all lower the observed hit rate.

## Reproduce Phase 2

Prerequisites: a configured Modal account. The command creates one- and two-L4 functions, runs them
sequentially, and keeps checkpoints in named persistent Volumes:

```bash
uv sync --extra dev --extra charts --extra cloud
uv run modal run infra/modal_app.py \
  --experiment-id phase2-reproduction \
  --output results/phase2/raw/modal-results.json
uv run python scripts/plot_training.py results/phase2/raw/modal-results.json \
  --output-dir charts/phase2
```

The checked-in measurement is tied to implementation commit `c237d74`; hashes and adapter integrity
checks are in [results/phase2/provenance.json](results/phase2/provenance.json).

## Lifecycle scorecard

| Capability | Evidence | Status |
|---|---|---|
| JSONL conversation parsing and exact chat-template tokenization | Analyzer CLI + sample corpus | **Complete** |
| Repeated-prefix discovery and block-size comparison | JSON/Markdown reports + charts | **Complete** |
| Prediction versus vLLM Prometheus counters | `compare` command | **Complete (GPU observation pending)** |
| Ray Data preprocessing | Revision-pinned distributed tokenization + raw stats | **Complete** |
| Ray Train + PyTorch FSDP LoRA | Measured 1/2-L4 runs + validated adapters | **Complete** |
| Interrupted-run recovery | Step-8 failure, step-6 restore, timed resume | **Complete** |
| vLLM + Ray Serve base/adapter serving | Deployment and smoke tests | Phase 3 |
| Prefix caching, concurrency, prompt-length matrix | Load-test report and raw request traces | Phase 3 |
| Three-minute demo video | Linked recording | Phase 4 |
| Ray/vLLM upstream issue or PR | Maintainer-reviewed contribution | Phase 4 |

## Bottleneck investigation

The strongest measured bottleneck is model scale: two-GPU FSDP is slower than one GPU for this 0.5B,
short-sequence workload. Collective and orchestration costs dominate the few seconds of training
compute, yielding 0.459× speedup; Ray Data actor/tokenizer startup similarly dominates only 160 rows.
The offline analyzer separately shows that prefix reuse is constrained by full-block alignment and
cache namespaces. See the training investigation in
[docs/phase-2-report.md](docs/phase-2-report.md#bottleneck-investigation) and cache analysis in
[docs/phase-1-report.md](docs/phase-1-report.md#bottleneck-investigation).

## What failed and why

The most consequential Phase-2 failure was a recovery run that looked successful while PEFT warned
that every adapter key was missing. FSDP state had been filtered twice, creating a valid but empty
safetensors file. The save path now filters exactly once; all final adapters were downloaded and
verified to contain 192 tensors. Other integration failures covered obsolete Ray Train V1 arguments,
mixed FSDP parameter dtypes, and autograd-incompatible inference views. The complete evidence is in
[docs/phase-2-report.md](docs/phase-2-report.md#what-failed-and-why).

## Design notes

- Target model: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), small enough for budget-conscious 1/2-GPU comparisons while retaining a real chat template and LoRA-serving path.
- The analyzer is inspired by, but independent from, the open [vLLM offline prefix-cache analyzer RFC](https://github.com/vllm-project/vllm/issues/47993). It neither copies an implementation nor claims compatibility with a future vLLM CLI.
- vLLM exposes token-counting counters for [`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`](https://docs.vllm.ai/en/latest/design/metrics/), which is why the comparison uses token hit ratio rather than request hit ratio.
- Ray Train recovery follows the current [Train V2 fault-tolerance pattern](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html), using persistent checkpoints and a stable run name rather than deprecated `Trainer.restore` APIs.
- Training uses Modal single-node 1/2-GPU functions and persistent Volumes; the serving phase will reuse that infrastructure. Modal's multi-GPU container behavior is documented in its [GPU guide](https://modal.com/docs/guide/gpu).

## Repository map

```text
src/ray_vllm_lab/analyzer/  standalone analyzer, report, and metrics comparison
src/ray_vllm_lab/training/  Ray Data, Ray Train V2, FSDP/LoRA, checkpoint recovery
infra/                      Modal GPU functions and pinned training environment
data/sample/                 agentic JSONL and synthetic Prometheus fixture
results/                     raw manifests, provenance, and human-readable reports
charts/                      generated cache and training evidence
tests/                       deterministic unit tests (no network or GPU)
docs/                        benchmark contract, phase report, demo, upstream plan
```

## Roadmap

The phases, acceptance criteria, and artifact boundaries are in [docs/roadmap.md](docs/roadmap.md).
The next phase is production inference: serve the base model and validated LoRA adapter through vLLM
and Ray Serve, run the declared caching/concurrency/prompt matrix, and compare the analyzer's static
prediction with isolated vLLM cache counters.

## License

MIT
