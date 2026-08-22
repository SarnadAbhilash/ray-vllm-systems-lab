# Phase 1 report — offline prefix-cache workload analyzer

## Outcome

Phase 1 delivers a standalone, no-GPU preflight tool for agentic JSONL workloads. It parses plain
prompts, OpenAI chat rows, and OpenAI Batch request bodies; renders chat templates with the selected
Hugging Face tokenizer; compares 8/16/32-token KV block sizes; identifies shared prefixes and
divergence points; accounts for adapter and cache-salt namespaces; emits JSON and Markdown; and
compares its prediction with vLLM Prometheus counters.

The result is an upper bound under an infinite cache, not a simulator of vLLM scheduling. That
boundary is part of the design rather than a footnote: Phase 3 will quantify where observed runtime
behavior departs from the static estimate.

## Evidence

- `tests/` exercises arrival-order reuse, full-block accounting, namespace partitioning, all supported
  input shapes, chat tokenization, Prometheus parsing, and predicted/observed comparison.
- `data/sample/agentic_requests.jsonl` contains two repeated-prefix agent families plus intentional
  adapter and cache-salt partitions.
- `results/sample/prefix-cache-report.json` is the stable, machine-readable output.
- `results/sample/prefix-cache-report.md` is the same evidence formatted for inspection.
- `charts/cacheability_by_block_size.png` and `charts/token_accounting.png` are generated only from
  that JSON report.

| Block size | Prompt tokens | Full-block tokens | Reusable tokens | Predicted hit ratio |
|---:|---:|---:|---:|---:|
| 8 | 809 | 776 | 360 | 46.4% |
| 16 | 809 | 720 | 336 | 46.7% |
| 32 | 809 | 640 | 288 | 45.0% |

## Bottleneck investigation

The offline workload has three sources of lost reuse:

1. **First-compute cost.** The first request in each prompt family must populate every block. Static
   reuse is inherently arrival-order dependent.
2. **Block granularity.** Only complete blocks are cache candidates. Increasing block size reduces
   hash/query overhead but makes the reusable boundary coarser and grows non-cacheable prompt tails.
3. **Namespace fragmentation.** vLLM-style cache identity must distinguish LoRA adapters and cache
   salts. The sample includes one adapter-specific support request and one tenant-salted travel
   request so the analyzer cannot report attractive but invalid cross-namespace reuse.

The key Phase-3 experiment is whether runtime hit ratio tracks the predicted ranking across block
sizes and workload shapes. It may not track the absolute value because concurrent requests compete
for finite cache capacity and scheduling changes which blocks are resident when queried.

## What failed and why

| Failure | Cause | Resolution | Lesson carried forward |
|---|---|---|---|
| Initial `uv sync` could not build the editable package | Package metadata referenced `README.md` before repository documentation had been created | Added the README, then repeated a clean dependency sync | CI must build from repository state, never from an assumed file-generation order |
| First real chat-template run raised an optional-dependency error | Transformers does not install Jinja by default, but `apply_chat_template` requires it | Declared Jinja as a direct runtime dependency and regenerated the lockfile | A tokenizer-only path still needs an integration test against the selected production tokenizer |
| GPU train/serve metrics are absent in Phase 1 | They have not been run; inventing or extrapolating them would undermine the benchmark | Kept every GPU field explicitly pending until controlled Modal runs produce raw artifacts | Every headline number needs a command, raw artifact, hardware descriptor, and commit SHA |

## Interpretation limits

- The hash chain is an independent SHA-256 prototype; it models parent-linked block identity but does
  not import vLLM private helpers or promise bit-for-bit hash compatibility.
- The cache begins empty, requests arrive in file order, and previously computed blocks remain forever.
- Partial blocks, eviction, preemption, offload, scheduler interleaving, and multi-instance routing are
  excluded.
- Text-only prompts are supported in Phase 1. Multimodal cache keys are rejected explicitly.
- Observed counters aggregate matching metric series. Production comparisons should isolate a model,
  replica, and experiment window before calculating the ratio.

## Next decision gate

Phase 2 is accepted only when one command can run the same LoRA fine-tuning configuration on one and
two GPUs, a controlled worker interruption resumes from a persistent checkpoint, and the report
contains raw tokens/second, peak allocated GPU memory, scaling efficiency, recovery time, and fixed
before/after quality results.
