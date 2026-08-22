# Phase 2 training results

Measured on 2026-08-22 with NVIDIA L4 GPUs. Both scaling runs used the same model revision,
dataset revision, global batch size of 8, 12 optimizer steps, and 192-token sequence cap. Throughput
excludes model load, evaluation, checkpoint I/O, and two warm-up steps. This is one measured run per
condition, so the values are a transparent systems probe rather than a statistically stable model
benchmark.

| Measurement | 1 × L4 | 2 × L4 FSDP | Recovery probe |
|---|---:|---:|---:|
| Training input tokens/s | 2,734.6 | 1,256.5 | 1,263.6 |
| Peak allocated GPU memory/device | 3.557 GiB | 3.371 GiB | 3.373 GiB |
| Peak process GPU memory/device | 4.615 GiB | 4.305 GiB | 4.418 GiB |
| Mean GPU utilization, full fit | 6.2% | 14.6% | 12.5% |
| Mean GPU utilization while active | 26.8% | 43.6% | 43.2% |
| Ray Train fit wall time | 43.2 s | 61.6 s | 71.1 s |
| Restore path (model + adapter + optimizer) | — | — | 2.52 s |
| Failure-to-first-recovered-step | — | — | 21.78 s |
| Replayed optimizer steps | — | — | 2 |

Two-GPU speedup was **0.459×** and scaling efficiency was **22.97%**. FSDP reduced peak allocated
memory by 5.2% per device, but the model was too small to amortize collectives, parameter-view
management, and duplicated orchestration.

| Held-out quality | Base | Two-GPU LoRA | Change |
|---|---:|---:|---:|
| Cross-entropy loss | 2.1553 | 2.0271 | -5.95% |
| Perplexity | 8.6308 | 7.5921 | -12.03% |
| Teacher-forced next-token accuracy | 53.52% | 54.85% | +1.33 pp |

Ray Data materialized 128 training and 32 evaluation examples containing 21,345 input tokens. The
one-GPU run measured 26.92 seconds end to end (5.94 examples/s, 792.8 tokens/s); actor startup and
tokenizer loading dominated the four small 32-row batches.

Raw telemetry and all `nvidia-smi` samples are in
[`raw/modal-results.json`](raw/modal-results.json). The complete per-run manifests live under
[`raw/manifests/`](raw/manifests/), and [`provenance.json`](provenance.json) records hashes, the exact
benchmark source commit, remote adapter paths, and adapter tensor integrity.
