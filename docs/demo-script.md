# Three-minute demo script (recording plan)

This is the shot list for Phase 4, not a claim that the final video has already been recorded.

**0:00–0:25 — problem and architecture.** Show the README problem statement and trace the path from
Ray Data to Ray Train/FSDP, checkpoint/LoRA, vLLM, Ray Serve, and observability.

**0:25–0:55 — GPU training evidence.** Launch or replay the same one-command run for one and two GPUs.
Show the persistent checkpoint, injected failure, recovery log, and the training/scaling table.

**0:55–1:30 — serving evidence.** Query both the base model and named LoRA adapter. Show streamed tokens
and then the concurrency/prompt-shape matrix with TTFT, TPOT, tail latency, and throughput.

**1:30–2:25 — systems differentiator.** Run the offline analyzer on the agentic JSONL, explain full-block
alignment and namespace fragmentation, display the block-size charts, then compare the prediction
with an isolated vLLM `/metrics` capture.

**2:25–2:50 — bottleneck and failure.** Show the largest predicted/observed gap, the trace that explains
it, and one failed experiment retained in the artifact index.

**2:50–3:00 — reproducibility.** End on the exact reproduction commands, commit SHA, hardware manifest,
and upstream issue/PR link.

