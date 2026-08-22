# Upstream contribution plan

The standalone analyzer is related to vLLM issue
[#47993](https://github.com/vllm-project/vllm/issues/47993), but this repository does not copy its
proposed CLI or represent work as an upstream implementation. The RFC specifically raises a layering
question around private KV-cache hashing helpers; importing those helpers here would pre-empt the
maintainers' design decision.

Before upstream code is attempted:

1. Complete the local predicted-versus-observed study and publish the workload, assumptions, and gap.
   **Complete:** 24 cache-on conditions produced 0.28 percentage points mean absolute error and 0.95
   points maximum error; the raw manifests and limitations are checked in.
2. Ask maintainers whether the experiment is useful and whether a stable helper/API is wanted.
3. Prefer a bounded first contribution: analyzer test cases, metrics documentation, a benchmark corpus
   schema, or observability guidance.
4. Open code only after maintainers confirm direction; link the resulting issue/PR and discussion in
   the final repository.

This keeps the prototype independent while respecting the active design discussion.
