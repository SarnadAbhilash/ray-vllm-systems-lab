# Offline prefix-cache analysis

Model/tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`  
Requests: 10

| Block size | Prompt tokens | Full-block tokens | Reusable tokens | Estimated hit ratio |
|---:|---:|---:|---:|---:|
| 8 | 809 | 776 | 360 | 46.4% |
| 16 | 809 | 720 | 336 | 46.7% |
| 32 | 809 | 640 | 288 | 45.0% |

## Strongest shared prefixes

### 8-token blocks

- 64 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 56 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 56 tokens shared by `support-01`, `support-02`, `support-03`, `support-batch-04`
- 48 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 48 tokens shared by `support-01`, `support-02`, `support-03`, `support-batch-04`

### 16-token blocks

- 64 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 48 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 48 tokens shared by `support-01`, `support-02`, `support-03`, `support-batch-04`
- 32 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 32 tokens shared by `support-01`, `support-02`, `support-03`, `support-batch-04`

### 32-token blocks

- 64 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 32 tokens shared by `travel-01`, `travel-02`, `travel-03`, `travel-04`
- 32 tokens shared by `support-01`, `support-02`, `support-03`, `support-batch-04`

## Interpretation limits

- Requests are evaluated in JSONL arrival order with an initially empty cache.
- The estimate assumes infinite cache capacity and no eviction, preemption, or KV offload.
- Only complete blocks are reusable; partial tail blocks are excluded.
- The standalone SHA-256 chain models block identity but does not import or claim exact vLLM internals.
- Adapter IDs and cache salts partition cache identity when present.
