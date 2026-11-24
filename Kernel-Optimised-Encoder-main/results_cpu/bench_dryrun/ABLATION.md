
### Run 2026-11-23 22:10 - CPU dry run

Reference throughput file: results_cpu\ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 reference-like (FP32, pad 512, batch 32) |  | 300 | 8192 | 3.5 | 0.97 | 1.0 | 0 |  |  |  |  |
| S2 + length-aware token-budget scheduler | fp16 sort budget bucket res32 | 300 | 8192 | 17.8 | 4.94 | 1.0 | 0 |  |  |  |  |
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph res32 | 300 | 8192 | 18.6 | 5.17 | 1.0 | 0 |  |  |  |  |

### Run 2026-11-23 22:19 - CPU dry run, after small-call plan fix

Reference throughput file: results_cpu/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2 + length-aware token-budget scheduler | fp16 sort budget bucket res32 | 300 | 8192 | 18.2 | 5.05 | 1.0 | 0 |  |  |  |  |
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph res32 | 300 | 8192 | 19.1 | 5.32 | 1.0 | 0 |  |  |  |  |
