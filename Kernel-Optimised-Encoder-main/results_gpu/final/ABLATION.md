
### Run 2026-11-24 04:56 - final 100k

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph short32<=32 | 100000 | 8192 | 604.9 | 21.3 | 0.999999 | 0 | 4.4 | 8.24 | 1.0 | 1622 |
