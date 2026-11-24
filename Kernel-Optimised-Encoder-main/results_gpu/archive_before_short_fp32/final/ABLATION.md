
### Run 2026-11-24 02:45 - final 100k

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph res32 | 100000 | 8192 | 621.3 | 21.87 | 0.999991 | 0 | 2.48 | 3.95 | 0.999997 | 970 |
