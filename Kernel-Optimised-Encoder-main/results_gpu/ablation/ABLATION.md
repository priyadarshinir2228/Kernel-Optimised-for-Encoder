
### Run 2026-11-24 00:04 - ablation 20k

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 + FP16, fused QKV, SDPA | fp16 res32 | 20000 | 65536 | 85.9 | 3.02 | 0.999991 | 0 |  |  |  | 2650 |
| S2 + length-aware token-budget scheduler | fp16 sort budget bucket res32 | 20000 | 65536 | 427.3 | 15.04 | 0.999991 | 0 |  |  |  | 2650 |

### Run 2026-11-24 00:22 - ablation 20k + live

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 + async host pipeline, CUDA Graph live path | fp16 sort budget bucket overlap graph res32 | 20000 | 65536 | 426.5 | 15.02 | 0.999991 | 0 | 3.34 | 5.64 | 0.999997 | 5496 |
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 65536 | 460.4 | 16.21 | 0.999991 | 0 | 3.34 | 5.77 | 0.999997 | 5500 |
| S5 + torch.compile | fp16 sort budget bucket overlap cls_only compile graph res32 | 20000 | 65536 | 493.7 | 17.38 | 0.99986 | 0 | 3.31 | 5.65 | 0.999997 | 3228 |

### Run 2026-11-24 00:23 - RES32 off

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 + CLS-only last layer | fp16 sort budget bucket overlap cls_only graph | 20000 | 65536 | 511.9 | 18.02 | 0.999663 | 0 |  |  |  | 5556 |

### Run 2026-11-24 00:24 - O2b off

Reference throughput file: results_gpu/hardware/ref_throughput.json

| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S4 + CLS-only last layer | fp16 sort budget overlap cls_only graph res32 | 20000 | 65536 | 569.4 | 20.04 | 0.999991 | 0 |  |  |  | 2752 |
