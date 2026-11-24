
### Run 2026-11-23 18:33

| Stage | Passages/s | Speed-up | Min cos | Violations | Live p50 ms | Live p99 ms | Live min cos |
|---|---|---|---|---|---|---|---|
| S1 + FP16, fused QKV, SDPA | 127.3 | 4.48 | 0.999663 | 0 |  |  |  |
| S2 + length-aware token-budget scheduler | 624.5 | 21.99 | 0.999663 | 0 |  |  |  |

### Run 2026-11-23 19:02

| Stage | Passages/s | Speed-up | Min cos | Violations | Live p50 ms | Live p99 ms | Live min cos |
|---|---|---|---|---|---|---|---|
| S3 + async host pipeline, CUDA Graph live path | 612.2 | 21.55 | 0.999663 | 0 | 2.2 | 3.67 | 0.999982 |
| S4 + CLS-only last layer | 654.6 | 23.05 | 0.999663 | 0 | 2.17 | 3.58 | 0.999982 |

### Run 2026-11-23 19:12

| Stage | Passages/s | Speed-up | Min cos | Violations | Live p50 ms | Live p99 ms | Live min cos |
|---|---|---|---|---|---|---|---|
| S4 + CLS-only last layer | 666.1 | 23.45 | 0.999663 | 0 | 2.19 | 3.6 | 0.999982 |
