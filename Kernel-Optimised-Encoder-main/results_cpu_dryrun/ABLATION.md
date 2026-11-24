
### Run 2026-11-23 10:15

| Stage | Passages/s | Speed-up | Min cos | Violations | Live p50 ms | Live p99 ms | Live min cos |
|---|---|---|---|---|---|---|---|
| S0 reference-like (FP32, pad 512, batch 32) | 4.0 | 1.02 | 1.0 | 0 |  |  |  |
| S2 + length-aware token-budget scheduler | 17.8 | 4.55 | 1.0 | 0 |  |  |  |
| S4 + CLS-only last layer | 18.9 | 4.83 | 1.0 | 0 |  |  |  |
