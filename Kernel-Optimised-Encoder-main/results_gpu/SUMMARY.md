# PS-09 results summary (results_gpu)

Generated from result files by scripts/collect_results.py.

## Hardware and baseline (measured)

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute 8.6, 5.69 GiB, 20 SMs, BF16 True)
- Software: Python 3.14.10, torch 2.15.0+cu132 (CUDA 13.2), transformers 5.24.0, tokenizers 0.23.3; OS Linux-7.3.5-arch1-1-x86_64-with-glibc2.44
- Reference baseline: **28.4 passages/s** (reference.py, FP32, pad 512, first 5000 passages, 176 s)

## Accuracy on the official references (measured)

| Run | Passages min cos | Passages mean | Queries min cos | Max abs err | Verdict |
|---|---|---|---|---|---|
| validate_default | 0.999999 | 0.9999997 | 1.000000 | 0.00055 | PASS |
| validate_fp16_plain | 0.999663 | 0.9999993 | 0.999982 | 0.00316 | PASS |
| validate_bf16 | 0.997342 | 0.9999801 | 0.998891 | 0.00768 | FAIL |
| validate_lax_reduction | 0.999991 | 0.9999998 | 0.999997 | 0.00062 | PASS |
| validate_fp16_accum | 0.999636 | 0.9999962 | 0.999972 | 0.00325 | PASS |

Edge-case suite: ALL PASS

## Ablation (official bench.py)

| Stage | Tag | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 | Live p99 | Peak MiB | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ablation 20k | fp16 res32 | 20000 | 65536 | 85.9 | 3.02 | 0.999991 | 0 |  |  | 2650 | ok |
| 2 | ablation 20k | fp16 sort budget bucket res32 | 20000 | 65536 | 427.3 | 15.04 | 0.999991 | 0 |  |  | 2650 | ok |
| 3 | ablation 20k + live | fp16 sort budget bucket overlap graph res32 | 20000 | 65536 | 426.5 | 15.02 | 0.999991 | 0 | 3.34 | 5.64 | 5496 | ok |
| 4 | ablation 20k + live | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 65536 | 460.4 | 16.21 | 0.999991 | 0 | 3.34 | 5.77 | 5500 | ok |
| 5 | ablation 20k + live | fp16 sort budget bucket overlap cls_only compile graph res32 | 20000 | 65536 | 493.7 | 17.38 | 0.99986 | 0 | 3.31 | 5.65 | 3228 | ok |
| 4 | RES32 off | fp16 sort budget bucket overlap cls_only graph | 20000 | 65536 | 511.9 | 18.02 | 0.999663 | 0 |  |  | 5556 | ok |
| 4 | O2b off | fp16 sort budget overlap cls_only graph res32 | 20000 | 65536 | 569.4 | 20.04 | 0.999991 | 0 |  |  | 2752 | ok |

## Token-budget sweep (official bench.py, stage 4)

| Stage | Tag | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 | Live p99 | Peak MiB | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | budget sweep | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 8192 | 621.1 | 21.87 | 0.999991 | 0 |  |  | 736 | ok |
| 4 | budget sweep | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 16384 | 626.6 | 22.06 | 0.999991 | 0 |  |  | 1096 | ok |
| 4 | budget sweep | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 32768 | 623.5 | 21.95 | 0.999991 | 0 |  |  | 2326 | ok |
| 4 | budget sweep | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 65536 | 617.1 | 21.73 | 0.999991 | 0 |  |  | 5500 | ok |
| 4 | budget sweep | fp16 sort budget bucket overlap cls_only graph res32 | 20000 | 131072 | 609.5 | 21.46 | 0.999991 | 0 |  |  | 4958 | ok |

## Final run (official bench.py)

| Stage | Tag | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 | Live p99 | Peak MiB | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | final 100k | fp16 sort budget bucket overlap cls_only graph short32<=32 | 100000 | 8192 | 604.9 | 21.3 | 0.999999 | 0 | 4.4 | 8.24 | 1622 | ok |

## Live-query latency detail (non-official replay of bench.py timing)

| Run | p50 ms | p90 | p95 | p99 | max | service p50 | service p99 | min cos |
|---|---|---|---|---|---|---|---|---|
| realtime_default | 4.31 | 4.61 | 6.16 | 8.17 | 10.94 | 4.17 | 4.42 | 1.000000 |
| service_graph_on | - | - | - | - | - | 3.71 | 3.82 | 1.000000 |
| service_graph_off | - | - | - | - | - | 4.05 | 4.61 | 1.000000 |

## Profiler (torch.profiler, 2,000 passages; wall time includes profiler overhead)

| Run | GPU kernel ms | GEMM | Attention | Norm+elementwise | Memcpy | Other | Device busy | Idle gaps >100us | Tokenise share |
|---|---|---|---|---|---|---|---|---|---|
| reference | 139886 | 46.3% | 0.0% | 3.5% | 0.0% | 50.2% | 1.000 | 0 | 0.007 |
| stage1 | 35219 | 38.0% | 0.0% | 12.0% | 0.0% | 50.1% | 1.000 | 0 | 0.003 |
| stage2 | 7486 | 37.4% | 0.0% | 12.5% | 0.0% | 50.1% | 1.000 | 0 | 0.013 |
| default | 6661 | 43.1% | 0.0% | 6.8% | 0.0% | 50.1% | 1.000 | 0 | 0.014 |

## Full 100k accuracy check (non-official)

- Submitted config vs FP32 path on all 100,000 passages: min cos **0.999992**, mean 0.9999997, max abs error 0.00064, violations (<0.999) **0**, below 0.9995: 0
- FP32 path vs official reference (first 2,000): min cos 1.000000
- Worst input: #86483 (41 tokens) cos 0.999992

