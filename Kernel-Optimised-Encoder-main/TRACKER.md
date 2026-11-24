# PS-09 implementation tracker

Status key: ✅ completed and verified · 🧪 implemented, verified on CPU only · ⏳ GPU validation pending ·
⛔ blocked · ❌ failed / rejected · ⬜ not started.
Evidence labels: **measured** (file named) · **CPU emulation** · **reported** (teammate, not reproduced) · **estimate**.

## Phases

| # | Phase | Status | Evidence | Result |
|---|---|---|---|---|
| 0 | Analysis and planning | ✅ | `docs/*.pdf` | Mean 100.2 tokens; 80.4% padding (re-measured exactly, `results_cpu/workload_stats.json`) |
| 1 | Code and correctness harness | ✅ | `team.py`, `validate.py`, `tests/test_correctness.py` | CPU FP32: 34/34 edge-case checks pass, min cos 1.0000000 (stages 0, 2, 4) |
| 2 | GPU baseline and "before" profile | ✅ | `results_gpu/hardware/ref_throughput.json`, `results_gpu/profiler/reference/` | 28.4 p/s (RTX 3050, 5,000 passages, recovered file); 75% FP32 GEMM, 136 idle gaps >100 µs |
| 3 | S1: FP16, fused QKV, SDPA | ✅ | `results_gpu/ablation/` | 85.9 p/s (3.0x), 20k |
| 4 | S2: token-budget scheduler (+ O2b bucket split) | ✅ | `results_gpu/ablation/`, `results_gpu/repeats/` | 427 p/s (15x) at budget 65,536; O2b +2.9% (3 interleaved rounds) |
| 5 | S3: async pipeline + CUDA Graph live path | ✅ | `results_gpu/ablation/`, `results_gpu/latency/` | FP16 query service 2.49 → 1.76 ms with graphs |
| 6 | S4: CLS-only last layer | ✅ | `results_gpu/ablation/` | exact; +8% in the 20k ablation |
| 7 | S5: torch.compile | ❌ | `results_gpu/ablation/` | 494 p/s but setup 197 s and min cos 0.99986; not submitted |
| 8 | Precision (P1 → P2) | ✅ | `results_gpu/final/full_check*.json`, `results_gpu/repeats_p2/` | 100k check: plain FP16 2 violations, FP32 residual 1; **P2 (≤ 32 tokens in FP32): 0, min 0.999992** |
| 9 | Budget sweep, full 100k validation, "after" profiles | ✅ | `results_gpu/token_budget/`, `results_gpu/final/`, `results_gpu/profiler/` | budget 8,192; **604.9 p/s, 21.3x** on 100k; 7 idle gaps, 99.7% busy |
| 10 | Report, demo, Q&A | ✅ | `reports/PS09_Final_Report.pdf`, `results_gpu/demo_gpu_output.txt`, `docs/judge_qa.md` | GPU demo: 20.6x on 2,000, min cos 0.999999, live p50 3.72 ms |

## Task list (session 2026-11-23, Windows laptop, no GPU)

| Task | Status | Evidence / files | Notes / blocker |
|---|---|---|---|
| Windows dev environment | ✅ | `.venv`, `requirements.txt`, `requirements-verified-windows-cpu.txt` | Python 3.12.10 (winget, user scope), torch 2.15.0+cpu, transformers 5.24.0 |
| Official harness read and contract checked | ✅ | `data/bench.py`, `data/reference.py` | setup() untimed; encode(all) timed incl. tokenisation; live replay FIFO one query per call |
| Edge cases vs reference.py (CPU) | ✅ | `results_cpu/test_correctness.json` | lengths 2-512 at every bucket boundary, empty, Unicode, >512, duplicates, ordering, repeated calls |
| Accuracy gate (CPU, default config) | ✅ | `results_cpu/validate_default_300.json` | 300 passages + 100 queries, min cos 1.000000 (FP32 on CPU) |
| Bug: debug read during graph capture (team.py) | ✅ | `team.py` `Encoder.forward` | still guarded by `is_current_stream_capturing()` |
| Fix: producer thread could hang after an error | 🧪 | `team.py` `_put`, `_encode` | stop event; thread always joins |
| Fix: index copy was not truly async | 🧪 | `team.py` `_build` | index tensor now pinned; chunk offset added before pinning |
| Fix: small calls padded heavily (O2b min fill) | 🧪 | `team.py` `_plan` | min batch before a bucket split capped at a quarter of the call's tokens: 200 passages 43.8% → 10.1% padding; 100k unchanged |
| Robust setup on unknown GPUs | ✅ | `team.py` `setup`, `_warmup`, `runtime_info` | ran on the RTX 3050 with no variables set; warm-up peak 0.94 GiB at budget 8,192. Note: the 75% check uses allocated memory at warm-up, while nvidia-smi peaks include the allocator cache and graphs, so at 65,536 the device peak was ~90% without halving - one reason 8,192 was chosen |
| O2b length-bucket split | ✅ | `team.py` `_plan`; `results_gpu/repeats/` | padding 10.0% → 3.7% (exact); GPU +2.9% (609.6 → 627.6 p/s, 3 rounds) |
| P1 FP32 residual stream | ❌ superseded | `team.py` `Encoder(res32)`, `results_gpu/archive_before_short_fp32/` | GPU: worst on official 2,000 0.999663 → 0.999991, but 100k check still 1 violation; 12% slower. Replaced by P2 |
| cuBLAS reduced-precision reductions off | ✅ | `results_gpu/accuracy/validate_lax_reduction.json` | no measurable effect on this GPU (0.999991 both ways); kept off as a safe default |
| BF16 | ❌ | `results_cpu/precision_study.json` | min cos 0.9918, 8/2000 under 0.9995 (CPU emulation) |
| Issue A: GPU artifacts missing | ✅ | `results_gpu/earlier_code/`, `results_gpu/hardware/` | recovered from `~/Desktop/ps09/results/` on the RTX machine (earlier code: 666 p/s, 23.45x on 100k with plain FP16) |
| Issue B: run_all.ps1 ran stage 0 on 100k | ✅ | `run_all.ps1` | now the Windows CPU workflow; never runs stage 0 on 100k; hands over to run_all.sh if a CUDA GPU exists |
| Issue C: stale installer | ✅ | `scripts/build_installer.py`, `install_ps09.sh`, `.gitattributes` | installer generated from canonical files, LF endings; `--check` run by run_all.ps1 |
| Issue D: GPU tests | ✅ | `results_gpu/accuracy/test_correctness.json` | 29 edge cases on GPU, stages 1-4: all pass |
| Linux GPU workflow | ✅ | `run_all.sh` | ran end to end on the RTX 3050 (Arch Linux), quick path and `--full`, exit 0 |
| Report generator + PDF | ✅ | `scripts/make_report.py`, `reports/` | rebuilt on Linux with GPU results (Brave headless for the PDF); profiler breakdown fixed (annotation double-count, attention mis-classified as GEMM) |
| Demo script | 🧪 | `demo.py`, `results_cpu/demo_cpu_output.txt` | run on CPU (`--n 200`): all sections OK; GPU run pending |

## Results log

| When (IST) | Where | What | Result | Label |
|---|---|---|---|---|
| 23 Nov, 15:40 | Cloud CPU, FP32 | `validate.py` stage 4, 300 passages + 60 queries | min cos 1.000000 / 1.000000 → PASS | reported |
| 23 Nov, 15:45 | Cloud CPU, FP32 | `validate.py` stages 0 and 2, 100 passages + 20 queries | min cos 1.000000 → PASS | reported |
| 23 Nov, 15:55 | Cloud CPU, FP32 | Official `bench.py` via `run_ablation.py`, first 300 passages, budget 8192 | S0 4.0 p/s (1.02×) · S2 17.8 p/s (4.55×) · S4 18.9 p/s (4.83×); files in `results_cpu_dryrun/` | reported (files in repo) |
| 23 Nov, 16:55 | Linux RTX machine | `run_all.sh` steps 1-4 | OK; step 5 crashed (debug read during graph capture), fixed | reported |
| 23 Nov, 17:00 | Cloud CPU, FP32 | Re-test after fix, `PS09_DEBUG=1` | PASS, max abs activation 16.7 | reported |
| 23 Nov, 17:42 | RTX 3050 6GB Laptop, Linux | reference baseline (5,000 passages, FP32, pad 512) | **28.4 passages/s** (176 s) | reported (file not in repo) |
| 23 Nov, 18:05 | RTX 3050 6GB Laptop, Linux | `validate.py` stage 4 (FP16), 2,000 passages + 1,500 queries | passages min **0.999663**, queries min **0.999982**; max activation 16.4 → PASS | reported (file not in repo) |
| 23 Nov, 18:10 | RTX 3050 6GB Laptop, Linux | Ablation stage 0 on 100k | stopped (≈1 h at 28 p/s) | reported |
| 23 Nov, 21:52 | Windows i7-13620H, CPU FP32 | `tests/test_correctness.py` stages 0/2/4, 29 edge cases | 34/34 checks pass, min cos 1.0000000 | measured |
| 23 Nov, 22:10 | Windows, CPU emulation | `scripts/precision_study.py`, 2,000 passages + 1,500 queries | FP16 min 0.999879 · FP16+res32 0.999967 · last-layer FP32 no gain · BF16 0.991831 (fail) | CPU emulation |
| 23 Nov, 22:20 | Windows, CPU | `scripts/workload_stats.py`, 100k passages | padding: reference 80.4%, S4 old 10.0% → O2b 3.7%; work reduction 5.25x → 5.62x | measured (exact) |
| 23 Nov, 22:30 | Windows, native CPU FP16 | `tests/test_lowprec_cpu.py`, 13 passages incl. worst 8 | FP16 min 0.999680 (#19, 8 tokens) · FP16+res32 0.999960 | measured |
| 23 Nov, 22:10 | Windows i7-13620H, CPU FP32 | Official `bench.py` via `run_ablation.py`, 300 passages, budget 8192 | S0 3.5 p/s (0.97×) · S2 17.8 (4.94×) · S4 18.6 (**5.17×**); 0 violations, min cos 1.0 | measured (not a GPU score) |
| 23 Nov, 22:50 | Windows i7-13620H, CPU FP32 | same, after the small-call plan fix | S2 18.2 p/s (5.05×) · S4 19.1 p/s (**5.32×**); 0 violations, min cos 1.0; files `results_cpu/bench_dryrun/` | measured (not a GPU score) |
| 24 Nov, 00:37 | RTX 3050 6GB Laptop, Linux | `run_all.sh` quick path (tests, accuracy, ablation, sweep, latency, profiles) | all OK except BF16 (expected fail) | measured |
| 24 Nov, 01:40 | RTX 3050, Linux | 3 interleaved rounds, budget 16,384 | default 627.6 · RES32 off 701.9 · O2b off 609.6 p/s | measured |
| 24 Nov, 03:00 | RTX 3050, Linux | `run_all.sh --full`, P1 config | 621.3 p/s (21.9x); 100k check **1 violation** (0.99779) | measured, archived |
| 24 Nov, 05:40 | RTX 3050, Linux | `run_all.sh --full`, P2 config (submitted) | **604.9 p/s (21.3x)**, 0 violations, 100k min 0.999992; live p50 4.4 / p99 8.2 ms | measured |
| 24 Nov, 06:30 | RTX 3050, Linux | 3 interleaved rounds, budget 8,192 | P2 598.7 · P1 619.0 · plain FP16 692.2 p/s | measured |
| 24 Nov, 06:50 | RTX 3050, Linux | `demo.py` | 20.6x on 2,000; min cos 0.999999; live p50 3.72 ms | measured |
| 23 Nov, 22:45 | Windows, CPU | `demo.py --n 200 --queries 30` | min cos 1.000000; 4.90× vs reference on 200 passages; padding 79.8% → 10.1%; `results_cpu/demo_cpu_output.txt` | measured (not a GPU score) |

## Decisions

- **Submitted config (24 Nov):** stage 4, token budget 8,192, P2 short inputs (≤ 32 tokens) in FP32, FP32 residual off.
  Official bench.py on 100k: 604.9 p/s (21.3x), 0 violations; all 100k ≥ 0.999992 vs FP32; live p50 4.4 / p99 8.2 ms.
- **P2 replaces P1** (24 Nov): the full 100k check found 1 violation with the FP32 residual (0.99779, 16 tokens) and 2
  with plain FP16; every FP16 result < 0.9999 was ≤ 23 tokens. P2 gives 0 at 598.7 p/s vs 619.0 (P1) and 692.2 (plain),
  3 interleaved rounds each.
- **Live queries in FP32** (24 Nov): +2 ms p50 accepted for zero precision risk on the hidden trace.
- **Budget 8,192** (24 Nov): sweep within 1% of the fastest, a third of the memory; 65,536 used ~90% of VRAM and timings
  drifted 34% between identical runs.
- *Superseded:*
  **FP32 residual stream on by default** (P1): native FP16 kernels lose more precision than ideal rounding, the worst
  inputs are short passages, and one violation zeroes the run. Cost is element-wise only (estimate 5-10%); the GPU
  ablation row "RES32 off" measures it. Revisit only if the 100k check of plain FP16 stays above 0.9995 AND the speed
  gain is > 3%.
- **O2b on** in stages 2-5: exact 7% less work at the default budget.
- O6 (padding-free varlen attention) dropped: residual padding after O2/O2b is 3.7%.
- Live path: `bench.py` calls `encode([q])` one at a time, so no cross-query batching. A CUDA Graph per bucket
  (32/64/128) is used; all practice queries fit the 32 bucket (max 25 tokens).
- Baseline: the 5,000-passage reference sample is valid because `reference.py`'s cost per passage is independent of
  the text (pad 512). The 1-hour full reference run is not repeated.
- The internal accuracy gate is a minimum cosine ≥ 0.9995; the official limit is 0.999.
- Defaults the organisers get: `DEFAULT_STAGE`, `DEFAULT_TOKEN_BUDGET`, `DEFAULT_RES32` at the top of `team.py`.

## Next commands

Nothing is required on the GPU. To reproduce: `bash run_all.sh && bash run_all.sh --full` (restartable), then
`python scripts/make_report.py` for the report and PDF.
