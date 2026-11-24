# PS-09 demo guide and presentation outline

For: the two of us presenting to the judges (10% of the score for the demo, 25% for the profiler report).

## Before the session (GPU machine)

```bash
cd ~/Desktop/ps09            # or the git checkout
source .venv/bin/activate
python demo.py --data data   # dry run once: downloads/caches the model, ~3-5 min
```

Have these open in tabs: `reports/PS09_Final_Report.pdf`, `results_gpu/SUMMARY.md`,
`results_gpu/profiler/reference/trace.json.gz` and `results_gpu/profiler/default/trace.json.gz` in
https://ui.perfetto.dev (drag the file in), and the architecture diagram (report Section 8).

## Live demo run sheet (about 6 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | `python demo.py --data data`, section 1 | "Same model, same weights. This is what runs: FP16 Tensor Cores for long inputs, exact FP32 for inputs of 32 tokens or less, sorted token-budget batches, CUDA Graphs." |
| 0:45 | Section 2: cosine per text | "Empty string, one word, a 900-word text cut at 512 tokens, Unicode: every cosine is above 0.9995. The rule is 0.999 on every input." |
| 1:30 | Section 3: padding | "The reference computes 512 positions per passage; the average passage has 100 tokens. 80% of its work is padding." |
| 2:15 | Section 4: throughput | "Same passages, same machine, end to end including tokenisation: about 590 passages/s vs 29 (20.6x on this run). The official bench.py number on all 100k is 604.9 p/s, 21.3x." |
| 3:15 | Section 5: live queries | "One query at a time, like the harness: a CUDA Graph replay, about 3.7 ms. Queries are short, so they run in FP32 - we chose zero precision risk over 2 ms." |
| 3:45 | Section 6: search | "And it still does its job: on a small demo library, each question finds the right passage." |
| 4:15 | Perfetto: reference trace vs ours | "Before: FP32 GEMMs on padding, 136 idle gaps. After: Tensor-Core GEMMs, 7 idle gaps, GPU busy 99.7% of the time." |
| 5:15 | Report: ablation chart | "Each step measured on its own with the official harness, so you can see what each idea is worth." |

If the GPU is unavailable: `python demo.py --data data --n 200` runs on any CPU (numbers labelled as not
representative), and the report holds the GPU evidence.

## Slide outline (8 slides)

1. **Problem in one line**: millions of passages, one modest GPU, same model, cosine ≥ 0.999 on every input.
2. **Where the time goes**: 80.4% padding (length histogram); FP32 GEMMs; per-batch syncs; launch-bound queries.
3. **Our design**: architecture diagram (report Section 8): two paths, offline throughput and live latency.
4. **Accuracy first**: precision gate; worst cases are short passages; FP32 residual stream; BF16 rejected (chart).
5. **Ablation**: throughput per stage (chart) + token-budget sweep.
6. **Profiler before / after**: GPU-time breakdown, idle gaps, kernels per passage.
7. **Live latency**: p50 / p99, burst queueing vs service time, CUDA Graph on/off.
8. **Results and limits**: final speed-up, 100k check, what we would do next.

## Key talking points

- We changed *how* the model runs, never *what* it computes: same weights, same 12 layers, same operations.
- Every optimisation is a switch, and every switch was measured alone with the official harness.
- Accuracy is engineered, not hoped for: we found the worst-case inputs (short passages), found the cause (FP16
  residual stream) and fixed it at the cost of element-wise work only.
- The scheduler counts exact token positions: 51.2 M → 10.4 M for 100k passages.
- Nothing is tuned to the practice data: the scheduler works from token lengths, budgets adapt to the GPU's memory,
  and every GPU feature has a fallback.
