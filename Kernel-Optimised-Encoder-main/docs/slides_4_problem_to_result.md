# PS-09 in 4 slides: from problem to result

All numbers measured on an NVIDIA GeForce RTX 3050 6GB Laptop GPU with the official `bench.py`.
Visuals refer to `reports/PS09_Explainer.pdf` (diagrams) and `showcase/index.html` (charts).

---

## Slide 1 — The problem: a good model, run wastefully

**Title:** Same model, same GPU, make it faster, never change the answers

**On the slide**
- A library embeds millions of passages with BAAI/bge-base-en-v1.5 (12-layer BERT) on one modest GPU
- The reference runs at **28.4 passages/s**: 100,000 passages take about an hour
- Rules: no new model, no INT8, no weight changes; FP16 allowed
- Gate: cosine **≥ 0.999 on every input**, or the run scores **zero**
- Score: throughput 50% (log scale, 40× = full) · profiler report 25% · live latency 15% · demo 10%

**Visual:** the padding tape (explainer page 3 or the showcase toggle in "Reference" mode).

**Speaker note:** "The judges run our one file on 100,000 passages we have never seen. Speed earns points, but a single
wrong answer anywhere scores zero, so every decision had to be measured for both."

---

## Slide 2 — Diagnosis and design: where the time went, and what we changed

**Title:** 80% of the work was padding, and all of it was FP32

**On the slide**
| Bottleneck (measured) | Our fix |
|---|---|
| 80.4% of token positions are padding to 512 | Sort by length, 8,192-token batches, pad to ×8 only: padding **3.7%** |
| 75% of GPU time in FP32 GEMMs | FP16 Tensor Cores, fused QKV GEMM, fused SDPA attention |
| GPU idles while the CPU tokenises (136 gaps) | Tokenise in a background thread, pinned memory, async copies |
| Last layer computes rows nobody uses | CLS-only last layer (exact) |
| Hundreds of kernel launches per query | CUDA Graph replay: one launch per query |

**Visual:** the system diagram (explainer page 4) or the layer diagram, reference vs ours (page 5).

**Speaker note:** "The model is not the problem; the way it is run is. Every change keeps the same weights and the
same maths. We can switch each one off, so each was measured on its own with the official harness."

---

## Slide 3 — The finding that mattered: checking all 100,000, not 2,000

**Title:** The official check sees 2,000 passages. The rule covers all 100,000

**On the slide**
- Plain FP16 passed the official 2,000 at 0.999663 and looked safe
- We embedded **all 100,000** practice passages and compared with our FP32 path (which matches the reference exactly)
- Plain FP16: **2 failures** (worst 0.9944). FP32 residual stream: still **1 failure**. Both would score zero
- Every result below 0.9999 was a passage of **≤ 23 tokens**
- Fix: inputs **≤ 32 tokens run in FP32** (33% of passages, only 5.5% of tokens) → **0 failures, worst 0.999992**, for 3% of speed

**Visual:** the precision chart (showcase, "Precision" section): worst error per length band, three variants, the
0.999 line.

**Speaker note:** "This is the result we are proudest of. The fastest version we built would most likely have scored
zero on the hidden set. We found it because we tested the rule as written, on every passage."

---

## Slide 4 — Result

**Title:** 21.3× faster, zero violations, evidence for every claim

**On the slide**
- **604.9 passages/s vs 28.4 → 21.3×** (official `bench.py`, 100,000 passages)
- **0 of 100,000** passages below 0.999; worst 0.999992
- Live queries: **p50 4.4 ms, p99 8.2 ms**, bursts included
- Profiler: GPU time for 2,000 passages **69.7 s → 3.3 s**; idle gaps **136 → 7**; GPU busy 99.7%
- Peak GPU memory 1.6 GB of 6 GB; adapts to an unknown GPU; one file, no settings

**Visual:** the stage ladder chart (28 → 86 → 427 → 605 passages/s) and the before/after profiler bars.

**Speaker note:** "21× sits at about 83% of the throughput scale. What is left in the profile is 79% matrix maths,
the model's own work. We chose to give up the last 10% of speed for answers we can guarantee."
