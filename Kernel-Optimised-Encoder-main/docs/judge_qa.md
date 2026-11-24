# Likely judge questions and answers

For: the presenting team. All numbers are measured on the RTX 3050 6GB Laptop GPU (`results_gpu/`, report Sections 13-18).

**Headline numbers?** 604.9 passages/s vs 28.4 for the reference = **21.3x** (official `bench.py`, 100k passages);
0 violations; worst cosine over all 100k passages 0.999992; live p50 4.4 ms, p99 8.2 ms; peak GPU memory 1.6 GB.

**Did you change the model?**
No. `team.py` loads the same BAAI/bge-base-en-v1.5 checkpoint and applies the same 12 BERT layers with the same
weights. We rewrote the forward pass by hand only so we could choose kernels (fused QKV, SDPA), skip unused last-layer
rows, and control precision. FP16 is explicitly allowed; INT8 and weight changes are not used.

**Is the CLS-only last layer really exact?**
Yes. In the last layer the CLS output depends on the keys and values of every token, which we still compute for all
tokens. The query, attention output, projection, FFN and LayerNorms are row-wise, so only the CLS row is needed. The
FP32 tests show cosine 1.0000000 with the reference.

**How do you know you never fall below 0.999?**
Four layers of evidence: edge-case tests against `reference.py` (lengths at every bucket boundary, empty, Unicode,
>512 tokens); the official 2,000 passage + 1,500 query references with an internal gate of 0.9995; a full check of
all 100k passages against our FP32 path (which matches the official reference); and a precision study that found the
worst-case inputs. The 100k check is the one that mattered: the official harness checks only 2,000 passages, and there
plain FP16 looked safe (0.999663), but across all 100,000 it failed on 2 passages (worst 0.9944) and even an FP32
residual stream failed on 1 (0.9978). All of them were short. So inputs of ≤ 32 tokens now run in FP32 (exact
reference maths) and the rest in FP16, where the 100k worst case is ≥ 0.99999.

**Why is the worst case a short passage?**
With few tokens the CLS vector is built from little context, so the same absolute rounding error is a larger relative
change in its direction. Every FP16 result below 0.9999 on the 100k set was ≤ 23 tokens; above 32 tokens nothing was.

**Isn't FP32 slow? Why is that acceptable?**
Short passages are 33% of the passages but only 5.5% of the tokens, and they are batched separately, so FP32 there
costs little. It also replaced the FP32 residual stream, which cost 12% on every passage and still let one through.

**Why a cut-off of 32 tokens and not 24?**
The worst FP16 result between 24 and 32 tokens was 0.99996 - safe, but 32 gives margin for hidden data, matches the
smallest live CUDA Graph bucket (so every typical query runs in FP32 too), and costs only 2 more points of tokens.

**Why not BF16?**
BF16 has an 8-bit mantissa (FP16 has 11). In our study its worst passage dropped to 0.9918 and 8 of 2,000 passages
were under 0.9995. It would risk a zero score.

**Why not INT8 or a smaller model?** Not allowed by the rules, and not needed.

**Where does the speed-up come from?**
Mostly from not computing padding (5.6x less work, exact count) and from FP16 Tensor Cores; the async pipeline keeps
the GPU busy; the CLS-only last layer removes ~7% more work. Measured (20k passages): FP16 alone 85.9 p/s (3.0x),
+ scheduler 427 p/s (15x), + pipeline and CLS-only 460-620 p/s (16-22x), final 100k run 604.9 p/s (21.3x).

**Why not 40x?**
The model's FLOPs are fixed. After removing padding, the remaining work is compute-bound GEMMs on a laptop GPU. The
ceiling is roughly (padding gain) × (FP16 vs FP32 GEMM throughput on this GPU). Measured: the profile of the final
config is 79% GEMM time with the GPU busy 99.7% of the span, so what is left is the model's own matrix maths. Plain FP16
everywhere would give ~16% more (692 p/s) but fails the tolerance on 2 of 100k passages.

**Why is live latency 4.4 ms and not 2 ms?**
Short queries run in FP32 for the same precision reason as short passages. An FP32 query reads 440 MB of weights, about
3.7 ms on this GPU. FP16 would be ~2.2 ms, but one violation on the hidden trace scores zero; latency is 15%.

**Why is p99 higher than p50?**
`bench.py` sends queries one at a time in arrival order. In a burst of 7 queries within 10 ms the last one waits for
the six before it, so p99 ≈ burst size × service time. We cannot batch across calls, so we minimised service time
with CUDA Graphs. Our latency script reports queueing and service time separately.

**What happens on a GPU you have never seen?**
FP16 needs only CUDA. The token budget halves automatically if warm-up exceeds 75% of VRAM, and any batch that runs
out of memory is split and retried. CUDA Graph capture or `torch.compile` failures fall back to the eager path. No GPU
model, path or environment variable is required.

**Is the 28.4 passages/s baseline fair?**
It is `reference.py` itself on the same GPU over 5,000 passages. Because the reference pads every input to 512 tokens,
its time per passage does not depend on the text, so the sample is representative of the full set.

**Is anything tuned to the practice set?**
No. The scheduler uses only token lengths; the budget is chosen by a sweep with headroom rules; no embeddings or
inputs are cached; the hidden set goes through exactly the same code.

**What would you do with more time?**
Fused Triton kernels for bias+GELU and bias+residual+LayerNorm if profiles show element-wise work is still large;
multi-stream execution; a varlen attention path for flatter length distributions.

**An earlier version ran at 666 p/s (23.5x). Why submit a slower one?**
That version ran everything in plain FP16. It passed the official 2,000-passage check, but our check of all 100,000
practice passages found 2 below 0.999. On the hidden set that would most likely score zero. The submitted version
trades ~10% of throughput (still 21.3x) for 0 violations across all 100,000.
