"""PS-09 demo for the judges: original encoder vs our optimised encoder, side by side, on this machine.

  python demo.py --data data            (GPU machine: ~3-5 min; on a CPU-only laptop use --n 200)

What it shows (every number is measured live by this script; the official scores come from bench.py):
  1. hardware and what our encoder actually runs (precision, batching, CUDA Graphs ...)
  2. the same texts through reference.py (FP32, pad 512) and team.py -> cosine per text, minimum cosine
  3. why the reference is slow: token positions computed vs real tokens (padding)
  4. throughput on the same N passages: reference vs ours, speed-up
  5. live queries: service time per query (p50 / p99)
  6. it still works as search: top passages for a few questions, using our embeddings
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W = 78


def box(title):
    print("\n" + "=" * W + f"\n {title}\n" + "=" * W, flush=True)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n", type=int, help="passages for the throughput comparison (default 2000 on GPU, 200 on CPU)")
    ap.add_argument("--queries", type=int, default=200)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    except Exception:
        pass
    import warnings; warnings.filterwarnings("ignore")
    import torch
    cuda = torch.cuda.is_available()
    n = a.n or (2000 if cuda else 200)
    d = Path(a.data)
    sys.path.insert(0, str(d))
    import reference
    import team

    passages = [json.loads(l)["text"] for l in open(d / "practice_passages.jsonl", encoding="utf-8")][:max(n, 2000)]
    trace = [json.loads(l)["text"] for l in open(d / "practice_trace.jsonl", encoding="utf-8")][: a.queries]

    box("1. Hardware and configuration")
    print(f" device: {torch.cuda.get_device_name(0) if cuda else 'CPU only (numbers NOT representative of a GPU)'}")
    print(" loading the original encoder (reference.py) ...", flush=True)
    tok, ref_model = reference.load("BAAI/bge-base-en-v1.5")
    print(" loading our encoder (team.py: setup() loads, converts, warms up, captures CUDA Graphs) ...", flush=True)
    t = time.time(); team.setup(); print(f" team.setup() took {time.time() - t:.1f} s (not timed by bench.py)")
    info = team.runtime_info()
    for k in ("gpu", "compute_capability", "vram_gib", "dtype", "res32", "token_budget", "graph_buckets", "compiled", "notes"):
        if k in info:
            print(f"   {k:20s} {info[k]}")
    print(f"   {'optimisations':20s} {', '.join(k for k, v in info['cfg'].items() if v)}")

    box("2. Same texts, same meaning: cosine(reference, ours)")
    sample = ["", "library", trace[0], trace[1], passages[19], passages[0], max(passages[:2000], key=len),
              "Café naïve façade — 東京大学 📚", " ".join(["semantic"] * 900)]
    labels = ["empty string", "one word", "query", "query", "short passage", "medium passage", "longest passage",
              "unicode", "900 words (truncated to 512 tokens)"]
    r = reference.encode(tok, ref_model, sample)
    o = team.encode(sample)
    c = (r * o).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1))
    for lab, s, ci in zip(labels, sample, c):
        ntok = len(tok(s, truncation=True, max_length=512)["input_ids"])
        print(f"   {lab:38s} {ntok:4d} tokens   cos = {ci:.6f}")
    print(f" minimum cosine {c.min():.6f}  (rule: >= 0.999 on every input; our internal gate 0.9995)")

    box("3. Why the reference is slow: padding")
    lengths = np.array([len(e.ids) for e in team._S["fast"].encode_batch(passages[:n])])
    team_pos = 0
    for idx in team._plan(lengths):
        team_pos += len(idx) * team._pad8(int(lengths[idx].max()))
    print(f" {n} passages, {lengths.sum():,} real tokens (mean {lengths.mean():.0f}, median {np.median(lengths):.0f})")
    print(f" reference computes {n * 512:,} token positions  -> {1 - lengths.sum() / (n * 512):.1%} is padding")
    print(f" ours computes      {team_pos:,} token positions  -> {1 - lengths.sum() / team_pos:.1%} is padding")

    box(f"4. Throughput on the same {n} passages (end to end, including tokenisation)")
    texts = passages[:n]
    reference.encode(tok, ref_model, texts[:32]); sync()
    t = time.perf_counter(); R = reference.encode(tok, ref_model, texts); sync(); tr = time.perf_counter() - t
    team.encode(texts[:64])
    t = time.perf_counter(); O = team.encode(texts); to = time.perf_counter() - t
    cc = (R * O).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(O, axis=1))
    print(f" reference (FP32, pad 512, batch 32): {n / tr:8.1f} passages/s   ({tr:.1f} s)")
    print(f" ours                               : {n / to:8.1f} passages/s   ({to:.1f} s)")
    print(f" speed-up                           : {tr / to:8.2f} x")
    print(f" accuracy on these {n}: min cos {cc.min():.6f}, mean {cc.mean():.7f}, violations {(cc < 0.999).sum()}")
    del ref_model
    if cuda:
        torch.cuda.empty_cache()

    box(f"5. Live queries ({len(trace)} from the practice trace, one encode([q]) call each)")
    svc = []
    for q in trace:
        t = time.perf_counter(); team.encode([q]); svc.append((time.perf_counter() - t) * 1000)
    svc = np.array(svc)
    print(f" service time per query: p50 {np.percentile(svc, 50):.2f} ms, p99 {np.percentile(svc, 99):.2f} ms, "
          f"mean {svc.mean():.2f} ms  ({'CUDA Graph replay' if team._S['graphs'] else 'eager path'})")
    print(" (bench.py's live p50/p99 also include waiting behind earlier queries of a burst)")

    box("6. It still works as semantic search (small demo library, written for this demo)")
    lib = ["Photosynthesis converts light energy into chemical energy stored in glucose, using water and carbon dioxide.",
           "The Art of War is an ancient Chinese military treatise attributed to the general and strategist Sun Tzu.",
           "World War I began in 1914 after the assassination of Archduke Franz Ferdinand in Sarajevo.",
           "Backpropagation computes the gradient of a neural network's loss with respect to its weights.",
           "The French Revolution of 1789 abolished the monarchy and led to the rise of Napoleon Bonaparte.",
           "Mitochondria are organelles that produce most of the cell's supply of ATP through respiration.",
           "A hash table stores key-value pairs and offers average constant-time lookup.",
           "Supply and demand determine the market price of a good in a competitive market."]
    L = team.encode(lib)
    for q in ("who wrote the art of war", "when did the first world war start", "how do plants make food from sunlight",
              "how are neural networks trained"):
        e = team.encode([q])[0]
        best = np.argsort(-(L @ e))[:2]
        print(f" Q: {q}")
        for b in best:
            print(f"    {L[b] @ e:.3f}  {lib[b][:90]}")

    box("Summary")
    print(f" min cosine {min(c.min(), cc.min()):.6f} | {n / to:.0f} vs {n / tr:.0f} passages/s = {tr / to:.1f}x | "
          f"live p50 {np.percentile(svc, 50):.2f} ms")


if __name__ == "__main__":
    main()
