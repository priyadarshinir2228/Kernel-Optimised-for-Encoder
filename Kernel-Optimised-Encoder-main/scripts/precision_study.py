"""NON-OFFICIAL diagnostic: FP16 precision study that runs on a CPU (no GPU needed).

GPU FP16 inference with PyTorch = FP16 storage + FP32 accumulation inside GEMMs / LayerNorm / softmax / GELU, with
every op output rounded to FP16. This script emulates exactly that on the CPU: it computes in FP32 and rounds every
stored tensor (weights and op outputs) to FP16. It then compares against the official FP32 reference embeddings
(practice_ref_first2000.npy, practice_ref_queries.npy) and reports the worst cases.

It also evaluates precision mitigations, so the decision can be made with evidence before touching the GPU:
  fp32         no rounding (sanity check: should give min cosine ~1.0)
  fp16         plain FP16 path (what team.py stage 4 runs on the GPU)
  fp16_last32  layers 1-11 FP16, the (CLS-only) last layer in FP32
  fp16_res32   FP16 GEMMs, but the residual stream and LayerNorms kept in FP32
  bf16         BF16 instead of FP16 (8-bit mantissa; expected to be worse)

What it does NOT emulate: cuBLAS split-K reduced-precision reductions and kernel-specific summation order. Those are
second-order; the GPU validate.py run remains the final word. Results are labelled "CPU emulation" everywhere.

Usage: python scripts/precision_study.py --data data [--limit 2000] [--queries 1500] [--variants fp16 fp16_last32]
       [--out results/precision_study.json]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PS09_STAGE", "4")
import team  # noqa: E402

ROUND = {"fp16": torch.float16, "bf16": torch.bfloat16}


def make_round(dt):
    if dt is None:
        return lambda x: x
    return lambda x: x.to(dt).float()


def rounded_weights(enc, rnd):
    L = []
    for P in enc.layers:
        L.append({k: (tuple(rnd(t) for t in v) if isinstance(v, tuple) else rnd(v)) for k, v in P.items()})
    return dict(word=rnd(enc.word), pos=rnd(enc.pos), typ0=rnd(enc.typ0), eln=tuple(rnd(t) for t in enc.eln), layers=L)


def forward(enc, W32, Wr, ids, attn, variant):
    """Emulated team.Encoder.forward (CLS-only last layer), returns L2-normalised CLS [B,768] float64."""
    dt = {"fp32": None, "bf16": torch.bfloat16}.get(variant, torch.float16)
    r = make_round(dt)
    ident = lambda x: x
    res32 = variant == "fp16_res32"
    rs = ident if res32 else r           # rounding applied to the residual stream / LN outputs
    D, H, dh, eps = enc.d, enc.heads, enc.dh, enc.eps
    W = Wr if dt is not None else W32
    B, T = ids.shape
    h = rs(rs(W["word"][ids] + W["pos"][:T]) + W["typ0"])
    h = rs(F.layer_norm(h, (D,), W["eln"][0], W["eln"][1], eps))
    neg = torch.finfo(dt).min if dt is not None else torch.finfo(torch.float32).min
    mask = (1 - attn[:, None, None, :].float()) * neg
    n = len(enc.layers)
    for i in range(n):
        last = i == n - 1
        if last and variant == "fp16_last32":
            P, rr, rrs = W32["layers"][i], ident, ident
        else:
            P, rr, rrs = W["layers"][i], r, rs
        x_in = rr(h)  # GEMM input is FP16 even when the residual stream is FP32
        if not last:
            qkv = rr(F.linear(x_in, P["wqkv"], P["bqkv"])).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            res = h
        else:
            kv = rr(F.linear(x_in, P["wqkv"][D:], P["bqkv"][D:])).view(B, T, 2, H, dh).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            res = h[:, :1]
            q = rr(F.linear(x_in[:, :1], P["wqkv"][:D], P["bqkv"][:D])).view(B, 1, H, dh).transpose(1, 2)
        ctx = rr(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        ctx = ctx.transpose(1, 2).reshape(B, -1, D)
        o = rr(F.linear(ctx, P["wo"], P["bo"]))
        x = rrs(F.layer_norm(rrs(o + res), (D,), P["ln1"][0], P["ln1"][1], eps))
        y = rr(F.linear(rr(F.gelu(rr(F.linear(rr(x), P["wi"], P["bi"])))), P["wf"], P["bf"]))
        h = rrs(F.layer_norm(rrs(y + x), (D,), P["ln2"][0], P["ln2"][1], eps))
    return F.normalize(h[:, 0].double(), dim=-1)


def embed(enc, W32, Wr, texts, variant, budget=16384):
    ids_l, lengths = team._tokenise(texts)
    order = np.argsort(-lengths, kind="stable")
    out = torch.empty((len(texts), 768), dtype=torch.float64)
    i = 0
    while i < len(order):
        T = team._pad8(int(lengths[order[i]]))
        idx = order[i:i + max(1, budget // T)]
        ids = torch.zeros((len(idx), T), dtype=torch.long)
        attn = torch.zeros((len(idx), T), dtype=torch.long)
        for rr_, j in enumerate(idx):
            ids[rr_, :lengths[j]] = torch.tensor(ids_l[j]); attn[rr_, :lengths[j]] = 1
        out[torch.from_numpy(idx)] = forward(enc, W32, Wr, ids, attn, variant)
        i += len(idx)
    return out.numpy(), lengths


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def summary(c, lengths, texts, k=5):
    w = np.argsort(c)[:k]
    return {"n": int(len(c)), "min": float(c.min()), "p0.1": float(np.percentile(c, 0.1)), "p1": float(np.percentile(c, 1)),
            "mean": float(c.mean()), "below_0.9995": int((c < 0.9995).sum()), "below_0.999": int((c < 0.999).sum()),
            "worst": [{"i": int(i), "cos": float(c[i]), "tokens": int(lengths[i]), "text": texts[i][:100]} for i in w]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16", "fp16_last32", "fp16_res32", "bf16"])
    ap.add_argument("--out", default=str(ROOT / "results" / "precision_study.json"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    team.CFG.update(fp16=0)  # load FP32 master weights on the CPU
    team.setup()
    enc = team._S["enc"]
    W32 = rounded_weights(enc, lambda x: x)
    d = Path(a.data)
    texts = [json.loads(l)["text"] for l in open(d / "practice_passages.jsonl", encoding="utf-8")][: a.limit]
    ref = np.load(d / "practice_ref_first2000.npy")[: a.limit]
    texts = texts[: len(ref)]
    qs = [json.loads(l)["text"] for l in open(d / "practice_trace.jsonl", encoding="utf-8")][: a.queries]
    qref = np.load(d / "practice_ref_queries.npy")[: len(qs)]
    res = {"what": "CPU emulation of GPU FP16/BF16 numerics (non-official diagnostic)", "torch": torch.__version__,
           "n_passages": len(texts), "n_queries": len(qs), "variants": {}}
    for v in a.variants:
        dt = {"fp32": None, "bf16": torch.bfloat16}.get(v, torch.float16)
        Wr = rounded_weights(enc, make_round(dt)) if dt is not None else W32
        t = time.time()
        e, L = embed(enc, W32, Wr, texts, v)
        qe, qL = embed(enc, W32, Wr, qs, v) if qs else (None, None)
        r = {"seconds": round(time.time() - t, 1), "passages": summary(cos(e, ref), L, texts)}
        if qs:
            r["queries"] = summary(cos(qe, qref), qL, qs)
        res["variants"][v] = r
        p, q = r["passages"], r.get("queries", {})
        print(f"{v:12s} passages min={p['min']:.6f} p0.1={p['p0.1']:.6f} mean={p['mean']:.7f} <0.9995:{p['below_0.9995']}"
              f" | queries min={q.get('min', float('nan')):.6f}  ({r['seconds']} s)", flush=True)
        print("             worst passages:", [(w["i"], round(w["cos"], 6), w["tokens"]) for w in p["worst"]], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
