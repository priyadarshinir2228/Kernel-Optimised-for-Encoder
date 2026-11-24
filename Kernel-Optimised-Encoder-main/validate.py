"""Innovation B: accuracy gate. Runs team.encode() on the reference sets and reports the WORST case.

Usage: python validate.py --data DATA_DIR [--limit N] [--queries N] [--gate 0.9995] [--out results/validate_stageX.json]
DATA_DIR holds practice_passages.jsonl, practice_ref_first2000.npy, practice_trace.jsonl, practice_ref_queries.npy.
Set PS09_STAGE / PS09_* flags exactly as for the run you want to check. PS09_DEBUG=1 also logs max |activation|
per layer, to spot FP16 overflow risk (FP16 max is 65504).
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def report(name, c, texts, lengths, gate, k=10, max_err=None):
    worst = np.argsort(c)[:k]
    r = {"n": int(len(c)), "min": float(c.min()), "p0.1": float(np.percentile(c, 0.1)), "mean": float(c.mean()),
         "max_abs_error": float(max_err) if max_err is not None else None,
         "violations_0.999": int((c < 0.999).sum()), "below_gate": int((c < gate).sum()),
         "worst": [{"i": int(i), "cos": float(c[i]), "tokens": int(lengths[i]), "text": texts[i][:80]} for i in worst]}
    flag = "PASS" if r["min"] >= gate else ("WARN (passes 0.999, under internal gate)" if r["min"] >= 0.999 else "FAIL")
    print(f"[{name}] n={r['n']} min={r['min']:.6f} p0.1={r['p0.1']:.6f} mean={r['mean']:.6f} -> {flag}")
    return r, flag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=1500); ap.add_argument("--gate", type=float, default=0.9995)
    ap.add_argument("--out")
    a = ap.parse_args()
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import team
    import torch
    d = Path(a.data)
    t = time.time(); team.setup(); setup_s = time.time() - t
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    texts = [json.loads(l)["text"] for l in open(d / "practice_passages.jsonl", encoding="utf-8")][: a.limit]
    ref = np.load(d / "practice_ref_first2000.npy")[: a.limit]
    texts = texts[: len(ref)]
    lengths = np.array([len(e.ids) for e in team._S["fast"].encode_batch(texts)])
    t = time.time(); emb = team.encode(texts); enc_s = time.time() - t
    assert emb.shape == (len(texts), 768), emb.shape
    res = {"stage": team.STAGE, "cfg": team.CFG, "token_budget": team.TOKEN_BUDGET, "setup_s": setup_s, "encode_s": enc_s,
           "runtime": team.runtime_info()}
    res["passages"], f1 = report("passages", cos(emb, ref), texts, lengths, a.gate, max_err=np.abs(emb - ref).max())
    f2 = "SKIP"
    if a.queries:
        qs = [json.loads(l)["text"] for l in open(d / "practice_trace.jsonl", encoding="utf-8")][: a.queries]
        qref = np.load(d / "practice_ref_queries.npy")[: len(qs)]
        qemb = np.stack([team.encode([x])[0] for x in qs])  # one at a time, exactly like bench.py's live replay
        ql = np.array([len(e.ids) for e in team._S["fast"].encode_batch(qs)])
        res["queries"], f2 = report("queries", cos(qemb, qref), qs, ql, a.gate, max_err=np.abs(qemb - qref).max())
    if torch.cuda.is_available():
        res["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if team.DEBUG and team._S["enc"].maxabs:
        res["max_abs_activation"] = max(team._S["enc"].maxabs)
        print(f"max |activation| seen: {res['max_abs_activation']:.1f} (FP16 limit 65504)")
    res["verdict"] = "FAIL" if "FAIL" in (f1, f2) else ("WARN" if "WARN" in (f1, f2) else "PASS")
    print("VERDICT:", res["verdict"])
    print("runtime:", json.dumps(res["runtime"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if res["verdict"] != "FAIL" else 1)
