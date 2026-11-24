"""NON-OFFICIAL full-dataset accuracy check (all 100,000 practice passages).

Official reference embeddings exist only for the first 2,000 passages. To check the other 98,000:
  1. `run` the FP32 path of team.py (PS09_FP16=0: same weights, FP32, sorted batches, no TF32) over all passages.
     Its first 2,000 rows are compared with the official reference to show it IS the reference maths
     (expected min cosine ~1.0). Takes ~10-15 min on an RTX 3050 instead of ~1 h for reference.py.
     Optionally use reference.py itself instead: python DATA/reference.py --passages ... --out ref100k.npy (~1 h).
  2. `run` the submitted configuration (no PS09_* variables) over all passages.
  3. `compare` the two: minimum / mean cosine, max abs error, violations, worst inputs, plus both against the official
     2,000-passage reference.

Usage:
  PS09_STAGE=2 PS09_FP16=0 python scripts/full_check.py run --data DATA --out results_gpu/final/fp32_100k.npy
  python scripts/full_check.py run --data DATA --out results_gpu/final/submitted_100k.npy
  python scripts/full_check.py compare --data DATA --a results_gpu/final/submitted_100k.npy \
      --b results_gpu/final/fp32_100k.npy --out results_gpu/final/full_check.json
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def load_texts(data, limit=None):
    return [json.loads(l)["text"] for l in open(Path(data) / "practice_passages.jsonl", encoding="utf-8")][:limit]


def run(a):
    import torch
    import team
    t = time.time(); team.setup(); setup_s = time.time() - t
    texts = load_texts(a.data, a.limit)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); emb = team.encode(texts); secs = time.perf_counter() - t
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out, emb.astype(np.float32))
    ref = np.load(Path(a.data) / "practice_ref_first2000.npy")
    k = min(len(ref), len(emb))
    c = cos(emb[:k], ref[:k])
    meta = {"file": a.out, "n": len(texts), "seconds": secs, "passages_per_second": len(texts) / secs,
            "setup_seconds": setup_s, "runtime": team.runtime_info(), "finite": bool(np.isfinite(emb).all()),
            "official_ref_first2000": {"min": float(c.min()), "mean": float(c.mean()), "violations_0.999": int((c < 0.999).sum())}}
    if torch.cuda.is_available():
        meta["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    json.dump(meta, open(a.out.replace(".npy", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def compare(a):
    A, B = np.load(a.a), np.load(a.b)
    assert A.shape == B.shape, (A.shape, B.shape)
    texts = load_texts(a.data, len(A))
    c = cos(A, B)
    err = np.abs(A.astype(np.float64) - B.astype(np.float64)).max(1)
    worst = np.argsort(c)[:10]
    ref = np.load(Path(a.data) / "practice_ref_first2000.npy")
    k = min(len(ref), len(A))
    ca, cb = cos(A[:k], ref[:k]), cos(B[:k], ref[:k])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("PS09_MODEL", "BAAI/bge-base-en-v1.5"))
    res = {"what": "non-official full-dataset comparison", "a": a.a, "b": a.b, "n": int(len(A)),
           "a_vs_b": {"min": float(c.min()), "p0.01": float(np.percentile(c, 0.01)), "p0.1": float(np.percentile(c, 0.1)),
                      "mean": float(c.mean()), "max_abs_error": float(err.max()),
                      "violations_0.999": int((c < 0.999).sum()), "below_0.9995": int((c < 0.9995).sum()),
                      "worst": [{"i": int(i), "cos": float(c[i]), "max_abs_err": float(err[i]),
                                 "tokens": len(tok(texts[i], truncation=True, max_length=512)["input_ids"]),
                                 "text": texts[i][:100]} for i in worst]},
           "a_vs_official_ref_first2000": {"min": float(ca.min()), "mean": float(ca.mean()), "violations_0.999": int((ca < 0.999).sum())},
           "b_vs_official_ref_first2000": {"min": float(cb.min()), "mean": float(cb.mean()), "violations_0.999": int((cb < 0.999).sum())}}
    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--data", required=True); r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int)
    c = sub.add_parser("compare"); c.add_argument("--data", required=True); c.add_argument("--a", required=True)
    c.add_argument("--b", required=True); c.add_argument("--out")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
