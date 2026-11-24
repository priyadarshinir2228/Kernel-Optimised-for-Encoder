"""Correctness tests: team.encode() against the official reference.encode() (FP32, pad 512) on edge cases.

Runs on CPU (FP32 path: checks the maths of the hand-written BERT forward, the scheduler, ordering, truncation,
CLS-only last layer) and, when CUDA is available, also checks the GPU-only paths (FP16, CUDA Graph live buckets,
async pipeline) against the same reference.

Usage:  python tests/test_correctness.py --data data [--stages 0 2 4] [--out results/test_correctness.json]
Exit code 0 = all checks passed.
"""
import argparse, importlib, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def text_with_tokens(tok, n_total):
    """A text whose tokenised length (including [CLS] and [SEP]) is exactly n_total (n_total >= 2)."""
    t = " ".join(["the"] * (n_total - 2))
    got = len(tok(t)["input_ids"])
    assert got == n_total, (n_total, got)
    return t


def edge_cases(tok, passages):
    cases = {}
    for n in (2, 3, 31, 32, 33, 34, 63, 64, 65, 66, 127, 128, 129, 130, 511, 512):
        cases[f"len{n}"] = text_with_tokens(tok, n)
    cases.update({
        "empty": "",
        "space": "   ",
        "one_word": "library",
        "short_query": "who wrote the art of war",
        "over_512_words": " ".join(["semantic"] * 900),
        "over_512_mixed": " ".join(passages[:12]),
        "unicode": "Café naïve façade — 東京大学 研究 📚 Ελληνικά עברית العربية",
        "repeated": "data " * 200,
        "punctuation": "!!! ??? ... ,,, ;;; ::: ((( ))) [[[ ]]]",
        "newlines_tabs": "line one\nline two\n\tindented\r\nwindows line",
        "numbers": "3.14159 2.71828 1.41421 6.02e23 299792458 m/s",
        "medium_passage": passages[0],
        "long_passage": max(passages[:2000], key=len),
    })
    return cases


def run_stage(stage, texts, names, ref, ref_by_name, cuda):
    os.environ["PS09_STAGE"] = str(stage)
    sys.modules.pop("team", None)  # fresh module so PS09_STAGE is re-read and state is clean
    team = importlib.import_module("team")
    t = time.time(); team.setup(); setup_s = time.time() - t
    gate = 0.9995
    res = {"stage": stage, "cfg": team.CFG, "setup_s": round(setup_s, 2)}
    print(f"\n== stage {stage} cfg={team.CFG} device={team._S['device']} dtype={team._S['dtype']}", flush=True)

    # 1. whole edge-case list in one call (mixed lengths, scheduler + ordering)
    emb = team.encode(texts)
    check("shape [N,768] float32", emb.shape == (len(texts), 768) and emb.dtype == np.float32, str(emb.shape))
    check("finite", bool(np.isfinite(emb).all()))
    norms = np.linalg.norm(emb, axis=1)
    check("L2-normalised", bool(np.abs(norms - 1).max() < 1e-3), f"max |norm-1|={np.abs(norms - 1).max():.2e}")
    c = cos(emb, ref)
    worst = int(np.argmin(c))
    check(f"batch min cos >= {gate}", c.min() >= gate, f"min={c.min():.7f} at {names[worst]}")
    res["batch_min_cos"] = float(c.min())
    res["batch_worst"] = names[worst]

    # 2. one at a time (live path; on GPU with graphs this exercises every bucket boundary)
    single = np.stack([team.encode([t])[0] for t in texts])
    cs = cos(single, ref)
    w = int(np.argmin(cs))
    check(f"single-query min cos >= {gate}", cs.min() >= gate, f"min={cs.min():.7f} at {names[w]}")
    res["single_min_cos"] = float(cs.min())

    # 3. ordering: reversed input gives reversed output
    rev = team.encode(texts[::-1])[::-1]
    check("order preserved (reversed input)", bool(np.allclose(rev, emb, atol=2e-3)),
          f"max diff={np.abs(rev - emb).max():.2e}")

    # 4. duplicates give identical embeddings within a call
    dup = team.encode([texts[0], texts[5], texts[0], texts[5]])
    check("duplicates identical", bool(np.allclose(dup[0], dup[2], atol=1e-6) and np.allclose(dup[1], dup[3], atol=1e-6)))

    # 5. repeated calls / no stale state: alternate two different single queries
    a1 = team.encode([ref_by_name["short_query"][0]])[0]
    b1 = team.encode([ref_by_name["unicode"][0]])[0]
    a2 = team.encode([ref_by_name["short_query"][0]])[0]
    check("no stale output between calls", bool(np.allclose(a1, a2, atol=1e-6) and not np.allclose(a1, b1, atol=1e-2)))
    check("live output matches reference", cos(a1[None], ref_by_name["short_query"][1][None])[0] >= gate)

    # 6. empty list
    z = team.encode([])
    check("empty list -> [0,768]", z.shape == (0, 768))

    # 7. returned array is not aliased to internal buffers (caller may keep it)
    x1 = team.encode([texts[1]])
    x1_copy = x1.copy()
    team.encode([texts[2]])
    check("returned array not overwritten by next call", bool(np.array_equal(x1, x1_copy)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--stages", type=int, nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    cuda = torch.cuda.is_available()
    stages = a.stages or ([1, 2, 3, 4] if cuda else [0, 2, 4])
    sys.path.insert(0, a.data)
    import reference
    tok, m = reference.load(os.environ.get("PS09_MODEL", "BAAI/bge-base-en-v1.5"))
    passages = [json.loads(l)["text"] for l in open(Path(a.data) / "practice_passages.jsonl", encoding="utf-8")][:2000]
    cases = edge_cases(tok, passages)
    names, texts = list(cases), list(cases.values())
    print(f"reference.encode on {len(texts)} edge cases (FP32, pad 512, device {m.device}) ...", flush=True)
    ref = reference.encode(tok, m, texts)
    ref_by_name = {n: (t, r) for n, t, r in zip(names, texts, ref)}
    lens = [len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in texts]
    print("token lengths:", dict(zip(names, lens)))
    check("truncation keeps <= 512 tokens", max(lens) == 512)
    del m
    out = {"cuda": cuda, "torch": torch.__version__, "n_cases": len(texts), "stages": []}
    for st in stages:
        out["stages"].append(run_stage(st, texts, names, ref, ref_by_name, cuda))
    out["failed"] = FAILS
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
