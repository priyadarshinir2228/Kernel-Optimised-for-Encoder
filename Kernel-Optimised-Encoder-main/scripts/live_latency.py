"""NON-OFFICIAL live-query latency study (the official numbers come from bench.py --trace).

Replays practice_trace.jsonl exactly like bench.py's live() (real-time arrivals, one query per encode([q]) call, FIFO,
latency = completion time - arrival time, so it includes queueing behind earlier queries of a burst) and in addition
records the service time of every call, so the p99 can be explained (burst queueing vs slow individual calls).

Modes:
  realtime  (default) same timing as bench.py: about 4 minutes for the 1,500-query trace
  service   back-to-back calls, no arrival schedule: pure per-query service time (a few seconds)

Usage:
  python scripts/live_latency.py --data DATA [--mode realtime|service] [--out results_gpu/latency/graph_on.json]
  PS09_GRAPH=0 python scripts/live_latency.py --data DATA --out results_gpu/latency/graph_off.json
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pct(x, q):
    return float(np.percentile(x, q))


def stats(ms):
    ms = np.asarray(ms)
    return {"p50": pct(ms, 50), "p90": pct(ms, 90), "p95": pct(ms, 95), "p99": pct(ms, 99), "mean": float(ms.mean()),
            "max": float(ms.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--mode", choices=["realtime", "service"], default="realtime")
    ap.add_argument("--limit", type=int); ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    import team
    t = time.time(); team.setup(); setup_s = time.time() - t
    d = Path(a.data)
    trace = [json.loads(l) for l in open(d / "practice_trace.jsonl", encoding="utf-8")][: a.limit]
    ref = np.load(d / "practice_ref_queries.npy")[: len(trace)]
    for r in trace[:20]:  # warm-up on a few queries (bench.py does not do this, but setup() already warmed the graphs)
        team.encode([r["text"]])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lat, svc, outs = [], [], []
    start = time.perf_counter()
    for r in trace:
        if a.mode == "realtime":
            wait = r["t"] - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
        t0 = time.perf_counter()
        outs.append(np.asarray(team.encode([r["text"]]))[0])
        t1 = time.perf_counter()
        svc.append((t1 - t0) * 1000)
        if a.mode == "realtime":
            lat.append((t1 - start - r["t"]) * 1000)
    e = np.stack(outs)
    c = (e * ref).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref, axis=1))
    svc = np.asarray(svc)
    res = {"what": "non-official live latency study; realtime mode replays bench.py's live() timing",
           "mode": a.mode, "n": len(trace), "setup_s": round(setup_s, 2), "runtime": team.runtime_info(),
           "service_ms": stats(svc), "min_cosine": float(c.min()), "violations_0.999": int((c < 0.999).sum()),
           "slowest_service": [{"i": int(i), "ms": float(svc[i]), "text": trace[i]["text"][:80]} for i in np.argsort(-svc)[:5]]}
    if lat:
        lat = np.asarray(lat)
        queue_ms = lat - svc
        res["latency_ms"] = stats(lat)
        res["queue_ms"] = stats(queue_ms)
        res["p99_explained"] = ("queueing behind burst arrivals" if pct(queue_ms, 99) > pct(svc, 99)
                                else "slow individual calls")
        res["queries_per_second_capacity"] = 1000 / float(svc.mean())
    if torch.cuda.is_available():
        res["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "runtime"}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
