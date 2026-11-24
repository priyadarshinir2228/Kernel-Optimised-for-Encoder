"""Runs the official bench.py once per stage and appends the results to <out-dir>/ablation.csv + <out-dir>/ABLATION.md.

Usage:
  # 1) reference throughput on this GPU (once; reference.py pads every input to 512 tokens, so its time per passage
  #    does not depend on the text and the first 5,000 passages give a valid estimate)
  python DATA/reference.py --passages DATA/practice_passages.jsonl --out results_gpu/ref5k.npy --limit 5000 \
      | tail -1 > results_gpu/hardware/ref_throughput.json
  # 2) all stages
  python run_ablation.py --data DATA --stages 1 2 3 4 5 [--limit N] [--no-trace] [--out-dir results_gpu/ablation]
Each stage runs in a fresh process with PS09_STAGE set. Extra PS09_* variables in the environment are passed through
(e.g. PS09_TOKEN_BUDGET=32768 to sweep the budget) and are recorded in the row.
Peak GPU memory is sampled with nvidia-smi every 200 ms while bench.py runs (device-level memory.used minus the idle
level before the run; includes the CUDA context and PyTorch's cache, so it is an upper bound on what the model needs).
"""
import argparse, csv, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {0: "S0 reference-like (FP32, pad 512, batch 32)", 1: "S1 + FP16, fused QKV, SDPA",
         2: "S2 + length-aware token-budget scheduler", 3: "S3 + async host pipeline, CUDA Graph live path",
         4: "S4 + CLS-only last layer", 5: "S5 + torch.compile"}
FLAGS = ["fp16", "sort", "budget", "bucket", "overlap", "cls_only", "compile", "graph"]


def stage_cfg(st):
    """The flags team.py will use for this stage (preset + PS09_<FLAG> overrides), without importing torch."""
    src = (HERE / "team.py").read_text(encoding="utf-8")
    ns = {}
    start = src.index("STAGES = {")
    exec(src[start:src.index("}\n", start) + 2], ns)  # the literal STAGES dict only
    cfg = dict(ns["STAGES"][st])
    for k in cfg:
        if f"PS09_{k.upper()}" in os.environ:
            cfg[k] = int(os.environ[f"PS09_{k.upper()}"])
    return cfg


def res32_on(cfg):
    """FP32 residual stream is active when the stage uses FP16 and PS09_RES32 (or team.py's DEFAULT_RES32) is 1."""
    if not cfg.get("fp16"):
        return False
    if "PS09_RES32" in os.environ:
        return os.environ["PS09_RES32"] == "1"
    for line in (HERE / "team.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEFAULT_RES32"):
            return line.split("=")[1].split("#")[0].strip() == "1"
    return False


def short_fp32(cfg):
    """P2 cut-off (tokens) when the stage uses FP16: PS09_SHORT_FP32 or team.py's DEFAULT_SHORT_FP32; 0 = off."""
    if not cfg.get("fp16"):
        return 0
    if "PS09_SHORT_FP32" in os.environ:
        return int(os.environ["PS09_SHORT_FP32"])
    for line in (HERE / "team.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEFAULT_SHORT_FP32"):
            return int(line.split("=")[1].split("#")[0].strip())
    return 0


def default_budget():
    src = (HERE / "team.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if line.startswith("DEFAULT_TOKEN_BUDGET"):
            return line.split("=")[1].split("#")[0].strip()
    return "?"


class GpuMemSampler:
    """Samples device memory.used (MiB) with nvidia-smi in a background thread."""

    def __init__(self, period_ms=200):
        self.ok = shutil.which("nvidia-smi") is not None
        self.period, self.samples, self._stop = period_ms / 1000, [], threading.Event()
        self.idle = self._read() if self.ok else None

    def _read(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            return int(out[0])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.period)

    def __enter__(self):
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self.ok:
            self._t.join()

    def peak_mib(self):
        if not self.samples or self.idle is None:
            return ""
        return max(self.samples) - self.idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--limit", type=int); ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--ref-throughput", help="reference throughput JSON (default: <out-dir>/../hardware/ref_throughput.json,"
                                             " then results/ref_throughput.json)")
    ap.add_argument("--tag", default="", help="free-text label stored in each row (e.g. 'budget sweep')")
    a = ap.parse_args()
    d = Path(a.data); res_dir = Path(a.out_dir); res_dir.mkdir(parents=True, exist_ok=True)
    ref_tp = a.ref_throughput
    if not ref_tp:
        for c in (res_dir.parent / "hardware" / "ref_throughput.json", HERE / "results" / "ref_throughput.json"):
            if c.exists() and c.stat().st_size > 0:
                ref_tp = str(c); break
    if not ref_tp:
        print("WARNING: no reference throughput file; speed-up column will be empty", flush=True)
    budget = os.environ.get("PS09_TOKEN_BUDGET", default_budget())
    rows = []
    for st in a.stages:
        suffix = f"_b{budget}" if "PS09_TOKEN_BUDGET" in os.environ else ""
        suffix += f"_n{a.limit}" if a.limit else ""
        out = res_dir / f"bench_stage{st}{suffix}.json"
        log = res_dir / f"bench_stage{st}{suffix}.log"
        cmd = [sys.executable, str(d / "bench.py"), "--team", str(HERE / "team.py"),
               "--passages", str(d / "practice_passages.jsonl"), "--ref", str(d / "practice_ref_first2000.npy"),
               "--out", str(out)]
        if not a.no_trace:
            cmd += ["--trace", str(d / "practice_trace.jsonl"), "--ref-queries", str(d / "practice_ref_queries.npy")]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if ref_tp:
            cmd += ["--ref-throughput", ref_tp]
        env = dict(os.environ, PS09_STAGE=str(st))
        cfg = stage_cfg(st)
        print(f"\n=== {NAMES.get(st, st)}  flags={cfg}  budget={budget} ===\n", " ".join(cmd), flush=True)
        if out.exists():
            out.unlink()  # never report a stale file from an earlier run
        t = time.time()
        with GpuMemSampler() as mem, open(log, "w") as lf:
            p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        wall = round(time.time() - t, 1)
        print(open(log).read()[-3000:], flush=True)
        base = {"when": time.strftime("%Y-%m-%d %H:%M"), "stage": st, "name": NAMES.get(st), "tag": a.tag,
                "flags": " ".join([k for k in FLAGS if cfg.get(k)] + (["res32"] if res32_on(cfg) else []) + ([f"short32<={short_fp32(cfg)}"] if short_fp32(cfg) else [])), "token_budget": budget,
                "peak_gpu_mib": mem.peak_mib(), "wall_s": wall, "result_file": str(out.relative_to(HERE)) if out.is_relative_to(HERE) else str(out)}
        if p.returncode != 0 or not out.exists():
            rows.append(dict(base, status=f"FAILED (exit {p.returncode}, see {log.name})")); continue
        r = json.load(open(out))
        live = r.get("live", {})
        ok = r.get("shape_ok") and r.get("n_violations") == 0 and (not live or live.get("n_violations", 0) == 0)
        rows.append(dict(base, status="ok" if ok else "ACCURACY FAIL",
                         n_passages=r["n_passages"], passages_per_s=round(r["passages_per_second"], 1),
                         seconds=round(r["seconds"], 2), speedup=round(r["speedup"], 2) if "speedup" in r else "",
                         min_cos=round(r["min_cosine"], 6) if r.get("min_cosine") is not None else "",
                         violations=r["n_violations"],
                         live_p50_ms=round(live["p50_ms"], 2) if live else "",
                         live_p99_ms=round(live["p99_ms"], 2) if live else "",
                         live_min_cos=round(live["min_cosine"], 6) if live.get("min_cosine") is not None else "",
                         live_violations=live.get("n_violations", "") if live else "",
                         setup_s=round(r["setup_seconds"], 1)))
    keys = ["when", "stage", "name", "tag", "flags", "status", "n_passages", "passages_per_s", "seconds", "speedup",
            "min_cos", "violations", "live_p50_ms", "live_p99_ms", "live_min_cos", "live_violations", "peak_gpu_mib",
            "setup_s", "token_budget", "wall_s", "result_file"]
    f = res_dir / "ablation.csv"; new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow(r)
    md = ["| Stage | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 ms | Live p99 ms | Live min cos | Peak GPU MiB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] in ("ok", "ACCURACY FAIL"):
            md.append(f"| {r['name']}{' **ACCURACY FAIL**' if r['status'] != 'ok' else ''} | {r['flags']} | {r['n_passages']} | "
                      f"{r['token_budget']} | {r['passages_per_s']} | {r['speedup']} | {r['min_cos']} | {r['violations']} | "
                      f"{r['live_p50_ms']} | {r['live_p99_ms']} | {r['live_min_cos']} | {r['peak_gpu_mib']} |")
        else:
            md.append(f"| {r['name']} | {r['flags']} | | {r['token_budget']} | {r['status']} | | | | | | | |")
    head = f"\n### Run {time.strftime('%Y-%m-%d %H:%M')}{(' - ' + a.tag) if a.tag else ''}\n\nReference throughput file: {ref_tp}\n\n"
    open(res_dir / "ABLATION.md", "a").write(head + "\n".join(md) + "\n")
    print("\n".join(md))
    sys.exit(0 if all(r["status"] == "ok" for r in rows) else 1)


if __name__ == "__main__":
    main()
