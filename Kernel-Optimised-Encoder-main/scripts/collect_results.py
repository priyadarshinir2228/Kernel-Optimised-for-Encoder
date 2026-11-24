"""Builds a Markdown summary of every result file that exists (nothing is typed in by hand; missing results are
reported as PENDING). Used by run_all.sh (results_gpu/SUMMARY.md) and by scripts/make_report.py.

Usage: python scripts/collect_results.py [--results results_gpu]
"""
import argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def rows(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8")))
    except Exception:
        return []


def fmt(x, nd=6):
    if x is None or x == "":
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summarise(R):
    R = Path(R)
    out = []
    env = jload(R / "hardware" / "env.json")
    ref = jload(R / "hardware" / "ref_throughput.json")
    out.append("## Hardware and baseline (measured)\n")
    if env:
        g = env.get("gpu", {})
        out.append(f"- GPU: {g.get('name', '-')} (compute {g.get('compute_capability', '-')}, {g.get('vram_gib', '-')} GiB, "
                   f"{g.get('sm_count', '-')} SMs, BF16 {g.get('bf16_supported', '-')})")
        out.append(f"- Software: Python {env.get('python')}, torch {env.get('torch')} (CUDA {env.get('torch_cuda')}), "
                   f"transformers {env.get('transformers')}, tokenizers {env.get('tokenizers')}; OS {env.get('os')}")
    else:
        out.append("- Hardware report: PENDING (run `bash run_all.sh --step env` on the GPU machine)")
    if ref:
        out.append(f"- Reference baseline: **{ref['passages_per_second']:.1f} passages/s** (reference.py, FP32, pad 512, "
                   f"first {ref['n']} passages, {ref['seconds']:.0f} s)")
    else:
        out.append("- Reference baseline: PENDING")

    out.append("\n## Accuracy on the official references (measured)\n")
    out.append("| Run | Passages min cos | Passages mean | Queries min cos | Max abs err | Verdict |\n|---|---|---|---|---|---|")
    for name in ("validate_default", "validate_fp16_plain", "validate_bf16", "validate_lax_reduction", "validate_fp16_accum"):
        v = jload(R / "accuracy" / f"{name}.json")
        if v:
            p, q = v.get("passages", {}), v.get("queries", {})
            out.append(f"| {name} | {fmt(p.get('min'))} | {fmt(p.get('mean'), 7)} | {fmt(q.get('min'))} | "
                       f"{fmt(p.get('max_abs_error'), 5)} | {v.get('verdict')} |")
        else:
            out.append(f"| {name} | PENDING | | | | |")
    t = jload(R / "accuracy" / "test_correctness.json")
    out.append(f"\nEdge-case suite: {'PENDING' if not t else ('ALL PASS' if not t['failed'] else 'FAILED: ' + ', '.join(t['failed']))}")

    for title, sub in (("Ablation (official bench.py)", "ablation"), ("Token-budget sweep (official bench.py, stage 4)", "token_budget"),
                       ("Final run (official bench.py)", "final")):
        out.append(f"\n## {title}\n")
        rs = rows(R / sub / "ablation.csv")
        if not rs:
            out.append("PENDING")
            continue
        out.append("| Stage | Tag | Flags | N | Budget | Passages/s | Speed-up | Min cos | Viol. | Live p50 | Live p99 | Peak MiB | Status |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            out.append(f"| {r['stage']} | {r.get('tag', '')} | {r.get('flags', '')} | {r.get('n_passages', '')} | {r['token_budget']} | "
                       f"{r.get('passages_per_s', '')} | {r.get('speedup', '')} | {r.get('min_cos', '')} | {r.get('violations', '')} | "
                       f"{r.get('live_p50_ms', '')} | {r.get('live_p99_ms', '')} | {r.get('peak_gpu_mib', '')} | {r['status']} |")

    out.append("\n## Live-query latency detail (non-official replay of bench.py timing)\n")
    out.append("| Run | p50 ms | p90 | p95 | p99 | max | service p50 | service p99 | min cos |\n|---|---|---|---|---|---|---|---|---|")
    for name in ("realtime_default", "service_graph_on", "service_graph_off"):
        v = jload(R / "latency" / f"{name}.json")
        if not v:
            out.append(f"| {name} | PENDING | | | | | | | |"); continue
        L, S = v.get("latency_ms", {}), v["service_ms"]
        out.append(f"| {name} | {fmt(L.get('p50'), 2)} | {fmt(L.get('p90'), 2)} | {fmt(L.get('p95'), 2)} | {fmt(L.get('p99'), 2)} | "
                   f"{fmt(L.get('max'), 2)} | {S['p50']:.2f} | {S['p99']:.2f} | {v['min_cosine']:.6f} |")

    out.append("\n## Profiler (torch.profiler, 2,000 passages; wall time includes profiler overhead)\n")
    out.append("GPU time = sum of kernel + memcpy durations from the exported trace (`profile_run.py --resummarise` recomputes it). "
               "*of which FP32 GEMM*: the reference's GEMMs, or in the default config the short-input (P2) path.\n")
    out.append("| Run | GPU kernel ms | GEMM | of which FP32 GEMM | Attention | Norm+elementwise | Memcpy | Other | Device busy | Idle gaps >100us | Tokenise share |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("reference", "stage1", "stage2", "default"):
        v = jload(R / "profiler" / name / "summary.json")
        if not v:
            out.append(f"| {name} | PENDING | | | | | | | | | |"); continue
        sh, tl = v.get("gpu_time_share", {}), v.get("timeline", {})
        pc = lambda k: f"{100 * sh.get(k, 0):.1f}%"
        out.append(f"| {name} | {v['gpu_kernel_ms']:.0f} | {pc('gemm')} | {100 * v.get('fp32_gemm_share', 0):.1f}% | {pc('attention')} | {pc('norm+elementwise')} | {pc('memcpy')} | "
                   f"{pc('other')} | {fmt(tl.get('device_busy_fraction_of_span'), 3)} | {tl.get('idle_gaps_over_100us', '-')} | "
                   f"{fmt(v.get('tokenise_share_of_wall'), 3)} |")

    out.append("\n## Full 100k accuracy check (non-official)\n")
    fc = jload(R / "final" / "full_check.json")
    if fc:
        ab = fc["a_vs_b"]
        out.append(f"- Submitted config vs FP32 path on all {fc['n']:,} passages: min cos **{ab['min']:.6f}**, mean {ab['mean']:.7f}, "
                   f"max abs error {ab['max_abs_error']:.5f}, violations (<0.999) **{ab['violations_0.999']}**, below 0.9995: {ab['below_0.9995']}")
        out.append(f"- FP32 path vs official reference (first 2,000): min cos {fc['b_vs_official_ref_first2000']['min']:.6f}")
        out.append(f"- Worst input: #{ab['worst'][0]['i']} ({ab['worst'][0]['tokens']} tokens) cos {ab['worst'][0]['cos']:.6f}")
    else:
        out.append("PENDING (`bash run_all.sh --step full`)")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results_gpu"))
    a = ap.parse_args()
    print(f"# PS-09 results summary ({a.results})\n\nGenerated from result files by scripts/collect_results.py.\n")
    print(summarise(a.results))


if __name__ == "__main__":
    main()
