"""Builds the final technical report from docs/technical_report.md + the result files.

docs/technical_report.md is the editable source. Blocks between <!-- AUTO:name --> and <!-- /AUTO:name --> are
regenerated from result files every time this runs (edit everything else freely). Charts are drawn only from result
files that exist; a missing result shows as PENDING, never as a placeholder number.

Outputs: docs/technical_report.md (AUTO blocks refreshed), reports/charts/*.png, reports/PS09_Final_Report.html,
         reports/PS09_Final_Report.pdf (via headless Microsoft Edge / Chrome / Chromium if one is installed).

Usage: python scripts/make_report.py [--gpu results_gpu] [--no-pdf]
"""
import argparse, base64, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_results import jload, rows, summarise  # noqa: E402

SRC = ROOT / "docs" / "technical_report.md"
OUT = ROOT / "reports"
CH = OUT / "charts"

# chart styling (reference palette, light mode: the report is printed)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, INK2, GRID, SURF = "#a3a29d", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "savefig.dpi": 160, "savefig.bbox": "tight"})
    return plt


def bars(plt, labels, values, colors, title, xlabel, path, fmt="{:,.0f}", ref_lines=(), log=False):
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(labels) + 1.2))
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.55, edgecolor=SURF, linewidth=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
    for yi, v in zip(y, values):
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left", color=INK2, fontsize=9)
    for x, lab in ref_lines:
        ax.axvline(x, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(x, 0.99, f" {lab} ", color=INK2, fontsize=8, va="top", ha="left", rotation=90,
                transform=ax.get_xaxis_transform())
    ax.set_title(title, pad=10); ax.set_xlabel(xlabel)
    ax.margins(x=0.18)
    fig.savefig(path); plt.close(fig)
    return path


def charts(cpu, gpu):
    plt = plt_setup()
    CH.mkdir(parents=True, exist_ok=True)
    made = {}
    ws = jload(cpu / "workload_stats.json") or jload(ROOT / "results" / "workload_stats.json")
    if ws:
        h = ws["passage_tokens"]["histogram"]
        made["lengths"] = bars(plt, list(h), [100 * v for v in h.values()], [BLUE] * len(h),
                               f"Passage length distribution ({ws['n_passages']:,} practice passages)",
                               "share of passages (%)", CH / "lengths.png", fmt="{:.1f}%")
        s = ws["stages_at_budget_65536"]
        abl = ws.get("bucket_ablation_stage4", {})
        off = abl.get("budget=65536 bucket=0 min_fill=-", s["4"])
        lab = ["reference.py (pad 512)", "S4 sorted token budget (no O2b)", "S4 submitted (+ O2b bucket split)"]
        vals = [ws["reference_token_positions"] / 1e6, off["token_positions"] / 1e6, s["4"]["token_positions"] / 1e6]
        made["positions"] = bars(plt, lab, vals, [GREY, BLUE, BLUE],
                                 "Token positions computed for 100k passages (millions; real tokens = "
                                 f"{s['4']['real_tokens'] / 1e6:.1f} M)", "token positions (millions, exact count)",
                                 CH / "positions.png", fmt="{:.1f} M")
    ps = jload(cpu / "precision_study.json")
    if ps:
        names = {"fp32": "FP32 (sanity)", "fp16": "FP16", "fp16_last32": "FP16, last layer FP32",
                 "fp16_res32": "FP16 + FP32 residual (submitted)", "bf16": "BF16"}
        v = ps["variants"]
        keys = [k for k in names if k in v and k != "fp32"]
        err = [max(1 - v[k]["passages"]["min"], 1e-9) for k in keys]
        made["precision"] = bars(plt, [names[k] for k in keys], err,
                                 [BLUE if k == "fp16_res32" else GREY for k in keys],
                                 "Worst-case error 1 - cos on 2,000 reference passages (CPU emulation; lower is better)",
                                 "1 - min cosine (log scale)", CH / "precision.png", fmt="{:.1e}",
                                 ref_lines=((1e-3, "rule 0.999"), (5e-4, "gate 0.9995")), log=True)
    if gpu:
        ab = [r for r in rows(gpu / "ablation" / "ablation.csv") if r["status"] == "ok" and r.get("passages_per_s")]
        ref = jload(gpu / "hardware" / "ref_throughput.json")
        if ab and ref:
            lab = ["reference.py (5k sample)"] + [f"S{r['stage']} {r.get('tag', '')}".replace("ablation 20k + live", "").replace("ablation 20k", "").strip()
                                                 + f" [{r['flags']}]" for r in ab]
            vals = [ref["passages_per_second"]] + [float(r["passages_per_s"]) for r in ab]
            made["ablation"] = bars(plt, lab, vals, [GREY] + [BLUE] * len(ab),
                                    "Throughput by stage (official bench.py, same GPU)", "passages / second",
                                    CH / "ablation.png", fmt="{:,.0f}")
        sw = [r for r in rows(gpu / "token_budget" / "ablation.csv") if r["status"] == "ok"]
        if sw:
            sw.sort(key=lambda r: int(r["token_budget"]))
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            x = [int(r["token_budget"]) for r in sw]; y = [float(r["passages_per_s"]) for r in sw]
            ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2)
            ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels([f"{v // 1024}k" for v in x])
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:,.0f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", color=INK2, fontsize=9)
            ax.set_title("Token-budget sweep, stage 4 (official bench.py)"); ax.set_xlabel("token budget per batch")
            ax.set_ylabel("passages / second"); ax.margins(y=0.25)
            fig.savefig(CH / "budget.png"); plt.close(fig); made["budget"] = CH / "budget.png"
        prof = {n: jload(gpu / "profiler" / n / "summary.json") for n in ("reference", "stage1", "stage2", "default")}
        prof = {n: p for n, p in prof.items() if p and p.get("gpu_ms_by_category")}
        if prof:
            cats = ["gemm", "attention", "norm+elementwise", "memcpy", "other"]
            cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
            fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(prof) + 1.6))
            names = list(prof)[::-1]
            left = [0.0] * len(names)
            for c, col in zip(cats, cols):
                vals = [prof[n]["gpu_ms_by_category"].get(c, 0) / 1000 for n in names]
                ax.barh(names, vals, left=left, color=col, edgecolor=SURF, linewidth=2, height=0.55, label=c)
                left = [l + v for l, v in zip(left, vals)]
            for i, tot in enumerate(left):
                ax.text(tot, i, f"  {tot:.1f} s", va="center", color=INK2, fontsize=9)
            ax.grid(axis="y", visible=False); ax.margins(x=0.15)
            ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=8)
            ax.set_title("GPU kernel time by category, 2,000 passages (torch.profiler)"); ax.set_xlabel("GPU seconds")
            fig.savefig(CH / "profile.png"); plt.close(fig); made["profile"] = CH / "profile.png"
    return made


def auto_blocks(cpu, gpu, made):
    b = {}
    ws = jload(cpu / "workload_stats.json") or jload(ROOT / "results" / "workload_stats.json")
    if ws:
        pt = ws["passage_tokens"]
        s = ws["stages_at_budget_65536"]
        lines = [f"Measured with the model's tokenizer on all {ws['n_passages']:,} practice passages (exact counts, "
                 "`scripts/workload_stats.py`).\n",
                 f"- Tokens per passage: mean **{pt['mean']:.1f}**, median {pt['median']:.0f}, p10 {pt['p10']:.0f}, "
                 f"p90 {pt['p90']:.0f}, p99 {pt['p99']:.0f}, max {pt['max']}; {pt['truncated_at_512']} reach the 512 limit.",
                 f"- Padding in the reference (every input padded to 512): **{ws['reference_padding_fraction']:.1%}** of all token positions.",
                 f"- Queries: {ws['queries']['n']} over {ws['queries']['duration_s']:.0f} s, mean {ws['queries']['mean_tokens']:.1f} tokens, "
                 f"max {ws['queries']['max_tokens']}; {ws['queries']['gaps_under_10ms']} arrivals come < 10 ms after the previous one (bursts).\n",
                 "| Plan | Batches | Token positions | Padding | GFLOPs (analytical) | Work reduction vs reference |",
                 "|---|---|---|---|---|---|",
                 f"| reference.py | 3,125 | {ws['reference_token_positions']:,} | {ws['reference_padding_fraction']:.1%} | {ws['reference_gflops']:,.0f} | 1.00x |"]
        for st in ("2", "4"):
            r = s[st]
            lines.append(f"| S{st} | {r['batches']} | {r['token_positions']:,} | {r['padding_fraction']:.1%} | {r['gflops_total']:,.0f} | "
                         f"{r['flop_reduction_vs_reference']:.2f}x |")
        abl = ws.get("bucket_ablation_stage4", {})
        k_off, k_on = "budget=65536 bucket=0 min_fill=-", "budget=65536 bucket=1 min_fill=0.25"
        if k_off in abl and k_on in abl:
            lines.append(f"\nLength-bucket split (O2b) at the 65,536-token budget: padding {abl[k_off]['padding_fraction']:.1%} -> "
                         f"{abl[k_on]['padding_fraction']:.1%}, work reduction {abl[k_off]['flop_reduction_vs_reference']:.2f}x -> "
                         f"{abl[k_on]['flop_reduction_vs_reference']:.2f}x.")
        for k in ("lengths", "positions"):
            if k in made:
                lines.append(f"\n![{k}](../reports/charts/{made[k].name})")
        b["workload"] = "\n".join(lines)
    ps = jload(cpu / "precision_study.json")
    lp = jload(cpu / "test_lowprec_cpu.json")
    if ps:
        lines = [f"CPU emulation of GPU FP16 numerics against the official FP32 reference ({ps['n_passages']:,} passages, "
                 f"{ps['n_queries']:,} queries; `scripts/precision_study.py`). Label: **measured, CPU emulation**.\n",
                 "| Variant | Passages min cos | p0.1 | Mean | < 0.9995 | Queries min cos | Worst passage (tokens) |",
                 "|---|---|---|---|---|---|---|"]
        for k, v in ps["variants"].items():
            p, q = v["passages"], v.get("queries", {})
            w = p["worst"][0]
            lines.append(f"| {k} | {p['min']:.6f} | {p['p0.1']:.6f} | {p['mean']:.7f} | {p['below_0.9995']} | "
                         f"{q.get('min', float('nan')):.6f} | #{w['i']} ({w['tokens']}) |")
        if lp:
            lines.append(f"\nNative `torch.float16` kernels (CPU, `tests/test_lowprec_cpu.py`, the 8 worst passages + 5 others): "
                         f"plain FP16 min cos **{lp['min']['fp16']:.6f}**, FP16 + FP32 residual **{lp['min']['fp16_res32']:.6f}**.")
        if "precision" in made:
            lines.append(f"\n![precision](../reports/charts/{made['precision'].name})")
        b["precision"] = "\n".join(lines)
    tc = jload(cpu / "test_correctness.json")
    if tc:
        lines = [f"`tests/test_correctness.py` on CPU ({tc['n_cases']} edge cases, torch {tc['torch']}): "
                 f"**{'ALL PASS' if not tc['failed'] else 'FAILED: ' + ', '.join(tc['failed'])}**.\n",
                 "| Stage | Flags | Batch min cos | Single-call min cos |", "|---|---|---|---|"]
        for s in tc["stages"]:
            lines.append(f"| {s['stage']} | {' '.join(k for k, v in s['cfg'].items() if v)} | {s['batch_min_cos']:.7f} | {s['single_min_cos']:.7f} |")
        b["cpu_tests"] = "\n".join(lines)
    dr = rows(cpu / "bench_dryrun" / "ablation.csv")
    old = rows(ROOT / "results_cpu_dryrun" / "ablation.csv")
    if dr or old:
        lines = ["CPU dry runs of the official `bench.py` (FP32 on a CPU; they prove the scheduler gains are real and exact, "
                 "but they are **not** GPU scores).\n", "| Source | Stage | N | Passages/s | Speed-up vs reference on same CPU | Min cos |",
                 "|---|---|---|---|---|---|"]
        for src, rs in (("teammate, cloud CPU (reported, files in results_cpu_dryrun/)", old), ("this laptop, i7-13620H (measured)", dr)):
            for r in rs:
                if r.get("passages_per_s"):
                    lines.append(f"| {src} | {r['stage']} | {r.get('n_passages', 300)} | {r['passages_per_s']} | {r['speedup']}x | {r['min_cos']} |")
        b["cpu_bench"] = "\n".join(lines)
    if gpu and gpu.exists():
        g = summarise(gpu)
        for k in ("ablation", "budget", "profile"):
            if k in made:
                g += f"\n![{k}](../reports/charts/{made[k].name})\n"
        b["gpu"] = g
    else:
        b["gpu"] = ("**PENDING - no GPU results in this checkout.** Run `bash run_all.sh` (and `--full`) on the RTX machine, "
                    "copy `results_gpu/` here, then re-run `python scripts/make_report.py`.\n\n"
                    "Previously reported by the teammate (TRACKER.md, files not in the repository): reference 28.4 passages/s on the "
                    "RTX 3050 Laptop GPU (5,000 passages); stage-4 FP16 min cos 0.999663 (passages) / 0.999982 (queries).")
    return b


def refresh(md, blocks):
    for k, v in blocks.items():
        pat = re.compile(rf"(<!-- AUTO:{k} -->)(.*?)(<!-- /AUTO:{k} -->)", re.S)
        md = pat.sub(lambda m: f"{m.group(1)}\n{v}\n{m.group(3)}", md)
    return md


CSS = """
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fff;max-width:900px;margin:32px auto;
     padding:0 16px;line-height:1.5;font-size:14px}
h1{font-size:26px;margin-bottom:4px} h2{font-size:19px;border-bottom:1px solid #e4e3df;padding-bottom:4px;margin-top:28px}
h3{font-size:15px} table{border-collapse:collapse;margin:10px 0;font-size:12px;width:100%}
th,td{border:1px solid #e4e3df;padding:4px 6px;text-align:left;vertical-align:top} th{background:#f3f2ef}
code{background:#f3f2ef;padding:1px 4px;border-radius:3px;font-size:12px} pre{background:#f3f2ef;padding:10px;overflow-x:auto}
img{max-width:100%} blockquote{border-left:3px solid #2a78d6;margin:8px 0;padding:2px 12px;color:#52514e}
@media print{body{margin:0;max-width:none} h2{page-break-after:avoid} table,img{page-break-inside:avoid}}
"""


def to_html(md):
    import markdown
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])

    def inline(m):  # embed charts so the HTML/PDF is self-contained
        p = (SRC.parent / m.group(1)).resolve()
        if p.exists():
            return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
        return m.group(0)
    body = re.sub(r'src="([^"]+\.png)"', inline, body)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>PS-09 Final Report</title><style>{CSS}</style></head><body>{body}</body></html>"


def find_browser():
    cands = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for c in cands:
        if os.path.exists(c):
            return c
    for n in ("msedge", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave", "brave-browser"):
        if shutil.which(n):
            return shutil.which(n)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=str(ROOT / "results_gpu")); ap.add_argument("--cpu", default=str(ROOT / "results_cpu"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    gpu = Path(a.gpu) if Path(a.gpu).exists() else None
    made = charts(Path(a.cpu), gpu)
    md = refresh(SRC.read_text(encoding="utf-8"), auto_blocks(Path(a.cpu), gpu, made))
    SRC.write_text(md, encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    html = OUT / "PS09_Final_Report.html"
    html.write_text(to_html(md), encoding="utf-8")
    print("wrote", SRC.relative_to(ROOT), "and", html.relative_to(ROOT), "charts:", ", ".join(made))
    if a.no_pdf:
        return
    br = find_browser()
    if not br:
        print("no Edge/Chrome/Chromium found: open the HTML and print to PDF"); return
    pdf = OUT / "PS09_Final_Report.pdf"
    subprocess.run([br, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.resolve().as_uri()],
                   capture_output=True, timeout=180)
    print("wrote", pdf.relative_to(ROOT) if pdf.exists() else "NO PDF (browser failed)")


if __name__ == "__main__":
    main()
