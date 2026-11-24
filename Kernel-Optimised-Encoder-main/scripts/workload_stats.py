"""NON-OFFICIAL analysis (CPU only, exact): the practice workload and the work each stage's batch plan sends to the GPU.

Tokenises the passages with the model's tokenizer, then replays team._plan / team._build sizing for each stage and
counts token positions processed (batch x padded length), padding fraction, number of batches, and the analytical
FLOPs. No model forward is run, so this is exact and fast; it is the "padding" evidence for the report.

Usage: python scripts/workload_stats.py --data data [--limit N] [--budgets 16384 32768 65536 131072]
       [--out results/workload_stats.json]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D, FF, L = 768, 3072, 12


def flops_per_seq(T, cls_only=False):
    """Forward FLOPs (2 x MACs) of BERT-base for one sequence padded to T (linear layers + attention)."""
    lin = 2 * T * (3 * D * D + D * D + 2 * D * FF)          # QKV, output projection, FFN
    att = 2 * 2 * T * T * D                                   # QK^T and PV
    total = L * (lin + att)
    if cls_only:  # last layer: K,V for all tokens; Q, attention, output projection, FFN for the CLS row only
        last_full = lin + att
        last_cls = 2 * T * 2 * D * D + 2 * (D * D + D * D + 2 * D * FF) + 2 * 2 * T * D
        total += last_cls - last_full
    return total


def plan_stats(team, lengths, cfg, budget):
    team.CFG.update(cfg)
    team.TOKEN_BUDGET = budget
    n = len(lengths)
    positions = flops = 0
    batches = 0
    shapes = []
    for start in range(0, n, team.CHUNK if cfg.get("overlap") else n):
        part = lengths[start:start + (team.CHUNK if cfg.get("overlap") else n)]
        for idx in team._plan(part):
            T = team._pad8(int(part[idx].max())) if (cfg["sort"] or cfg["budget"]) else team.MAX_LEN
            positions += len(idx) * T
            flops += len(idx) * flops_per_seq(T, bool(cfg["cls_only"]))
            batches += 1
            shapes.append((len(idx), T))
    real = int(lengths.sum())
    return {"batches": batches, "token_positions": int(positions), "real_tokens": real,
            "padding_fraction": 1 - real / positions, "gflops_total": flops / 1e9,
            "largest_batch_rows": max(s[0] for s in shapes), "largest_batch_tokens": max(s[0] * s[1] for s in shapes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--limit", type=int)
    ap.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--out", default=str(ROOT / "results" / "workload_stats.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import team
    tok = AutoTokenizer.from_pretrained(team.MODEL)
    fast = tok.backend_tokenizer
    fast.enable_truncation(team.MAX_LEN); fast.no_padding()
    d = Path(a.data)
    texts = [json.loads(l)["text"] for l in open(d / "practice_passages.jsonl", encoding="utf-8")][: a.limit]
    t = time.perf_counter()
    lengths = np.array([len(e.ids) for e in fast.encode_batch(texts)], dtype=np.int64)
    tok_s = time.perf_counter() - t
    qs = [json.loads(l)["text"] for l in open(d / "practice_trace.jsonl", encoding="utf-8")]
    qlen = np.array([len(e.ids) for e in fast.encode_batch(qs)])
    trace_t = np.array([json.loads(l)["t"] for l in open(d / "practice_trace.jsonl", encoding="utf-8")])
    gaps = np.diff(trace_t)

    bins = [0, 16, 32, 64, 128, 256, 384, 513]
    hist = {f"{bins[i]}-{bins[i + 1] - 1}": float(((lengths >= bins[i]) & (lengths < bins[i + 1])).mean()) for i in range(len(bins) - 1)}
    res = {"what": "exact workload analysis (tokenizer only, no model run)", "n_passages": len(texts),
           "tokenise_seconds_single_call": round(tok_s, 2), "cpu_count": os.cpu_count(),
           "passage_tokens": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                              "p10": float(np.percentile(lengths, 10)), "p90": float(np.percentile(lengths, 90)),
                              "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max()),
                              "truncated_at_512": int((lengths >= 512).sum()), "histogram": hist},
           "first_20k_vs_all_mean_tokens": [float(lengths[:20000].mean()), float(lengths.mean())],
           "queries": {"n": len(qs), "mean_tokens": float(qlen.mean()), "max_tokens": int(qlen.max()),
                       "duration_s": float(trace_t[-1]), "mean_gap_ms": float(gaps.mean() * 1000),
                       "gaps_under_10ms": int((gaps < 0.010).sum()),
                       "bucket_coverage": {str(b): float((qlen <= b).mean()) for b in (16, 32, 64, 128)}}}
    ref_pos = len(texts) * 512
    res["reference_token_positions"] = ref_pos
    res["reference_padding_fraction"] = 1 - lengths.sum() / ref_pos
    res["reference_gflops"] = len(texts) * flops_per_seq(512) / 1e9

    from team import STAGES
    stages = {}
    for st in (0, 1, 2, 3, 4):
        stages[st] = plan_stats(team, lengths, dict(STAGES[st]), 65536)
        stages[st]["flop_reduction_vs_reference"] = res["reference_gflops"] / stages[st]["gflops_total"]
    res["stages_at_budget_65536"] = stages
    res["budget_sweep_stage4"] = {b: plan_stats(team, lengths, dict(STAGES[4]), b) for b in a.budgets}
    for b, s in res["budget_sweep_stage4"].items():
        s["flop_reduction_vs_reference"] = res["reference_gflops"] / s["gflops_total"]
    # O2b ablation: length-bucket splitting off / on with different minimum fills, at each budget
    abl = {}
    mf0 = team.MIN_FILL
    for b in a.budgets:
        for bucket, mf in ((0, mf0), (1, 0.125), (1, 0.25), (1, 0.5)):
            team.MIN_FILL = mf
            s = plan_stats(team, lengths, dict(STAGES[4], bucket=bucket), b)
            s["flop_reduction_vs_reference"] = res["reference_gflops"] / s["gflops_total"]
            abl[f"budget={b} bucket={bucket} min_fill={mf if bucket else '-'}"] = s
    team.MIN_FILL = mf0
    res["bucket_ablation_stage4"] = abl

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"passages {len(texts)}: mean {lengths.mean():.1f} tokens, median {np.median(lengths):.0f}, max {lengths.max()}, "
          f"reference padding {res['reference_padding_fraction']:.1%}, tokenise {tok_s:.1f}s")
    print("stage | batches | positions | padding | GFLOPs | FLOP reduction")
    for st, s in stages.items():
        print(f"S{st} | {s['batches']} | {s['token_positions']:,} | {s['padding_fraction']:.1%} | {s['gflops_total']:.0f} | {s['flop_reduction_vs_reference']:.2f}x")
    print("budget sweep (stage 4):")
    for b, s in res["budget_sweep_stage4"].items():
        print(f"  {b}: batches {s['batches']}, padding {s['padding_fraction']:.1%}, max rows {s['largest_batch_rows']}, "
              f"FLOP reduction {s['flop_reduction_vs_reference']:.2f}x")
    print("bucket ablation (stage 4):")
    for k, s in abl.items():
        print(f"  {k}: batches {s['batches']}, padding {s['padding_fraction']:.1%}, "
              f"FLOP reduction {s['flop_reduction_vs_reference']:.2f}x")
    print("queries:", res["queries"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
