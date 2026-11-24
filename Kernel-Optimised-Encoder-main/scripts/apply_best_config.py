"""Picks the best token budget from a sweep (run_ablation.py rows) and, with --apply, writes it into team.py's defaults.

Rule: among rows with status "ok" (0 violations) choose the highest passages/s; ties within 2% go to the SMALLER
budget (less memory, more headroom on an unknown GPU). Rows whose peak GPU memory exceeds 70% of this GPU's VRAM are
rejected, so the default keeps headroom for the organisers' hidden inputs.

Usage: python scripts/apply_best_config.py --sweep results_gpu/token_budget/ablation.csv [--stage 4] [--apply]
"""
import argparse, csv, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def vram_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**20
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True); ap.add_argument("--stage", type=int)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.sweep)))
    vram = vram_mib()
    cands = []
    for r in rows:
        if r["status"] != "ok" or (a.stage is not None and int(r["stage"]) != a.stage):
            continue
        if vram and r.get("peak_gpu_mib") and float(r["peak_gpu_mib"]) > 0.7 * vram:
            print(f"reject budget {r['token_budget']}: peak {r['peak_gpu_mib']} MiB > 70% of {vram:.0f} MiB")
            continue
        cands.append(r)
    if not cands:
        print("no valid rows; defaults unchanged"); sys.exit(1)
    best_tp = max(float(r["passages_per_s"]) for r in cands)
    near = [r for r in cands if float(r["passages_per_s"]) >= 0.98 * best_tp]
    pick = min(near, key=lambda r: int(r["token_budget"]))
    print("| budget | passages/s | min cos | peak GPU MiB |\n|---|---|---|---|")
    for r in sorted(cands, key=lambda r: int(r["token_budget"])):
        print(f"| {r['token_budget']} | {r['passages_per_s']} | {r['min_cos']} | {r.get('peak_gpu_mib', '')} |")
    print(f"\nRECOMMENDED: stage {pick['stage']}, token budget {pick['token_budget']} ({pick['passages_per_s']} p/s; "
          f"fastest was {best_tp} p/s; ties within 2% go to the smaller budget)")
    if a.apply:
        p = ROOT / "team.py"
        s = p.read_text(encoding="utf-8")
        s2 = re.sub(r"^DEFAULT_TOKEN_BUDGET = \d+", f"DEFAULT_TOKEN_BUDGET = {int(pick['token_budget'])}", s, flags=re.M)
        s2 = re.sub(r"^DEFAULT_STAGE = \d+", f"DEFAULT_STAGE = {int(pick['stage'])}", s2, flags=re.M)
        p.write_text(s2, encoding="utf-8")
        print("team.py defaults updated" if s2 != s else "team.py already has these defaults")


if __name__ == "__main__":
    main()
