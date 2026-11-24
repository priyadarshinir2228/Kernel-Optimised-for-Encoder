"""Exercises team.Encoder's FP16 and FP16+res32 code paths on the CPU (native torch.float16 kernels; slow, so only a
few passages: the worst cases from scripts/precision_study.py plus a spread of lengths). Checks that both paths run,
that res32 is at least as accurate as plain FP16 on these inputs, and both pass the 0.9995 gate.

Usage: python tests/test_lowprec_cpu.py --data data
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import team  # noqa: E402

WORST = [19, 826, 1170, 657, 37, 1851, 613, 1707]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / "data")); ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModel
    torch.set_grad_enabled(False)
    team.CFG.update(fp16=0); team.setup()
    d = Path(a.data)
    texts = [json.loads(l)["text"] for l in open(d / "practice_passages.jsonl", encoding="utf-8")][:2000]
    ref = np.load(d / "practice_ref_first2000.npy")
    idx = WORST + [0, 5, 100, 400, 999]
    hf = AutoModel.from_pretrained(team.MODEL).float().eval()
    ids_l, lengths = team._tokenise([texts[i] for i in idx])
    res = {}
    for name, res32 in (("fp16", False), ("fp16_res32", True)):
        enc = team.Encoder(hf, torch.float16, torch.device("cpu"), res32=res32)
        out = []
        for j, ids in enumerate(ids_l):  # one at a time: exact length, no padding
            t = torch.tensor([ids])
            out.append(enc.forward(t, torch.ones_like(t), use_mask=False, cls_only=True)[0].numpy())
        e = np.stack(out); r = ref[idx]
        c = (e * r).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1))
        res[name] = c
        print(f"{name:11s} min cos {c.min():.6f}  per input: {np.round(c, 6).tolist()}")
    ok = res["fp16"].min() >= 0.9995 and res["fp16_res32"].min() >= 0.9995 and res["fp16_res32"].min() >= res["fp16"].min() - 1e-6
    print("RESULT:", "PASS" if ok else "FAIL")
    if a.out:
        json.dump({"what": "native torch.float16 CPU kernels, team.Encoder, one passage per call (no padding)",
                   "passage_index": idx, "cos": {k: [float(x) for x in v] for k, v in res.items()},
                   "min": {k: float(v.min()) for k, v in res.items()}, "pass": bool(ok)}, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
