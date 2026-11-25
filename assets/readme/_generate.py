"""Generate the animated README SVGs (CSS keyframes only: GitHub strips JS, so anime.js-style motion is baked in).

Every element's resting style is its FINAL state; keyframes only describe the journey there, so
prefers-reduced-motion (animation:none) shows the finished figure.
"""
import math
import os
import sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

SANS = "'Segoe UI',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif"
MONO = "ui-monospace,'SFMono-Regular','Cascadia Mono',Consolas,Menlo,monospace"
C = dict(bg="#0b1016", panel="#111922", line="#223040", grid="#1a2531", ink="#e6edf3", mute="#8b98a5",
         dim="#5d6b78", blue="#3b8ff0", orange="#f07a3c", green="#22c48a", red="#f25f5c", yellow="#f2c94c",
         pad="#2c3946", violet="#a78bfa")
EASE = "cubic-bezier(.16,1,.3,1)"  # anime.js outExpo-like


class SVG:
    def __init__(self, w, h, title, dur=9.0, loop=True):
        self.w, self.h, self.title, self.dur, self.loop = w, h, title, dur, loop
        self.body, self.kf, self.n = [], [], 0

    def add(self, s):
        self.body.append(s)

    def anim(self, frm, to, t0, t1, extra_from=""):
        """Return a class that animates from `frm` CSS to `to` CSS between t0..t1 seconds, then holds."""
        self.n += 1
        name = f"k{self.n}"
        D = self.dur
        p0, p1 = 100 * t0 / D, 100 * t1 / D
        if self.loop:
            self.kf.append(
                f"@keyframes {name}{{0%,{p0:.2f}%{{{frm}{extra_from};animation-timing-function:{EASE}}}"
                f"{p1:.2f}%,100%{{{to}}}}}"
                f".{name}{{animation:{name} {D}s linear infinite both}}")
        else:
            self.kf.append(
                f"@keyframes {name}{{0%{{{frm}{extra_from}}}100%{{{to}}}}}"
                f".{name}{{animation:{name} {t1 - t0:.2f}s {EASE} {t0:.2f}s both}}")
        return name

    # common motions
    def grow_x(self, t0, t1):
        return self.anim("transform:scaleX(0)", "transform:scaleX(1)", t0, t1) + " ox"

    def grow_y(self, t0, t1):
        return self.anim("transform:scaleY(0)", "transform:scaleY(1)", t0, t1) + " oy"

    def fade_up(self, t0, t1, dy=14):
        return self.anim(f"opacity:0;transform:translateY({dy}px)", "opacity:1;transform:translateY(0)", t0, t1)

    def fade(self, t0, t1):
        return self.anim("opacity:0", "opacity:1", t0, t1)

    def render(self):
        D = self.dur
        loopfade = ""
        if self.loop:
            loopfade = (f"@keyframes lf{{0%{{opacity:0}}2%,95%{{opacity:1}}100%{{opacity:0}}}}"
                        f".lf{{animation:lf {D}s linear infinite both}}")
        css = (f"text{{font-family:{SANS}}}.mono{{font-family:{MONO}}}"
               f".mute{{fill:{C['mute']}}}.dim{{fill:{C['dim']}}}"
               ".ox{transform-box:fill-box;transform-origin:0% 50%}"
               ".oy{transform-box:fill-box;transform-origin:50% 100%}"
               ".oc{transform-box:fill-box;transform-origin:50% 50%}"
               + loopfade + "".join(self.kf) +
               "@media (prefers-reduced-motion:reduce){*{animation:none!important}.motion{display:none}}")
        wrap_open = '<g class="lf">' if self.loop else "<g>"
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{esc(self.title)}">\n<title>{esc(self.title)}</title>\n'
                f"<style>{css}</style>\n"
                f'<defs><linearGradient id="glow" x1="0" y1="0" x2="1" y2="1">'
                f'<stop offset="0" stop-color="#16324f" stop-opacity=".55"/><stop offset=".55" stop-color="{C["bg"]}" stop-opacity="0"/></linearGradient>'
                f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="6" height="6" fill="{C["pad"]}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#3a4958" stroke-width="2"/></pattern></defs>\n'
                f'<rect width="{self.w}" height="{self.h}" rx="16" fill="{C["bg"]}"/>'
                f'<rect width="{self.w}" height="{self.h}" rx="16" fill="url(#glow)"/>'
                f'<rect x=".5" y=".5" width="{self.w - 1}" height="{self.h - 1}" rx="15.5" fill="none" stroke="{C["line"]}"/>\n'
                + wrap_open + "\n" + "\n".join(self.body) + "\n</g></svg>\n")

    def save(self, name):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(self.render())
        print("wrote", name)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def T(x, y, s, size=13, cls="", weight=400, anchor="start", fill=None, extra=""):
    f = f' fill="{fill or C["ink"]}"'
    c = f' class="{cls}"' if cls else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{f}{c}{extra}>'
            f"{esc(s)}</text>")


def header(s, kicker, title, sub=None):
    s.add(T(32, 40, kicker, 11, "mono mute", 600, extra=' letter-spacing="1.5"'))
    s.add(T(32, 68, title, 21, "", 700))
    if sub:
        s.add(T(32, 90, sub, 13, "mute"))


# =====================================================================================================================
# 1. HERO
def hero():
    s = SVG(960, 420, "604.9 passages per second, 21.3x faster than the reference encoder", loop=False)
    s.add(T(40, 50, "PS-09 · KERNEL OPTIMISATION FOR ENCODERS", 11, "mono mute", 600, extra=' letter-spacing="2"'))
    s.add(f'<g class="{s.fade_up(0, .7)}">' + T(40, 92, "Kernel-Optimised Encoder", 38, "", 800) + "</g>")
    s.add(f'<g class="{s.fade_up(.15, .85)}">' + T(40, 120, "BAAI/bge-base-en-v1.5 · same weights · same maths · one RTX 3050 6 GB Laptop GPU", 14, "mute") + "</g>")
    # odometer 604.9
    s.add(f'<defs><clipPath id="odo"><rect x="36" y="150" width="270" height="96"/></clipPath></defs>')
    digits = [("6", 66), ("0", 120), ("4", 174), (".", 214), ("9", 254)]
    s.add('<g clip-path="url(#odo)">')
    for i, (d, cx) in enumerate(digits):
        if d == ".":
            s.add(T(cx, 228, ".", 84, "mono", 700, "middle", C["blue"]))
            continue
        target = 10 + int(d)
        cls = s.anim("transform:translateY(0)", f"transform:translateY(-{target * 100}px)", .2 + .12 * i, 1.9 + .25 * i)
        col = "".join(T(cx, 228 + k * 100, str(k % 10), 84, "mono", 700, "middle", C["blue"]) for k in range(20))
        s.add(f'<g class="{cls}" style="transform:translateY(-{target * 100}px)">{col}</g>')
    s.add("</g>")
    s.add(f'<g class="{s.fade(1.6, 2.4)}">' + T(306, 226, "passages / s", 16, "mute", 600) + "</g>")
    s.add(T(40, 268, "official bench.py · all 100,000 practice passages · end to end, tokenisation included", 12, "dim"))
    # bars
    scale = 300 / 604.9
    rows = [("reference.py  (FP32, pad 512)", 28.4, C["mute"], .4), ("team.py  (this repo)", 604.9, C["blue"], .7)]
    for j, (lab, v, col, t0) in enumerate(rows):
        y = 300 + j * 36
        s.add(T(40, y + 13, lab, 12, "mute"))
        s.add(f'<rect x="250" y="{y}" width="300" height="18" rx="4" fill="{C["panel"]}"/>')
        s.add(f'<rect class="{s.grow_x(t0, t0 + 1.6)}" x="250" y="{y}" width="{v * scale:.1f}" height="18" rx="4" fill="{col}"/>')
        s.add(f'<g class="{s.fade(t0 + 1.0, t0 + 1.8)}">' + T(250 + v * scale + 8 if v < 400 else 250 + v * scale - 8, y + 13,
              f"{v:g} p/s", 12, "mono", 700, "start" if v < 400 else "end", C["ink"] if v < 400 else "#06121f") + "</g>")
    s.add(T(40, 392, "Measured on the same GPU · sources linked under every figure", 11, "dim"))
    # tiles
    tiles = [("21.3×", "speed-up vs reference.py", "28.4 → 604.9 passages/s", C["blue"]),
             ("0", "tolerance violations", "of 100,000 · min cos 0.999992", C["green"]),
             ("4.4 ms", "live-query p50", "p99 8.2 ms · 1,500-query trace", C["orange"]),
             ("1.6 GB", "peak GPU memory", "of 6 GB · token budget 8,192", C["yellow"])]
    for i, (big, l1, l2, col) in enumerate(tiles):
        x = 590 + (i % 2) * 180
        y = 150 + (i // 2) * 128
        g = s.fade_up(.9 + .15 * i, 1.8 + .15 * i, 18)
        s.add(f'<g class="{g}"><rect x="{x}" y="{y}" width="166" height="114" rx="12" fill="{C["panel"]}" stroke="{C["line"]}"/>'
              f'<rect x="{x}" y="{y + 18}" width="3" height="34" rx="1.5" fill="{col}"/>'
              + T(x + 18, y + 48, big, 30, "", 800, fill=col) + T(x + 18, y + 76, l1, 12, "", 600)
              + T(x + 18, y + 96, l2, 11, "mute") + "</g>")
    s.save("hero.svg")


# =====================================================================================================================
# 2. PADDING (the existing solution)
def padding():
    s = SVG(960, 380, "reference.py pads every passage to 512 tokens: 80.4 percent of the work is padding", 9)
    header(s, "THE EXISTING SOLUTION · reference.py", "Every passage is padded to 512 tokens",
           "8 passages at typical practice-set lengths, to scale. Blue = real tokens, hatched = padding the GPU still computes.")
    lens = [11, 249, 67, 34, 459, 100, 23, 140]
    x0, W = 40, 520
    k = W / 512
    for i, L in enumerate(lens):
        y = 116 + i * 28
        s.add(f'<rect class="{s.fade(1.6 + .05 * i, 2.4 + .05 * i)}" x="{x0 + L * k:.1f}" y="{y}" width="{(512 - L) * k:.1f}" height="18" fill="url(#hatch)"/>')
        s.add(f'<rect class="{s.grow_x(.3 + .08 * i, 1.4 + .08 * i)}" x="{x0}" y="{y}" width="{L * k:.1f}" height="18" rx="2" fill="{C["blue"]}"/>')
        s.add(T(x0 + W + 8, y + 13, f"{L} tok", 11, "mono mute"))
    # mean marker & 512 marker
    mx = x0 + 100 * k
    s.add(f'<g class="{s.fade(1.2, 2)}"><line x1="{mx:.1f}" y1="106" x2="{mx:.1f}" y2="338" stroke="{C["yellow"]}" stroke-dasharray="3 3"/>'
          + T(mx, 354, "mean 100", 11, "mono", 600, "middle", C["yellow"]) + "</g>")
    s.add(f'<line x1="{x0 + W}" y1="106" x2="{x0 + W}" y2="338" stroke="{C["mute"]}" stroke-dasharray="2 4"/>')
    s.add(T(x0 + W, 354, "512", 11, "mono mute", 600, "middle"))
    # stats
    px = 640
    s.add(f'<g class="{s.fade_up(2.4, 3.4)}">' + T(px, 160, "80.4%", 58, "", 800, fill=C["red"])
          + T(px, 186, "of all token positions are padding", 13, "", 600) + "</g>")
    s.add(f'<g class="{s.fade_up(2.8, 3.8)}">'
          + T(px, 228, "51.2 M", 22, "mono", 700) + T(px + 104, 228, "positions computed", 12, "mute")
          + T(px, 258, "10.0 M", 22, "mono", 700, fill=C["blue"]) + T(px + 104, 258, "real tokens", 12, "mute")
          + T(px, 296, "100,000 passages · mean 100 tokens · median 67", 11, "dim")
          + T(px, 314, "plus FP32 maths, a full last layer and a .cpu()", 11, "dim")
          + T(px, 330, "sync after every batch of 32", 11, "dim") + "</g>")
    s.save("padding.svg")


# =====================================================================================================================
# 3. LENGTHS
def lengths():
    s = SVG(960, 330, "Passage length distribution of the 100,000 practice passages", 8)
    header(s, "THE WORKLOAD · scripts/workload_stats.py", "Most passages are short; none reach 512",
           "Share of 100,000 practice passages by token length (exact counts with the model's tokenizer).")
    bins = [("0-15", 15.3), ("16-31", 16.7), ("32-63", 16.5), ("64-127", 23.9), ("128-255", 18.2), ("256-383", 6.7), ("384-512", 2.8)]
    x0, bw, gap, base, hmax = 60, 70, 22, 290, 160
    for i, (lab, v) in enumerate(bins):
        x = x0 + i * (bw + gap)
        h = v / 25 * hmax
        col = C["orange"] if i < 2 else C["blue"]
        s.add(f'<rect class="{s.grow_y(.2 + .1 * i, 1.3 + .1 * i)}" x="{x}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" rx="4" fill="{col}"/>')
        s.add(f'<g class="{s.fade(1.1 + .1 * i, 1.7 + .1 * i)}">' + T(x + bw / 2, base - h - 8, f"{v}%", 12, "mono", 700, "middle") + "</g>")
        s.add(T(x + bw / 2, base + 18, lab, 11, "mono mute", 400, "middle"))
    s.add(f'<line x1="{x0 - 10}" y1="{base}" x2="{x0 + 7 * (bw + gap) - gap + 10}" y2="{base}" stroke="{C["line"]}"/>')
    px = 712
    s.add(f'<g class="{s.fade_up(2, 3)}">' + T(px, 140, "mean 100 · median 67", 14, "mono", 700)
          + T(px, 162, "p10 11 · p90 249", 12, "mono mute") + T(px, 180, "p99 459 · max 502", 12, "mono mute")
          + f'<rect x="{px}" y="206" width="10" height="10" rx="2" fill="{C["orange"]}"/>'
          + T(px + 18, 216, "≤ 32 tokens: 33% of passages", 12, "", 600)
          + T(px + 18, 234, "but only 5.5% of tokens", 12, "mute")
          + T(px + 18, 252, "→ these get the FP32 path (P2)", 12, "", 600, fill=C["orange"]) + "</g>")
    s.save("lengths.svg")


# =====================================================================================================================
# 4. PIPELINE
def pipeline():
    s = SVG(960, 470, "Architecture of team.py: offline batch path and live query path", 10)
    header(s, "THE PROPOSED SOLUTION · team.py", "Same model, re-engineered execution",
           "setup() (untimed) loads FP16 + FP32 weight copies, warms up at real shapes and captures CUDA Graphs.")

    def box(x, y, w, h, t1, t2, col, t0):
        cls = s.fade_up(t0, t0 + .8, 10)
        s.add(f'<g class="{cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{C["panel"]}" stroke="{col}" stroke-opacity=".75"/>'
              + T(x + w / 2, y + 24, t1, 12.5, "", 700, "middle") + T(x + w / 2, y + 42, t2, 10.5, "mute", 400, "middle") + "</g>")

    def lane(y, label, col):
        s.add(T(32, y, label, 10, "mono", 700, fill=col, extra=' letter-spacing="1.5"'))

    lane(124, "CPU · PRODUCER THREAD", C["mute"])
    lane(250, "GPU · CONSUMER", C["blue"])
    lane(376, "LIVE PATH · encode([query])", C["orange"])
    w, h, gx = 166, 54, 18
    top = [("encode(texts)", "N passages, any order"), ("Rust tokenizer", "16k-text chunks, no pad"),
           ("Sort by length", "short → long"), ("Token-budget batches", "8,192 tok · split at ×8"),
           ("Pinned memory", "ready for async copy")]
    gpu = [("Async H2D copy", "non-blocking"), ("12-layer encoder", "FP16 >32 tok · FP32 ≤32"),
           ("CLS-only last layer", "K,V all · rest CLS"), ("L2-norm + scatter", "back to input order"),
           ("1 × D2H copy", "→ [N, 768] float32")]
    live = [("1 query", "≈12 tokens"), ("Tokenise", "~0.1 ms"), ("Pick bucket", "32 · 64 · 128"),
            ("CUDA Graph replay", "32-graph runs FP32"), ("[1, 768]", "p50 4.4 ms")]
    xs = [32 + i * (w + gx) for i in range(5)]
    for i, (a, b) in enumerate(top):
        box(xs[i], 134, w, h, a, b, C["mute"], .2 + .12 * i)
    for i, (a, b) in enumerate(gpu):
        box(xs[i], 260, w, h, a, b, C["blue"], .9 + .12 * i)
    for i, (a, b) in enumerate(live):
        box(xs[i], 386, w, h, a, b, C["orange"], 1.6 + .12 * i)
    # connectors
    paths = []
    for row_y in (161, 287, 413):
        for i in range(4):
            s.add(f'<path d="M{xs[i] + w} {row_y} h{gx}" stroke="{C["line"]}" stroke-width="2"/>')
    pq = f"M{xs[4] + w / 2} 188 V224 H{xs[0] + w - 24} V260"
    s.add(f'<path d="{pq}" fill="none" stroke="{C["line"]}" stroke-width="2" stroke-dasharray="4 4"/>')
    s.add(T((xs[0] + xs[4] + w) / 2, 218, "queue of ready batches · GPU never waits for the tokenizer", 10.5, "mute", 400, "middle"))
    off = f"M{xs[0] + 10} 161 H{xs[4] + w / 2} V224 H{xs[0] + w - 24} V287 H{xs[4] + w - 10}"
    liv = f"M{xs[0] + 10} 413 H{xs[4] + w - 10}"
    s.add('<g class="motion">')
    for j in range(6):
        s.add(f'<circle r="4" fill="{C["blue"]}"><animateMotion dur="6s" begin="{j}s" repeatCount="indefinite" path="{off}"/></circle>')
    for j in range(3):
        s.add(f'<circle r="4" fill="{C["orange"]}"><animateMotion dur="2.4s" begin="{j * .8:.1f}s" repeatCount="indefinite" path="{liv}"/></circle>')
    s.add("</g>")
    s.save("pipeline.svg")


# =====================================================================================================================
# 5. O1 fused QKV + FP16 + SDPA
def o1():
    s = SVG(960, 330, "O1: FP16 tensor cores, fused QKV projection and fused SDPA attention: 3.02x", 9)
    header(s, "O1 · FP16 TENSOR CORES + FUSED QKV + SDPA", "Fewer, faster kernels", None)
    # panel 1: QKV merge
    px = 40
    s.add(T(px, 108, "1  Fuse Q, K, V projections", 13, "", 700))
    cols = [("Wq", C["blue"]), ("Wk", C["green"]), ("Wv", C["orange"])]
    for i, (n, col) in enumerate(cols):
        x_final = px + 30 + i * 52
        x_start = px + 10 + i * 82
        cls = s.anim(f"transform:translateX({x_start - x_final}px)", "transform:translateX(0)", .4, 1.6)
        s.add(f'<g class="{cls}"><rect x="{x_final}" y="128" width="52" height="96" fill="{col}" fill-opacity=".22" stroke="{col}"/>'
              + T(x_final + 26, 182, n, 14, "mono", 700, "middle", col) + "</g>")
    s.add(f'<g class="{s.fade(1.5, 2.2)}"><rect x="{px + 30}" y="128" width="156" height="96" fill="none" stroke="{C["ink"]}" stroke-width="2" rx="2"/>'
          + T(px + 108, 250, "3 GEMMs → 1 GEMM", 12, "mono", 700, "middle") + T(px + 108, 268, "linear(x, [Wq;Wk;Wv]) — exact", 11, "mute", 400, "middle") + "</g>")
    # panel 2: FP32 -> FP16
    px = 340
    s.add(T(px, 108, "2  FP32 → FP16 on Tensor Cores", 13, "", 700))
    for i in range(32):
        x = px + i * 8
        if i < 16:
            s.add(f'<rect x="{x}" y="140" width="6" height="22" rx="1" fill="{C["mute"]}"/>')
            s.add(f'<rect x="{x}" y="196" width="6" height="22" rx="1" fill="{C["blue"]}" class="{s.fade(.6 + i * .04, 1.0 + i * .04)}"/>')
        else:
            s.add(f'<rect x="{x}" y="140" width="6" height="22" rx="1" fill="{C["mute"]}" opacity=".55"/>')
    s.add(T(px, 180, "FP32 · 4 bytes · CUDA cores", 11, "mono mute"))
    s.add(T(px, 236, "FP16 · 2 bytes · Tensor Cores (sm_86)", 11, "mono", 400, fill=C["blue"]))
    s.add(T(px, 262, "GEMM reductions stay FP32 (cuBLAS", 11, "mute"))
    s.add(T(px, 278, "reduced-precision reductions disabled)", 11, "mute"))
    # panel 3: SDPA
    px = 640
    s.add(T(px, 108, "3  One fused attention kernel", 13, "", 700))
    ops = ["Q·Kᵀ", "scale+mask", "softmax", "·V"]
    for i, o in enumerate(ops):
        x_final = px + i * 68
        cls = s.anim(f"transform:translateX({i * 8}px)", "transform:translateX(0)", 1.0, 2.0)
        s.add(f'<g class="{cls}"><rect x="{x_final}" y="140" width="66" height="34" rx="6" fill="{C["panel"]}" stroke="{C["line"]}"/>'
              + T(x_final + 33, 162, o, 11, "mono", 600, "middle") + "</g>")
    s.add(f'<g class="{s.fade_up(1.8, 2.6)}"><rect x="{px}" y="190" width="270" height="34" rx="6" fill="{C["violet"]}" fill-opacity=".18" stroke="{C["violet"]}"/>'
          + T(px + 135, 212, "scaled_dot_product_attention", 12, "mono", 700, "middle", C["violet"]) + "</g>")
    s.add(T(px, 250, "flash / memory-efficient kernels: the", 11, "mute"))
    s.add(T(px, 266, "L×L score matrix never leaves the chip", 11, "mute"))
    # result
    s.add(f'<g class="{s.fade_up(2.6, 3.4)}">' + f'<rect x="32" y="292" width="896" height="1" fill="{C["line"]}"/>'
          + T(40, 318, "Stage 1 result (official bench.py, 20k passages):", 12, "mute")
          + T(352, 318, "28.4 → 85.9 passages/s", 13, "mono", 700) + T(560, 318, "3.02×", 15, "mono", 800, fill=C["blue"])
          + T(620, 318, "· GPU kernel time 69.7 s → 17.6 s per 2,000", 12, "mute") + "</g>")
    s.save("o1-fp16-fused.svg")


# =====================================================================================================================
# 6. SCHEDULER
def scheduler():
    s = SVG(960, 420, "O2 and O2b: length-aware token-budget scheduler removes padding: 15x", 10)
    header(s, "O2 + O2b · LENGTH-AWARE TOKEN-BUDGET SCHEDULER", "Sort, pack by token budget, pad only to ×8",
           "12 passages: first padded to 512 → sorted by length → each batch padded only to its own longest row (×8).")
    lens = [180, 24, 96, 410, 40, 132, 16, 250, 64, 300, 8, 88]
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    rank = {i: r for r, i in enumerate(order)}
    batches = [order[0:5], order[5:9], order[9:12]]   # token-budget style: many short rows, few long rows
    bmax = {}
    for b in batches:
        m = max(lens[i] for i in b)
        m = int(math.ceil(m / 8) * 8)
        for i in b:
            bmax[i] = m
    x0, W, rh, gap, y0 = 40, 480, 16, 5, 118
    k = W / 512
    # batch gap: add extra spacing between batches in sorted layout
    def ypos(r):
        extra = 0 if r < 5 else (10 if r < 9 else 20)
        return y0 + r * (rh + gap) + extra
    for i, L in enumerate(lens):
        y_unsorted = y0 + i * (rh + gap) + (0 if i < 5 else (10 if i < 9 else 20))
        y_sorted = ypos(rank[i])
        move = s.anim(f"transform:translateY({y_unsorted - y_sorted}px)", "transform:translateY(0)", 1.6, 2.8)
        pad_final = (bmax[i] - L) * k
        pad_init = (512 - L) * k
        sc = pad_final / pad_init if pad_init else 0
        shrink = s.anim("transform:scaleX(1)", f"transform:scaleX({sc:.4f})", 3.2, 4.4)
        s.add(f'<g class="{move}">'
              f'<rect class="{shrink} ox" style="transform:scaleX({sc:.4f})" x="{x0 + L * k:.1f}" y="{y_sorted}" width="{pad_init:.1f}" height="{rh}" fill="url(#hatch)"/>'
              f'<rect x="{x0}" y="{y_sorted}" width="{L * k:.1f}" height="{rh}" rx="2" fill="{C["blue"]}"/></g>')
    # batch brackets
    for bi, b in enumerate(batches):
        r0, r1 = min(rank[i] for i in b), max(rank[i] for i in b)
        ya, yb = ypos(r0) - 3, ypos(r1) + rh + 3
        m = bmax[b[0]]
        s.add(f'<g class="{s.fade(4.2 + .2 * bi, 5 + .2 * bi)}"><rect x="{x0 - 6}" y="{ya}" width="{m * k + 12:.1f}" height="{yb - ya}" rx="5" fill="none" stroke="{C["green"]}" stroke-dasharray="4 3"/>'
              + T(x0 + m * k + 14, (ya + yb) / 2 + 4, f"batch {bi + 1} · pad to {m}", 10.5, "mono", 600, fill=C["green"]) + "</g>")
    # phase captions
    s.add(f'<g class="{s.anim("opacity:1", "opacity:0", 1.4, 1.8)}" style="opacity:0">' + T(x0, 404, "① reference: every row padded to 512", 11.5, "mono", 600, fill=C["red"]) + "</g>")
    s.add(f'<g class="{s.fade(4.4, 5)}">' + T(x0, 404, "③ ours: sorted, packed, padded to each batch's max (×8)", 11.5, "mono", 600, fill=C["green"]) + "</g>")
    # stats
    px = 640
    s.add(f'<g class="{s.fade_up(4.6, 5.6)}">'
          + T(px, 138, "token positions for 100k passages", 12, "mute")
          + f'<rect x="{px}" y="148" width="260" height="14" rx="3" fill="{C["pad"]}"/>'
          + f'<rect x="{px}" y="148" width="{260 * 10.4 / 51.2:.1f}" height="14" rx="3" fill="{C["blue"]}"/>'
          + T(px, 182, "51.2 M → 10.4 M", 20, "mono", 700)
          + T(px, 222, "padding", 12, "mute") + T(px + 90, 222, "80.4% → 3.7%", 14, "mono", 700, fill=C["green"])
          + T(px, 248, "work (FLOPs)", 12, "mute") + T(px + 90, 248, "5.62× less", 14, "mono", 700, fill=C["green"])
          + T(px, 274, "batches", 12, "mute") + T(px + 90, 274, "3,125 → 379", 14, "mono", 700)
          + f'<rect x="{px}" y="296" width="280" height="1" fill="{C["line"]}"/>'
          + T(px, 324, "85.9 → 427.3 p/s", 18, "mono", 800, fill=C["blue"])
          + T(px, 346, "15.0× vs reference (stage 2, 20k)", 12, "mute")
          + T(px, 366, "O2b bucket split: +2.9% (3 interleaved runs)", 11, "dim") + "</g>")
    s.save("scheduler.svg")


# =====================================================================================================================
# 7. OVERLAP
def overlap():
    s = SVG(960, 360, "O3: overlapped host pipeline: idle gaps 136 to 7", 9)
    header(s, "O3 · OVERLAPPED HOST PIPELINE", "Keep the GPU fed",
           "Tokenise in a background thread, pinned memory, async copies, one sync at the very end.")
    x0, W = 150, 560

    def lanes(y, label, cpu, gpu, gaps, t0):
        s.add(T(40, y - 10, label, 11, "mono", 700, fill=C["ink"]))
        s.add(T(40, y + 13, "CPU", 11, "mono mute"))
        s.add(T(40, y + 41, "GPU", 11, "mono mute"))
        reveal = s.anim("transform:scaleX(0)", "transform:scaleX(1)", t0, t0 + 2.2)
        s.add(f'<defs><clipPath id="c{y}"><rect class="{reveal} ox" x="{x0}" y="{y - 4}" width="{W}" height="64"/></clipPath></defs>')
        g = [f'<g clip-path="url(#c{y})">']
        for a, b in cpu:
            g.append(f'<rect x="{x0 + a * W:.1f}" y="{y}" width="{(b - a) * W:.1f}" height="18" rx="3" fill="{C["mute"]}"/>')
        for a, b in gpu:
            g.append(f'<rect x="{x0 + a * W:.1f}" y="{y + 28}" width="{(b - a) * W:.1f}" height="18" rx="3" fill="{C["blue"]}"/>')
        for a, b in gaps:
            g.append(f'<rect x="{x0 + a * W:.1f}" y="{y + 28}" width="{(b - a) * W:.1f}" height="18" rx="3" fill="{C["red"]}" fill-opacity=".28" stroke="{C["red"]}" stroke-dasharray="3 2"/>')
        g.append("</g>")
        s.add("".join(g))
        ph = s.anim(f"transform:translateX(0)", f"transform:translateX({W}px)", t0, t0 + 2.2)
        s.add(f'<line class="{ph} motion" x1="{x0}" y1="{y - 6}" x2="{x0}" y2="{y + 52}" stroke="{C["yellow"]}" stroke-width="2"/>')

    # before: serial
    cpu, gpu, gaps = [], [], []
    t = 0
    for _ in range(5):
        cpu.append((t, t + .06)); gaps.append((t, t + .06)); t += .06
        gpu.append((t, t + .12)); t += .12
        cpu.append((t, t + .02)); gaps.append((t, t + .02)); t += .02   # .cpu() sync
    lanes(132, "reference.py  — tokenise, compute, .cpu() sync, repeat", cpu, gpu, gaps, .3)
    # after: overlapped
    cpu2 = [(i * .07, i * .07 + .06) for i in range(8)]
    gpu2 = []
    t = .065
    while t < .98:
        gpu2.append((t, min(t + .085, 1))); t += .087
    lanes(232, "team.py  — producer thread runs ahead, GPU never waits", cpu2, gpu2, [(0, .062)], 2.6)
    px = 750
    s.add(f'<g class="{s.fade_up(4.6, 5.6)}">' + T(px, 150, "idle gaps > 100 µs", 12, "mute")
          + T(px, 186, "136 → 7", 30, "mono", 800, fill=C["green"])
          + T(px, 206, "per 2,000 passages (profiler)", 11, "dim")
          + T(px, 250, "GPU busy", 12, "mute") + T(px, 280, "99.7%", 26, "mono", 800, fill=C["blue"])
          + T(px, 300, "of the device span", 11, "dim") + "</g>")
    s.add(T(40, 336, "Schematic timeline (not to scale) · numbers: results_gpu/profiler/{reference,default}/summary.json", 11, "dim"))
    s.save("overlap.svg")


# =====================================================================================================================
# 8. CLS-only
def cls_only():
    s = SVG(960, 350, "O5: CLS-only last layer, exact, plus 8 percent", 9)
    header(s, "O5 · CLS-ONLY LAST LAYER", "The 12th layer only needs one output row",
           "The embedding is the [CLS] row. It attends to every token's K and V (kept), but nothing else needs the other rows.")
    toks = ["[CLS]", "the", "art", "of", "war", "is", "an", "anc…", "[SEP]"]
    rows = [("K, V projection", True), ("Q projection", False), ("attention", False), ("out-proj + LayerNorm", False), ("FFN + LayerNorm", False)]
    x0, y0, cw, ch = 210, 128, 58, 30
    for j, t in enumerate(toks):
        s.add(T(x0 + j * cw + cw / 2 - 2, y0 - 10, t, 11, "mono", 600 if j == 0 else 400, "middle", C["yellow"] if j == 0 else C["mute"]))
    for r, (name, keep) in enumerate(rows):
        y = y0 + r * (ch + 6)
        s.add(T(40, y + 19, name, 12, "", 600))
        for j in range(len(toks)):
            x = x0 + j * cw
            on = keep or j == 0
            col = C["yellow"] if j == 0 else (C["green"] if keep else C["blue"])
            if on:
                s.add(f'<rect x="{x}" y="{y}" width="{cw - 6}" height="{ch}" rx="5" fill="{col}" fill-opacity=".85"/>')
            else:
                d = s.anim(f"fill-opacity:.85;fill:{C['blue']}", f"fill-opacity:1;fill:{C['panel']}", 1.2 + .06 * j + .1 * r, 2.2 + .06 * j + .1 * r)
                s.add(f'<rect class="{d}" x="{x}" y="{y}" width="{cw - 6}" height="{ch}" rx="5" fill="{C["panel"]}" stroke="{C["line"]}"/>')
    px = 760
    s.add(f'<g class="{s.fade_up(3, 4)}">' + T(px, 150, "exact", 26, "", 800, fill=C["green"])
          + T(px, 172, "same CLS output · exact maths", 11, "mute")
          + T(px, 222, "+8%", 26, "mono", 800, fill=C["blue"])
          + T(px, 244, "426.5 → 460.4 p/s (S3 → S4)", 11, "mute")
          + T(px, 324, "a mean-length passage", 11, "dim") + T(px, 288, "rows dropped", 12, "mute") + T(px, 308, "8 of 9 here · 99 of 100 for", 11, "dim") + "</g>")
    s.save("cls-only.svg")


# =====================================================================================================================
# 9. CUDA graph
def graphs():
    s = SVG(960, 330, "O7: CUDA Graphs for live queries", 8)
    header(s, "O7 · CUDA GRAPHS FOR LIVE QUERIES", "One replay instead of ~200 launches",
           "bench.py sends one query per encode() call. Small kernels are launch-bound, so each bucket (32/64/128 tok) is captured once.")
    x0, W, msmax = 170, 560, 2.6
    k = W / msmax
    s.add(T(40, 142, "eager", 12, "mono", 700))
    s.add(T(40, 158, "FP16 service", 10.5, "mute"))
    n = 70
    span = 2.49 * k
    for i in range(n):
        x = x0 + i * span / n
        s.add(f'<rect class="{s.fade(.2 + i * .025, .3 + i * .025)}" x="{x:.1f}" y="128" width="{span / n * .55:.2f}" height="30" rx="1" fill="{C["mute"]}"/>')
    s.add(f'<g class="{s.fade(2, 2.6)}">' + T(x0 + span + 10, 148, "2.49 ms", 13, "mono", 700) + "</g>")
    s.add(T(40, 210, "CUDA Graph", 12, "mono", 700, fill=C["blue"]))
    s.add(T(40, 226, "FP16 service", 10.5, "mute"))
    s.add(f'<rect class="{s.grow_x(2.4, 3.4)}" x="{x0}" y="196" width="{1.76 * k:.1f}" height="30" rx="4" fill="{C["blue"]}"/>')
    s.add(f'<g class="{s.fade(3.2, 3.8)}">' + T(x0 + 1.76 * k + 10, 216, "1.76 ms", 13, "mono", 700, fill=C["blue"]) + "</g>")
    for ms in (0, .5, 1, 1.5, 2, 2.5):
        x = x0 + ms * k
        s.add(f'<line x1="{x:.1f}" y1="240" x2="{x:.1f}" y2="246" stroke="{C["line"]}"/>' + T(x, 260, f"{ms:g} ms", 10, "mono dim", 400, "middle"))
    s.add(f'<g class="{s.fade_up(3.6, 4.6)}">'
          + T(40, 296, "Submitted: the 32-token graph runs the FP32 copy (P2) → service 3.71 ms (graph) vs 4.05 ms (eager).", 12, "mute")
          + T(40, 316, "Official bench.py live trace (1,500 queries, bursts included):", 12, "mute")
          + T(398, 316, "p50 4.4 ms · p99 8.2 ms", 13, "mono", 700, fill=C["orange"]) + "</g>")
    s.save("cuda-graphs.svg")


# =====================================================================================================================
# 10. PRECISION / P2
def precision():
    s = SVG(960, 430, "P2: the full 100k check found 2 FP16 violations on short passages; FP32 for 32 tokens or less fixes them", 11)
    header(s, "P2 · THE 100,000-PASSAGE CHECK", "Checking all 100k, not the official 2k, found 2 failures",
           "Plain FP16 vs our FP32 path: the 9 worst passages. y = 1 − cosine (log). Every failure is a short passage.")
    X0, X1, Y0, Y1 = 90, 640, 120, 370   # plot box
    xl0, xl1 = math.log10(4), math.log10(512)
    yl0, yl1 = -6, -2

    def px(t):
        return X0 + (math.log10(t) - xl0) / (xl1 - xl0) * (X1 - X0)

    def py(e):
        return Y1 - (math.log10(e) - yl0) / (yl1 - yl0) * (Y1 - Y0)

    s.add(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}" fill="{C["panel"]}" rx="4"/>')
    for e in (-6, -5, -4, -3, -2):
        y = py(10 ** e)
        s.add(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{C["grid"]}"/>' + T(X0 - 8, y + 4, f"1e{e}", 10, "mono dim", 400, "end"))
    for t in (8, 16, 32, 64, 128, 256, 512):
        x = px(t)
        s.add(f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y1}" stroke="{C["grid"]}"/>' + T(x, Y1 + 16, str(t), 10, "mono dim", 400, "middle"))
    s.add(T((X0 + X1) / 2, Y1 + 32, "passage length (tokens, log)", 11, "mute", 400, "middle"))
    # band: > 32 tokens all <= 8e-6
    s.add(f'<rect x="{px(33):.1f}" y="{py(8e-6):.1f}" width="{X1 - px(33):.1f}" height="{Y1 - py(8e-6):.1f}" fill="{C["blue"]}" fill-opacity=".14"/>')
    s.add(T(px(140), py(2.2e-6), "all ~67,000 passages > 32 tokens: 1 − cos ≤ 8e-6", 10.5, "", 600, "middle", C["blue"]))
    # P2 zone
    s.add(f'<rect class="{s.grow_x(3.6, 4.6)}" x="{X0}" y="{Y0}" width="{px(32) - X0:.1f}" height="{Y1 - Y0}" fill="{C["green"]}" fill-opacity=".10" stroke="{C["green"]}" stroke-dasharray="4 3"/>')
    s.add(f'<g class="{s.fade(4.2, 4.8)}">' + T(X0 + 8, Y0 + 18, "P2: ≤ 32 tokens → FP32", 11, "mono", 700, fill=C["green"]) + "</g>")
    # thresholds
    for val, lab, col in ((1e-3, "rule 0.999", C["red"]), (5e-4, "gate 0.9995", C["yellow"])):
        y = py(val)
        s.add(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{col}" stroke-dasharray="5 4"/>' + T(X1 - 4, y - 5, lab, 10, "mono", 600, "end", col))
    pts = [(16, 1 - .99440155), (8, 1 - .99889230), (11, 1 - .99911098), (23, 1 - .99912410), (22, 1 - .99923381),
           (9, 1 - .99952843), (8.4, 1 - .99955531), (18, 1 - .99958750), (7.6, 1 - .99966445)]
    for i, (t, e) in enumerate(pts):
        x, y = px(t), py(e)
        col = C["red"] if e > 1e-3 else C["orange"]
        drop = py(1.3e-6) - y
        appear = s.fade(.3 + .12 * i, .9 + .12 * i)
        fall = s.anim("transform:translateY(0)", f"transform:translateY({drop:.1f}px)", 5 + .06 * i, 6.2 + .06 * i)
        recol = s.anim(f"fill:{col}", f"fill:{C['green']}", 5 + .06 * i, 6.2 + .06 * i)
        s.add(f'<g class="{appear}"><g class="{fall}" style="transform:translateY({drop:.1f}px)">'
              f'<circle class="{recol}" cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{C["green"]}" stroke="{C["bg"]}" stroke-width="1.5"/></g></g>')
    s.add(f'<g class="{s.anim("opacity:1", "opacity:0", 4.8, 5.4)}" style="opacity:0">'
          + T(px(16) + 10, py(pts[0][1]) + 4, "#31022 · 16 tok · cos 0.99440", 10.5, "mono", 600, fill=C["red"]) + "</g>")
    # submitted worst
    s.add(f'<g class="{s.fade(6.2, 7)}"><path d="M{px(41):.1f} {py(8.4e-6) - 6:.1f} l6 6 l-6 6 l-6 -6z" fill="{C["green"]}"/>'
          + T(px(41) + 10, py(8.4e-6) - 10, "submitted worst: #86483 · 41 tok · 0.999992", 10.5, "mono", 600, fill=C["green"]) + "</g>")
    # stats
    sx = 680
    s.add(f'<g class="{s.anim("opacity:1", "opacity:0", 4.8, 5.4)}" style="opacity:0">'
          + T(sx, 160, "plain FP16", 12, "mute") + T(sx, 200, "2 violations", 28, "mono", 800, fill=C["red"])
          + T(sx, 222, "5 below gate · 28 below 0.9999", 11, "dim")
          + T(sx, 240, "all ≤ 23 tokens — invisible in the", 11, "dim")
          + T(sx, 256, "official 2,000-passage check", 11, "dim") + "</g>")
    s.add(f'<g class="{s.fade_up(6, 7)}">'
          + T(sx, 160, "submitted (P2)", 12, "mute") + T(sx, 200, "0 violations", 28, "mono", 800, fill=C["green"])
          + T(sx, 222, "min cos 0.999992 over 100,000", 11, "dim")
          + T(sx, 240, "FP32 on 5.5% of tokens costs ~3%", 11, "dim")
          + T(sx, 256, "of throughput (598.7 vs 619.0 p/s)", 11, "dim") + "</g>")
    s.add(T(sx, 320, "FP32 residual stream (P1) still", 11, "dim"))
    s.add(T(sx, 336, "left 1 violation (0.99779) → replaced", 11, "dim"))
    s.add(T(sx, 352, "BF16: min 0.99734 → rejected", 11, "dim"))
    s.add(T(40, 422, "Source: results_gpu/final/full_check_fp16plain.json, full_check.json", 11, "dim"))
    s.save("precision-p2.svg")


# =====================================================================================================================
# 11. ABLATION staircase
def ablation():
    s = SVG(960, 450, "Throughput by stage, official bench.py", 9)
    header(s, "ABLATION · OFFICIAL bench.py · RTX 3050 6 GB LAPTOP", "Each optimisation, measured on its own",
           "Passages per second; each bar adds one stage on top of the previous one.")
    bars = [("reference", "FP32 · pad 512", 28.4, "1.0×", C["mute"]),
            ("S1", "FP16 · QKV · SDPA", 85.9, "3.0×", C["blue"]),
            ("S2", "+ token-budget sched.", 427.3, "15.0×", C["blue"]),
            ("S3", "+ async pipeline · graphs", 426.5, "15.0×", C["blue"]),
            ("S4", "+ CLS-only last layer", 460.4, "16.2×", C["blue"]),
            ("submitted", "S4 · budget 8,192 · P2", 604.9, "21.3×", C["green"])]
    x0, bw, gap, base, hmax = 60, 112, 34, 358, 210
    for i, (n, sub, v, sp, col) in enumerate(bars):
        x = x0 + i * (bw + gap)
        h = v / 604.9 * hmax
        s.add(f'<rect class="{s.grow_y(.2 + .25 * i, 1.4 + .25 * i)}" x="{x}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" rx="6" fill="{col}"/>')
        s.add(f'<g class="{s.fade_up(1 + .25 * i, 1.8 + .25 * i, 8)}">' + T(x + bw / 2, base - h - 28, f"{v:g}", 16, "mono", 800, "middle")
              + T(x + bw / 2, base - h - 10, sp, 12, "mono", 700, "middle", col) + "</g>")
        s.add(T(x + bw / 2, base + 20, n, 12.5, "", 700, "middle"))
        s.add(T(x + bw / 2, base + 36, sub, 10, "mute", 400, "middle"))
    s.add(f'<line x1="{x0 - 10}" y1="{base}" x2="{x0 + 6 * (bw + gap) - gap + 10}" y2="{base}" stroke="{C["line"]}"/>')
    s.add(T(40, 418, "S1–S4: 20k passages, budget 65,536, one run each (laptop timings drifted at ~90% VRAM). Submitted: all 100k passages, budget 8,192.", 10.5, "dim"))
    s.add(T(40, 434, "Not submitted: S5 torch.compile 493.7 p/s but min cos 0.99986 and 197 s setup.  Source: results_gpu/ablation/, results_gpu/final/", 10.5, "dim"))
    s.save("ablation.svg")


# =====================================================================================================================
# 12. PROFILER
def profiler():
    s = SVG(960, 360, "GPU kernel time by category, torch.profiler, 2,000 passages", 9)
    header(s, "PROFILER · torch.profiler · 2,000 PASSAGES", "Where the GPU time goes",
           "Sum of kernel durations from the exported Chrome traces (results_gpu/profiler/*/summary.json).")
    runs = [("reference", 69674, (.749, .179, .071), 136), ("stage 1", 17588, (.654, .106, .240), 77),
            ("stage 2", 3735, (.687, .063, .250), 29), ("submitted", 3325, (.790, .073, .136), 7)]
    cols = (C["blue"], C["orange"], C["green"])
    x0, W, y0 = 140, 600, 122
    k = W / 69674
    for i, (n, ms, sh, gaps) in enumerate(runs):
        y = y0 + i * 50
        s.add(T(40, y + 20, n, 13, "", 700))
        x = x0
        t0 = .3 + .35 * i
        segs = []
        for f, c in zip(sh, cols):
            w = ms * f * k
            segs.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w, .8):.1f}" height="28" fill="{c}"/>')
            x += w
        s.add(f'<g class="{s.grow_x(t0, t0 + 1.4)}">' + "".join(segs) + "</g>")
        s.add(f'<g class="{s.fade(t0 + .9, t0 + 1.6)}">' + T(x + 10, y + 19, f"{ms / 1000:.1f} s", 13, "mono", 700) + "</g>")
        s.add(T(920, y + 19, str(gaps), 13, "mono", 700, "end", C["red"] if gaps > 50 else C["green"]))
    s.add(T(920, 108, "idle gaps >100µs", 10.5, "mute", 400, "end"))
    lx = 140
    for lab, c in zip(("GEMM", "attention", "norm + element-wise"), cols):
        s.add(f'<rect x="{lx}" y="328" width="10" height="10" rx="2" fill="{c}"/>' + T(lx + 16, 337, lab, 11, "mute"))
        lx += 130 if lab != "attention" else 110
    s.add(f'<g class="{s.fade_up(2.2, 3)}">' + T(560, 290, "21× less GPU time", 20, "mono", 800, fill=C["green"])
          + T(560, 310, "75% FP32 GEMM → FP16 Tensor-Core GEMM", 11, "mute") + "</g>")
    s.save("profiler.svg")


for fn in (hero, padding, lengths, pipeline, o1, scheduler, overlap, cls_only, graphs, precision, ablation, profiler):
    fn()
