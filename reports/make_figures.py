"""Generates the SVG figures of reports/F1_CONSTRUCT_paper.md (no dependencies). Data are the aggregate numbers in
explore/ovmer/p17_ovmerd_report_2026-11-24.md. Run: python3 reports/make_figures.py"""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
TXT, TXT2, GRID, SURF = "#191919", "#666666", "#e7e7e7", "#ffffff"
SER = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]  # validated categorical slots 1-5 (light)
FONT = "font-family='Helvetica, Arial, sans-serif'"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, color=TXT, anchor="start", weight="normal", italic=False):
    st = " font-style='italic'" if italic else ""
    return f"<text x='{x}' y='{y}' {FONT} font-size='{size}' fill='{color}' text-anchor='{anchor}' font-weight='{weight}'{st}>{esc(s)}</text>"


def box(x, y, w, h, lines, fill="#f5f7fa", stroke="#c9d1db", title=None, tcolor=TXT, sw=1.5):
    o = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'/>"]
    yy = y + 22
    if title:
        o.append(text(x + w / 2, yy, title, 14, tcolor, "middle", "bold")); yy += 20
    for ln in lines:
        o.append(text(x + w / 2, yy, ln, 12, TXT2, "middle")); yy += 17
    return "".join(o)


def arrow(x1, y1, x2, y2, label=None):
    o = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{TXT2}' stroke-width='2' marker-end='url(#ah)'/>"
    if label:
        o += text((x1 + x2) / 2, (y1 + y2) / 2 - 8, label, 11, TXT2, "middle", italic=True)
    return o


def svg(w, h, body):
    defs = ("<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
            f"<path d='M0,0 L10,5 L0,10 z' fill='{TXT2}'/></marker></defs>")
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'><rect width='{w}' height='{h}' fill='{SURF}'/>{defs}{body}</svg>"


def fig_pipeline():
    b = []
    b.append(text(20, 28, "Figure 1. Inference pipeline. Stages A-B follow the official OV-MER protocol; stage C (CONSTRUCT) is our contribution.", 13, TXT, weight="bold"))
    b.append(box(20, 60, 140, 120, ["video frames", "audio track", "English subtitle"], title="Clip x"))
    b.append(arrow(160, 120, 205, 120))
    b.append(box(205, 50, 215, 140, ["Qwen2.5-Omni-7B (zero-shot)", "official OV-MER prompt", "1 greedy + N sampled", "descriptions (T=1.0, top-p 0.95)", "cut invented 'Human:' turns"], title="A. Describe"))
    b.append(arrow(420, 120, 480, 120))
    b.append(text(450, 108, "N+1", 11, TXT2, "middle", italic=True)); b.append(text(450, 140, "texts", 11, TXT2, "middle", italic=True))
    b.append(box(480, 50, 215, 140, ["Qwen2.5-7B-Instruct (vLLM)", "official extraction prompt", "(<= 4 labels per description)", "T=0.7, top-p 0.8", "-> one label list per text"], title="B. Extract"))
    b.append(arrow(695, 120, 745, 120))
    b.append(text(720, 108, "S_1..S_N", 11, TXT2, "middle", italic=True))
    b.append(box(745, 50, 205, 140, ["build candidate pool", "Monte Carlo expected EW-F", "R(Y) = mean_i U(Y, S_i)", "greedy set construction", "-> final label set Y*"], title="C. CONSTRUCT", fill="#eaf2fd", stroke=SER[0], tcolor=SER[0], sw=2.5))
    b.append(arrow(847, 190, 847, 230))
    b.append(box(690, 230, 260, 70, ["map labels to emotion-wheel groups,", "P / R / F over the corpus, 5 wheels"], title="EW-F evaluation (official)"))
    # example strip
    b.append(text(20, 240, "Example (one clip, N = 4):", 13, TXT, weight="bold"))
    ex = [("S_1", "worried, sad, angry"), ("S_2", "worried, sad, surprised"), ("S_3", "worried, sad, happy"), ("S_4", "worried, sad, disgusted")]
    for i, (k, v) in enumerate(ex):
        b.append(text(30, 265 + 18 * i, f"{k}:  {{{v}}}", 12, TXT2))
    b.append(text(330, 265, "Best single list (MBR):  0.75", 12, TXT2))
    b.append(text(330, 283, "CONSTRUCT {worried, sad}:  0.80", 12, SER[0], weight="bold"))
    b.append(text(330, 301, "(expected F, one-wheel toy example)", 11, TXT2, italic=True))
    return svg(970, 330, "".join(b))


def fig_construct():
    b = [text(20, 28, "Figure 2. CONSTRUCT on the toy clip of Fig. 1: candidates and Monte Carlo expected score per step.", 13, TXT, weight="bold")]
    rows = [("step 1", "{worried}", 0.500, True), ("step 1", "{sad}", 0.500, False), ("step 1", "{angry}", 0.125, False),
            ("step 2", "{worried, sad}", 0.800, True), ("step 2", "{worried, angry}", 0.500, False), ("step 2", "{worried, anxious*}", 0.500, False),
            ("step 3", "{worried, sad, angry}", 0.750, False), ("step 3", "{worried, sad, happy}", 0.750, False)]
    x0, y0, W, bh = 300, 60, 480, 22
    for i, (st, lab, v, sel) in enumerate(rows):
        y = y0 + i * (bh + 8) + (12 if i >= 3 else 0) + (12 if i >= 6 else 0)
        b.append(text(20, y + 16, st, 12, TXT2))
        b.append(text(285, y + 16, lab, 12, TXT, "end", "bold" if sel else "normal"))
        w = W * v
        b.append(f"<rect x='{x0}' y='{y}' width='{w:.1f}' height='{bh}' rx='4' fill='{SER[0] if sel else '#b6c6d8'}'/>")
        b.append(text(x0 + w + 6, y + 16, f"{v:.3f}" + ("  <- kept" if sel else ""), 12, TXT, weight="bold" if sel else "normal"))
    b.append(text(20, 345, "Step 1 tie (worried = sad) is broken by sample frequency. Stop: no step-3 addition beats 0.800, so Y* = {worried, sad}. *'anxious' = same group as 'worried'.", 11, TXT2, italic=True))
    b.append(text(20, 362, "Scores: one wheel; per-sample F = 2|G(S) ∩ G(Y)| / (|G(S)| + |G(Y)|), averaged over the four samples of Fig. 1.", 11, TXT2, italic=True))
    return svg(920, 380, "".join(b))


def fig_ncurve():
    Ns = [1, 2, 4, 8, 16]
    data = [("CONSTRUCT-D", [55.56, 61.80, 66.44, 67.90, 69.07]), ("MultiMER rule", [48.52, 60.50, 64.60, 66.97, 68.72]),
            ("MBR", [53.38, 57.81, 63.36, 66.48, 68.05]), ("Agent-MER vote", [48.00, 59.97, 63.35, 63.88, 68.27]),
            ("TOPK", [48.00, 56.37, 60.52, 64.74, 64.92])]
    L, T, W, H = 80, 60, 600, 300
    ymin, ymax = 45, 72
    xs = lambda i: L + W * i / (len(Ns) - 1)
    ys = lambda v: T + H * (1 - (v - ymin) / (ymax - ymin))
    b = [text(20, 28, "Figure 3. EW-F on OV-MERD+ vs. number of samples N (same samples for every aggregator; Qwen2.5-Omni-7B).", 13, TXT, weight="bold")]
    for v in range(45, 73, 5):
        b.append(f"<line x1='{L}' y1='{ys(v):.1f}' x2='{L + W}' y2='{ys(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        b.append(text(L - 8, ys(v) + 4, str(v), 11, TXT2, "end"))
    for i, n in enumerate(Ns):
        b.append(text(xs(i), T + H + 20, str(n), 11, TXT2, "middle"))
    b.append(text(L + W / 2, T + H + 42, "samples per clip (N, log scale)", 12, TXT2, "middle"))
    b.append(text(22, T + H / 2, "EW-F", 12, TXT2, "middle"))
    for (name, ys_), c in zip(data, SER):
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(ys_))
        b.append(f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2'/>")
        for i, v in enumerate(ys_):
            b.append(f"<circle cx='{xs(i):.1f}' cy='{ys(v):.1f}' r='4.5' fill='{c}' stroke='{SURF}' stroke-width='2'/>")
    # direct labels at right end, de-overlapped
    ends = sorted([(ys(d[1][-1]), d[0], c, d[1][-1]) for d, c in zip(data, SER)])
    last = -99
    for y, name, c, v in ends:
        y = max(y, last + 15); last = y
        b.append(f"<rect x='{L + W + 10}' y='{y - 9}' width='10' height='10' rx='2' fill='{c}'/>")
        b.append(text(L + W + 25, y, f"{name} ({v:.2f})", 12, TXT))
    b.append(text(L, T + H + 70, "N = 16 values: 5-repetition cross-fitting (Table 3); N < 16: one repetition. Tuned rules choose k / tau / beta by 5-fold cross-fitting.", 11, TXT2, italic=True))
    return svg(920, 450, "".join(b))


def fig_ceiling():
    rows = [("Omni-7B greedy", 60.46, "#b6c6d8"), ("CONSTRUCT (label-free)", 68.91, SER[0]), ("MER-R1 (SFT + RL, published)", 70.68, "#9a9a9a"),
            ("Oracle: best sampled list", 84.73, "#d9d9d9"), ("Oracle: best subset of sampled labels", 91.48, "#d9d9d9")]
    L, T, W, bh = 300, 60, 520, 26
    b = [text(20, 28, "Figure 4. Headroom on OV-MERD+: the 16 samples already contain far better label sets (oracles use the ground truth).", 13, TXT, weight="bold")]
    for v in range(0, 101, 20):
        x = L + W * v / 100
        b.append(f"<line x1='{x}' y1='{T - 5}' x2='{x}' y2='{T + 5 * (bh + 10)}' stroke='{GRID}' stroke-width='1'/>")
        b.append(text(x, T + 5 * (bh + 10) + 16, str(v), 11, TXT2, "middle"))
    for i, (lab, v, c) in enumerate(rows):
        y = T + i * (bh + 10)
        b.append(text(L - 10, y + 18, lab, 12, TXT, "end"))
        b.append(f"<rect x='{L}' y='{y}' width='{W * v / 100:.1f}' height='{bh}' rx='4' fill='{c}'/>")
        b.append(text(L + W * v / 100 + 6, y + 18, f"{v:.2f}", 12, TXT, weight="bold"))
    b.append(text(20, T + 5 * (bh + 10) + 40, "EW-F level 1. Light-grey bars are not methods: they pick with the ground truth and bound what any sample-based decoder can reach.", 11, TXT2, italic=True))
    return svg(920, 290, "".join(b))


for name, f in (("fig1_pipeline.svg", fig_pipeline), ("fig2_construct_steps.svg", fig_construct), ("fig3_n_curve.svg", fig_ncurve), ("fig4_headroom.svg", fig_ceiling)):
    open(os.path.join(OUT, name), "w").write(f()); print("wrote", name)
