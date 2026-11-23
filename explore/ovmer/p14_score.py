"""P14': confirmation of the constructive emotion-wheel MBR decoder (CONSTRUCT) on STANDARD open-vocabulary benchmarks.
CONSTRUCT was designed on P13 outputs (exploratory; explore/ovmer/p13_report_2026-11-23.md). Runs on the USER's Kaggle; writes
aggregate scores only. Replaces the earlier P14 design (fresh EmoS-MELD + OV-MERD subset), which was never run.

PRE-REGISTERED (2026-11-23, before any P14' output exists):
  Benchmarks:
    OV-MERD+ (532 clips; the benchmark reported by AffectGPT 62.52, AffectGPT-R1 / AffectGPT-RL 68.39, EW-F level1). 332 of its
      clips were used to design CONSTRUCT -> reported for the comparison with published numbers, NOT part of the decision.
    MER2025-FG (1,000 clips, MER2025 fine-grained track test labels, AffectGPT-RL/dataset/mer2025ov1000-emotion.csv) -> PRIMARY.
    EmoS-MELD-fresh (400 EmoS-MELD clips disjoint from P13; GT = GoEmotions labels with annotator share >= 0.5, excluding neutral
      and unsure) -> PRIMARY.
    Media: MER2026 video_7z/video_track2_train_human/video_split.zip (OV-MERD+, MER2025-FG); MELD train (EmoS-MELD-fresh).
  Backbones (zero-shot; same prompt, 8 frames <= 336 px, audio, subtitle as P13):
    Qwen/Qwen2.5-Omni-3B fp16, 48 samples = 3 seeds x 16; Qwen/Qwen2.5-Omni-7B 4-bit NF4 LM, 16 samples = 1 seed (time budget).
  Decoders (frozen code, p13_mbr.decode): GREEDY; TOPK; MBR; CONSTRUCT.
  Metric: corpus EW-F level1 (mean of 5 wheels; ew_metric.py); level2 as guard. Protocol note: the model outputs the label list
    directly (no LLM label-extraction step as in the official AffectGPT pipeline); this is stated next to published numbers.
  Statistics: differences averaged over the backbone's seeds; paired bootstrap over clips (2,000 draws, same resample for all seeds).
  HOLDS iff on the 4 primary cells (2 backbones x {MER2025-FG, EmoS-MELD-fresh}):
    CONSTRUCT - GREEDY >= +3.0 with CI lower bound > 0 in all 4 cells;
    CONSTRUCT - TOPK >= +1.5 with CI lower bound > 0 in >= 3 of 4 cells and > 0 (point estimate) in all 4;
    level2 CONSTRUCT - GREEDY >= 0 (point estimate) in all 4 cells.
  A cell with < 90% of its clips decoded is "missing"; the verdict is then labelled "partial" and computed on the remaining cells.
Usage: python p14_score.py --items items.jsonl --outs o3b.jsonl,o7b.jsonl --names 3B,7B --seeds 3,1 --mapping wheel_mapping.npz --report r.md
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import decode  # noqa: E402

DEC = ("GREEDY", "TOPK", "MBR", "CONSTRUCT")
PRIMARY = ("MER2025-FG", "EmoS-MELD-fresh")
ALL = ("MER2025-FG", "EmoS-MELD-fresh", "OV-MERD+")


def main(a):
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    L = ["# P14': constructive emotion-wheel MBR on standard OV benchmarks (aggregate scores only)",
         "Published OV-MERD+ EW-F (level1): AffectGPT 62.52; AffectGPT-R1 / AffectGPT-RL 68.39 (fine-tuned 7B + RL; official pipeline)."]
    cells = {}
    for name, path, ns in zip(a.names.split(","), a.outs.split(","), map(int, a.seeds.split(","))):
        R = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
        R = [r for r in R if r.get("ok")]
        for ds in ALL:
            RR = [r for r in R if r["ds"] == ds]; n_exp = sum(1 for it in items.values() if it["ds"] == ds)
            if n_exp == 0:
                continue
            if len(RR) < 0.9 * n_exp:
                L.append(f"\n## {name} / {ds}: missing ({len(RR)} of {n_exp} clips decoded)"); continue
            G = [items[r["key"]]["gt"] for r in RR]
            D = [[decode(r, ew, r["samples"][16 * s:16 * (s + 1)]) for r in RR] for s in range(ns)]
            L.append(f"\n## {name} / {ds} (n={len(RR)} of {n_exp}; {ns} seed(s) x 16 samples)")
            L.append("| decoder | EW-F L1 mean ± sd | P | R | EW-F L2 | mean #labels |"); L.append("|---|---|---|---|---|---|")
            for k in DEC:
                sc = [ew.corpus(G, [d[k] for d in D[s]]) for s in range(ns)]; f2 = [ew.corpus(G, [d[k] for d in D[s]], "level2")[0] for s in range(ns)]
                L.append(f"| {k} | {np.mean([x[0] for x in sc]):.2f} ± {np.std([x[0] for x in sc]):.2f} | {np.mean([x[1] for x in sc]):.2f} | "
                         f"{np.mean([x[2] for x in sc]):.2f} | {np.mean(f2):.2f} | {np.mean([len(d[k]) for s in range(ns) for d in D[s]]):.2f} |")
            n = len(RR); rng = np.random.default_rng(1)

            def diff(x, y, ix, level="level1"):
                g = [G[i] for i in ix]
                return float(np.mean([ew.corpus(g, [D[s][i][x] for i in ix], level)[0] - ew.corpus(g, [D[s][i][y] for i in ix], level)[0] for s in range(ns)]))
            full = np.arange(n); res = {}
            for x, y in (("CONSTRUCT", "GREEDY"), ("CONSTRUCT", "TOPK"), ("CONSTRUCT", "MBR")):
                bs = [diff(x, y, rng.integers(0, n, n)) for _ in range(2000)]
                res[y] = (diff(x, y, full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), 2.8 * float(np.std(bs)))
                L.append(f"- CONSTRUCT - {y} = {res[y][0]:+.2f} [{res[y][1]:+.2f}, {res[y][2]:+.2f}] (MDE {res[y][3]:.2f})")
            l2 = diff("CONSTRUCT", "GREEDY", full, "level2"); L.append(f"- level2 CONSTRUCT - GREEDY = {l2:+.2f}")
            cells[(name, ds)] = dict(g=res["GREEDY"], t=res["TOPK"], l2=l2)
    prim = [(nm, ds) for nm in a.names.split(",") for ds in PRIMARY]; have = [c for c in prim if c in cells]
    ok = bool(have)
    if have:
        ok &= all(cells[c]["g"][0] >= 3.0 and cells[c]["g"][1] > 0 and cells[c]["l2"] >= 0 and cells[c]["t"][0] > 0 for c in have)
        need = 3 if len(have) == 4 else len(have)
        ok &= sum(cells[c]["t"][0] >= 1.5 and cells[c]["t"][1] > 0 for c in have) >= need
    tag = "" if len(have) == len(prim) else f" (partial: {len(have)} of {len(prim)} primary cells)"
    L.append(f"\n**P14' verdict: CONSTRUCT {'HOLDS' if ok else 'DOES NOT HOLD'}{tag}**")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "outs", "names", "mapping", "report"):
        ap.add_argument("--" + k)
    ap.add_argument("--seeds", default="3,1")
    main(ap.parse_args())
