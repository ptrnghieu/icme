"""P14' (single-protocol version, replaces the never-run 09a/09b/09c design): everything uses the OFFICIAL OV-MER protocol, so every
number is directly comparable with the published OV-MERD+ results.

PRE-REGISTERED (2026-11-23, before any P14' output exists):
  Protocol (official, zeroQiaoba/AffectGPT): the MLLM writes a free-text emotion description with the official OV-MER prompt
    (p13_mbr.PROMPT_OFFICIAL); Qwen/Qwen2.5-7B-Instruct extracts the labels with the official extraction prompt and sampling
    parameters (temperature 0.7, top_p 0.8, repetition_penalty 1.05; 4-bit NF4 on T4, fixed seed) (p14_official.py).
  Backbone: Qwen/Qwen2.5-Omni-3B, zero-shot (8 frames <= 336 px, audio, subtitle). Greedy description + 16 sampled descriptions.
  Decoders applied to the extracted label lists (frozen code, p13_mbr.decode): GREEDY, TOPK, MBR, CONSTRUCT.
  Data: OV-MERD+ (532; published benchmark; 332 of its clips were seen when CONSTRUCT was designed -> comparison table only),
    MER2025-FG (1,000; PRIMARY), EmoS-MELD-fresh (400 clips disjoint from P13; PRIMARY).
  Metric: corpus EW-F level1 (ew_metric.py); level2 as guard. Paired bootstrap over clips, 2,000 draws.
  HOLDS iff on BOTH primary sets: CONSTRUCT - GREEDY >= +3.0 with CI lower bound > 0; CONSTRUCT - TOPK > 0 (point estimate);
    level2 CONSTRUCT - GREEDY >= 0; AND CONSTRUCT - TOPK >= +1.5 with CI lower bound > 0 on at least one of the two primary sets.
  A set with < 90% of its clips extracted is "missing" and the verdict is labelled "partial".
Usage: python p14o_score.py --items items.jsonl --labels l1.jsonl,l2.jsonl,l3.jsonl --mapping wheel_mapping.npz --report r.md
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import decode  # noqa: E402

PRIMARY = ("MER2025-FG", "EmoS-MELD-fresh")


def main(a):
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    R = [json.loads(l) for p in a.labels.split(",") if os.path.exists(p) for l in open(p)]; R = [r for r in R if r.get("ok")]
    L = ["# P14' (official protocol only): Qwen2.5-Omni-3B zero-shot + label decoders (aggregate scores only)"]; cells = {}
    for ds in ("OV-MERD+",) + PRIMARY:
        RR = [r for r in R if r["ds"] == ds]; n_exp = sum(1 for it in items.values() if it["ds"] == ds)
        if len(RR) < 0.9 * max(n_exp, 1):
            L.append(f"\n## {ds}: missing ({len(RR)} of {n_exp})"); continue
        G = [items[r["key"]]["gt"] for r in RR]; D = [decode(r, ew) for r in RR]
        L.append(f"\n## {ds} (n={len(RR)} of {n_exp})"); L.append("| decoder | EW-F L1 | P | R | EW-F L2 | mean #labels |"); L.append("|---|---|---|---|---|---|")
        for k in ("GREEDY", "TOPK", "MBR", "CONSTRUCT"):
            f = ew.corpus(G, [d[k] for d in D]); f2 = ew.corpus(G, [d[k] for d in D], "level2")[0]
            L.append(f"| {k} | {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} | {f2:.2f} | {np.mean([len(d[k]) for d in D]):.2f} |")
        if ds == "OV-MERD+":
            L.append("| *published: AffectGPT (SFT 7B)* | *62.52* | | | | |"); L.append("| *published: AffectGPT-R1 / -RL (SFT + RL 7B)* | *68.39* | | | | |")
        n = len(RR); rng = np.random.default_rng(1); res = {}
        for y in ("GREEDY", "TOPK", "MBR"):
            pt = ew.corpus(G, [d["CONSTRUCT"] for d in D])[0] - ew.corpus(G, [d[y] for d in D])[0]; bs = []
            for _ in range(2000):
                ix = rng.integers(0, n, n); g = [G[i] for i in ix]
                bs.append(ew.corpus(g, [D[i]["CONSTRUCT"] for i in ix])[0] - ew.corpus(g, [D[i][y] for i in ix])[0])
            res[y] = (pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            L.append(f"- CONSTRUCT - {y} = {pt:+.2f} [{res[y][1]:+.2f}, {res[y][2]:+.2f}]")
        l2 = ew.corpus(G, [d["CONSTRUCT"] for d in D], "level2")[0] - ew.corpus(G, [d["GREEDY"] for d in D], "level2")[0]
        L.append(f"- level2 CONSTRUCT - GREEDY = {l2:+.2f}"); cells[ds] = dict(r=res, l2=l2)
    have = [ds for ds in PRIMARY if ds in cells]; ok = bool(have)
    ok &= all(cells[d]["r"]["GREEDY"][0] >= 3.0 and cells[d]["r"]["GREEDY"][1] > 0 and cells[d]["r"]["TOPK"][0] > 0 and cells[d]["l2"] >= 0 for d in have)
    ok &= any(cells[d]["r"]["TOPK"][0] >= 1.5 and cells[d]["r"]["TOPK"][1] > 0 for d in have)
    tag = "" if len(have) == len(PRIMARY) else f" (partial: {', '.join(have) or 'no primary set'})"
    L.append(f"\n**P14' verdict: CONSTRUCT {'HOLDS' if ok else 'DOES NOT HOLD'}{tag}**")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "labels", "mapping", "report"):
        ap.add_argument("--" + k)
    main(ap.parse_args())
