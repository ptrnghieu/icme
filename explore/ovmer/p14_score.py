"""P14: confirmation of the constructive emotion-wheel MBR decoder (CONSTRUCT), designed on P13 outputs (exploratory, see
explore/ovmer/p13_report_2026-11-23.md). Runs on the USER's Kaggle; writes aggregate scores only.

PRE-REGISTERED (2026-11-23, before any P14 output exists):
  Backbones: Qwen/Qwen2.5-Omni-3B (fp16) and Qwen/Qwen2.5-Omni-7B (4-bit NF4 language model; fp16 audio/vision encoders); zero-shot,
    same prompt, frames (8, <= 336 px), audio and subtitle handling as P13 (explore/ovmer/p13_mbr.py).
  Samples: 48 per clip (temperature 1.0, top_p 0.95), split in order into 3 disjoint sets of 16 = 3 seeds; greedy decoded once.
  Data: PRIMARY = EmoS-MELD-fresh: 400 EmoS-MELD clips (GT = GoEmotions labels with annotator share >= 0.5, excluding neutral and
    unsure) sampled with seed 1 from the clips NOT used in P13. SECONDARY = OV-MERD (the same 332 clips as P13, used when the decoder
    was designed; new samples only).
  Decoders (frozen code, p13_mbr.decode): GREEDY; TOPK (labels ranked by sample frequency, k = median sample length);
    MBR (best sampled list under the emotion-wheel utility); CONSTRUCT (constructive MBR: <= 12 in-vocabulary candidates by
    frequency, add the label that most raises the mean utility against the samples, stop when no gain or at 6 labels).
  Metric: corpus EW-F (level1, mean of 5 wheels; explore/ovmer/ew_metric.py); level2 as guard.
  Statistics: per seed compute EW-F of each decoder; differences averaged over the 3 seeds; paired bootstrap over clips (2,000 draws,
    the same clip resample applied to all seeds).
  HOLDS iff, for BOTH backbones on EmoS-MELD-fresh: CONSTRUCT - TOPK >= +1.5 with CI lower bound > 0, AND CONSTRUCT - GREEDY >= +3.0
    with CI lower bound > 0; AND on all four (backbone, dataset) cells CONSTRUCT - TOPK > 0 (point estimate) and level2
    CONSTRUCT - GREEDY >= 0 (point estimate). A backbone whose run fails to produce >= 90% of clips is reported as missing and the
    verdict is computed on the remaining backbone with the label "partial".
Usage: python p14_score.py --items items.jsonl --outs out_3b.jsonl,out_7b.jsonl --names 3B,7B --mapping wheel_mapping.npz --report r.md
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import decode  # noqa: E402

DEC = ("GREEDY", "TOPK", "MBR", "CONSTRUCT")


def main(a):
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    L = ["# P14: constructive emotion-wheel MBR, confirmation (aggregate scores only)"]; verdict = {}; complete = {}
    for name, path in zip(a.names.split(","), a.outs.split(",")):
        if not os.path.exists(path):
            L.append(f"\n## {name}: output missing"); complete[name] = False; continue
        R = [json.loads(l) for l in open(path)]; R = [r for r in R if r.get("ok")]
        for ds in ("EmoS-MELD-fresh", "OV-MERD"):
            RR = [r for r in R if r["ds"] == ds]; n_exp = sum(1 for it in items.values() if it["ds"] == ds)
            complete[(name, ds)] = len(RR) >= 0.9 * n_exp
            if not RR:
                L.append(f"\n## {name} / {ds}: no clips"); continue
            G = [items[r["key"]]["gt"] for r in RR]
            D = [[decode(r, ew, r["samples"][16 * s:16 * (s + 1)]) for r in RR] for s in range(3)]
            L.append(f"\n## {name} / {ds} (n={len(RR)} of {n_exp}; 3 seeds x 16 samples)")
            L.append("| decoder | EW-F L1 mean ± sd | EW-F L2 mean | mean #labels |"); L.append("|---|---|---|---|")
            for k in DEC:
                f1 = [ew.corpus(G, [d[k] for d in D[s]])[0] for s in range(3)]; f2 = [ew.corpus(G, [d[k] for d in D[s]], "level2")[0] for s in range(3)]
                L.append(f"| {k} | {np.mean(f1):.2f} ± {np.std(f1):.2f} | {np.mean(f2):.2f} | {np.mean([len(d[k]) for s in range(3) for d in D[s]]):.2f} |")
            n = len(RR); rng = np.random.default_rng(1)

            def diff(x, y, ix, level="level1"):
                g = [G[i] for i in ix]
                return float(np.mean([ew.corpus(g, [D[s][i][x] for i in ix], level)[0] - ew.corpus(g, [D[s][i][y] for i in ix], level)[0] for s in range(3)]))
            full = np.arange(n); res = {}
            for x, y in (("CONSTRUCT", "TOPK"), ("CONSTRUCT", "GREEDY"), ("CONSTRUCT", "MBR")):
                bs = [diff(x, y, rng.integers(0, n, n)) for _ in range(2000)]
                res[(x, y)] = (diff(x, y, full), np.percentile(bs, 2.5), np.percentile(bs, 97.5), 2.8 * np.std(bs))
                L.append(f"- {x} - {y} = {res[(x, y)][0]:+.2f} [{res[(x, y)][1]:+.2f}, {res[(x, y)][2]:+.2f}] (MDE {res[(x, y)][3]:.2f})")
            l2 = diff("CONSTRUCT", "GREEDY", full, "level2"); L.append(f"- level2 CONSTRUCT - GREEDY = {l2:+.2f}")
            verdict[(name, ds)] = dict(ct=res[("CONSTRUCT", "TOPK")], cg=res[("CONSTRUCT", "GREEDY")], l2=l2)
    names = [nm for nm in a.names.split(",") if complete.get((nm, "EmoS-MELD-fresh"))]
    ok = bool(names)
    for nm in names:
        v = verdict[(nm, "EmoS-MELD-fresh")]
        ok &= v["ct"][0] >= 1.5 and v["ct"][1] > 0 and v["cg"][0] >= 3.0 and v["cg"][1] > 0
        for ds in ("EmoS-MELD-fresh", "OV-MERD"):
            if (nm, ds) in verdict:
                ok &= verdict[(nm, ds)]["ct"][0] > 0 and verdict[(nm, ds)]["l2"] >= 0
    tag = "" if len(names) == len(a.names.split(",")) else f" (partial: {', '.join(names) or 'none'})"
    L.append(f"\n**P14 verdict: CONSTRUCT {'HOLDS' if ok else 'DOES NOT HOLD'}{tag}**")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "outs", "names", "mapping", "report"):
        ap.add_argument("--" + k)
    main(ap.parse_args())
