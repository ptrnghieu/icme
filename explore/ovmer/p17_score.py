"""P17 scorer: CONSTRUCT / CONSTRUCT-D vs standard decoders and vs the calibrated aggregation rules of Agent-MER and MultiMER,
with 5-fold cross-fitting so that every clip is scored and no parameter is chosen on the clip it is scored on.

PRE-REGISTERED 2026-11-24, before the full P17 run (only the 21-clip timing run exists; its descriptions were inspected for
STRUCTURE only, with p17_inspect.py; no label was extracted and no score was computed).
P17 data and protocol:
  - phase 1: p17_describe.py (Qwen2.5-Omni-7B thinker, bf16, transformers 5.24.0, audio from the video track; greedy + 16 samples;
    OV-MERD+ 3 sample seeds);
  - phase 2: p17_extract_vllm.py (Qwen2.5-7B-Instruct, vLLM, official AffectGPT-RL prompt and parameters, no translation);
  - sets: OV-MERD+ 532, MER2025-FG 1,000, EmoS-MELD all 2,732 clips with a non-empty GT.
Cut rule (fixed from the timing run): --cut turn, i.e. drop everything from the first invented dialogue turn ("Human:" in 215 of 218
  first markers, "Assistant:" 2, "User:" 1); the model's own text before it is kept. Reason: Omni-7B invents a "Human:" turn in
  44-73% of descriptions. 'turn' is the smallest change to the official input (which is the full text): it keeps multi-paragraph
  answers (47-89% stay multi-line), whereas 'firstline' would drop 33-60% of the model's own text. Sensitivity runs: extraction
  seed 1; the older prompt without the four-label limit.

Inputs per clip: greedy + 16 sampled label lists from the same backbone and extractor. Every method aggregates the same lists.

Methods (one parameter each, grids fixed here):
  CONSTRUCT-D: beta in {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4} (p16_density.construct_beta).
  Agent-MER vote: k in 1..16.
  MultiMER rule (NORM_MAP + rank-decayed voting, [3, 6] clip): tau in {0.05, ..., 1.00}.
  MultiMER rule without the clip: same tau grid.
  Parameter-free, reported: GREEDY, TOPK, MBR, CONSTRUCT (p13_mbr.decode, frozen).

Cross-fitting:
  5 folds, the same folds for every method. The parameter is chosen on the 4 other folds (corpus EW-F level 1), then applied to
  the held-out fold.
  Repeated with 5 fold seeds; per clip the prediction of each repetition is kept, and scores are averaged over the 5 repetitions.

Statistics:
  Paired bootstrap over clips, 2,000 draws per set; the same clip draws are used for every repetition.
  Pooled effect = clip-weighted mean of the per-set differences; its CI comes from the same bootstrap, with each set drawn
  independently.

Power (computed before registration from the P14′ data and the P16 bootstrap; explore/ovmer/p16_report_2026-11-24.md):
  expected SE of the pooled CONSTRUCT-D - rule difference about 0.31 points; the P16 effect, about +1.1, gives power about 0.9 at
  alpha 0.05.

PRIMARY (decides P17): for BOTH calibrated rules (Agent-MER vote, MultiMER rule with clip), the pooled CONSTRUCT-D - rule is > 0
  with CI lower bound > 0, AND on no set is CONSTRUCT-D worse than either rule with CI upper bound < 0.
BENCHMARK ROW (OV-MERD+, comparable with published numbers, no OV-MERD+ label used): parameter-free CONSTRUCT (frozen,
  beta = 1) and GREEDY (the zero-shot baseline), each as mean ± sd over the 3 sample seeds (16 samples per seed).
  CONSTRUCT-D (cross-fitted) uses OV-MERD+ labels to choose beta and is therefore an analysis, not a benchmark entry.
SECONDARY: per-set differences; CONSTRUCT - GREEDY / TOPK / MBR per set; the curve over N = 1, 2, 4, 8, 16 samples (first N
  samples, for every method).

Aggregate scores only.
Usage: python p17_score.py --items items.jsonl --labels labels.jsonl --mapping wheel_mapping.npz
        --multimer_src MultiMER/track2/build_ensemble_v16.py --report r.md [--sets A,B,C] [--reps 5] [--boot 2000] [--ncurve 1]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import parse, decode  # noqa: E402
from p14_prior_aggregators import load_norm_map, mm_normalize, vote_k, multimer_rule  # noqa: E402
from p16_density import Fast, construct_beta, BETAS, TAUS  # noqa: E402

TUNED = ("CONSTRUCT-D", "Agent-MER vote", "MultiMER rule", "MultiMER rule (no clip)")
FIXED = ("GREEDY", "TOPK", "MBR", "CONSTRUCT")
RULES = ("Agent-MER vote", "MultiMER rule")


def families(R, ew, NM, fast, nS=16):
    S = [[parse(x) for x in r["samples"][:nS]] for r in R]; GR = [parse(r["greedy"]) for r in R]
    SN = [[[y for y in (mm_normalize(x, NM) for x in s) if y] for s in ss] for ss in S]
    D = [decode(r, ew, r["samples"][:nS]) for r in R]
    fam = {"CONSTRUCT-D": {b: fast.pr([construct_beta(s, g, ew, fast, b) for s, g in zip(S, GR)]) for b in BETAS},
           "Agent-MER vote": {k: fast.pr([vote_k(s, k) for s in S]) for k in range(1, nS + 1)},
           "MultiMER rule": {t: fast.pr([multimer_rule(s, t) for s in SN]) for t in TAUS},
           "MultiMER rule (no clip)": {t: fast.pr([multimer_rule(s, t, lo=1, hi=8) for s in SN]) for t in TAUS}}
    for k in FIXED:
        fam[k] = {0: fast.pr([d[k] for d in D])}
    return fam


def crossfit(fam, n, reps):
    """Per method: list over repetitions of the per-clip (P, R, M) rows assembled from held-out folds, and the chosen parameters."""
    out = {m: [] for m in fam}; chosen = {m: [] for m in fam}
    for rep in range(reps):
        fold = np.random.default_rng(rep).permutation(n) % 5
        for m, grid in fam.items():
            P = np.zeros((n, 5)); R = np.zeros((n, 5)); M = np.zeros((n, 5))
            for f in range(5):
                tr = (fold != f).astype(float); te = fold == f
                p = max(grid, key=lambda q: Fast.f(grid[q], tr)); chosen[m].append(p)
                P[te], R[te], M[te] = grid[p][0][te], grid[p][1][te], grid[p][2][te]
            out[m].append((P, R, M))
    return out, chosen


def score(rows, w):
    return float(np.mean([Fast.f(x, w) for x in rows]))


def main(a):
    ew = EW(a.mapping); NM = load_norm_map(a.multimer_src)
    items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    R = [json.loads(l) for l in open(a.labels)]; R = [r for r in R if r.get("ok")]
    sets = a.sets.split(",")
    L = ["# P17: CONSTRUCT-D vs decoders and calibrated aggregation rules (5-fold cross-fitting; aggregate scores only)",
         f"labels: {os.path.basename(a.labels)}; repetitions {a.reps}; bootstrap {a.boot}"]
    per = {}
    for ds in sets:
        RR = [r for r in R if r["ds"] == ds]; n = len(RR); n_exp = sum(1 for it in items.values() if it["ds"] == ds)
        if n < 0.9 * max(n_exp, 1):
            L.append(f"\n## {ds}: missing ({n} of {n_exp})"); continue
        gts = [items[r["key"]]["gt"] for r in RR]; fast = Fast(ew, gts)
        fam = families(RR, ew, NM, fast); cf, chosen = crossfit(fam, n, a.reps)
        rng = np.random.default_rng(7); W = [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(a.boot)]
        one = np.ones(n); sc = {m: score(cf[m], one) for m in fam}
        bs = {m: np.array([score(cf[m], w) for w in W]) for m in fam}
        per[ds] = dict(n=n, sc=sc, bs=bs)
        L.append(f"\n## {ds} (n={n})"); L.append("| method | EW-F L1 (cross-fitted) | parameter chosen (mode) | CONSTRUCT-D - method [95% CI] |")
        L.append("|---|---|---|---|")
        for m in TUNED + FIXED:
            mode = max(set(chosen[m]), key=chosen[m].count) if m in TUNED else "-"
            d = "" if m == "CONSTRUCT-D" else (f"{sc['CONSTRUCT-D'] - sc[m]:+.2f} [{np.percentile(bs['CONSTRUCT-D'] - bs[m], 2.5):+.2f}, "
                                                 f"{np.percentile(bs['CONSTRUCT-D'] - bs[m], 97.5):+.2f}]")
            L.append(f"| {m} | {sc[m]:.2f} | {mode} | {d} |")
        for y in ("GREEDY", "TOPK", "MBR"):
            dd = bs["CONSTRUCT"] - bs[y]
            L.append(f"- CONSTRUCT - {y} = {sc['CONSTRUCT'] - sc[y]:+.2f} [{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]")
        nseed = min(len(r["samples"]) for r in RR) // 16
        if nseed > 1:
            per_seed = {k: [] for k in ("GREEDY", "TOPK", "MBR", "CONSTRUCT")}
            for s in range(nseed):
                Ds = [decode(r, ew, r["samples"][16 * s:16 * (s + 1)]) for r in RR]
                for k in per_seed:
                    per_seed[k].append(ew.corpus(gts, [d[k] for d in Ds])[0])
            L.append(f"- benchmark row ({nseed} sample seeds, no label used): " +
                     ", ".join(f"{k} {np.mean(v):.2f} ± {np.std(v):.2f}" for k, v in per_seed.items()))
        if a.ncurve:
            row = []
            for nS in (1, 2, 4, 8):
                famN = families(RR, ew, NM, fast, nS); cfN, _ = crossfit(famN, n, 1)
                row.append(f"N={nS}: " + ", ".join(f"{m} {score(cfN[m], one):.2f}" for m in ("CONSTRUCT-D", "Agent-MER vote", "MultiMER rule", "MBR", "TOPK")))
            L.append("- sample-count curve (1 repetition): " + " | ".join(row))
        print("\n".join(L[-14:]), flush=True)
    have = [ds for ds in sets if ds in per]; N = sum(per[d]["n"] for d in have) or 1
    L.append("\n## Pooled over sets (clip-weighted)" + ("" if len(have) == len(sets) else f" (partial: {', '.join(have)})"))
    ok = bool(have)
    for m in RULES + ("MultiMER rule (no clip)", "MBR", "TOPK"):
        pt = sum(per[d]["n"] / N * (per[d]["sc"]["CONSTRUCT-D"] - per[d]["sc"][m]) for d in have)
        bsd = sum(per[d]["n"] / N * (per[d]["bs"]["CONSTRUCT-D"] - per[d]["bs"][m]) for d in have) if have else np.zeros(1)
        lo, hi = np.percentile(bsd, [2.5, 97.5]); L.append(f"- CONSTRUCT-D - {m} (pooled) = {pt:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        if m in RULES:
            ok &= pt > 0 and lo > 0
    worse = any(np.percentile(per[d]["bs"]["CONSTRUCT-D"] - per[d]["bs"][m], 97.5) < 0 for d in have for m in RULES)
    ok &= not worse
    L.append(f"\n**P17 verdict (primary): CONSTRUCT-D {'HOLDS' if ok else 'DOES NOT HOLD'}** (significantly worse than a calibrated rule on some set: {worse})")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L[-8:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "labels", "mapping", "multimer_src", "report"):
        ap.add_argument("--" + k)
    ap.add_argument("--sets", default="OV-MERD+,MER2025-FG,EmoS-MELD")
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ncurve", type=int, default=1)
    main(ap.parse_args())
