"""P16: density-calibrated CONSTRUCT vs the aggregation rules of Agent-MER and MultiMER, all calibrated on the same labelled split.

PRE-REGISTERED 2026-11-24, before any P16 output exists. Motivation (exploratory finding,
explore/ovmer/p14_prior_aggregators_report_2026-11-24.md): corpus EW-F depends on how many labels annotators give per clip
(OV-MERD+ 3.84, MER2025-FG 6.23, EmoS-MELD-fresh 1.34), but the model's own samples hold about 2.5 labels. Uncalibrated
CONSTRUCT therefore leans to precision where annotations are dense (MER2025-FG: P 73 / R 60) and to recall where they are
sparse (EmoS-MELD-fresh).

Method CONSTRUCT-D:
  CONSTRUCT (p13_mbr.construct) with the per-sample utility F1 replaced by F_beta against each sample,
    (1 + beta^2) |G ∩ Q| / (beta^2 |G| + |Q|) per wheel (G = groups of the sample, Q = groups of the candidate set),
    averaged over the 5 wheels and over the 16 samples.
  beta > 1 favours recall and gives more labels. beta is the only parameter, chosen on the calibration split.
  Grid: {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4}. Candidates: the 12 most frequent in-vocabulary labels. Up to 8 labels.

Baselines, each calibrated on the same split:
  - Agent-MER vote (label in >= k of 16 samples): k in 1..16.
  - MultiMER rule (NORM_MAP + rank-decayed voting, keep score >= tau * max, clipped to [3, 6] words, as in the paper and code):
    tau in {0.05, 0.10, ..., 1.00}.
  - MultiMER rule without the [3, 6] clip: same tau grid.
  - CONSTRUCT as frozen in P14' (beta = 1, up to 6 labels): no parameter.

Protocol:
  - Data: the P14' amendment-1 label lists (16 samples per clip).
  - Per dataset, 20 random splits (seeds 0..19): 30% calibration, 70% test.
  - Each method's parameter is the one with the highest corpus EW-F level 1 on the calibration part.
  - Score on the test part. Differences are CONSTRUCT-D minus each baseline.
  - CI: 200 paired bootstrap draws over test clips per split, pooled over the 20 splits (4,000 draws); 2.5 and 97.5 percentiles.
    The point estimate is the mean over splits.

HOLDS iff, for the two calibrated voting rules (Agent-MER vote and MultiMER rule with clip):
  on at least 2 of the 3 sets, CONSTRUCT-D beats the better of the two with a pooled CI lower bound > 0;
  and on no set is CONSTRUCT-D worse than either of them with a pooled CI upper bound < 0.
The no-clip MultiMER variant and frozen CONSTRUCT are reported, not part of the verdict.
Aggregate scores only.
Usage: python p16_density.py --items items.jsonl --labels labels.jsonl --mapping wheel_mapping.npz
        --multimer_src MultiMER/track2/build_ensemble_v16.py --report r.md
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW, WHEELS  # noqa: E402
from p13_mbr import parse, construct  # noqa: E402
from p14_prior_aggregators import load_norm_map, mm_normalize, vote_k, multimer_rule  # noqa: E402

SETS = ("OV-MERD+", "MER2025-FG", "EmoS-MELD-fresh")
BETAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
TAUS = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2))


class Fast:
    """Per-clip precision / recall per wheel for a list of predictions; corpus EW-F on any multiset of clip indices."""

    def __init__(self, ew, gts):
        self.ew = ew; self.cache = {}
        self.G = [[ew.groups(g, w) for w in WHEELS] for g in gts]

    def grp(self, label, w):
        k = (label, w)
        if k not in self.cache:
            self.cache[k] = self.ew.groups([label], w)
        return self.cache[k]

    def pr(self, preds):
        n = len(preds); P = np.zeros((n, 5)); R = np.zeros((n, 5)); M = np.zeros((n, 5))
        for i, p in enumerate(preds):
            for j, w in enumerate(WHEELS):
                G = self.G[i][j]
                if not G:
                    continue
                M[i, j] = 1; Q = set().union(*[self.grp(x, w) for x in p]) if p else set()
                if Q:
                    inter = len(G & Q); P[i, j] = inter / len(Q); R[i, j] = inter / len(G)
        return P, R, M

    @staticmethod
    def f(PRM, wts):  # wts: per-clip multiplicity (bootstrap counts or a 0/1 mask)
        P, R, M = PRM; W = M * wts[:, None]; d = W.sum(0)
        p = np.where(d > 0, (P * W).sum(0) / np.maximum(d, 1e-12), 0); r = np.where(d > 0, (R * W).sum(0) / np.maximum(d, 1e-12), 0)
        return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0))) * 100


def construct_beta(S, G, ew, fast, beta, n_cand=12, kmax=8):
    cnt = {}
    for s in S:
        for x in set(s):
            cnt[x] = cnt.get(x, 0) + 1
    for x in G:
        cnt.setdefault(x, 0)
    cands = [x for x, _ in sorted(cnt.items(), key=lambda z: (-z[1], z[0])) if ew.in_vocab(x)][:n_cand]
    SG = [[set().union(*[fast.grp(x, w) for x in s]) if s else set() for w in WHEELS] for s in S if s]
    b2 = beta * beta

    def util(cur):
        if not SG:
            return 0.0
        Q = [set().union(*[fast.grp(x, w) for x in cur]) if cur else set() for w in WHEELS]; tot = 0.0
        for sg in SG:
            fs = [(1 + b2) * len(g & q) / (b2 * len(g) + len(q)) for g, q in zip(sg, Q) if g]
            tot += float(np.mean(fs)) if fs else 0.0
        return tot / len(SG)

    cur, best = [], -1.0
    while len(cur) < kmax:
        gain = None
        for x in cands:
            if x not in cur:
                u = util(cur + [x])
                if gain is None or u > gain[0]:
                    gain = (u, x)
        if gain is None or gain[0] <= best:
            break
        best = gain[0]; cur.append(gain[1])
    return cur or (cands[:1] if cands else G)


def main(a):
    ew = EW(a.mapping); NM = load_norm_map(a.multimer_src)
    items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    R = [json.loads(l) for l in open(a.labels)]; R = [r for r in R if r.get("ok")]
    L = ["# P16: density-calibrated CONSTRUCT vs calibrated Agent-MER / MultiMER rules (aggregate scores only)",
         "20 splits x (30% calibration / 70% test); CI = pooled paired bootstrap (20 x 200 draws)."]
    verdict_sets = []; worse_any = False
    for ds in SETS:
        RR = [r for r in R if r["ds"] == ds]; gts = [items[r["key"]]["gt"] for r in RR]; n = len(RR)
        S = [[parse(x) for x in r["samples"]] for r in RR]; GR = [parse(r["greedy"]) for r in RR]
        SN = [[[y for y in (mm_normalize(x, NM) for x in s) if y] for s in ss] for ss in S]
        fast = Fast(ew, gts)
        fam = {"CONSTRUCT-D": {b: fast.pr([construct_beta(s, g, ew, fast, b) for s, g in zip(S, GR)]) for b in BETAS},
               "Agent-MER vote": {k: fast.pr([vote_k(s, k) for s in S]) for k in range(1, 17)},
               "MultiMER rule": {t: fast.pr([multimer_rule(s, t) for s in SN]) for t in TAUS},
               "MultiMER rule (no clip)": {t: fast.pr([multimer_rule(s, t, lo=1, hi=8) for s in SN]) for t in TAUS},
               "CONSTRUCT (frozen)": {0: fast.pr([construct(s, g, ew) for s, g in zip(S, GR)])}}
        print(ds, "predictions done", flush=True)
        test_sc = defaultdict(list); chosen = defaultdict(list); diffs = defaultdict(list)
        for seed in range(20):
            rng = np.random.default_rng(seed); perm = rng.permutation(n); nc = int(round(0.3 * n))
            cal = np.zeros(n); cal[perm[:nc]] = 1; tst = np.zeros(n); tst[perm[nc:]] = 1; tidx = perm[nc:]
            pick = {m: max(grid, key=lambda p: Fast.f(grid[p], cal)) for m, grid in fam.items()}
            for m in fam:
                chosen[m].append(pick[m]); test_sc[m].append(Fast.f(fam[m][pick[m]], tst))
            brng = np.random.default_rng(1000 + seed)
            for _ in range(200):
                w = np.bincount(brng.choice(tidx, len(tidx)), minlength=n).astype(float)
                fd = Fast.f(fam["CONSTRUCT-D"][pick["CONSTRUCT-D"]], w)
                for m in fam:
                    if m != "CONSTRUCT-D":
                        diffs[m].append(fd - Fast.f(fam[m][pick[m]], w))
        L.append(f"\n## {ds} (n={n}; calibration {int(round(0.3 * n))}, test {n - int(round(0.3 * n))})")
        L.append("| method | test EW-F L1 (mean ± sd over splits) | chosen parameter (mode) | CONSTRUCT-D - method [pooled 95% CI] |")
        L.append("|---|---|---|---|")
        res = {}
        for m in fam:
            mode = max(set(chosen[m]), key=chosen[m].count)
            if m == "CONSTRUCT-D":
                d = ""
            else:
                pt = float(np.mean(np.array(test_sc["CONSTRUCT-D"]) - np.array(test_sc[m])))
                lo, hi = np.percentile(diffs[m], [2.5, 97.5]); res[m] = (pt, lo, hi); d = f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            L.append(f"| {m} | {np.mean(test_sc[m]):.2f} ± {np.std(test_sc[m]):.2f} | {mode} | {d} |")
        best = max(("Agent-MER vote", "MultiMER rule"), key=lambda m: np.mean(test_sc[m]))
        win = res[best][1] > 0; worse = any(res[m][2] < 0 for m in ("Agent-MER vote", "MultiMER rule"))
        L.append(f"- better calibrated rule: {best}; CONSTRUCT-D beats it with CI > 0: {win}; worse than a calibrated rule (CI < 0): {worse}")
        verdict_sets.append(win); worse_any |= worse
        print("\n".join(L[-(len(fam) + 4):]), flush=True)
    ok = sum(verdict_sets) >= 2 and not worse_any
    L.append(f"\n**P16 verdict: CONSTRUCT-D {'HOLDS' if ok else 'DOES NOT HOLD'}** (wins on {sum(verdict_sets)} of 3 sets; significantly worse somewhere: {worse_any})")
    open(a.report, "w").write("\n".join(L) + "\n"); print(L[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "labels", "mapping", "multimer_src", "report"):
        ap.add_argument("--" + k)
    main(ap.parse_args())
