"""P14' follow-up (descriptive, added 2026-11-24 after reading the full text of the two closest OV-MER aggregation papers):
compare CONSTRUCT with the aggregation rules of
  - Agent-MER (Lai et al., ACM MM 2025, doi 10.1145/3746027.3762011, Sec. 3.5, Eq. 3): a label is kept if it appears in at least
    k of the N candidate sets; k tuned on OV-MERD (Sec. 4.4, Fig. 3).
  - MultiMER (Bao et al., ACM MM 2026, doi 10.1145/3767308.3838611, Sec. 2.2-2.3; released code track2/build_ensemble_v16.py):
    hand-written NORM_MAP normalisation (211 rules) + rank-decayed weighted voting, keep words with score >= tau * max,
    clipped to [3, 6] words; tau = 0.29 (paper). Here every sample is one voter with weight 1. The train-vocabulary boost is
    disabled (beta = 1): its vocabulary comes from the MER2026 human-annotated clips, which include the evaluation sets.
applied to the SAME 16 sampled label lists per clip (P14' amendment 1 outputs).
Reported: k-of-N voting with k chosen on OV-MERD+ (as in Agent-MER) and the oracle k chosen on each test set (an upper bound
for that rule); MultiMER rule with the paper's tau and with the oracle tau per set (upper bound); CONSTRUCT with and without
NORM_MAP. Paired bootstrap CONSTRUCT - rule, 1,000 draws. Aggregate scores only.
Usage: python p14_prior_aggregators.py --items items.jsonl --labels labels.jsonl --mapping wheel_mapping.npz
        --multimer_src MultiMER/track2/build_ensemble_v16.py --report r.md
"""
import argparse, ast, json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import construct, parse  # noqa: E402

SETS = ("OV-MERD+", "MER2025-FG", "EmoS-MELD-fresh")


def load_norm_map(src):  # read the dict literal from the released source without executing the file
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NORM_MAP" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("NORM_MAP not found")


def mm_normalize(w, NM):  # MultiMER normalize()
    w = w.strip().lower()
    if " " in w:
        return None
    if w in NM:
        return NM[w]
    c = re.sub(r"[^a-z_-]", "", w)
    if not c:
        return None
    for v in (c, c.replace("-", "_"), c.replace("-", "")):
        if v in NM:
            return NM[v]
    r = c.replace("-", "_")
    return r if 2 <= len(r) <= 20 else None


def vote_k(S, k):
    cnt = defaultdict(int)
    for s in S:
        for x in set(s):
            cnt[x] += 1
    return sorted([x for x, c in cnt.items() if c >= k], key=lambda x: (-cnt[x], x))


def multimer_rule(S, tau, decay=0.5, lo=3, hi=6):  # build_ensemble_v16.build_ensemble with unit voter weights, beta = 1
    sc = defaultdict(float)
    for s in S:
        for r, w in enumerate(s[:8]):
            sc[w] += 1.0 / (r * decay + 1)
    if not sc:
        return []
    srt = sorted(sc.items(), key=lambda x: (-x[1], x[0])); best = srt[0][1]; out = []
    for w, v in srt:
        if len(out) < lo:
            out.append(w)
        elif v >= tau * best and len(out) < hi:
            out.append(w)
        else:
            break
    return out


def main(a):
    ew = EW(a.mapping); NM = load_norm_map(a.multimer_src)
    items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    R = [json.loads(l) for l in open(a.labels)]; R = [r for r in R if r.get("ok")]
    L = ["# P14' follow-up: CONSTRUCT vs the aggregation rules of Agent-MER and MultiMER on the same 16 samples (aggregate only)",
         f"NORM_MAP rules loaded: {len(NM)}"]
    data = {}
    for ds in SETS:
        RR = [r for r in R if r["ds"] == ds]
        G = [items[r["key"]]["gt"] for r in RR]
        S = [[parse(x) for x in r["samples"]] for r in RR]
        SN = [[[y for y in (mm_normalize(x, NM) for x in s) if y] for s in ss] for ss in S]
        GR = [parse(r["greedy"]) for r in RR]
        data[ds] = (G, S, SN, GR)
    # Agent-MER: k chosen on OV-MERD+ (their development set), then applied everywhere
    G0, S0, _, _ = data["OV-MERD+"]
    k_dev = max(range(1, 17), key=lambda k: ew.corpus(G0, [vote_k(s, k) for s in S0])[0])
    L.append(f"Agent-MER k chosen on OV-MERD+: k = {k_dev} of 16")
    for ds in SETS:
        G, S, SN, GR = data[ds]; n = len(G)
        P = {}
        P["CONSTRUCT"] = [construct(s, g, ew) for s, g in zip(S, GR)]  # identical to p13_mbr.decode
        P["CONSTRUCT + NORM_MAP"] = [construct(s, [y for y in (mm_normalize(x, NM) for x in g) if y], ew) for s, g in zip(SN, GR)]
        P[f"Agent-MER vote k={k_dev} (dev-tuned)"] = [vote_k(s, k_dev) for s in S]
        ks = {k: ew.corpus(G, [vote_k(s, k) for s in S])[0] for k in range(1, 17)}; k_or = max(ks, key=ks.get)
        P[f"Agent-MER vote k={k_or} (oracle on this set)"] = [vote_k(s, k_or) for s in S]
        P["MultiMER rule tau=0.29 (paper)"] = [multimer_rule(s, 0.29) for s in SN]
        taus = {t: ew.corpus(G, [multimer_rule(s, t) for s in SN])[0] for t in np.round(np.arange(0.05, 1.0001, 0.05), 2)}
        t_or = max(taus, key=taus.get)
        P[f"MultiMER rule tau={t_or} (oracle on this set)"] = [multimer_rule(s, t_or) for s in SN]
        P[f"MultiMER rule tau={t_or}, no [3,6] clip (oracle)"] = [multimer_rule(s, t_or, lo=1, hi=8) for s in SN]
        L.append(f"\n## {ds} (n={n})"); L.append("| aggregator | EW-F L1 | EW-F L2 | mean #labels | CONSTRUCT - aggregator [95% CI] |")
        L.append("|---|---|---|---|---|")
        rng = np.random.default_rng(1); boots = [rng.integers(0, n, n) for _ in range(1000)]
        for name, pr in P.items():
            f1 = ew.corpus(G, pr)[0]; f2 = ew.corpus(G, pr, "level2")[0]
            if name == "CONSTRUCT":
                d = ""
            else:
                pt = ew.corpus(G, P["CONSTRUCT"])[0] - f1
                bs = [ew.corpus([G[i] for i in ix], [P["CONSTRUCT"][i] for i in ix])[0] - ew.corpus([G[i] for i in ix], [pr[i] for i in ix])[0] for ix in boots]
                d = f"{pt:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
            L.append(f"| {name} | {f1:.2f} | {f2:.2f} | {np.mean([len(x) for x in pr]):.2f} | {d} |")
        print("\n".join(L[-len(P) - 3:]), flush=True)
    open(a.report, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "labels", "mapping", "multimer_src", "report"):
        ap.add_argument("--" + k)
    main(ap.parse_args())
