"""POST-HOC power / effect-size check for the H intervention (added 2026-11-21, AFTER seeing results).

This does NOT change the pre-registered decision in INTERVENTION.md. It asks: given how much the manipulation
moved alignment, how big an accuracy effect would H predict, and could 8 pairs have detected it?
Predicted effect = (slope of last-10 test acc on alignment across pilot actor-protocol runs) x (observed Delta align).
That slope is an UPPER-BOUND reading of H: it assumes the whole pilot association is caused by alignment.

Usage: python analyze_intervention_power.py --root /data/cremad --pilot runs --interv runs_interv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr


def last10(d):
    L = [json.loads(x) for x in open(os.path.join(d, "log.jsonl"))]
    return 100 * np.mean([x["test"]["acc"] for x in L if "test" in x and x.get("epoch", -1) >= 90])


def load(runs, human):
    rows = []
    for m in sorted(glob.glob(os.path.join(runs, "*", "metrics.json"))):
        r = json.load(open(m))
        if r["method"] in ("audio_only", "visual_only"):
            continue
        d = os.path.dirname(m)
        p = pd.read_csv(os.path.join(d, "preds_test.csv")).merge(human, on="clip").dropna(subset=["d_human"])
        rows.append(dict(split=r["split"], seed=r["seed"], method=r["method"], cond=r.get("aux_weights"),
                         acc=last10(d), align=spearmanr(p.rel_lf, p.d_human).correlation))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--interv", required=True)
    a = ap.parse_args()
    human = pd.read_csv(os.path.join(a.root, "human_reliability.csv"))[["clip", "d_human"]]
    P = load(a.pilot, human)
    A = P[P.split.str.startswith("actor")]
    sl = stats.linregress(A["align"], A["acc"])
    print(f"Pilot, actor protocol, joint + OGM-GE (n = {len(A)}): slope of last-10 acc on alignment = "
          f"{sl.slope:.1f} points per unit (SE {sl.stderr:.1f}), r = {sl.rvalue:.2f}")
    I = load(a.interv, human)
    W = I.pivot_table(index=["split", "seed"], columns="cond", values=["acc", "align"])
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    for x, y in [("human", "shuffled"), ("human", "anti"), ("anti", "shuffled"), ("human", "uniform")]:
        da = W["acc"][x] - W["acc"][y]
        dl = W["align"][x] - W["align"][y]
        n, sd = len(da), da.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        pred = sl.slope * dl.mean()
        mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * sd / np.sqrt(n)
        need = int(np.ceil((z * sd / abs(pred)) ** 2)) if pred else -1
        print(f"{x} - {y}: Delta align {dl.mean():+.3f} | Delta acc {da.mean():+.2f} [{da.mean() - h:+.2f}, "
              f"{da.mean() + h:+.2f}] | SD(diff) {sd:.2f} | predicted by pilot slope {pred:+.2f} | "
              f"MDE(80%, n={n}) {mde:.2f} | pairs for 80% power at predicted effect: {need}")
    X = I[I.cond.isin(["human", "shuffled", "anti"])].copy()  # same multiset of weights, same lambda
    X["pair"] = X.split + "_" + X.seed.astype(str)
    X["acc_c"] = X.acc - X.groupby("pair").acc.transform("mean")
    X["al_c"] = X["align"] - X.groupby("pair")["align"].transform("mean")
    s2 = stats.linregress(X.al_c, X.acc_c)
    dof = len(X) - 1 - X.pair.nunique()
    se = s2.stderr * np.sqrt((len(X) - 2) / dof)
    h = stats.t.ppf(0.975, dof) * se
    print(f"EXPLORATORY within-pair slope over human/shuffled/anti (n = {len(X)} runs, {X.pair.nunique()} pairs): "
          f"{s2.slope:.1f} [{s2.slope - h:.1f}, {s2.slope + h:.1f}] points per unit")


if __name__ == "__main__":
    main()
