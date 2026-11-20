"""Analyse the H intervention (pre-registered in INTERVENTION.md, 2026-11-20).

Per (fold, seed) pair: Delta acc = acc(human) - acc(shuffled), with acc = mean test accuracy of the last
10 epochs (primary; no test-based or val-based checkpoint choice); Delta align = alignment(human) -
alignment(shuffled), alignment = Spearman(rel_lf, d_human) over test clips at the val-selected checkpoint.
95% CIs: paired t over pairs (decision) and a two-level bootstrap (pairs, then clips) for reference.

Usage: python analyze_intervention.py --root /data/cremad --runs runs_interv [--out intervention_report.md]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

CONDS = ["human", "shuffled", "anti", "uniform"]


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean() - h, x.mean() + h


def load(runs_dir, human):
    rows, preds = [], {}
    for m in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        r = json.load(open(m))
        if r.get("method") != "human_aux":
            continue
        d = os.path.dirname(m)
        p = pd.read_csv(os.path.join(d, "preds_test.csv")).merge(human, on="clip", how="left")
        p["share_v"] = p["kl_drop_v"] / (p["kl_drop_v"] + p["kl_drop_a"] + 1e-12)
        q = p.dropna(subset=["d_human"])
        rows.append({"fold": r["split"], "seed": r["seed"], "cond": r["aux_weights"],
                     "acc_last10": 100 * r["last10_test_acc"], "acc_valsel": 100 * r["test_acc"],
                     "align": spearmanr(q["rel_lf"], q["d_human"]).correlation, "share_v": q["share_v"].mean()})
        preds[(r["split"], r["seed"], r["aux_weights"])] = p
    return pd.DataFrame(rows), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", default="runs_interv")
    ap.add_argument("--out", default="intervention_report.md")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    human = pd.read_csv(os.path.join(a.root, "human_reliability.csv"))[["clip", "d_human"]]
    R, P = load(a.runs, human)
    L = []
    say = lambda s="": (print(s), L.append(s))  # noqa: E731
    if R.empty:
        raise SystemExit("no human_aux runs found")
    say(f"Runs: {len(R)} | per condition: {R.cond.value_counts().to_dict()}")
    say()
    say("## Means per condition (over all runs)")
    say("| condition | n | acc last-10 (%) | acc val-selected (%) | alignment | visual share |")
    say("|---|---|---|---|---|---|")
    for c in CONDS:
        g = R[R.cond == c]
        if len(g):
            say(f"| {c} | {len(g)} | {g.acc_last10.mean():.2f} ± {g.acc_last10.std(ddof=0):.2f} | "
                f"{g.acc_valsel.mean():.2f} | {g.align.mean():.3f} ± {g.align.std(ddof=0):.3f} | {g.share_v.mean():.3f} |")
    say()

    W = R.pivot_table(index=["fold", "seed"], columns="cond", values=["acc_last10", "acc_valsel", "align", "share_v"])
    say("## Paired contrasts over (fold, seed) pairs")
    res = {}
    for other in ["shuffled", "anti", "uniform"]:
        ok = W.dropna(subset=[("acc_last10", "human"), ("acc_last10", other)])
        if ok.empty:
            continue
        da = (ok[("acc_last10", "human")] - ok[("acc_last10", other)]).to_numpy()
        dv = (ok[("acc_valsel", "human")] - ok[("acc_valsel", other)]).to_numpy()
        dl = (ok[("align", "human")] - ok[("align", other)]).to_numpy()
        ds = (ok[("share_v", "human")] - ok[("share_v", other)]).to_numpy()
        res[other] = (da, dl, ds, ok.index.tolist())
        lo, hi = t_ci(da)
        llo, lhi = t_ci(dl)
        say(f"- human - {other} (n pairs = {len(da)}): Delta acc last-10 = {da.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
            f"(pairs > 0: {(da > 0).sum()}/{len(da)}); Delta acc val-selected = {dv.mean():+.2f}; "
            f"Delta align = {dl.mean():+.3f} [{llo:+.3f}, {lhi:+.3f}]; Delta visual share = {ds.mean():+.3f}")
    say()

    # two-level bootstrap for the primary contrast (pairs, then clips of the val-selected predictions)
    if "shuffled" in res:
        da, dl, ds, keys = res["shuffled"]
        rng = np.random.default_rng(0)
        clip_d = []
        for f, s in keys:
            ph, ps = P.get((f, s, "human")), P.get((f, s, "shuffled"))
            m = ph[["clip", "correct"]].merge(ps[["clip", "correct"]], on="clip", suffixes=("_h", "_s"))
            clip_d.append((m["correct_h"] - m["correct_s"]).to_numpy())
        boots = []
        for _ in range(a.boot):
            pick = rng.integers(0, len(clip_d), len(clip_d))
            boots.append(100 * np.mean([clip_d[k][rng.integers(0, len(clip_d[k]), len(clip_d[k]))].mean() for k in pick]))
        say(f"- Reference: two-level bootstrap of Delta acc (val-selected checkpoints; pairs then clips): "
            f"{100*np.mean([c.mean() for c in clip_d]):+.2f} [{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]")
        say()

    say("## Decision (rule fixed in INTERVENTION.md before any run)")
    if "shuffled" not in res or len(res["shuffled"][0]) < 2:
        say("- Outcome: UNDETERMINED - fewer than 2 complete human/shuffled pairs")
    else:
        da, dl, ds, _ = res["shuffled"]
        manip = dl.mean() > 0 and t_ci(dl)[0] > 0
        effect = da.mean() > 0 and t_ci(da)[0] > 0
        say(f"- Manipulation check (Delta align > 0, CI excludes 0): {'PASSED' if manip else 'FAILED'}")
        say(f"- Effect (Delta acc last-10 > 0, CI excludes 0): {'YES' if effect else 'NO'}")
        if abs(ds.mean()) > 0.1:
            say(f"- Note: |Delta visual share| = {abs(ds.mean()):.3f} > 0.1, balancing strength differs between conditions")
        outcome = ("H SUPPORTED (causal)" if (manip and effect) else
                   "H NOT SUPPORTED" if manip else "INCONCLUSIVE - manipulation check failed")
        say(f"- Outcome: {outcome}")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
