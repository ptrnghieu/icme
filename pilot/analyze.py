"""Aggregate pilot runs and print the go/no-go summary.

G1  Protocol: test accuracy under the standard (actor-overlapping) split vs actor-disjoint folds;
    validation-selected vs OGM-GE-style test-selected checkpoints; OGM-GE gain over joint.
G2  Q1 "does balancing move reliance in the right direction?"
    - align  = Spearman(phi_v - phi_a, d_human) over test clips  (d_human = r_face - r_voice)
    - share_v = mean visual Shapley share
    - targeting = Spearman(Delta_i, d_human_i), Delta_i = (phi_v - phi_a)_OGM-GE - (phi_v - phi_a)_joint,
      paired by split and seed: > 0 means OGM-GE boosts visual reliance more on clips where humans rely on the face;
      ~ 0 with share_v increasing means a uniform boost.
G3  Q2 / H "does human-aligned reliance generalise to unseen actors?"
    G3a model level: over AV runs of the actor protocol, partial Spearman between alignment on the validation
        actors and test accuracy on the unseen test actors, controlling for validation accuracy
        (ranks z-scored within fold); bootstrap CI over runs.
    G3b sample level: on unseen-actor test clips, difference in correctness between clips where the sign of the
        model's reliance (phi_v - phi_a) agrees with the human sign (d_human), stratified by human AV accuracy
        quintile (difficulty control); bootstrap CI over clips.

Usage: python analyze.py --root /data/cremad --runs runs [--out pilot_report.md]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def load_runs(runs_dir):
    out = []
    for m in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        d = os.path.dirname(m)
        r = json.load(open(m))
        r["dir"] = d
        r["protocol"] = "standard" if r["split"] == "standard" else "actor"
        r["label"] = r["method"] + (f"(a={r['alpha']:g})" if r["method"] == "ogm_ge" else "")
        out.append(r)
    return pd.DataFrame(out)


def preds(run_dir, part, human):
    p = os.path.join(run_dir, f"preds_{part}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).merge(human[["clip", "r_voice", "r_face", "r_av", "d_human"]], on="clip", how="left")
    if "phi_a" in df:
        df["rel"] = df["phi_v"] - df["phi_a"]
        df["share_v"] = df["phi_v"] / (df["phi_v"].abs() + df["phi_a"].abs() + 1e-8)
    return df


def boot_ci(f, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    vals = [v for v in vals if np.isfinite(v)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def fmt(m, s):
    return f"{100*m:.2f} ± {100*s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="pilot_report.md")
    args = ap.parse_args()
    human = pd.read_csv(os.path.join(args.root, "human_reliability.csv"))
    R = load_runs(args.runs)
    if R.empty:
        raise SystemExit("no finished runs found")
    L = []
    say = lambda s="": (print(s), L.append(s))  # noqa: E731

    # ---------------- G1
    say("## G1 Protocol (test accuracy, %)")
    say("| protocol | method | n runs | val-selected | test-selected (OGM-GE style) |")
    say("|---|---|---|---|---|")
    for (pr, lab), g in R.groupby(["protocol", "label"]):
        say(f"| {pr} | {lab} | {len(g)} | {fmt(g.test_acc.mean(), g.test_acc.std(ddof=0))} | "
            f"{fmt(g.test_acc_oracle_best_epoch.mean(), g.test_acc_oracle_best_epoch.std(ddof=0))} |")
    for pr, g in R.groupby("protocol"):
        og = g[g.method == "ogm_ge"]
        jt = g[g.method == "joint"]
        if og.empty or jt.empty:
            continue
        best_alpha = og.groupby("alpha").val_acc.mean().idxmax()
        gain = og[og.alpha == best_alpha].test_acc.mean() - jt.test_acc.mean()
        say(f"- {pr}: OGM-GE alpha chosen on val = {best_alpha:g}; gain over joint on test = {100*gain:+.2f} points")
    say()

    # ---------------- G2
    say("## G2 Q1: alignment of model modality reliance with humans (test clips)")
    av = R[R.method.isin(["joint", "ogm_ge"])].copy()
    rows = []
    for _, r in av.iterrows():
        d = preds(r["dir"], "test", human)
        if d is None or d["d_human"].isna().all():
            continue
        d = d.dropna(subset=["d_human"])
        rho = spearmanr(d["rel"], d["d_human"]).correlation
        rho_score = spearmanr(np.log(d["pv_true"] + 1e-8) - np.log(d["pa_true"] + 1e-8), d["d_human"]).correlation
        rows.append({"run": r["run"], "protocol": r["protocol"], "split": r["split"], "seed": r["seed"],
                     "label": r["label"], "align_shap": rho, "align_score": rho_score,
                     "share_v": d["share_v"].mean()})
    A = pd.DataFrame(rows)
    if not A.empty:
        say("| protocol | method | align (Shapley) | align (OGM-GE score) | mean visual share |")
        say("|---|---|---|---|---|")
        for (pr, lab), g in A.groupby(["protocol", "label"]):
            say(f"| {pr} | {lab} | {g.align_shap.mean():.3f} ± {g.align_shap.std(ddof=0):.3f} | "
                f"{g.align_score.mean():.3f} | {g.share_v.mean():.3f} |")
        # targeting: paired OGM-GE vs joint (same split & seed)
        tg = []
        for _, r in av[av.method == "ogm_ge"].iterrows():
            j = av[(av.method == "joint") & (av.split == r["split"]) & (av.seed == r["seed"])]
            if j.empty:
                continue
            do = preds(r["dir"], "test", human)
            dj = preds(j.iloc[0]["dir"], "test", human)
            m = do.merge(dj[["clip", "rel", "share_v"]], on="clip", suffixes=("", "_j")).dropna(subset=["d_human"])
            delta = m["rel"] - m["rel_j"]
            tg.append({"protocol": r["protocol"], "label": r["label"],
                       "targeting": spearmanr(delta, m["d_human"]).correlation,
                       "d_share_v": (m["share_v"] - m["share_v_j"]).mean()})
        T = pd.DataFrame(tg)
        if not T.empty:
            say()
            say("Targeting (OGM-GE minus joint, paired): Spearman(Delta reliance, d_human); change in visual share")
            for (pr, lab), g in T.groupby(["protocol", "label"]):
                say(f"- {pr} {lab}: targeting = {g.targeting.mean():.3f} ± {g.targeting.std(ddof=0):.3f}, "
                    f"delta visual share = {g.d_share_v.mean():+.3f} (n={len(g)})")
    say()

    # ---------------- G3a
    say("## G3 Q2 / H: does human-aligned reliance predict accuracy on unseen actors?")
    act = av[av.protocol == "actor"]
    g3 = []
    for _, r in act.iterrows():
        dv = preds(r["dir"], "val", human)
        if dv is None:
            continue
        dv = dv.dropna(subset=["d_human"])
        g3.append({"split": r["split"], "align_val": spearmanr(dv["rel"], dv["d_human"]).correlation,
                   "val_acc": r["val_acc"], "test_acc": r["test_acc"]})
    G = pd.DataFrame(g3)
    if len(G) >= 6:
        def z(x):
            x = rankdata(x)
            return (x - x.mean()) / (x.std() + 1e-12)
        for c in ["align_val", "val_acc", "test_acc"]:
            G[c + "_z"] = G.groupby("split")[c].transform(z)

        def partial(idx):
            s = G.iloc[idx]
            X = np.c_[np.ones(len(s)), s["val_acc_z"]]
            rx = s["align_val_z"] - X @ np.linalg.lstsq(X, s["align_val_z"], rcond=None)[0]
            ry = s["test_acc_z"] - X @ np.linalg.lstsq(X, s["test_acc_z"], rcond=None)[0]
            if rx.std() < 1e-9 or ry.std() < 1e-9:
                return np.nan
            return np.corrcoef(rx, ry)[0, 1]
        pr = partial(np.arange(len(G)))
        if not np.isfinite(pr):
            say(f"- G3a (n={len(G)} runs): undefined - no variance left in alignment or test accuracy after "
                f"controlling for val accuracy (test acc values: {sorted(G.test_acc.round(4).unique())[:5]})")
        else:
            lo, hi = boot_ci(partial, len(G))
            say(f"- G3a (model level, n={len(G)} runs): partial rank corr(align on val actors, test acc | val acc) = "
                f"{pr:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    else:
        say(f"- G3a: not enough actor-protocol AV runs ({len(G)}); need >= 6")

    # ---------------- G3b
    res = []
    for sp, g in act.groupby("split"):
        frames = []
        for _, r in g.iterrows():
            d = preds(r["dir"], "test", human)
            if d is not None:
                frames.append(d.dropna(subset=["d_human", "r_av"]).assign(run=r["run"]))
        if not frames:
            continue
        D = pd.concat(frames)
        D = D[D["d_human"] != 0]
        D["agree"] = (np.sign(D["rel"]) == np.sign(D["d_human"])).astype(int)
        D["stratum"] = pd.qcut(D["r_av"].rank(method="first"), 5, labels=False)
        arr = D[["agree", "correct", "stratum"]].to_numpy()
        # cluster bootstrap: resample clips (a clip appears once per run), keep all its rows
        clip_codes = pd.factorize(D["clip"])[0]
        rows_of = [np.flatnonzero(clip_codes == k) for k in range(clip_codes.max() + 1)]

        def strat_diff(cidx, by_clip=True):
            idx = np.concatenate([rows_of[k] for k in cidx]) if by_clip else cidx
            s = arr[idx]
            diffs, ws = [], []
            for k in range(5):
                q = s[s[:, 2] == k]
                a1, a0 = q[q[:, 0] == 1][:, 1], q[q[:, 0] == 0][:, 1]
                if len(a1) and len(a0):
                    diffs.append(a1.mean() - a0.mean())
                    ws.append(len(q))
            return np.average(diffs, weights=ws) if diffs else np.nan
        est = strat_diff(np.arange(len(arr)), by_clip=False)
        lo, hi = boot_ci(strat_diff, len(rows_of), B=1000)
        res.append((sp, est, lo, hi))
        say(f"- G3b {sp}: P(correct | reliance agrees with humans) - P(correct | disagrees), stratified by human AV "
            f"accuracy = {100*est:+.2f} points, 95% CI [{100*lo:+.2f}, {100*hi:+.2f}] "
            f"(clips={len(rows_of)}, clip-runs={len(arr)}, cluster bootstrap over clips)")
    say()

    # ---------------- decision support
    say("## Decision support (rule defined in README before running)")
    say("- GO (strong): G3a CI excludes 0 with positive estimate, OR G3b positive with CI excluding 0 in every fold.")
    say("- GO (weak, discuss): only G2 shows a uniform visual boost (targeting ~ 0 while visual share rises) AND G1 "
        "shows the OGM-GE gain shrinking by more than half under the actor protocol.")
    say("- NO-GO: neither.")
    open(args.out, "w").write("\n".join(L) + "\n")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
