"""Aggregate pilot runs and print the go/no-go summary.

Reliance measures (per test/val clip, joint and OGM-GE models)
  rel_lf (PRIMARY, label-free, since 2026-11-20): KL(p_av || p_audio-only) - KL(p_av || p_visual-only)
          = Shapley(visual) - Shapley(audio) with value v(S) = -KL(p_av || p_S). Positive = the prediction depends
          more on the visual stream. Does not use the true label.
  rel     (label-based, pre-registered on 2026-11-19): phi_v - phi_a, Shapley values of log p(true class).
          Kept for transparency. It is structurally tied to correctness: a clip predicted correctly thanks to audio
          gets rel < 0, and since humans mostly rely on the face (d_human > 0), correct audio-driven predictions
          are counted as "disagreeing with humans". Do not use it for decisions.
  d_human = r_face - r_voice (human accuracy face-only minus voice-only, CREMA-D crowd ratings).

G1  Protocol: test accuracy, standard (actor-overlapping) split vs actor-disjoint folds; validation-selected vs
    OGM-GE-style test-selected checkpoints; OGM-GE gain over joint. Plus the eval-frame check (centre vs first frame).
G2  Q1 alignment = Spearman(rel_lf, d_human) over test clips; visual share; targeting = Spearman(Delta rel_lf,
    d_human) for OGM-GE minus joint, paired by split and seed (> 0: OGM-GE raises visual reliance more where
    humans rely on the face; ~ 0 with visual share rising: uniform boost).
G3  Q2 / H
    G3a (decision) model level: over AV runs of the actor protocol, partial rank correlation between alignment on
        the validation actors (unseen in training) and test accuracy on the unseen test actors, controlling for
        validation accuracy (ranks z-scored within fold); bootstrap CI over runs.
    G3b (descriptive only) sample level: correctness when the sign of rel_lf agrees vs disagrees with d_human,
        stratified by human AV accuracy; cluster bootstrap over clips. Confounded by how competent each branch is,
        so it is not a decision criterion.

Only runs with dump_version >= 2 are used for anything that involves video frames; audio-only runs of any version
are used (audio does not depend on the frame choice).

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
        r["dump_version"] = r.get("dump_version", 1)
        out.append(r)
    return pd.DataFrame(out)


def preds(run_dir, part, human):
    p = os.path.join(run_dir, f"preds_{part}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).merge(human[["clip", "r_voice", "r_face", "r_av", "d_human"]], on="clip", how="left")
    if "phi_a" in df:
        df["rel"] = df["phi_v"] - df["phi_a"]
    if "rel_lf" in df:
        df["share_v_lf"] = df["kl_drop_v"] / (df["kl_drop_v"] + df["kl_drop_a"] + 1e-12)
    return df


def sp(x, y):
    return spearmanr(x, y).correlation


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
    R_all = load_runs(args.runs)
    if R_all.empty:
        raise SystemExit("no finished runs found")
    keep = (R_all.method == "audio_only") | (R_all.dump_version >= 2)
    R = R_all[keep].copy()
    L = []
    say = lambda s="": (print(s), L.append(s))  # noqa: E731
    say(f"Runs found: {len(R_all)}; used: {len(R)}; excluded (frame-dependent, dump_version < 2): {(~keep).sum()}")
    say()

    # ---------------- G1
    say("## G1 Protocol (test accuracy, %)")
    say("| protocol | method | n runs | val-selected | test-selected (OGM-GE style) |")
    say("|---|---|---|---|---|")
    for (pr, lab), g in R.groupby(["protocol", "label"]):
        say(f"| {pr} | {lab} | {len(g)} | {fmt(g.test_acc.mean(), g.test_acc.std(ddof=0))} | "
            f"{fmt(g.test_acc_oracle_best_epoch.mean(), g.test_acc_oracle_best_epoch.std(ddof=0))} |")
    gains = {}
    for pr, g in R.groupby("protocol"):
        og, jt = g[g.method == "ogm_ge"], g[g.method == "joint"]
        if og.empty or jt.empty:
            continue
        best_alpha = og.groupby("alpha").val_acc.mean().idxmax()
        gains[pr] = og[og.alpha == best_alpha].test_acc.mean() - jt.test_acc.mean()
        say(f"- {pr}: OGM-GE alpha chosen on val = {best_alpha:g}; gain over joint on test = {100*gains[pr]:+.2f} points")
    fr = R[R.test_acc_other_eval_frame.notna()] if "test_acc_other_eval_frame" in R else R.iloc[0:0]
    if len(fr):
        say()
        say("Eval-frame check (same checkpoint; test accuracy %, centre frame vs first frame):")
        rows = [(r.label, r.test_acc, r.test_acc_other_eval_frame["test_acc"]) for r in fr.itertuples()]
        F = pd.DataFrame(rows, columns=["label", "center", "first"])
        for lab, g in F.groupby("label"):
            say(f"- {lab}: centre {100*g.center.mean():.2f} vs first {100*g['first'].mean():.2f} "
                f"(diff {100*(g.center - g['first']).mean():+.2f}, n={len(g)})")
    say()

    # ---------------- G2
    say("## G2 Q1: alignment of model modality reliance with humans (test clips)")
    av = R[R.method.isin(["joint", "ogm_ge"])].copy()
    rows = []
    for _, r in av.iterrows():
        d = preds(r["dir"], "test", human)
        if d is None or "rel_lf" not in d:
            continue
        d = d.dropna(subset=["d_human"])
        rows.append({"protocol": r["protocol"], "label": r["label"], "align_lf": sp(d["rel_lf"], d["d_human"]),
                     "align_lb": sp(d["rel"], d["d_human"]), "share_v_lf": d["share_v_lf"].mean()})
    A = pd.DataFrame(rows)
    T = pd.DataFrame()
    if not A.empty:
        say("| protocol | method | n | align label-free (primary) | align label-based | visual share (label-free) |")
        say("|---|---|---|---|---|---|")
        for (pr, lab), g in A.groupby(["protocol", "label"]):
            say(f"| {pr} | {lab} | {len(g)} | {g.align_lf.mean():.3f} ± {g.align_lf.std(ddof=0):.3f} | "
                f"{g.align_lb.mean():.3f} | {g.share_v_lf.mean():.3f} |")
        tg = []
        for _, r in av[av.method == "ogm_ge"].iterrows():
            j = av[(av.method == "joint") & (av.split == r["split"]) & (av.seed == r["seed"])]
            if j.empty:
                continue
            do, dj = preds(r["dir"], "test", human), preds(j.iloc[0]["dir"], "test", human)
            if do is None or dj is None or "rel_lf" not in do or "rel_lf" not in dj:
                continue
            m = do.merge(dj[["clip", "rel_lf", "share_v_lf"]], on="clip", suffixes=("", "_j")).dropna(subset=["d_human"])
            tg.append({"protocol": r["protocol"], "label": r["label"],
                       "targeting": sp(m["rel_lf"] - m["rel_lf_j"], m["d_human"]),
                       "d_share_v": (m["share_v_lf"] - m["share_v_lf_j"]).mean()})
        T = pd.DataFrame(tg)
        if not T.empty:
            say()
            say("Targeting (OGM-GE minus joint, paired by split and seed; label-free):")
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
        if dv is None or "rel_lf" not in dv:
            continue
        dv = dv.dropna(subset=["d_human"])
        g3.append({"split": r["split"], "align_lf": sp(dv["rel_lf"], dv["d_human"]),
                   "align_lb": sp(dv["rel"], dv["d_human"]), "val_acc": r["val_acc"], "test_acc": r["test_acc"]})
    G = pd.DataFrame(g3)
    g3a = None
    if len(G) >= 6:
        def z(x):
            x = rankdata(x)
            return (x - x.mean()) / (x.std() + 1e-12)
        for c in ["align_lf", "align_lb", "val_acc", "test_acc"]:
            G[c + "_z"] = G.groupby("split")[c].transform(z)

        def partial(idx, col):
            s = G.iloc[idx]
            X = np.c_[np.ones(len(s)), s["val_acc_z"]]
            rx = s[col] - X @ np.linalg.lstsq(X, s[col], rcond=None)[0]
            ry = s["test_acc_z"] - X @ np.linalg.lstsq(X, s["test_acc_z"], rcond=None)[0]
            if rx.std() < 1e-9 or ry.std() < 1e-9:
                return np.nan
            return np.corrcoef(rx, ry)[0, 1]
        for col, tag in [("align_lf_z", "label-free, DECISION"), ("align_lb_z", "label-based, reference only")]:
            est = partial(np.arange(len(G)), col)
            if not np.isfinite(est):
                say(f"- G3a [{tag}] (n={len(G)} runs): undefined - no variance left after controlling for val accuracy")
                continue
            lo, hi = boot_ci(lambda i: partial(i, col), len(G))
            say(f"- G3a [{tag}] (n={len(G)} runs): partial rank corr(alignment on val actors, test acc | val acc) = "
                f"{est:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
            if col == "align_lf_z":
                g3a = (est, lo, hi)
    else:
        say(f"- G3a: not enough actor-protocol AV runs with label-free columns ({len(G)}); need >= 6")

    # ---------------- G3b (descriptive)
    for sp_, g in act.groupby("split"):
        frames = []
        for _, r in g.iterrows():
            d = preds(r["dir"], "test", human)
            if d is not None and "rel_lf" in d:
                frames.append(d.dropna(subset=["d_human", "r_av"]).assign(run=r["run"]))
        if not frames:
            continue
        D = pd.concat(frames)
        D = D[(D["d_human"] != 0) & (D["rel_lf"] != 0)]
        D["agree"] = (np.sign(D["rel_lf"]) == np.sign(D["d_human"])).astype(int)
        D["stratum"] = pd.qcut(D["r_av"].rank(method="first"), 5, labels=False)
        arr = D[["agree", "correct", "stratum"]].to_numpy()
        codes = pd.factorize(D["clip"])[0]
        rows_of = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]

        def strat_diff(cidx, by_clip=True):
            s = arr[np.concatenate([rows_of[k] for k in cidx])] if by_clip else arr[cidx]
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
        say(f"- G3b [descriptive] {sp_}: P(correct | label-free reliance agrees with humans) - P(correct | disagrees), "
            f"stratified by human AV accuracy = {100*est:+.2f} points, 95% CI [{100*lo:+.2f}, {100*hi:+.2f}] "
            f"(clips={len(rows_of)}, clip-runs={len(arr)}; share agreeing = {D['agree'].mean():.2f})")
    say()

    # ---------------- decision support
    say("## Decision support (rule amended 2026-11-20, before any OGM-GE result existed; see README)")
    strong = g3a is not None and g3a[0] > 0 and g3a[1] > 0
    weak_g2 = (not T.empty) and abs(T.targeting.mean()) < 0.05 and T.d_share_v.mean() > 0
    weak_g1 = ("standard" in gains and "actor" in gains and gains["standard"] > 0
               and gains["actor"] < 0.5 * gains["standard"])
    say(f"- GO (strong): G3a label-free estimate > 0 with 95% CI excluding 0 -> "
        f"{'MET' if strong else ('not computable yet' if g3a is None else 'not met')}")
    say(f"- GO (weak, discuss): G2 uniform visual boost (|targeting| < 0.05 and visual share rises) -> "
        f"{'MET' if weak_g2 else 'not met'}; AND G1 OGM-GE gain under the actor protocol < half of the standard-split "
        f"gain -> {'MET' if weak_g1 else 'not met'}")
    if strong:
        outcome = "GO (strong)"
    elif weak_g2 and weak_g1:
        outcome = "GO (weak)"
    elif g3a is None:
        # G3a could not be computed (fewer than 6 actor-protocol AV runs, or no variance left): the decisive test
        # has not been run, so no decision can be drawn from this report.
        outcome = "UNDETERMINED - G3a not computable yet (needs >= 6 actor-protocol AV runs); not a NO-GO"
    else:
        outcome = "NO-GO"
    say(f"- Outcome: {outcome}")
    open(args.out, "w").write("\n".join(L) + "\n")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
