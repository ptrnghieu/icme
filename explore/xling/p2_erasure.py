"""P2: does erasing CORPUS information from frozen nonverbal features help cross-corpus transfer? (mechanism transfer:
LEACE, Belrose et al. 2023; CORAL, Sun et al. 2016). Criterion FIXED BEFORE RESULTS (2026-11-21):
  HELPS iff, for audio(WavLM) or face(ViT-FER emb), TR(method) - TR(none) >= +0.10 with group-bootstrap 95% CI excluding 0,
  in >= 2 of 3 directions {MOSI->SIMS, SIMS->MOSI, CREMA-D->MELD}, and source in-domain bacc drops by <= 2 points.
  Directions where the target oracle is within 3 points of chance count as not supporting.
Erasers are fit on UNLABELED pooled features (source train + target train); emotion labels are never used.
Usage: python -I p2_erasure.py <dir> [out.md]
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
rng = np.random.default_rng(0)


def load(name, classes=None):
    d = np.load(os.path.join(R, name), allow_pickle=True); D = {k: d[k] for k in d.files}
    if "video_id" in D:
        keep = D["ok"].astype(bool) & (D["label"].astype(float) != 0); D = {k: v[keep] for k, v in D.items()}
        D["y"] = (D["label"].astype(float) > 0).astype(int); D["grp"] = D["video_id"]
    else:
        keep = D["ok"].astype(bool) & (np.isin(D["label"], classes) if classes is not None else True)
        D = {k: v[keep] for k, v in D.items()}; D["y"] = np.array([classes.index(c) for c in D["label"]]); D["grp"] = D["group"]
    return D


def leace(X, Z):
    """closed-form LEACE eraser for concept Z (one-hot) on X; returns function x -> erased x."""
    mu = X.mean(0); Xc = X - mu; Zc = Z - Z.mean(0)
    S = Xc.T @ Xc / len(X); evals, evecs = np.linalg.eigh(S); evals = np.clip(evals, 1e-8, None)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T; Winv = evecs @ np.diag(evals ** 0.5) @ evecs.T
    Sxz = Xc.T @ Zc / len(X); A = W @ Sxz
    U, s, _ = np.linalg.svd(A, full_matrices=False); U = U[:, s > 1e-8 * s.max()]
    P = Winv @ (U @ U.T) @ W
    return lambda x: x - (x - mu) @ P.T


def coral(Xs, Xt):
    """align target to source second-order statistics; returns (fs, ft)."""
    def cov(X):
        return np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-3
    es, Vs = np.linalg.eigh(cov(Xs)); et, Vt = np.linalg.eigh(cov(Xt))
    Wt = Vt @ np.diag(np.clip(et, 1e-6, None) ** -0.5) @ Vt.T; Cs = Vs @ np.diag(np.clip(es, 1e-6, None) ** 0.5) @ Vs.T
    ms, mt = Xs.mean(0), Xt.mean(0)
    return (lambda x: x), (lambda x: (x - mt) @ Wt @ Cs + ms)


def fit_eval(Xs_tr, ys_tr, Xs_va, ys_va, Xs_te, ys_te, Xt_te, yt_te):
    sc = StandardScaler().fit(Xs_tr); best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xs_tr), ys_tr)
        s = balanced_accuracy_score(ys_va, m.predict(sc.transform(Xs_va)))
        if best is None or s > best[0]:
            best = (s, m)
    m = best[1]
    return m.predict(sc.transform(Xs_te)), m.predict(sc.transform(Xt_te))


def oracle(T, key, Xt):
    tr, va, te = (T["mode"] == "train"), (T["mode"] == "valid"), (T["mode"] == "test")
    if va.sum() == 0:
        va = tr
    _, p = fit_eval(Xt[tr], T["y"][tr], Xt[va], T["y"][va], Xt[te], T["y"][te], Xt[te], T["y"][te])
    return p


def direction(S, T, nS, nT, chance):
    say(f"\n## {nS} -> {nT}")
    say("| features | method | source in-domain bacc | cross bacc | oracle bacc | TR |"); say("|---|---|---|---|---|---|")
    sup = False
    for key, nm in [("a_wavlm", "audio(WavLM)"), ("v_emb", "face(ViT-FER emb)")]:
        s_tr, s_va, s_te = (S["mode"] == "train"), (S["mode"] == "valid"), (S["mode"] == "test")
        t_tr, t_te = (T["mode"] != "test"), (T["mode"] == "test")
        Xs, Xt = S[key].astype(float), T[key].astype(float)
        pool = np.concatenate([Xs[s_tr], Xt[t_tr]]); Z = np.concatenate([np.zeros(s_tr.sum()), np.ones(t_tr.sum())])
        er = leace(pool, np.stack([1 - Z, Z], 1))
        fs_c, ft_c = coral(Xs[s_tr], Xt[t_tr])
        variants = {"none": (lambda x: x, lambda x: x), "LEACE(corpus)": (er, er), "CORAL": (fs_c, ft_c)}
        po = oracle(T, key, Xt); o = 100 * balanced_accuracy_score(T["y"][t_te], po)
        res = {}
        for vn, (fs, ft) in variants.items():
            pi, pc = fit_eval(fs(Xs[s_tr]), S["y"][s_tr], fs(Xs[s_va]), S["y"][s_va], fs(Xs[s_te]), S["y"][s_te], ft(Xt[t_te]), T["y"][t_te])
            ins = 100 * balanced_accuracy_score(S["y"][s_te], pi); c = 100 * balanced_accuracy_score(T["y"][t_te], pc)
            tr = (c - chance) / (o - chance) if o - chance >= 3 else np.nan
            res[vn] = (pc, ins, tr)
            say(f"| {nm} | {vn} | {ins:.1f} | {c:.1f} | {o:.1f} | {'undefined' if np.isnan(tr) else f'{tr:+.2f}'} |")
        if o - chance < 3:
            continue
        yt, g = T["y"][t_te], T["grp"][t_te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}
        for vn in ["LEACE(corpus)", "CORAL"]:
            d = []
            for _ in range(1000):
                s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
                ob = 100 * balanced_accuracy_score(yt[s], po[s]) - chance
                if ob > 1:
                    d.append((100 * balanced_accuracy_score(yt[s], res[vn][0][s]) - 100 * balanced_accuracy_score(yt[s], res["none"][0][s])) / ob)
            lo, hi = np.percentile(d, [2.5, 97.5]); dtr = res[vn][2] - res["none"][2]
            ok = dtr >= 0.10 and lo > 0 and (res["none"][1] - res[vn][1]) <= 2
            sup |= ok
            say(f"- {nm} {vn}: dTR = {dtr:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {'supports' if ok else 'no'}")
    return sup


M, Sm = load("mosi_p0.npz"), load("sims_p0.npz")
cls = sorted(set(np.load(os.path.join(R, "cremad_p0b.npz"), allow_pickle=True)["label"]) & set(np.load(os.path.join(R, "meld_p0b.npz"), allow_pickle=True)["label"]))
Cr, Me = load("cremad_p0b.npz", cls), load("meld_p0b.npz", cls)
v = [direction(M, Sm, "MOSI", "SIMS", 50.0), direction(Sm, M, "SIMS", "MOSI", 50.0), direction(Cr, Me, "CREMA-D", "MELD", 100.0 / len(cls))]
say(f"\nDirections supported: {sum(v)}/3 -> corpus erasure {'HELPS' if sum(v) >= 2 else 'DOES NOT HELP'}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
