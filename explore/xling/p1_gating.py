"""P1: premise test for R2-2 (per-sample modality weighting learned from HELD-OUT utility), in-domain, frozen features.

Criterion FIXED BEFORE RESULTS (2026-11-21):
  R2-2 premise HOLDS iff in >= 2 of the 4 datasets {MOSI, CH-SIMS, CREMA-D, MELD}:
    (i)  M - B2 > 0 on test balanced accuracy with group-bootstrap 95% CI excluding 0, and
    (ii) M >= B3 and M >= B4 (point estimates) in the same dataset.
  where B2 = global late-fusion weights chosen on valid, B3 = entropy-based confidence weighting (temperature on valid),
  B4 = same gate as M but trained on IN-SAMPLE (training-fit) utilities, M = gate trained on OUT-OF-FOLD utilities
  (5-fold cross-fitting grouped by speaker/video/dialogue). Also reported: B1 early concat, per-sample oracle (upper bound).

Usage: python -I p1_gating.py <dir with mosi_p0.npz sims_p0.npz cremad_p0b.npz meld_p0b.npz> [out.md]
"""
import itertools, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
CS = [0.001, 0.01, 0.1, 1.0]


def load(name):
    d = np.load(os.path.join(R, name), allow_pickle=True); D = {k: d[k] for k in d.files}
    if "video_id" in D:  # MOSI / SIMS: binary polarity, drop zeros
        keep = D["ok"].astype(bool) & (D["label"].astype(float) != 0)
        D = {k: v[keep] for k, v in D.items()}; D["y"] = (D["label"].astype(float) > 0).astype(int); D["grp"] = D["video_id"]
        mods = {"text": "text", "audio": "a_wavlm", "face": "v_emb"}
    else:
        keep = D["ok"].astype(bool); D = {k: v[keep] for k, v in D.items()}
        cls = sorted(set(D["label"])); D["y"] = np.array([cls.index(c) for c in D["label"]]); D["grp"] = D["group"]
        mods = {"audio": "a_wavlm", "face": "v_emb"}
        if np.abs(D["text"]).sum() > 0 and len(set(map(str, D["text"][:50].round(3).tolist()))) > 1:
            mods = {"text": "text", **mods}
    return D, mods


def lr(X, y, C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)


def unimodal(D, key, tr, va):
    sc = StandardScaler().fit(D[key][tr]); best = None
    for C in CS:
        m = lr(sc.transform(D[key][tr]), D["y"][tr], C)
        s = balanced_accuracy_score(D["y"][va], m.predict(sc.transform(D[key][va])))
        if best is None or s > best[0]:
            best = (s, C)
    return best[1]


def logp(D, key, fit_idx, pred_idx, C):
    sc = StandardScaler().fit(D[key][fit_idx]); m = lr(sc.transform(D[key][fit_idx]), D["y"][fit_idx], C)
    return np.log(np.clip(m.predict_proba(sc.transform(D[key][pred_idx])), 1e-6, 1))


def grid(M, step=0.1):
    n = int(round(1 / step)); return np.array([np.array(c) / n for c in itertools.product(range(n + 1), repeat=M) if sum(c) == n])


def fuse(LP, W):  # LP: [M, N, K]; W: [N, M] or [M]
    W = np.broadcast_to(W, (LP.shape[1], LP.shape[0])) if W.ndim == 1 else W
    return np.einsum("nm,mnk->nk", W, LP)


def gate_feats(LP):
    P = np.exp(LP); f = []
    for m in range(LP.shape[0]):
        p = P[m]; s = np.sort(p, 1)
        f += [s[:, -1], s[:, -1] - s[:, -2], -(p * np.log(p + 1e-12)).sum(1)]
    pred = P.argmax(2)
    for a, b in itertools.combinations(range(LP.shape[0]), 2):
        f.append((pred[a] == pred[b]).astype(float))
    return np.stack(f, 1)


def utilities(LP, y, G):  # per-sample index of best grid point (lowest fused NLL); ties -> lowest index
    nll = np.stack([-(fuse(LP, g)[np.arange(len(y)), y] - np.log(np.exp(fuse(LP, g)).sum(1))) for g in G], 1)
    return nll.argmin(1)


def train_gate(Z, t, Zva, LPva, yva, G):
    sc = StandardScaler().fit(Z); best = None
    for C in [0.01, 0.1, 1.0]:
        if len(np.unique(t)) < 2:
            break
        g = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Z), t)
        W = g.predict_proba(sc.transform(Zva)) @ G[g.classes_]
        s = balanced_accuracy_score(yva, fuse(LPva, W).argmax(1))
        if best is None or s > best[0]:
            best = (s, g)
    if best is None:
        return lambda Zx: np.broadcast_to(G[t[0]], (len(Zx), G.shape[1]))
    g = best[1]; return lambda Zx: g.predict_proba(sc.transform(Zx)) @ G[g.classes_]


def run(name, seed):
    D, mods = load(name); keys = list(mods.values()); M = len(keys)
    tr, va, te = (D["mode"] == "train"), (D["mode"] == "valid"), (D["mode"] == "test")
    itr, iva, ite = np.where(tr)[0], np.where(va)[0], np.where(te)[0]
    Cs = {k: unimodal(D, k, itr, iva) for k in keys}
    LPva = np.stack([logp(D, k, itr, iva, Cs[k]) for k in keys]); LPte = np.stack([logp(D, k, itr, ite, Cs[k]) for k in keys])
    LPin = np.stack([logp(D, k, itr, itr, Cs[k]) for k in keys])           # in-sample (training-fit) predictions
    LPoof = np.zeros_like(LPin)                                              # out-of-fold predictions
    rng = np.random.RandomState(seed); gtr = D["grp"][itr]; ug = np.unique(gtr); perm = dict(zip(ug, rng.permutation(len(ug))))
    for f_tr, f_ho in GroupKFold(5).split(itr, groups=[perm[g] for g in gtr]):
        for mi, k in enumerate(keys):
            LPoof[mi, f_ho] = logp(D, k, itr[f_tr], itr[f_ho], Cs[k])
    y_tr, y_va, y_te = D["y"][itr], D["y"][iva], D["y"][ite]; G = grid(M)
    res = {}
    # B1 early concat
    X = np.concatenate([D[k] for k in keys], 1); sc = StandardScaler().fit(X[itr]); best = None
    for C in CS:
        m = lr(sc.transform(X[itr]), y_tr, C); s = balanced_accuracy_score(y_va, m.predict(sc.transform(X[iva])))
        if best is None or s > best[0]:
            best = (s, m)
    res["B1 early concat"] = best[1].predict(sc.transform(X[ite]))
    # B2 global late fusion
    wg = G[np.argmax([balanced_accuracy_score(y_va, fuse(LPva, g).argmax(1)) for g in G])]
    res["B2 global weights (valid)"] = fuse(LPte, wg).argmax(1)
    # B3 entropy confidence weighting
    def conf_w(LP, T):
        H = np.stack([-(np.exp(LP[m]) * LP[m]).sum(1) for m in range(M)], 1); w = np.exp(-H / T); return w / w.sum(1, keepdims=True)
    Tb = max([0.05, 0.1, 0.2, 0.5, 1.0, 2.0], key=lambda T: balanced_accuracy_score(y_va, fuse(LPva, conf_w(LPva, T)).argmax(1)))
    res["B3 entropy confidence"] = fuse(LPte, conf_w(LPte, Tb)).argmax(1)
    # B4 / M gates
    Zva, Zte = gate_feats(LPva), gate_feats(LPte)
    g4 = train_gate(gate_feats(LPin), utilities(LPin, y_tr, G), Zva, LPva, y_va, G)
    res["B4 gate on training-fit utility"] = fuse(LPte, g4(Zte)).argmax(1)
    gm = train_gate(gate_feats(LPoof), utilities(LPoof, y_tr, G), Zva, LPva, y_va, G)
    res["M gate on out-of-fold utility"] = fuse(LPte, gm(Zte)).argmax(1)
    # oracle: per-sample best grid point on TEST labels (upper bound)
    corr = np.stack([fuse(LPte, g).argmax(1) == y_te for g in G], 1)
    res["oracle per-sample (upper bound)"] = np.where(corr.any(1), y_te, fuse(LPte, wg).argmax(1))
    return res, y_te, D["grp"][ite], list(mods)


def main():
    rng = np.random.default_rng(0)
    votes = 0
    for name, label in [("mosi_p0.npz", "MOSI"), ("sims_p0.npz", "CH-SIMS"), ("cremad_p0b.npz", "CREMA-D"), ("meld_p0b.npz", "MELD")]:
        runs = [run(name, s) for s in range(3)]
        y, g, mods = runs[0][1], runs[0][2], runs[0][3]
        say(f"\n## {label} (modalities {mods}; test n={len(y)}; 3 cross-fitting seeds)")
        say("| method | bacc (mean over seeds) | weighted F1 |"); say("|---|---|---|")
        means = {}
        for k in runs[0][0]:
            b = [100 * balanced_accuracy_score(y, r[0][k]) for r in runs]; w = [100 * f1_score(y, r[0][k], average="weighted") for r in runs]
            means[k] = np.mean(b); say(f"| {k} | {np.mean(b):.2f} ± {np.std(b):.2f} | {np.mean(w):.2f} |")
        ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}
        d = []
        for _ in range(1000):
            s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
            d.append(np.mean([100 * (balanced_accuracy_score(y[s], r[0]["M gate on out-of-fold utility"][s]) - balanced_accuracy_score(y[s], r[0]["B2 global weights (valid)"][s])) for r in runs]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff = means["M gate on out-of-fold utility"] - means["B2 global weights (valid)"]
        ok = lo > 0 and means["M gate on out-of-fold utility"] >= means["B3 entropy confidence"] and means["M gate on out-of-fold utility"] >= means["B4 gate on training-fit utility"]
        votes += ok
        say(f"- M - B2 = {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]; M - B3 = {means['M gate on out-of-fold utility'] - means['B3 entropy confidence']:+.2f}; "
            f"M - B4 = {means['M gate on out-of-fold utility'] - means['B4 gate on training-fit utility']:+.2f}; oracle headroom over B2 = "
            f"{means['oracle per-sample (upper bound)'] - means['B2 global weights (valid)']:+.2f} -> {'supports' if ok else 'does not support'}")
    say(f"\nDatasets supporting R2-2: {votes}/4 -> premise {'HOLDS' if votes >= 2 else 'DOES NOT HOLD'}")
    if OUT:
        open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
