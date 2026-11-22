"""P8: "neutral is absence, not a class" (round 4d, ideas/round4_original_ideas_2026-11-21.md). Frozen features, LR heads.

Criterion FIXED BEFORE RESULTS (2026-11-22):
  Datasets: CREMA-D (6 classes incl. NEU; intensity LO/MD/HI from file names; actor-disjoint split as in P0b),
            MELD (6 classes shared with CREMA-D incl. NEU; dev = train/valid as in P0b, test = test),
            CH-SIMS (3 classes: negative / neutral (label 0) / positive; intensity = |label|; official split).
  Features: CREMA-D audio(WavLM)+face(ViT-FER emb); MELD and CH-SIMS text(e5)+audio(WavLM)+face(ViT-FER emb), standardised concat.
  B1 flat: multinomial LR over all classes (class_weight balanced, C on valid).
  B2 flat + neutral-logit bias b tuned on valid (balanced accuracy).
  M  factorised: presence head p(emotional) = LR on neutral vs emotional, trained with sample weights = intensity level where
     available (CREMA-D: LO 1, MD 2, HI 3, XX 2; CH-SIMS: |label| scaled to 1..3; MELD: 1); category head = multinomial LR on
     emotional samples only; predict neutral iff p(emotional) < tau (tau tuned on valid by balanced accuracy), else category argmax.
  HOLDS iff M - max(B1, B2) >= +2 balanced-accuracy points with group-bootstrap 95% CI excluding 0 in >= 2 of 3 datasets.
  Reported before the decision: MDE = 2.8 x bootstrap SE of the difference. Descriptive: P(pred = neutral | emotional) by intensity.
Usage: python -I p8_neutral.py <dir with *_p0*.npz> <cremad_manifest.csv> [out.md]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

R, MAN = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
CS = [0.001, 0.01, 0.1, 1.0]


def best_lr(X, y, Xv, yv, w=None):
    sc = StandardScaler().fit(X); best = None
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y, sample_weight=w)
        s = balanced_accuracy_score(yv, m.predict(sc.transform(Xv)))
        if best is None or s > best[0]:
            best = (s, m)
    return sc, best[1]


def run(name, X, y, inten, mode, grp, neu):
    tr, va, te = mode == "train", mode == "valid", mode == "test"
    sc, m = best_lr(X[tr], y[tr], X[va], y[va]); P_va, P_te = m.predict_log_proba(sc.transform(X[va])), m.predict_log_proba(sc.transform(X[te]))
    ni = list(m.classes_).index(neu); p1 = m.classes_[P_te.argmax(1)]
    def shifted(P, b):
        Q = P.copy(); Q[:, ni] += b; return m.classes_[Q.argmax(1)]
    bb = max(np.linspace(-3, 3, 61), key=lambda b: balanced_accuracy_score(y[va], shifted(P_va, b))); p2 = shifted(P_te, bb)
    emo = y != neu; w = np.where(emo, inten, 1.0)
    scp, mp = best_lr(X[tr], emo[tr].astype(int), X[va], emo[va].astype(int), w[tr])
    scc, mc = best_lr(X[tr & emo], y[tr & emo], X[va & emo], y[va & emo])
    def fact(idx, tau):
        pe = mp.predict_proba(scp.transform(X[idx]))[:, 1]; cat = mc.predict(scc.transform(X[idx]))
        return np.where(pe < tau, neu, cat)
    tau = max(np.linspace(0.05, 0.95, 91), key=lambda t: balanced_accuracy_score(y[va], fact(va, t))); pm = fact(te, tau)
    yt, g = y[te], grp[te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}; rng = np.random.default_rng(0)
    b1, b2, bm = (100 * balanced_accuracy_score(yt, p) for p in (p1, p2, pm))
    base = p1 if b1 >= b2 else p2; d = []
    for _ in range(1000):
        s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
        d.append(100 * (balanced_accuracy_score(yt[s], pm[s]) - balanced_accuracy_score(yt[s], base[s])))
    lo, hi = np.percentile(d, [2.5, 97.5]); mde = 2.8 * np.std(d); diff = bm - max(b1, b2)
    ok = diff >= 2 and lo > 0
    say(f"\n## {name} (test n={te.sum()}, classes {list(m.classes_)})")
    say(f"- B1 flat {b1:.1f} | B2 flat + neutral bias (b={bb:+.1f}) {b2:.1f} | M factorised (tau={tau:.2f}) {bm:.1f}")
    say(f"- M - max(B1, B2) = {diff:+.1f} [{lo:+.1f}, {hi:+.1f}]; MDE {mde:.1f} -> {'supports' if ok else 'no'}")
    for nm, p in (("B1", p1), ("M", pm)):
        rows = []
        for lv in sorted(set(inten[te & emo])):
            msk = (inten[te] == lv) & (yt != neu)
            if msk.sum() >= 15:
                rows.append(f"intensity {lv:g}: {100 * (p[msk] == neu).mean():.0f}% (n={msk.sum()})")
        say(f"  - {nm}: P(pred=neutral | emotional) " + "; ".join(rows))
    return ok


def npz(f):
    d = np.load(os.path.join(R, f), allow_pickle=True); return {k: d[k] for k in d.files}


# CREMA-D
C = npz("cremad_p0b.npz"); man = pd.read_csv(MAN).set_index("id"); ok = C["ok"].astype(bool)
C = {k: v[ok] for k, v in C.items()}
lev = {"LO": 1.0, "MD": 2.0, "HI": 3.0, "XX": 2.0}
inten_c = np.array([lev.get(str(man.loc[i, "intensity"]) if i in man.index else "XX", 2.0) for i in C["id"]])
Xc = np.concatenate([C["a_wavlm"], C["v_emb"]], 1).astype(float)
v1 = run("CREMA-D", Xc, C["label"].astype(str), inten_c, C["mode"], C["group"], "NEU")
# MELD (6 classes shared with CREMA-D)
M = npz("meld_p0b.npz"); cls = set(C["label"]); ok = M["ok"].astype(bool) & np.isin(M["label"], list(cls)); M = {k: v[ok] for k, v in M.items()}
Xm = np.concatenate([M["text"], M["a_wavlm"], M["v_emb"]], 1).astype(float)
v2 = run("MELD", Xm, M["label"].astype(str), np.ones(len(Xm)), M["mode"], M["group"], "NEU")
# CH-SIMS (3-class)
S = npz("sims_p0.npz"); ok = S["ok"].astype(bool); S = {k: v[ok] for k, v in S.items()}
lab = S["label"].astype(float); ys = np.where(lab > 0, "POS", np.where(lab < 0, "NEG", "NEU"))
inten_s = 1.0 + 2.0 * np.abs(lab)  # |label| in [0, 1] -> 1..3
Xs = np.concatenate([S["text"], S["a_wavlm"], S["v_emb"]], 1).astype(float)
v3 = run("CH-SIMS (3-class)", Xs, ys, inten_s, S["mode"], S["video_id"], "NEU")
say(f"\nDatasets supporting: {v1 + v2 + v3}/3 -> {'HOLDS' if v1 + v2 + v3 >= 2 else 'DOES NOT HOLD'}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
