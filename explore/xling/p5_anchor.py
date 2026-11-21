"""P5: text-anchored target-specific nonverbal heads for cross-lingual MSA (no target labels).
Motivation (P0): text transfers across languages (TR 0.87-0.95 with target normalisation), audio/face lose their
discriminative direction (cross AUC 52-58) -> nonverbal heads must be learned ON THE TARGET; the transferable modality
supplies the supervision. Mechanism lineage: co-training (Blum & Mitchell 1998), cross-modal self-training in video DA.

Criterion FIXED BEFORE RESULTS (2026-11-21):
  Testable direction now: MOSI->SIMS (target nonverbal adds +5.8 bacc over text in-domain on SIMS; on MOSI it adds ~0,
  so SIMS->MOSI is reported descriptively, expected no gain).
  T  = source-trained text LR head, target-normalised features (scaler fit on target non-test pool).
  M  = T + target nonverbal head (concat wavlm+egemaps+face_emb+face_logits, scaler on target pool) trained on the target
       pool with T's pseudo-labels, keeping per-class samples with confidence >= per-class median; C by 5-fold CV against
       pseudo-labels; fusion = sum of log-probs (equal weight).
  B0 = source-trained late fusion (text + nonverbal, target-normalised), global weights chosen on source valid.
  B1 = same as M but pseudo-labels from B0 (fused anchor, CASP-like) instead of T.
  PREMISE HOLDS iff in MOSI->SIMS: M - T >= +2 with group-bootstrap 95% CI excluding 0, AND M > B0, AND M >= B1.
  Upper bound U = T + nonverbal head trained on TRUE target labels (target train split), same fusion.
Usage: python -I p5_anchor.py <dir> [out.md]
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
rng = np.random.default_rng(0)
NV = ["a_wavlm", "a_egemaps", "v_emb", "v_logits"]


def load(name):
    d = np.load(os.path.join(R, name), allow_pickle=True); D = {k: d[k] for k in d.files}
    keep = D["ok"].astype(bool) & (D["label"].astype(float) != 0); D = {k: v[keep] for k, v in D.items()}
    D["y"] = (D["label"].astype(float) > 0).astype(int); D["grp"] = D["video_id"]; D["nv"] = np.concatenate([D[k].astype(float) for k in NV], 1)
    return D


def lp(m, X):
    return np.log(np.clip(m.predict_proba(X), 1e-6, 1))


def fit_c(X, y, Xva=None, yva=None):
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        if Xva is not None:
            s = balanced_accuracy_score(yva, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y).predict(Xva))
        else:
            s = np.mean([balanced_accuracy_score(y[te], LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr]).predict(X[te]))
                         for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)])
        if best is None or s > best[0]:
            best = (s, C)
    return LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(X, y)


def select(P):
    y = P.argmax(1); c = P.max(1); keep = np.zeros(len(y), bool)
    for k in np.unique(y):
        i = y == k; keep[i] = c[i] >= np.median(c[i])
    return keep, y


def direction(S, T, nS, nT):
    s_tr, s_va = S["mode"] == "train", S["mode"] == "valid"; pool, te, ttr = T["mode"] != "test", T["mode"] == "test", T["mode"] == "train"
    yte = T["y"][te]
    st, tt = StandardScaler().fit(S["text"][s_tr]), StandardScaler().fit(T["text"][pool])
    mt = fit_c(st.transform(S["text"][s_tr]), S["y"][s_tr], st.transform(S["text"][s_va]), S["y"][s_va])
    Tp, Tte = lp(mt, tt.transform(T["text"][pool])), lp(mt, tt.transform(T["text"][te]))
    sn, tn = StandardScaler().fit(S["nv"][s_tr]), StandardScaler().fit(T["nv"][pool])
    mn = fit_c(sn.transform(S["nv"][s_tr]), S["y"][s_tr], sn.transform(S["nv"][s_va]), S["y"][s_va])
    va_t, va_n = lp(mt, st.transform(S["text"][s_va])), lp(mn, sn.transform(S["nv"][s_va]))
    ws = np.linspace(0, 1, 11); w = ws[np.argmax([balanced_accuracy_score(S["y"][s_va], (a * va_t + (1 - a) * va_n).argmax(1)) for a in ws])]
    B0p = w * Tp + (1 - w) * lp(mn, tn.transform(T["nv"][pool])); B0te = w * Tte + (1 - w) * lp(mn, tn.transform(T["nv"][te]))
    Xp, Xte = tn.transform(T["nv"][pool]), tn.transform(T["nv"][te])
    k1, y1 = select(np.exp(Tp)); mM = fit_c(Xp[k1], y1[k1]); M = Tte + lp(mM, Xte)
    k2, y2 = select(np.exp(B0p - np.logaddexp.reduce(B0p, 1, keepdims=True))); mB1 = fit_c(Xp[k2], y2[k2]); B1 = Tte + lp(mB1, Xte)
    ttr_i = np.where(ttr)[0]; Xtr = tn.transform(T["nv"][ttr_i]); mU = fit_c(Xtr, T["y"][ttr_i]); U = Tte + lp(mU, Xte)
    pl_acc = 100 * balanced_accuracy_score(T["y"][pool][k1], y1[k1])
    rows = {"T text-only (target-normalised)": Tte.argmax(1), "B0 source-trained fusion": B0te.argmax(1), "B1 fused-anchor pseudo-labels": B1.argmax(1),
            "M text-anchored target nonverbal head": M.argmax(1), "target nonverbal head alone (M's student)": lp(mM, Xte).argmax(1),
            "U upper bound (true target labels for nonverbal)": U.argmax(1)}
    say(f"\n## {nS} -> {nT} (source-valid fusion weight on text = {w:.1f}; kept pseudo-labels {k1.sum()}/{pool.sum()}, their true bacc {pl_acc:.1f})")
    say("| method | target test bacc |"); say("|---|---|")
    for k, p in rows.items():
        say(f"| {k} | {100 * balanced_accuracy_score(yte, p):.1f} |")
    g = T["grp"][te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}; d = []
    for _ in range(1000):
        s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
        d.append(100 * (balanced_accuracy_score(yte[s], rows["M text-anchored target nonverbal head"][s]) - balanced_accuracy_score(yte[s], rows["T text-only (target-normalised)"][s])))
    lo, hi = np.percentile(d, [2.5, 97.5]); b = {k: 100 * balanced_accuracy_score(yte, p) for k, p in rows.items()}
    dM = b["M text-anchored target nonverbal head"] - b["T text-only (target-normalised)"]
    ok = dM >= 2 and lo > 0 and b["M text-anchored target nonverbal head"] > b["B0 source-trained fusion"] and b["M text-anchored target nonverbal head"] >= b["B1 fused-anchor pseudo-labels"]
    say(f"- M - T = {dM:+.1f} [{lo:+.1f}, {hi:+.1f}]; M - B0 = {b['M text-anchored target nonverbal head'] - b['B0 source-trained fusion']:+.1f}; "
        f"M - B1 = {b['M text-anchored target nonverbal head'] - b['B1 fused-anchor pseudo-labels']:+.1f}; U - T = {b['U upper bound (true target labels for nonverbal)'] - b['T text-only (target-normalised)']:+.1f}")
    return ok


Mo, Si = load("mosi_p0.npz"), load("sims_p0.npz")
ok = direction(Mo, Si, "MOSI", "SIMS")
direction(Si, Mo, "SIMS", "MOSI")
say(f"\nPremise (MOSI->SIMS, pre-registered): {'HOLDS' if ok else 'DOES NOT HOLD'}; SIMS->MOSI descriptive only")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
