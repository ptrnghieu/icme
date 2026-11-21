"""P3: label-free model selection for cross-corpus emotion recognition (mechanism transfer: Agreement-on-the-Line,
Baek et al. NeurIPS 2022, ALine-D; ATC, Garg et al. ICLR 2022). Frozen features, LR heads.

Criterion FIXED BEFORE RESULTS (2026-11-21):
  Zoo per direction = every non-empty subset of feature blocks (early concat) x C in {0.001, 0.01, 0.1, 1}.
  Blocks: MOSI<->SIMS {text, wavlm, egemaps, face_emb, face_logits}; CREMA-D->MELD {wavlm, egemaps, face_emb, face_logits}.
  S0 = select by SOURCE valid bacc (standard practice). PRIMARY method = ALine-D estimate of target accuracy.
  Unlabeled target pool for S1-S3 = target non-test split; evaluation on target test (bacc).
  (P) Premise holds iff headroom (best zoo model on target test - S0) >= 3 points in >= 2 of 3 directions.
  (M) Method holds iff ALine-D - S0 >= +2 points with group-bootstrap 95% CI excluding 0 in >= 2 of 3 directions.
  S2 (agreement with zoo majority vote) and S3 (ATC) are reported descriptively only.
Usage: python -I p3_selection.py <dir> [out.md]
"""
import itertools, os, sys
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
rng = np.random.default_rng(0)
BLK = {"text": "text", "wavlm": "a_wavlm", "egemaps": "a_egemaps", "face_emb": "v_emb", "face_logits": "v_logits"}


def load(name, classes=None):
    d = np.load(os.path.join(R, name), allow_pickle=True); D = {k: d[k] for k in d.files}
    if "video_id" in D:
        keep = D["ok"].astype(bool) & (D["label"].astype(float) != 0); D = {k: v[keep] for k, v in D.items()}
        D["y"] = (D["label"].astype(float) > 0).astype(int); D["grp"] = D["video_id"]
    else:
        keep = D["ok"].astype(bool) & np.isin(D["label"], classes); D = {k: v[keep] for k, v in D.items()}
        D["y"] = np.array([classes.index(c) for c in D["label"]]); D["grp"] = D["group"]
    return D


def pinv(p):
    return norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))


def direction(S, T, nS, nT, blocks):
    s_tr, s_va = S["mode"] == "train", S["mode"] == "valid"; t_pool, t_te = T["mode"] != "test", T["mode"] == "test"
    models = []
    for r in range(1, len(blocks) + 1):
        for sub in itertools.combinations(blocks, r):
            Xs = np.concatenate([S[BLK[b]].astype(float) for b in sub], 1); Xt = np.concatenate([T[BLK[b]].astype(float) for b in sub], 1)
            sc = StandardScaler().fit(Xs[s_tr])
            for C in [0.001, 0.01, 0.1, 1.0]:
                m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xs[s_tr]), S["y"][s_tr])
                pv = m.predict_proba(sc.transform(Xs[s_va])); pp = m.predict_proba(sc.transform(Xt[t_pool])); pt = m.predict(sc.transform(Xt[t_te]))
                models.append(dict(name=f"{'+'.join(sub)} C={C}", va_pred=pv.argmax(1), va_conf=pv.max(1), pool_pred=pp.argmax(1), pool_conf=pp.max(1), te_pred=pt,
                                   va_bacc=balanced_accuracy_score(S["y"][s_va], pv.argmax(1)), va_acc=(pv.argmax(1) == S["y"][s_va]).mean(),
                                   te_bacc=100 * balanced_accuracy_score(T["y"][t_te], pt)))
    n = len(models); va = np.array([m["va_bacc"] for m in models]); te = np.array([m["te_bacc"] for m in models])
    # ALine-D (Baek et al. 2022, Alg. 2): fit (a, b) on probit agreement over pairs, then solve least squares for per-model OOD acc.
    idx = list(itertools.combinations(range(n), 2))
    agr_id = np.array([(models[i]["va_pred"] == models[j]["va_pred"]).mean() for i, j in idx])
    agr_ood = np.array([(models[i]["pool_pred"] == models[j]["pool_pred"]).mean() for i, j in idx])
    a, b = np.polyfit(pinv(agr_id), pinv(agr_ood), 1)
    acc_id = np.array([m["va_acc"] for m in models]); A = np.zeros((len(idx), n)); rhs = np.zeros(len(idx))
    for k, (i, j) in enumerate(idx):
        A[k, i] = A[k, j] = 0.5
        rhs[k] = pinv(agr_ood[k]) + a * ((pinv(acc_id[i]) + pinv(acc_id[j])) / 2 - pinv(agr_id[k]))
    est = np.linalg.lstsq(A, rhs, rcond=None)[0]
    maj = np.array([np.bincount(c, minlength=int(T["y"].max()) + 1).argmax() for c in np.stack([m["pool_pred"] for m in models], 1)])
    s2 = np.array([(m["pool_pred"] == maj).mean() for m in models])
    s3 = []
    for m in models:  # ATC: threshold t so that frac(conf_va > t) = acc_va, estimate = frac(conf_pool > t)
        t = np.quantile(m["va_conf"], 1 - m["va_acc"]); s3.append((m["pool_conf"] > t).mean())
    s3 = np.array(s3)
    sel = {"S0 source-valid bacc": int(np.argmax(va)), "S1 ALine-D (primary)": int(np.argmax(est)), "S2 agreement w/ zoo majority": int(np.argmax(s2)),
           "S3 ATC": int(np.argmax(s3)), "oracle (best on target test)": int(np.argmax(te))}
    say(f"\n## {nS} -> {nT} (zoo {n} models; probit agreement fit a={a:.2f}, b={b:.2f})")
    rho_id = spearmanr(va, te).correlation; rho_al = spearmanr(est, te).correlation
    say(f"- Spearman(source-valid bacc, target test bacc) = {rho_id:+.2f}; Spearman(ALine-D est, target test bacc) = {rho_al:+.2f}; "
        f"Spearman(S2, test) = {spearmanr(s2, te).correlation:+.2f}; Spearman(S3, test) = {spearmanr(s3, te).correlation:+.2f}")
    say("| rule | selected model | target test bacc |"); say("|---|---|---|")
    for k, i in sel.items():
        say(f"| {k} | {models[i]['name']} | {te[i]:.1f} |")
    head = te.max() - te[sel["S0 source-valid bacc"]]
    yt, g = T["y"][t_te], T["grp"][t_te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}
    p0, p1 = models[sel["S0 source-valid bacc"]]["te_pred"], models[sel["S1 ALine-D (primary)"]]["te_pred"]
    d = []
    for _ in range(1000):
        s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
        d.append(100 * (balanced_accuracy_score(yt[s], p1[s]) - balanced_accuracy_score(yt[s], p0[s])))
    lo, hi = np.percentile(d, [2.5, 97.5]); diff = te[sel["S1 ALine-D (primary)"]] - te[sel["S0 source-valid bacc"]]
    okm = diff >= 2 and lo > 0
    say(f"- headroom = {head:+.1f} -> {'premise yes' if head >= 3 else 'premise no'}; ALine-D - S0 = {diff:+.1f} [{lo:+.1f}, {hi:+.1f}] -> {'supports' if okm else 'no'}")
    return head >= 3, okm


M, Sm = load("mosi_p0.npz"), load("sims_p0.npz")
cls = sorted(set(np.load(os.path.join(R, "cremad_p0b.npz"), allow_pickle=True)["label"]) & set(np.load(os.path.join(R, "meld_p0b.npz"), allow_pickle=True)["label"]))
Cr, Me = load("cremad_p0b.npz", cls), load("meld_p0b.npz", cls)
b5 = ["text", "wavlm", "egemaps", "face_emb", "face_logits"]
r = [direction(M, Sm, "MOSI", "SIMS", b5), direction(Sm, M, "SIMS", "MOSI", b5), direction(Cr, Me, "CREMA-D", "MELD", b5[1:])]
say(f"\nPremise (headroom >= 3): {sum(x[0] for x in r)}/3 -> {'HOLDS' if sum(x[0] for x in r) >= 2 else 'DOES NOT HOLD'}")
say(f"Method (ALine-D - S0 >= 2, CI>0): {sum(x[1] for x in r)}/3 -> {'HOLDS' if sum(x[1] for x in r) >= 2 else 'DOES NOT HOLD'}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
