"""P4: modalities as conditionally independent 'annotators' on an UNLABELED target corpus (mechanism transfer:
Dawid & Skene 1979 crowd aggregation / unsupervised ensemble estimation, e.g. Platanios et al. 2014; Parisi et al. 2014).
Source-trained unimodal heads predict the target pool; unsupervised Dawid-Skene EM estimates the target class prior and
each head's confusion matrix; the DS posterior is the target-domain fusion rule. Frozen features, LR heads.

Criterion FIXED BEFORE RESULTS (2026-11-21):
  Heads = one LR head per feature block (MOSI<->SIMS: text, wavlm, egemaps, face_emb, face_logits; CREMA-D->MELD: wavlm,
  egemaps, face_emb, face_logits; >=3 heads needed for identifiability, so audio and face blocks enter separately).
  S0 = global late fusion of head log-probs, weights on a 0.1 simplex grid chosen on SOURCE valid bacc.
  M  = Dawid-Skene posterior; EM fit on target non-test pool (hard predictions, majority-vote init, 50 iters, add-0.5
       smoothing); applied to target test.
  HOLDS iff M - S0 >= +2 bacc points with group-bootstrap 95% CI excluding 0 in >= 2 of 3 directions.
  Reported descriptively: majority vote, one-coin weights, best single head by source valid, oracle global weights on test.
Usage: python -I p4_ds.py <dir> [out.md]
"""
import itertools, os, sys
import numpy as np
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


def head(S, T, b):
    s_tr, s_va = S["mode"] == "train", S["mode"] == "valid"
    Xs, Xt = S[BLK[b]].astype(float), T[BLK[b]].astype(float); sc = StandardScaler().fit(Xs[s_tr]); best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xs[s_tr]), S["y"][s_tr])
        s = balanced_accuracy_score(S["y"][s_va], m.predict(sc.transform(Xs[s_va])))
        if best is None or s > best[0]:
            best = (s, m)
    m = best[1]; lp = lambda X: np.log(np.clip(m.predict_proba(sc.transform(X)), 1e-6, 1))  # noqa: E731
    return best[0], lp(Xs[s_va]), lp(Xt)


def ds_fit(P, K, iters=50):
    """P: [N, H] hard predictions. returns prior [K], theta [H, K, K] (true -> predicted)."""
    N, H = P.shape
    q = np.stack([np.bincount(r, minlength=K) for r in P]).astype(float) + 1e-9; q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        pi = q.mean(0); th = np.zeros((H, K, K))
        for h in range(H):
            for k in range(K):
                th[h, :, k] = q[P[:, h] == k].sum(0)
        th = (th + 0.5) / (th + 0.5).sum(2, keepdims=True)
        lq = np.log(pi + 1e-12)[None] + sum(np.log(th[h][:, P[:, h]].T) for h in range(H)); q = np.exp(lq - lq.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    return pi, th


def ds_post(P, pi, th):
    lq = np.log(pi + 1e-12)[None] + sum(np.log(th[h][:, P[:, h]].T) for h in range(P.shape[1])); return lq.argmax(1)


def direction(S, T, nS, nT, blocks):
    K = int(max(S["y"].max(), T["y"].max())) + 1
    t_pool, t_te = T["mode"] != "test", T["mode"] == "test"; s_va = S["mode"] == "valid"
    hs = [head(S, T, b) for b in blocks]
    LPva = np.stack([h[1] for h in hs]); LPt = np.stack([h[2] for h in hs]); yva, yte = S["y"][s_va], T["y"][t_te]
    n = 10; G = np.array([np.array(c) / n for c in itertools.product(range(n + 1), repeat=len(blocks)) if sum(c) == n])
    fuse = lambda LP, w: np.einsum("m,mnk->nk", w, LP)  # noqa: E731
    w0 = G[np.argmax([balanced_accuracy_score(yva, fuse(LPva, g).argmax(1)) for g in G])]
    pred = {}
    pred["S0 global weights (source valid)"] = fuse(LPt[:, t_te], w0).argmax(1)
    Ppool, Pte = LPt[:, t_pool].argmax(2).T, LPt[:, t_te].argmax(2).T
    pi, th = ds_fit(Ppool, K)
    pred["M Dawid-Skene posterior (primary)"] = ds_post(Pte, pi, th)
    pred["majority vote"] = np.array([np.bincount(r, minlength=K).argmax() for r in Pte])
    acc = np.array([np.mean([th[h][k, k] for k in range(K)]) for h in range(len(blocks))])  # DS-estimated per-head balanced acc
    w1 = np.log(np.clip(acc, 1e-3, 1 - 1e-3) * (K - 1) / np.clip(1 - acc, 1e-3, 1)); w1 = np.clip(w1, 0, None)
    pred["one-coin weights from DS acc"] = fuse(LPt[:, t_te], w1 / max(w1.sum(), 1e-9)).argmax(1)
    bi = int(np.argmax([h[0] for h in hs])); pred[f"best single head on source valid ({blocks[bi]})"] = LPt[bi, t_te].argmax(1)
    pred["oracle global weights (target test)"] = fuse(LPt[:, t_te], G[np.argmax([balanced_accuracy_score(yte, fuse(LPt[:, t_te], g).argmax(1)) for g in G])]).argmax(1)
    say(f"\n## {nS} -> {nT} (heads {blocks}; S0 weights {np.round(w0, 1).tolist()})")
    true_bacc = [100 * balanced_accuracy_score(yte, LPt[h, t_te].argmax(1)) for h in range(len(blocks))]
    say("- per-head target test bacc (true): " + ", ".join(f"{b} {v:.1f}" for b, v in zip(blocks, true_bacc)))
    say("- per-head DS-estimated balanced acc (label-free): " + ", ".join(f"{b} {100 * v:.1f}" for b, v in zip(blocks, acc)))
    say(f"- DS-estimated target prior: {np.round(pi, 3).tolist()}; true target-test prior: {np.round(np.bincount(yte, minlength=K) / len(yte), 3).tolist()}")
    say("| rule | target test bacc |"); say("|---|---|")
    for k, p in pred.items():
        say(f"| {k} | {100 * balanced_accuracy_score(yte, p):.1f} |")
    g = T["grp"][t_te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}
    pm, p0 = pred["M Dawid-Skene posterior (primary)"], pred["S0 global weights (source valid)"]; d = []
    for _ in range(1000):
        s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))])
        d.append(100 * (balanced_accuracy_score(yte[s], pm[s]) - balanced_accuracy_score(yte[s], p0[s])))
    lo, hi = np.percentile(d, [2.5, 97.5]); diff = 100 * (balanced_accuracy_score(yte, pm) - balanced_accuracy_score(yte, p0))
    ok = diff >= 2 and lo > 0
    say(f"- M - S0 = {diff:+.1f} [{lo:+.1f}, {hi:+.1f}] -> {'supports' if ok else 'no'}")
    return ok


M, Sm = load("mosi_p0.npz"), load("sims_p0.npz")
cls = sorted(set(np.load(os.path.join(R, "cremad_p0b.npz"), allow_pickle=True)["label"]) & set(np.load(os.path.join(R, "meld_p0b.npz"), allow_pickle=True)["label"]))
Cr, Me = load("cremad_p0b.npz", cls), load("meld_p0b.npz", cls)
b5 = ["text", "wavlm", "egemaps", "face_emb", "face_logits"]
v = [direction(M, Sm, "MOSI", "SIMS", b5), direction(Sm, M, "SIMS", "MOSI", b5), direction(Cr, Me, "CREMA-D", "MELD", b5[1:])]
say(f"\nDirections supported: {sum(v)}/3 -> {'HOLDS' if sum(v) >= 2 else 'DOES NOT HOLD'}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
