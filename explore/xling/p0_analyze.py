"""P0 premise analysis (criterion fixed BEFORE results, 2026-11-21):

Binary polarity (label > 0 vs < 0; label == 0 dropped). For each feature set m and direction S->T:
  in-domain  = train S/train, select C on S/valid, test S/test
  cross      = same model, test T/test (source-fitted scaler)
  cross_tn   = same model, target features standardized with TARGET statistics (label-free, AdaBN-like)
  oracle_T   = train T/train, select C on T/valid, test T/test
  transfer ratio TR = (cross_bacc - 50) / (oracle_T_bacc - 50)   (balanced accuracy, %)
Premise "language shift hits text hardest" HOLDS iff TR(text) < TR(audio) and TR(text) < TR(face) in BOTH directions,
with video-cluster bootstrap 95% CIs of TR(text) - TR(other) excluding 0. Otherwise: does not hold.

F4 diagnostic (added BEFORE results): AUC transfer ATR = (AUC_cross - .5)/(AUC_oracle - .5), threshold-free.
If ATR >> TR for nonverbal features, the shift is mainly gain/threshold (expressivity) rather than direction.

Usage: python -I p0_analyze.py mosi_p0.npz sims_p0.npz [out.md]
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
LAST_SCORES = [None]


def load(p):
    d = np.load(p, allow_pickle=True)
    keep = d["ok"] & (d["label"] != 0)
    D = {k: d[k][keep] for k in d.files}
    D["y"] = (D["label"] > 0).astype(int)
    return D


FEATS = {"text(e5)": ["text"], "audio(eGeMAPS)": ["a_egemaps"], "audio(WavLM)": ["a_wavlm"],
         "face(ViT-FER emb)": ["v_emb"], "face(FER logits)": ["v_logits"], "fusion(T+A_wl+V)": ["text", "a_wavlm", "v_emb"],
         "nonverbal(A_wl+V)": ["a_wavlm", "v_emb"]}


def X(D, keys, idx):
    return np.concatenate([D[k][idx] for k in keys], 1)


def fit(D, keys):
    tr, va = D["mode"] == "train", D["mode"] == "valid"
    sc = StandardScaler().fit(X(D, keys, tr))
    best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X(D, keys, tr)), D["y"][tr])
        s = balanced_accuracy_score(D["y"][va], m.predict(sc.transform(X(D, keys, va))))
        if best is None or s > best[0]:
            best = (s, m, C)
    return sc, best[1], best[2]


def preds(sc, m, D, keys, target_norm=False):
    te = D["mode"] == "test"
    Z = X(D, keys, te)
    Z = StandardScaler().fit_transform(Z) if target_norm else sc.transform(Z)
    LAST_SCORES[0] = m.decision_function(Z)
    return m.predict(Z), D["y"][te], D["video_id"][te]


def bacc(p, y):
    return 100 * balanced_accuracy_score(y, p)


def boot_tr_paired(resA, resB, B=1000):
    """video-cluster bootstrap of TR(A) - TR(B) with the SAME resampled target videos for both feature sets.
    TR = (bacc(cross) - 50) / (bacc(oracle) - 50) on the target test set."""
    pcA, yc, vc, poA, yo, vo = resA[:6]
    pcB, poB = resB[0], resB[3]
    vids = np.unique(vc)
    idx = {v: np.where(vc == v)[0] for v in vids}
    out = []
    for _ in range(B):
        s = np.concatenate([idx[v] for v in rng.choice(vids, len(vids))])
        if len(np.unique(yc[s])) < 2:
            continue
        o_a, o_b = bacc(poA[s], yo[s]) - 50, bacc(poB[s], yo[s]) - 50
        if o_a <= 1 or o_b <= 1:
            continue
        out.append((bacc(pcA[s], yc[s]) - 50) / o_a - (bacc(pcB[s], yc[s]) - 50) / o_b)
    return np.array(out)


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None
    L = []
    say = lambda s="": (print(s), L.append(s))  # noqa: E731
    for nm, D in [("MOSI", A), ("SIMS", B)]:
        say(f"{nm}: n={len(D['y'])} (pos {D['y'].mean():.2f}); split {dict(zip(*np.unique(D['mode'], return_counts=True)))}; faces/clip {D['n_face'].mean():.2f}")
    verdict = {}
    for (sn, S), (tn, T) in [(("MOSI", A), ("SIMS", B)), (("SIMS", B), ("MOSI", A))]:
        say(f"\n## {sn} -> {tn} (balanced accuracy %, test sets)")
        say("| features | in-domain (S) | cross S->T | cross, target-normalized | oracle T->T | TR | TR (target-norm) | AUC cross | AUC oracle | ATR |")
        say("|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for f, keys in FEATS.items():
            sc, m, _ = fit(S, keys)
            pi, yi, _ = preds(sc, m, S, keys)
            pc, yc, vc = preds(sc, m, T, keys)
            auc_c = 100 * roc_auc_score(yc, LAST_SCORES[0])
            pn, yn, _ = preds(sc, m, T, keys, target_norm=True)
            sc2, m2, _ = fit(T, keys)
            po, yo, vo = preds(sc2, m2, T, keys)
            auc_o = 100 * roc_auc_score(yo, LAST_SCORES[0])
            o = bacc(po, yo)
            tr, trn = (bacc(pc, yc) - 50) / (o - 50), (bacc(pn, yn) - 50) / (o - 50)
            res[f] = (pc, yc, vc, po, yo, vo, pn)
            atr = (auc_c - 50) / (auc_o - 50)
            say(f"| {f} | {bacc(pi, yi):.1f} | {bacc(pc, yc):.1f} | {bacc(pn, yn):.1f} | {o:.1f} | {tr:+.2f} | {trn:+.2f} | {auc_c:.1f} | {auc_o:.1f} | {atr:+.2f} |")
        # premise test: text vs best audio / face
        for other in ["audio(WavLM)", "audio(eGeMAPS)", "face(ViT-FER emb)"]:
            d = boot_tr_paired(res["text(e5)"], res[other])
            lo, hi = np.percentile(d, [2.5, 97.5])
            say(f"- TR(text) - TR({other}) = {np.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] (bootstrap draws used: {len(d)})")
            verdict[(sn, other)] = hi < 0
        # descriptive (added before results): translate-test for text. Source text model, target test transcripts
        # machine-translated INTO the source language (opus-mt), embedded with the same e5 encoder.
        if len(sys.argv) > 4:
            tz = np.load(sys.argv[4], allow_pickle=True)
            key = lambda D: np.array([f"{v}_{c}" for v, c in zip(D["video_id"], D["clip_id"])])  # noqa: E731
            if tn == "SIMS":
                lut = dict(zip(tz["sims_clip"], tz["sims_en_emb"]))
            else:
                lut = dict(zip(tz["mosi_clip"], tz["mosi_zh_emb"]))
            sc, m, _ = fit(S, ["text"])
            te = T["mode"] == "test"
            Z = np.stack([lut[k] for k in key(T)[te]])
            pt = m.predict(sc.transform(Z))
            po, yo = res["text(e5)"][3], res["text(e5)"][4]
            o = bacc(po, yo)
            say(f"- translate-test text: cross {bacc(pt, T['y'][te]):.1f} (oracle {o:.1f}; TR {(bacc(pt, T['y'][te]) - 50) / (o - 50):+.2f})")
    holds = all(verdict[(s, o)] for s in ["MOSI", "SIMS"] for o in ["face(ViT-FER emb)"]) and \
        all(verdict[(s, "audio(WavLM)")] or verdict[(s, "audio(eGeMAPS)")] for s in ["MOSI", "SIMS"])
    say(f"\nPremise (text hit hardest, both directions, vs audio and face): {'HOLDS' if holds else 'DOES NOT HOLD'}")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
