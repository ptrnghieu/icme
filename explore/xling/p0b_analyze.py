"""P0b analysis (posed -> spontaneous), criteria fixed BEFORE results (2026-11-21).

Shared classes: labels present in both npz. For each feature set and direction S->T (multi-class, LR, C chosen on S/valid):
  bacc (macro recall, %), macro one-vs-rest AUC (%), TR = (bacc_cross - chance)/(bacc_oracle - chance), ATR = (AUC_cross - 50)/(AUC_oracle - 50)
  gain g = mean over target-test clips of [logit_true - mean logit_other] under the SOURCE model, divided by the same on SOURCE test.
F4 ("expressivity shift") predicts, for nonverbal features under posed -> spontaneous: g < 1 and ATR > TR (shift mostly in magnitude/threshold).
Cluster bootstrap (by group) 95% CI for g and for ATR - TR.
Usage: python -I p0b_analyze.py source.npz target.npz name_S name_T [out.md]
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
FEATS = {"audio(eGeMAPS)": ["a_egemaps"], "audio(WavLM)": ["a_wavlm"], "face(ViT-FER emb)": ["v_emb"], "face(FER logits)": ["v_logits"],
         "text(e5)": ["text"], "nonverbal(A_wl+V)": ["a_wavlm", "v_emb"]}


def load(p, classes=None):
    d = np.load(p, allow_pickle=True)
    D = {k: d[k] for k in d.files}
    keep = D["ok"].astype(bool)
    if classes is not None:
        keep &= np.isin(D["label"], classes)
    return {k: v[keep] for k, v in D.items()}


def X(D, keys, m):
    return np.concatenate([D[k][m] for k in keys], 1)


def fit(D, keys, classes):
    tr, va = D["mode"] == "train", D["mode"] == "valid"
    sc = StandardScaler().fit(X(D, keys, tr)); best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(sc.transform(X(D, keys, tr)), D["label"][tr])
        s = balanced_accuracy_score(D["label"][va], m.predict(sc.transform(X(D, keys, va))))
        if best is None or s > best[0]:
            best = (s, m)
    return sc, best[1]


def evaluate(sc, m, D, keys, classes):
    te = D["mode"] == "test"
    Z = sc.transform(X(D, keys, te)); y = D["label"][te]
    P = m.predict_proba(Z); L = m.decision_function(Z)
    order = [list(m.classes_).index(c) for c in classes]
    P, L = P[:, order], L[:, order]
    yi = np.array([classes.index(c) for c in y])
    margin = L[np.arange(len(yi)), yi] - (L.sum(1) - L[np.arange(len(yi)), yi]) / (len(classes) - 1)
    return dict(pred=np.array(classes)[P.argmax(1)], y=y, P=P, yi=yi, margin=margin, g=D["group"][te])


def bacc(r, idx=None):
    idx = slice(None) if idx is None else idx
    return 100 * balanced_accuracy_score(r["y"][idx], r["pred"][idx])


def auc(r, idx=None):
    idx = slice(None) if idx is None else idx
    try:
        return 100 * roc_auc_score(r["yi"][idx], r["P"][idx], multi_class="ovr", average="macro", labels=list(range(r["P"].shape[1])))
    except ValueError:
        return np.nan


def main():
    S0, T0 = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
    classes = sorted(set(S0["label"]) & set(T0["label"]))
    S, T = load(sys.argv[1], classes), load(sys.argv[2], classes)
    nS, nT = sys.argv[3], sys.argv[4]
    out = sys.argv[5] if len(sys.argv) > 5 else None
    chance = 100 / len(classes)
    L = []; say = lambda s="": (print(s), L.append(s))  # noqa: E731
    say(f"classes {classes}; {nS}: {len(S['label'])} clips, {nT}: {len(T['label'])} clips; chance bacc {chance:.1f}")
    say(f"\n## {nS} -> {nT}")
    say("| features | in-domain bacc | cross bacc | oracle bacc | TR | cross AUC | oracle AUC | ATR | gain g [95% CI] | ATR - TR [95% CI] |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for f, keys in FEATS.items():
        if any(np.abs(S[k]).sum() == 0 or np.abs(T[k]).sum() == 0 for k in keys):
            continue
        sc, m = fit(S, keys, classes)
        rs, rc = evaluate(sc, m, S, keys, classes), evaluate(sc, m, T, keys, classes)
        sc2, m2 = fit(T, keys, classes)
        ro = evaluate(sc2, m2, T, keys, classes)
        if bacc(ro) - chance < 3 or auc(ro) - 50 < 3:
            say(f"| {f} | {bacc(rs):.1f} | {bacc(rc):.1f} | {bacc(ro):.1f} | n/a (oracle near chance) | {auc(rc):.1f} | {auc(ro):.1f} | n/a | n/a | n/a |")
            continue
        TR = (bacc(rc) - chance) / (bacc(ro) - chance); ATR = (auc(rc) - 50) / (auc(ro) - 50)
        g = rc["margin"].mean() / rs["margin"].mean()
        gs, ds = [], []
        gT, gS = np.unique(rc["g"]), np.unique(rs["g"])
        iT = {v: np.where(rc["g"] == v)[0] for v in gT}; iS = {v: np.where(rs["g"] == v)[0] for v in gS}
        for _ in range(500):
            a = np.concatenate([iT[v] for v in rng.choice(gT, len(gT))]); b = np.concatenate([iS[v] for v in rng.choice(gS, len(gS))])
            gs.append(rc["margin"][a].mean() / rs["margin"][b].mean())
            ob, oa = bacc(ro, a) - chance, auc(ro, a) - 50
            if ob > 1 and oa > 1:
                ds.append((auc(rc, a) - 50) / oa - (bacc(rc, a) - chance) / ob)
        gl, gh = np.nanpercentile(gs, [2.5, 97.5]); dl, dh = np.nanpercentile(ds, [2.5, 97.5])
        say(f"| {f} | {bacc(rs):.1f} | {bacc(rc):.1f} | {bacc(ro):.1f} | {TR:+.2f} | {auc(rc):.1f} | {auc(ro):.1f} | {ATR:+.2f} | {g:.2f} [{gl:.2f}, {gh:.2f}] | {ATR - TR:+.2f} [{dl:+.2f}, {dh:+.2f}] |")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
