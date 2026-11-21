"""Descriptive (post-hoc) diagnostics for F4 after P0b: (1) label-free EM prior adaptation (Saerens et al. 2002) on MELD;
(2) within-CREMA-D intensity test on sentence IEO (LO/MD/HI): margin and AUC per intensity under a model trained without IEO."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
C = np.load(sys.argv[1], allow_pickle=True); M = np.load(sys.argv[2], allow_pickle=True)
cls = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
def em_prior(P, src_prior, it=100):
    pri = src_prior.copy()
    for _ in range(it):
        Q = P * (pri / src_prior); Q /= Q.sum(1, keepdims=True); pri = Q.mean(0)
    return Q, pri
for feat in [["a_wavlm"], ["a_egemaps"], ["v_emb"], ["a_wavlm", "v_emb"]]:
    X = lambda D, m: np.concatenate([D[k][m] for k in feat], 1)
    tr = (C["mode"] == "train") & np.isin(C["label"], cls); te = (M["mode"] == "test") & np.isin(M["label"], cls)
    sc = StandardScaler().fit(X(C, tr)); m = LogisticRegression(C=0.01, max_iter=4000, class_weight="balanced").fit(sc.transform(X(C, tr)), C["label"][tr])
    P = m.predict_proba(sc.transform(X(M, te))); y = M["label"][te]
    src_prior = np.full(6, 1 / 6)  # class_weight='balanced' -> effective uniform training prior
    Q, pri = em_prior(P, src_prior)
    b0 = 100 * balanced_accuracy_score(y, m.classes_[P.argmax(1)]); b1 = 100 * balanced_accuracy_score(y, m.classes_[Q.argmax(1)])
    true_pri = np.array([(y == c).mean() for c in m.classes_])
    pred_share = np.array([(m.classes_[P.argmax(1)] == c).mean() for c in m.classes_])
    print(f"{'+'.join(feat)}: bacc source-only {b0:.1f} -> EM-prior {b1:.1f} | MELD true prior {dict(zip(m.classes_, true_pri.round(2)))} | EM est {dict(zip(m.classes_, pri.round(2)))} | source-model predicted share {dict(zip(m.classes_, pred_share.round(2)))}")
# (2) intensity within CREMA-D
ids = C["id"]; sent = np.array([i.split('_')[1] for i in ids]); inten = np.array([i.split('_')[3] for i in ids])
for feat in [["a_wavlm"], ["v_emb"]]:
    X = lambda m: np.concatenate([C[k][m] for k in feat], 1)
    tr = (C["mode"] != "test") & (sent != "IEO"); sc = StandardScaler().fit(X(tr))
    m = LogisticRegression(C=0.01, max_iter=4000, class_weight="balanced").fit(sc.transform(X(tr)), C["label"][tr])
    for lev in ["LO", "MD", "HI"]:
        sel = (C["mode"] == "test") & (sent == "IEO") & (inten == lev)
        Lg = m.decision_function(sc.transform(X(sel))); y = C["label"][sel]; yi = np.array([list(m.classes_).index(c) for c in y])
        marg = Lg[np.arange(len(yi)), yi] - (Lg.sum(1) - Lg[np.arange(len(yi)), yi]) / 5
        P = m.predict_proba(sc.transform(X(sel)))
        auc = 100 * roc_auc_score(yi, P, multi_class="ovr", average="macro", labels=list(range(6))) if len(np.unique(yi)) == 5 or len(np.unique(yi)) == 6 else np.nan
        print(f"  CREMA-D IEO {lev} [{'+'.join(feat)}] n={sel.sum()} classes={sorted(set(y))}: bacc {100*balanced_accuracy_score(y, m.classes_[P.argmax(1)]):.1f}, mean true-class margin {marg.mean():.2f}")
