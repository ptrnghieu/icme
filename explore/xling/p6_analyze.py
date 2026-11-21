"""P6 premise test for idea E (post-production / "editorial" cues in edited-media emotion datasets).

Criterion FIXED BEFORE RESULTS (2026-11-21):
  Cue classes (AudioSet labels, rule fixed): MUSIC = labels matching (?i)\\bmusic\\b|instrument|orchestra|song|piano|guitar|violin|drum|synthesizer;
  AUDIENCE = (?i)laugh|giggle|chuckle|snicker|applause|clapping|cheering|crowd.  A clip "has the cue" if max prob in the class >= 0.2.
  Editorial feature vector E = probabilities of all MUSIC and AUDIENCE classes (no speech classes).
  For each dataset (CH-SIMS: Chinese film/TV; MELD: sitcom; MOSI: YouTube vlogs = negative control):
    (i)  prevalence = fraction of clips with MUSIC or AUDIENCE cue;
    (ii) E-only logistic regression (class_weight balanced, C chosen on valid; MELD: train on dev split as in P0b) -> test bacc,
         group-bootstrap 95% CI (video / dialogue groups) of (bacc - chance).
  Premise HOLDS on a dataset iff prevalence >= 15% AND E-only bacc - chance >= 3 points with CI excluding 0.
  Idea E proceeds to stage 3 (novelty) iff the premise HOLDS on CH-SIMS AND on MELD.
  Descriptive only: audio-model (WavLM LR from P0/P0b features) accuracy on clips whose E-only prediction disagrees vs agrees
  with the true label, split by cue presence.
Labels: MOSI/CH-SIMS binary polarity (label != 0); MELD 6 classes shared with CREMA-D (as P0b, surprise dropped).
Usage: python -I p6_analyze.py <run dir> [out.md]
"""
import os, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
rng = np.random.default_rng(0)
MU = re.compile(r"(?i)\bmusic\b|instrument|orchestra|song|piano|guitar|violin|drum|synthesizer")
AU = re.compile(r"(?i)laugh|giggle|chuckle|snicker|applause|clapping|cheering|crowd")


def fit(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr); best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        s = balanced_accuracy_score(yva, m.predict(sc.transform(Xva)))
        if best is None or s > best[0]:
            best = (s, m)
    return lambda X: best[1].predict(sc.transform(X))


def dataset(name, tagf, featf, binary):
    T = np.load(os.path.join(R, tagf), allow_pickle=True); F = np.load(os.path.join(R, featf), allow_pickle=True)
    labs = list(T["labels"]); mu = [i for i, l in enumerate(labs) if MU.search(l)]; au = [i for i, l in enumerate(labs) if AU.search(l)]
    if binary:
        kt = {(str(v), str(c)): i for i, (v, c) in enumerate(zip(T["video_id"], T["clip_id"]))}
        kf = [kt.get((str(v), str(c))) for v, c in zip(F["video_id"], F["clip_id"])]
    else:
        kt = {str(i): j for j, i in enumerate(T["id"])}; kf = [kt.get(str(i)) for i in F["id"]]
    keep = np.array([k is not None for k in kf]) & F["ok"].astype(bool)
    idx = np.array([k for k in kf if k is not None]); keep_t = T["ok"][np.array([k if k is not None else 0 for k in kf])].astype(bool)
    keep &= keep_t
    D = {k: F[k][keep] for k in F.files}; tg = T["tags"][np.array([k for k, kk in zip(kf, keep) if kk])]
    if binary:
        lab = D["label"].astype(float); nz = lab != 0; D = {k: v[nz] for k, v in D.items()}; tg = tg[nz]
        y = (D["label"].astype(float) > 0).astype(int); grp = D["video_id"]; chance = 50.0
    else:
        cls = sorted(set(np.load(os.path.join(R, "cremad_p0b.npz"), allow_pickle=True)["label"]) & set(D["label"]))
        m = np.isin(D["label"], cls); D = {k: v[m] for k, v in D.items()}; tg = tg[m]
        y = np.array([cls.index(c) for c in D["label"]]); grp = D["group"]; chance = 100.0 / len(cls)
    E = tg[:, mu + au]; has_mu = tg[:, mu].max(1) >= 0.2; has_au = tg[:, au].max(1) >= 0.2; has = has_mu | has_au
    tr, va, te = D["mode"] == "train", D["mode"] == "valid", D["mode"] == "test"
    if va.sum() == 0:
        va = tr
    pe = fit(E[tr], y[tr], E[va], y[va])(E[te]); be = 100 * balanced_accuracy_score(y[te], pe)
    g = grp[te]; ug = np.unique(g); ix = {v: np.where(g == v)[0] for v in ug}; d = []
    for _ in range(1000):
        s = np.concatenate([ix[v] for v in rng.choice(ug, len(ug))]); d.append(100 * balanced_accuracy_score(y[te][s], pe[s]) - chance)
    lo, hi = np.percentile(d, [2.5, 97.5]); prev = has.mean()
    holds = prev >= 0.15 and be - chance >= 3 and lo > 0
    say(f"\n## {name} (n={len(y)}, test n={te.sum()}, chance {chance:.1f})")
    say(f"- prevalence: music cue {100 * has_mu.mean():.1f}%, audience cue {100 * has_au.mean():.1f}%, either {100 * prev:.1f}%")
    say(f"- E-only (editorial audio tags) test bacc {be:.1f}; bacc - chance = {be - chance:+.1f} [{lo:+.1f}, {hi:+.1f}] -> premise {'HOLDS' if holds else 'does not hold'}")
    pa = fit(D["a_wavlm"][tr].astype(float), y[tr], D["a_wavlm"][va].astype(float), y[va])(D["a_wavlm"][te].astype(float))
    hte = has[te]; agree = pe == y[te]
    for nm, m in [("cue present, E-pred agrees with label", hte & agree), ("cue present, E-pred disagrees", hte & ~agree),
                  ("no cue, E-pred agrees", ~hte & agree), ("no cue, E-pred disagrees", ~hte & ~agree)]:
        if m.sum() >= 10:
            say(f"  - WavLM audio-model accuracy, {nm}: {100 * (pa[m] == y[te][m]).mean():.1f} (n={m.sum()})")
    return holds


r = {"CH-SIMS": dataset("CH-SIMS (film/TV)", "sims_p6.npz", "sims_p0.npz", True),
     "MELD": dataset("MELD (sitcom)", "meld_p6.npz", "meld_p0b.npz", False),
     "MOSI": dataset("MOSI (vlogs, negative control)", "mosi_p6.npz", "mosi_p0.npz", True)}
ok = r["CH-SIMS"] and r["MELD"]
say(f"\nIdea E premise (CH-SIMS AND MELD): {'HOLDS -> stage 3' if ok else 'DOES NOT HOLD -> ĐÃ LOẠI'}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
