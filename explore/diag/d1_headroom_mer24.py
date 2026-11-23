"""Protocol v3, step 1 (CLAUDE.md §4.7): headroom map of the real baseline on MER2024 (5-fold CV on Train&Val, WAF).
Descriptive (no pass/fail); the gate is applied afterwards: keep slices whose oracle ceiling >= 2 x MDE (MDE 1.32 WAF, e06) and
that also appear on M3ED. Writes aggregate statistics only (MER licence).
Baseline = early-fusion concat AVT (A_hub + V_clip mean + T_rob) -> multinomial LR (mer24_screen.oof_proba), fold seed 0.
Ceiling of a slice = WAF after setting the predictions of that slice's error clips to the true label, minus baseline WAF.
Usage: python -I d1_headroom_mer24.py <cache dir> <label-disdim.csv> [out.md]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mer24"))
import mer24_screen as M  # noqa: E402

CACHE, LAB = sys.argv[1:3]; OUT = sys.argv[3] if len(sys.argv) > 3 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
from transformers import AutoConfig  # noqa: E402
cfg = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"); AST = [cfg.id2label[i] for i in range(cfg.num_labels)]
D = M.load_cache(CACHE); y, val, cls = M.labels(D, LAB); ok = D["ok"].astype(bool); K = len(cls)
V = D["v_clip"].astype(np.float32).sum(1) / np.maximum(D["n_frames"], 1)[:, None]
F = {"A": D["a_hub"], "V": V, "T": D["t_rob"]}; F["AVT"] = np.concatenate([F["A"], F["V"], F["T"]], 1)
F = {k: v[ok].astype(np.float32) for k, v in F.items()}; y = y[ok]; N = len(y)
P = {k: M.oof_proba(X, y, 0, "cpu", K) for k, X in F.items()}
pred = {k: p.argmax(1) for k, p in P.items()}; base = M.waf(y, pred["AVT"]); acc = lambda p: 100 * (p == y).mean()  # noqa: E731
err = pred["AVT"] != y; ne = err.sum()


def ceiling(mask):
    p = pred["AVT"].copy(); m = mask & err; p[m] = y[m]; return M.waf(y, p) - base, m.sum()


say("# Headroom map, MER2024 (step 1 of protocol v3; aggregate statistics only)")
say(f"- clips {N}; baseline AVT WAF {base:.2f}, ACC {acc(pred['AVT']):.2f}; errors {ne} ({100 * ne / N:.1f}% of clips); MDE 1.32 -> gate 2 x MDE = 2.64 WAF")
say("- unimodal: " + ", ".join(f"{k} WAF {M.waf(y, pred[k]):.2f}" for k in "AVT"))
c = {k: pred[k] == y for k in "AVT"}; anyc = c["A"] | c["V"] | c["T"]
say("\n## 1. Error decomposition (AVT errors)")
say("| slice | AVT errors in slice | % of all errors | oracle ceiling (WAF) |"); say("|---|---|---|---|")
rows = [("some unimodal correct (fusion failure)", anyc), ("no unimodal correct", ~anyc)]
for nm, m in (("A correct only", c["A"] & ~c["V"] & ~c["T"]), ("V correct only", ~c["A"] & c["V"] & ~c["T"]), ("T correct only", ~c["A"] & ~c["V"] & c["T"]),
              ("two unimodal correct", (c["A"].astype(int) + c["V"] + c["T"]) == 2), ("all three correct", c["A"] & c["V"] & c["T"])):
    rows.append(("  " + nm, m))
for nm, m in rows:
    g, n = ceiling(m); say(f"| {nm} | {n} | {100 * n / ne:.1f} | {g:+.2f} |")

say("\n## 2. Fusion failures: calibration vs. information (clips where some unimodal is right and AVT is wrong)")
ff = anyc & err; conf = {k: P[k].max(1) for k in "AVT"}
top = np.stack([conf[k] for k in "AVT"], 1).argmax(1); sel = np.stack([pred[k] for k in "AVT"], 1)[np.arange(N), top]
say(f"- fusion-failure clips: {ff.sum()}; among them the MOST CONFIDENT unimodal is correct in {100 * (sel[ff] == y[ff]).mean():.1f}%")
for k in "AVT":
    m = ff & c[k]
    if m.sum():
        oth = [j for j in "AVT" if j != k]
        say(f"  - {k} correct ({m.sum()} clips): mean conf of {k} {conf[k][m].mean():.3f} vs wrong modalities " + ", ".join(f"{j} {conf[j][m & ~c[j]].mean():.3f}" for j in oth if (m & ~c[j]).sum()))
say(f"- confidence-selection rule (pick the most confident unimodal) over ALL clips: WAF {M.waf(y, sel):.2f} (baseline {base:.2f})")
Z = np.concatenate([np.log(P[k] + 1e-6) for k in "AVT"], 1); Ps = M.oof_proba(Z, y, 1, "cpu", K)
say(f"- late fusion (LR stacked on unimodal out-of-fold log-probs, nested CV): WAF {M.waf(y, Ps.argmax(1)):.2f}")


def ece(p, n_bin=10):
    cf, pr = p.max(1), p.argmax(1); b = np.minimum((cf * n_bin).astype(int), n_bin - 1)
    return 100 * sum(abs((pr[b == i] == y[b == i]).mean() - cf[b == i].mean()) * (b == i).mean() for i in range(n_bin) if (b == i).any())
say("- ECE (10 bins, %): " + ", ".join(f"{k} {ece(P[k]):.1f}" for k in ("A", "V", "T", "AVT")))

say("\n## 3. Error mass by class and by confusion (AVT)")
say("| true class | n | recall | errors | % of all errors | ceiling (WAF) |"); say("|---|---|---|---|---|---|")
for i, cn in enumerate(cls):
    m = y == i; g, n = ceiling(m); say(f"| {cn} | {m.sum()} | {100 * (pred['AVT'][m] == i).mean():.1f} | {n} | {100 * n / ne:.1f} | {g:+.2f} |")
neu = cls.index("neutral"); g, n = ceiling((y != neu) & (pred["AVT"] == neu)); say(f"- emotional -> predicted neutral: {n} errors ({100 * n / ne:.1f}%), ceiling {g:+.2f}")
g, n = ceiling((y == neu)); say(f"- neutral -> predicted emotional: {n} errors ({100 * n / ne:.1f}%), ceiling {g:+.2f}")

say("\n## 4. Error mass by clip property (AVT)")
dur = D["dur"][ok]; q = np.quantile(dur, [0.25, 0.5, 0.75]); bq = np.digitize(dur, q)
for b in range(4):
    g, n = ceiling(bq == b); say(f"- duration Q{b + 1}: n {np.sum(bq == b)}, error rate {100 * err[bq == b].mean():.1f}%, ceiling {g:+.2f}")
tags = D["tags"][ok].astype(np.float32); mus = tags[:, [i for i, l in enumerate(AST) if "music" in l.lower()]].max(1) >= 0.2
sub = D["sub"][ok] > 0.5
for nm, m in (("music cue", mus), ("no music cue", ~mus), ("subtitle", sub), ("no subtitle", ~sub)):
    g, n = ceiling(m); say(f"- {nm}: n {m.sum()}, error rate {100 * err[m].mean():.1f}%, ceiling {g:+.2f}")

say("\n## 5. Are errors concentrated on identities? (speaker x-vector clusters, k-means k=150, label-free)")
from sklearn.cluster import KMeans  # noqa: E402
spk = D["spk"][ok].astype(np.float32); lab = KMeans(150, n_init=4, random_state=0).fit_predict(spk)
sz = np.bincount(lab); er = np.array([err[lab == j].mean() for j in range(150)])
rng = np.random.default_rng(0); perm_sd = [np.std([err[rng.permutation(N)][lab == j].mean() for j in range(150)]) for _ in range(200)]
say(f"- cluster sizes median {np.median(sz):.0f}; sd of per-cluster error rate {np.std(er):.3f} vs permutation {np.mean(perm_sd):.3f} [{np.percentile(perm_sd, 2.5):.3f}, {np.percentile(perm_sd, 97.5):.3f}]")
hi = np.argsort(-er)[:15]; m = np.isin(lab, hi); g, n = ceiling(m)
say(f"- 15 worst clusters: {m.sum()} clips ({100 * m.mean():.1f}%), error rate {100 * err[m].mean():.1f}%, {100 * n / ne:.1f}% of errors, ceiling {g:+.2f}")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
