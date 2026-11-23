"""CPU/GPU screens on the cached MER2024 Train&Val features (built by mer24_features.py). Runs on the USER's Kaggle.
Writes ONLY aggregate statistics (MER licence). Each screen is a function below; its pre-registered criterion lives in its docstring.

Protocol for every screen (2404.17113 §3): five-fold cross-validation on Train&Val (5,030 clips, 6 classes), metric WAF (weighted F1).
We use 3 fold seeds (StratifiedKFold, shuffle) and out-of-fold predictions for all clips. Classifier: multinomial logistic regression
(L2, full-batch L-BFGS in torch), C in {0.01, 0.1, 1} chosen by WAF on an inner 10% split of each training fold.
MDE for a paired comparison = 2.8 x bootstrap SE (1000 resamples of clips) of the WAF difference.

Usage:
  python mer24_screen.py text     --cache DIR --labels label-disdim.csv --trans label-transcription.csv      (once; caches text features)
  python mer24_screen.py baseline --cache DIR --labels label-disdim.csv --out report.md
"""
import argparse, glob, os, re
import numpy as np

PAPER = {"A_hub": 73.02, "V_clip": 66.66, "T_rob": 52.66, "AVT": "79.31-80.34 (Tab. 5, attention fusion, Top1-Top6)"}


# --------------------------------------------------------------------------------------------------------------- data
def load_cache(cache):
    P = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(os.path.join(cache, "part_*.npz")))]
    D = {k: np.concatenate([p[k] for p in P]) for k in P[0].files}
    t = os.path.join(cache, "text.npz")
    if os.path.exists(t):
        T = np.load(t, allow_pickle=True); idx = {n: i for i, n in enumerate(T["name"])}
        sel = np.array([idx.get(str(n), -1) for n in D["name"]])
        for k in ("t_rob", "t_e5", "t_len"):
            D[k] = np.where((sel >= 0)[:, None] if T[k].ndim > 1 else (sel >= 0), T[k][np.maximum(sel, 0)], 0)
    return D


def labels(D, path):
    import pandas as pd
    L = pd.read_csv(path); L["name"] = L["name"].astype(str); L = L.set_index("name")
    cls = sorted(L["discrete"].unique()); names = [str(n) for n in D["name"]]
    y = np.array([cls.index(L.loc[n, "discrete"]) for n in names]); val = np.array([float(L.loc[n, "valence"]) for n in names])
    return y, val, cls


def cmd_text(a):
    import pandas as pd, torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = pd.read_csv(a.trans); T["name"] = T["name"].astype(str); texts = [str(t) if isinstance(t, str) else "" for t in T["chinese"]]
    out = {"name": T["name"].values, "t_len": np.array([len(t) for t in texts])}
    for key, mid, prefix in (("t_rob", "hfl/chinese-roberta-wwm-ext-large", ""), ("t_e5", "intfloat/multilingual-e5-base", "query: ")):
        tok = AutoTokenizer.from_pretrained(mid); mdl = AutoModel.from_pretrained(mid).to(dev).eval(); E = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = tok([prefix + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
                h = mdl(**b).last_hidden_state; mk = b["attention_mask"].unsqueeze(-1).float()
                E.append(((h * mk).sum(1) / mk.sum(1)).float().cpu().numpy())
        out[key] = np.concatenate(E).astype(np.float32)
    np.savez_compressed(os.path.join(a.cache, "text.npz"), **out); print("text features cached:", {k: v.shape for k, v in out.items()})


# --------------------------------------------------------------------------------------------------------------- model
def fit_lr(X, y, C, K, dev):
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(X.shape[1], K, device=dev, requires_grad=True); b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, history_size=20, line_search_fn="strong_wolfe"); lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + 0.5 * lam * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure); Wd, bd = W.detach(), b.detach()
    return lambda Z: torch.softmax(torch.as_tensor(Z, dtype=torch.float32, device=dev) @ Wd + bd, 1).cpu().numpy()


def oof_proba(X, y, seed, dev, K):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    P = np.zeros((len(y), K)); rng = np.random.default_rng(seed)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        tr = rng.permutation(tr); nv = max(1, len(tr) // 10); va, tr2 = tr[:nv], tr[nv:]
        mu, sd = X[tr2].mean(0), X[tr2].std(0) + 1e-6; best = None
        for C in (0.01, 0.1, 1.0):
            f = fit_lr((X[tr2] - mu) / sd, y[tr2], C, K, dev); s = f1_score(y[va], f((X[va] - mu) / sd).argmax(1), average="weighted")
            if best is None or s > best[0]:
                best = (s, C)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P[te] = fit_lr((X[tr] - mu) / sd, y[tr], best[1], K, dev)((X[te] - mu) / sd)
    return P


def waf(y, p):
    from sklearn.metrics import f1_score
    return 100 * f1_score(y, p, average="weighted")


def boot_delta(y, pa, pb, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        s = rng.integers(0, N, N); d.append(waf(y[s], pa[s]) - waf(y[s], pb[s]))
    d = np.array(d); return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(2.8 * d.std())


# --------------------------------------------------------------------------------------------------------------- screens
def screen_baseline(D, y, val, cls, dev, say):
    """Baseline reproduction + failure slices (channel (a) of protocol v2). Descriptive; no pass/fail criterion.
    Sanity rule (fixed before results): the pipeline is considered usable if A_hub, V_clip and AVT are each within 5 WAF points
    of the paper values (Tab. 4/5); otherwise later screens must report absolute numbers with that caveat."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    K = len(cls); ok = D["ok"].astype(bool)
    vclip = D["v_clip"].astype(np.float32); nf = np.maximum(D["n_frames"], 1)[:, None]; V = vclip.sum(1) / nf
    F = {"A_hub": D["a_hub"], "A_wavlm": D["a_wavlm"], "V_clip": V, "T_rob": D["t_rob"], "T_e5": D["t_e5"],
         "AV": np.concatenate([D["a_hub"], V], 1), "AT": np.concatenate([D["a_hub"], D["t_rob"]], 1),
         "VT": np.concatenate([V, D["t_rob"]], 1), "AVT": np.concatenate([D["a_hub"], V, D["t_rob"]], 1)}
    F = {k: v[ok].astype(np.float32) for k, v in F.items()}; y, val = y[ok], val[ok]
    say(f"- clips with features: {ok.sum()} / {len(ok)}; classes {cls}; class counts {np.bincount(y).tolist()}")
    say(f"- valence: min {np.nanmin(val):.2f}, max {np.nanmax(val):.2f}, mean {np.nanmean(val):.2f}; NaN {int(np.isnan(val).sum())}; distinct finite values {len(np.unique(val[np.isfinite(val)]))}")
    P = {k: [oof_proba(X, y, s, dev, K) for s in range(3)] for k, X in F.items()}
    say("\n| features | WAF mean ± sd (3 fold seeds) | ACC mean | paper Train&Val WAF |"); say("|---|---|---|---|")
    for k in F:
        w = [waf(y, p.argmax(1)) for p in P[k]]; acc = np.mean([100 * accuracy_score(y, p.argmax(1)) for p in P[k]])
        say(f"| {k} | {np.mean(w):.2f} ± {np.std(w):.2f} | {acc:.2f} | {PAPER.get(k, '')} |")
    pa, pb = P["AVT"][0].argmax(1), P["AV"][0].argmax(1); lo, hi, mde = boot_delta(y, pa, pb)
    say(f"\n- power: WAF(AVT) - WAF(AV), seed 0 = {waf(y, pa) - waf(y, pb):+.2f} [{lo:+.2f}, {hi:+.2f}]; **MDE for a paired WAF comparison ≈ {mde:.2f} points**")
    p = P["AVT"][0].argmax(1); neu = cls.index("neutral") if "neutral" in cls else None
    say("\n## Failure slices (AVT, seed 0; out-of-fold)")
    say("- per-class recall: " + ", ".join(f"{c} {100 * (p[y == i] == i).mean():.1f}" for i, c in enumerate(cls)))
    cm = confusion_matrix(y, p, labels=range(K)); off = [(cm[i, j], cls[i], cls[j]) for i in range(K) for j in range(K) if i != j]
    say("- top confusions (true -> pred, count): " + "; ".join(f"{a}->{b} {n}" for n, a, b in sorted(off, reverse=True)[:6]))
    fin = np.isfinite(val); say(f"- clips with a finite valence label: {fin.sum()} / {len(val)}")
    av = np.where(fin, np.abs(val), np.nan); qs = np.nanquantile(av, [0.25, 0.5, 0.75]); bins = np.where(fin, np.digitize(np.nan_to_num(av), qs), -1)
    say("- by |valence| quartile (Q1 lowest): " + "; ".join(f"Q{b + 1}: WAF {waf(y[bins == b], p[bins == b]):.1f} (n={(bins == b).sum()})" for b in range(4)))
    if neu is not None:
        emo = y != neu
        say("- neutral sink P(pred=neutral | true emotional) by |valence| quartile: " + "; ".join(
            f"Q{b + 1}: {100 * (p[emo & (bins == b)] == neu).mean():.1f}% (n={(emo & (bins == b)).sum()})" for b in range(4) if (emo & (bins == b)).sum() >= 20))
    pos = {cls.index(c) for c in ("happy",) if c in cls}; neg = {cls.index(c) for c in ("sad", "angry", "worried") if c in cls}
    inc = np.array([(yy in pos and v < 0) or (yy in neg and v > 0) for yy, v in zip(y, val)])
    say(f"- label-valence inconsistent clips (happy with valence<0, or sad/angry/worried with valence>0): {inc.sum()} ({100 * inc.mean():.1f}%); "
        f"AVT accuracy consistent {100 * (p[~inc] == y[~inc]).mean():.1f} vs inconsistent {100 * (p[inc] == y[inc]).mean():.1f}")
    for nm, arr in (("duration (s)", D["dur"][ok]), ("transcript length (chars)", D["t_len"][ok])):
        q = np.quantile(arr, [0.25, 0.5, 0.75]); bb = np.digitize(arr, q)
        say(f"- by {nm} quartile (cut points {np.round(q, 1).tolist()}): " + "; ".join(f"Q{b + 1} WAF {waf(y[bb == b], p[bb == b]):.1f}" for b in range(4)))
    labels_ast = AST_LABELS
    tags = D["tags"][ok].astype(np.float32)
    for nm, rx in (("music", r"(?i)\bmusic\b"), ("laughter", r"(?i)laugh"), ("crying", r"(?i)crying|sobbing|whimper"), ("shout/scream", r"(?i)shout|scream|yell")):
        idx = [i for i, l in enumerate(labels_ast) if re.search(rx, l)]
        has = tags[:, idx].max(1) >= 0.2
        if has.sum() >= 20:
            say(f"- {nm} cue (AST >= 0.2): {100 * has.mean():.1f}% of clips; AVT WAF with {waf(y[has], p[has]):.1f} vs without {waf(y[~has], p[~has]):.1f}")
    sub = D["sub"][ok] > 0.5
    say(f"- subtitle on middle frame: {100 * sub.mean():.1f}% of clips; AVT WAF with {waf(y[sub], p[sub]):.1f} vs without {waf(y[~sub], p[~sub]):.1f}")
    ca, cv, ct = (P[k][0].argmax(1) == y for k in ("A_hub", "V_clip", "T_rob"))
    say(f"- unimodal correctness: any correct {100 * (ca | cv | ct).mean():.1f}%; all three {100 * (ca & cv & ct).mean():.1f}%; exactly one {100 * ((ca.astype(int) + cv + ct) == 1).mean():.1f}%")
    for nm, msk in (("only A correct", ca & ~cv & ~ct), ("only V correct", ~ca & cv & ~ct), ("only T correct", ~ca & ~cv & ct), ("none correct", ~(ca | cv | ct))):
        say(f"  - {nm}: {msk.sum()} clips ({100 * msk.mean():.1f}%); AVT accuracy on them {100 * (p[msk] == y[msk]).mean():.1f}")


def screen_peakend(D, y, val, cls, dev, say):
    """Peak-end pooling screen (P10 on MER2024; pre-registered 2026-11-23, before any MER2024 result).
    Audio = chinese-hubert-large per 1-s segment (a_hub_seg, n_seg). Peak = segment farthest (L2) from the clip mean after
    per-dimension standardisation by the pooled segment sd (label-free). End = last valid segment, First = first, Random = one
    uniformly chosen segment (seed 0). MPE = mean+peak+end vs MFR = mean+first+random (equal dimension).
    PRIMARY: WAF(MPE) - WAF(MFR), out-of-fold, mean over 3 fold seeds. HOLDS iff >= +1.0 WAF AND the seed-0 bootstrap CI excludes 0
    AND the gain on clips >= 4 s exceeds the gain on clips < 4 s. Descriptive: ME-MF, MP-MR, and AVT with each audio pooling."""
    K = len(cls); ok = D["ok"].astype(bool) & (D["n_seg"] > 0)
    S = D["a_hub_seg"][ok].astype(np.float32); n = D["n_seg"][ok].astype(int); y, dur = y[ok], D["dur"][ok]
    mask = np.arange(S.shape[1])[None, :] < n[:, None]; sd = S[mask].std(0) + 1e-6; rng = np.random.default_rng(0); N = len(y)
    mean = (S * mask[..., None]).sum(1) / n[:, None]; first = S[:, 0]; end = S[np.arange(N), n - 1]
    dist = np.linalg.norm((S - mean[:, None]) / sd, axis=2); dist[~mask] = -1; peak = S[np.arange(N), dist.argmax(1)]
    rnd = S[np.arange(N), (rng.random(N) * n).astype(int)]
    V = (D["v_clip"].astype(np.float32).sum(1) / np.maximum(D["n_frames"], 1)[:, None])[ok]; T = D["t_rob"][ok].astype(np.float32)
    F = {"M": mean, "MF": np.c_[mean, first], "ME": np.c_[mean, end], "MP": np.c_[mean, peak], "MR": np.c_[mean, rnd],
         "MPE": np.c_[mean, peak, end], "MFR": np.c_[mean, first, rnd],
         "AVT_M": np.c_[mean, V, T], "AVT_MPE": np.c_[mean, peak, end, V, T], "AVT_MFR": np.c_[mean, first, rnd, V, T]}
    say(f"- clips: {N}; segments per clip median {np.median(n):.0f}; clips >= 4 s: {(dur >= 4).sum()} ({100 * (dur >= 4).mean():.1f}%)")
    P = {k: [oof_proba(X.astype(np.float32), y, s, dev, K).argmax(1) for s in range(3)] for k, X in F.items()}
    lg = dur >= 4
    say("\n| features | WAF mean ± sd | WAF < 4 s | WAF >= 4 s |"); say("|---|---|---|---|")
    for k in F:
        w = [waf(y, p) for p in P[k]]
        say(f"| {k} | {np.mean(w):.2f} ± {np.std(w):.2f} | {np.mean([waf(y[~lg], p[~lg]) for p in P[k]]):.2f} | {np.mean([waf(y[lg], p[lg]) for p in P[k]]):.2f} |")
    def dlt(a, b, m=None):
        m = np.ones(N, bool) if m is None else m
        return float(np.mean([waf(y[m], pa[m]) - waf(y[m], pb[m]) for pa, pb in zip(P[a], P[b])]))
    d0 = dlt("MPE", "MFR"); lo, hi, mde = boot_delta(y, P["MPE"][0], P["MFR"][0]); dl, dh = dlt("MPE", "MFR", ~lg), dlt("MPE", "MFR", lg)
    holds = d0 >= 1.0 and lo > 0 and dh > dl
    say(f"\n- PRIMARY WAF(MPE) - WAF(MFR) = {d0:+.2f} (seed-0 CI [{lo:+.2f}, {hi:+.2f}], MDE {mde:.2f}); < 4 s {dl:+.2f}; >= 4 s {dh:+.2f}")
    for a, b in (("ME", "MF"), ("MP", "MR"), ("AVT_MPE", "AVT_MFR"), ("AVT_MPE", "AVT_M")):
        l2, h2, _ = boot_delta(y, P[a][0], P[b][0]); say(f"- {a} - {b}: {dlt(a, b):+.2f} (seed-0 CI [{l2:+.2f}, {h2:+.2f}])")
    say(f"\n**Peak-end premise on MER2024 {'HOLDS' if holds else 'DOES NOT HOLD'}**")


AST_LABELS = []


SCREENS = {"baseline": screen_baseline, "peakend": screen_peakend}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("screen"); ap.add_argument("--cache"); ap.add_argument("--labels"); ap.add_argument("--trans", default="")
    ap.add_argument("--out", default="screen_report.md"); a = ap.parse_args()
    if a.screen == "text":
        return cmd_text(a)
    import torch
    from transformers import AutoConfig
    global AST_LABELS
    cfg = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"); AST_LABELS = [cfg.id2label[i] for i in range(cfg.num_labels)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = load_cache(a.cache); y, val, cls = labels(D, a.labels); out = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    say(f"# MER2024 screen `{a.screen}` (aggregate statistics only)")
    SCREENS[a.screen](D, y, val, cls, dev, say)
    open(a.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
