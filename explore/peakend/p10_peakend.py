"""P10 peak-end pooling premise test (round 4; ideas/round4_original_ideas_2026-11-21.md). CPU, frozen WavLM-base-plus.

PRE-REGISTERED (2026-11-23, before any result):
  Datasets / splits (official, community standard): CH-SIMS (train / valid / test), CMU-MOSI (train / valid / test).
  Target: continuous sentiment score (CH-SIMS label in [-1, 1]; MOSI in [-3, 3]); metric Pearson r on test (MSA standard), Acc-2
    (non-zero labels) reported too. Regressor: ridge, alpha in {1, 10, 100, 1000} chosen on the official valid split by r.
  Segments: WavLM layer-average, mean over each 1-s window (<= 20 windows). Peak = window farthest (L2) from the clip mean in
    feature space standardised with train statistics (label-free salience). End = last window. First = first window.
    Random = one window chosen uniformly (seed 0). Clips with one window: all equal the mean.
  Feature sets: M = mean; MF = mean+first; ME = mean+end; MP = mean+peak; MR = mean+random; MPE = mean+peak+end;
    MFR = mean+first+random (same dimension as MPE: capacity control).
  PRIMARY: delta_r = r(MPE) - r(MFR) on test. HOLDS iff, on BOTH datasets, delta_r >= +0.02 with 95% cluster-bootstrap CI
    (clusters = videos, 2000 draws) excluding 0, AND delta_r is larger for clips >= 4 s than for clips < 4 s (point estimates).
  Descriptive: ME - MF (recency), MP - MR (peak), duration-stratified r. MDE = 2.8 x bootstrap SE.
  This CPU pilot informs the MER2024 screen (`peakend` in explore/mer24/mer24_screen.py); a fail here closes the branch.
Usage: python -I p10_peakend.py <workdir> <sims Raw.zip> <sims label.csv> <mosi Raw.zip> <mosi label.csv> [out.md]
"""
import os, subprocess, sys, tempfile, warnings, zipfile
import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore"); torch.set_num_threads(4)
W = sys.argv[1]; OUT = sys.argv[6] if len(sys.argv) > 6 else None; os.makedirs(W, exist_ok=True)
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
from transformers import AutoFeatureExtractor, WavLMModel
wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); wm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()


@torch.no_grad()
def segments(x):
    x = x[: 16000 * 20]
    if len(x) < 16000:
        x = np.pad(x, (0, 16000 - len(x)))
    hs = wm(**wfe(x, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
    h = torch.stack(hs[1:]).mean(0)[0].numpy()
    return np.stack([h[i:i + 50].mean(0) for i in range(0, len(h), 50) if len(h[i:i + 50]) >= 10])


def extract(name, zpath, lpath):
    cache = os.path.join(W, f"{name}_seg.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return {k: d[k] for k in d.files}
    z = zipfile.ZipFile(zpath); names = set(z.namelist()); Lb = pd.read_csv(lpath, dtype={"clip_id": str, "video_id": str})
    segs, keep = [], []
    with tempfile.TemporaryDirectory() as td:
        for k, (v, c) in enumerate(zip(Lb.video_id, Lb.clip_id)):
            p = next((f"Raw/{v}/{cc}.mp4" for cc in (c, str(int(c)) if c.isdigit() else c, c.zfill(4)) if f"Raw/{v}/{cc}.mp4" in names), None)
            s = None
            if p is not None:
                f = os.path.join(td, "c.mp4"); open(f, "wb").write(z.read(p)); w = os.path.join(td, "a.wav")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f, "-ac", "1", "-ar", "16000", w], check=False)
                try:
                    x, _ = sf.read(w, dtype="float32"); s = (segments(x), len(x) / 16000)
                except Exception:  # noqa: BLE001
                    s = None
            segs.append(s); keep.append(s is not None)
            if k % 300 == 0:
                print(name, k, len(Lb), flush=True)
    d = dict(seg=np.array([s[0] if s else np.zeros((1, 768)) for s in segs], dtype=object), dur=np.array([s[1] if s else 0 for s in segs]),
             ok=np.array(keep), label=Lb.label.values.astype(float), mode=Lb["mode"].values, vid=Lb.video_id.values)
    np.savez_compressed(cache, **d); return d


def run(name, d):
    ok = d["ok"].astype(bool); S = d["seg"][ok]; y = d["label"][ok]; mode = d["mode"][ok]; vid = d["vid"][ok]; dur = d["dur"][ok]
    tr = mode == "train"; allseg = np.concatenate(list(S[tr])); mu, sd = allseg.mean(0), allseg.std(0) + 1e-6
    rng = np.random.default_rng(0)
    mean = np.stack([s.mean(0) for s in S]); first = np.stack([s[0] for s in S]); end = np.stack([s[-1] for s in S])
    peak = np.stack([s[np.argmax(np.linalg.norm((s - s.mean(0)) / sd, axis=1))] for s in S]); rnd = np.stack([s[rng.integers(len(s))] for s in S])
    F = {"M": mean, "MF": np.c_[mean, first], "ME": np.c_[mean, end], "MP": np.c_[mean, peak], "MR": np.c_[mean, rnd],
         "MPE": np.c_[mean, peak, end], "MFR": np.c_[mean, first, rnd]}
    va, te = mode == "valid", mode == "test"; P = {}
    for k, X in F.items():
        sc = StandardScaler().fit(X[tr]); best = None
        for a in (1, 10, 100, 1000):
            m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr]); r = pearsonr(m.predict(sc.transform(X[va])), y[va])[0]
            if best is None or r > best[0]:
                best = (r, m)
        P[k] = best[1].predict(sc.transform(X[te]))
    yt, vt, dt = y[te], vid[te], dur[te]
    say(f"\n## {name} (test n={te.sum()}; clips >= 4 s: {(dt >= 4).sum()}; median duration {np.median(dt):.1f} s)")
    say("| features | r (test) | Acc-2 (non-zero) | r, clips < 4 s | r, clips >= 4 s |"); say("|---|---|---|---|---|")
    nz = yt != 0
    for k in F:
        p = P[k]; a2 = 100 * np.mean((p[nz] > 0) == (yt[nz] > 0))
        say(f"| {k} | {pearsonr(p, yt)[0]:.3f} | {a2:.1f} | {pearsonr(p[dt < 4], yt[dt < 4])[0]:.3f} | {pearsonr(p[dt >= 4], yt[dt >= 4])[0]:.3f} |")
    uv = np.unique(vt); ix = {v: np.where(vt == v)[0] for v in uv}; rng = np.random.default_rng(1)

    def delta(a, b, mask=None):
        msk = np.ones(len(yt), bool) if mask is None else mask
        pt = pearsonr(P[a][msk], yt[msk])[0] - pearsonr(P[b][msk], yt[msk])[0]; bs = []
        for _ in range(2000):
            s = np.concatenate([ix[v] for v in rng.choice(uv, len(uv))]); s = s[msk[s]]
            if len(s) > 10:
                bs.append(pearsonr(P[a][s], yt[s])[0] - pearsonr(P[b][s], yt[s])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi, 2.8 * np.std(bs)
    d0, lo, hi, mde = delta("MPE", "MFR"); dl = delta("MPE", "MFR", dt < 4)[0]; dh = delta("MPE", "MFR", dt >= 4)[0]
    ok_ = d0 >= 0.02 and lo > 0 and dh > dl
    say(f"- PRIMARY delta_r = r(MPE) - r(MFR) = {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] (MDE {mde:.3f}); < 4 s {dl:+.3f}, >= 4 s {dh:+.3f} -> {'supports' if ok_ else 'no'}")
    for a, b, nm in (("ME", "MF", "recency (end vs first)"), ("MP", "MR", "peak (peak vs random)")):
        x = delta(a, b); say(f"- {nm}: {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]")
    return ok_


Zs, Ls, Zm, Lm = sys.argv[2:6]
say("# P10: peak-end pooling premise test")
v1 = run("CH-SIMS", extract("sims", Zs, Ls)); v2 = run("CMU-MOSI", extract("mosi", Zm, Lm))
say(f"\n**Peak-end premise {'HOLDS' if v1 and v2 else 'DOES NOT HOLD'}** (CH-SIMS {'yes' if v1 else 'no'}, MOSI {'yes' if v2 else 'no'})")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
