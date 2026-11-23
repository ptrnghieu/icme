"""P9 (branch E2, last pilot): does a standard audio emotion model follow the BACKGROUND MUSIC rather than the speaker?
Causal test by music swapping (Demucs htdemucs: vocals vs. accompaniment = drums+bass+other). CPU, frozen WavLM features.

PRE-REGISTERED (2026-11-23, before any swap result):
  Datasets and official splits (user decision 2026-11-23: community-standard splits):
    CH-SIMS: train = official train; evaluation pool = official valid+test; binary polarity (label != 0).
    M3ED:    train = 5,000 clips sampled (stratified, seed 0) from the official TV-independent train split; C chosen on 1,000 clips of
             the official val split; evaluation pool = 1,000 clips sampled (seed 0) from the official test split; 7 classes.
  Model: WavLM-base-plus layer-average mean (same function as P0) on the ORIGINAL audio -> logistic regression (balanced).
  Targets: evaluation clips with a music cue (AST max prob over music classes >= 0.2, the P6 rule) and a valenced label
    (CH-SIMS: all non-zero; M3ED: Happy = positive; Sad, Anger, Disgust, Fear = negative).
  For each target i: vocals v_i, accompaniment a_i. Donors from the same evaluation pool, with a music cue, from a different
    video / TV series, accompaniment rescaled to the RMS of a_i:
      OPP  = 3 donors whose label has the opposite valence;   SAME = 3 donors with the same label.
    Conditions: SELF = v_i + a_i (separation round trip), OPP_k = v_i + a_donor_k, SAME_k = v_i + a_donor_k, VOC = v_i alone.
  flip(cond) = fraction of donors for which the prediction differs from the SELF prediction (per target, then averaged).
  PRIMARY: delta_flip = mean_i[flip_OPP(i) - flip_SAME(i)], 95% cluster bootstrap over videos / series (2000 draws).
  E2 premise HOLDS iff delta_flip >= +5 points with CI excluding 0 on BOTH datasets. Otherwise branch E is closed for good.
  MDE (2.8 x bootstrap SE) is printed next to each estimate. Descriptive: accuracy under SELF / OPP / SAME / VOC; share of OPP
  flips that move to the donor's valence.
Usage: python -I p9_swap.py <workdir> <sims Raw.zip> <sims label.csv> <sims_p6.npz> <sims_p0.npz> <m3ed repo dir> <m3ed wav dir> [out.md]
"""
import json, os, subprocess, sys, tempfile, warnings, zipfile
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(4); warnings.filterwarnings("ignore")
W, SZ, SL, SP6, SP0, MREPO, MWAV = sys.argv[1:8]; OUT = sys.argv[8] if len(sys.argv) > 8 else None
os.makedirs(W, exist_ok=True); L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
from transformers import AutoFeatureExtractor, WavLMModel, ASTFeatureExtractor, ASTForAudioClassification
from demucs.pretrained import get_model
from demucs.apply import apply_model
wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); wm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
dem = get_model("htdemucs").eval(); VI = list(dem.sources).index("vocals")
import re
MU = re.compile(r"(?i)\bmusic\b|instrument|orchestra|song|piano|guitar|violin|drum|synthesizer")


@torch.no_grad()
def wavlm(x):
    x = x[: 16000 * 20]
    if len(x) < 16000:
        x = np.pad(x, (0, 16000 - len(x)))
    hs = wm(**wfe(x, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
    return torch.stack(hs[1:]).mean(0)[0].mean(0).numpy()


def load16(src, td):
    w = os.path.join(td, "a16.wav"); subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", w], check=False)
    return sf.read(w, dtype="float32")[0]


@torch.no_grad()
def separate(src, td):
    w = os.path.join(td, "a44.wav"); subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "2", "-ar", "44100", w], check=False)
    x = sf.read(w, dtype="float32", always_2d=True)[0]; t = torch.from_numpy(x.T.copy())
    ref = t.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8
    s = apply_model(dem, ((t - mu) / sd)[None], device="cpu", split=True, overlap=0.25, progress=False)[0] * sd + mu
    s = s.numpy(); voc = s[VI].mean(0); acc = s[[i for i in range(4) if i != VI]].sum(0).mean(0)
    return resample_poly(voc, 160, 441).astype(np.float32), resample_poly(acc, 160, 441).astype(np.float32)


def remix(v, a, ref_rms):
    n = len(v); a = np.resize(a, n) if len(a) > 0 else np.zeros(n, np.float32)
    r = np.sqrt(np.mean(a ** 2)) + 1e-8
    return (v + a * (ref_rms / r)).astype(np.float32)


def fit(Xtr, ytr, Xva, yva):
    from sklearn.metrics import balanced_accuracy_score
    sc = StandardScaler().fit(Xtr); best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        s = balanced_accuracy_score(yva, m.predict(sc.transform(Xva)))
        if best is None or s > best[0]:
            best = (s, m)
    return lambda X: best[1].predict(sc.transform(np.atleast_2d(X)))


def run_swaps(name, targets, pool, getsrc, label, valence, cluster, predict):
    """targets/pool: lists of ids; returns per-target dict."""
    if os.environ.get("P9_LIMIT"):  # local smoke test only
        n = int(os.environ["P9_LIMIT"]); targets, pool = targets[:n], pool[: 4 * n]
    cache = os.path.join(W, f"{name}_sep.npz"); sep = dict(np.load(cache, allow_pickle=True)["d"].item()) if os.path.exists(cache) else {}
    rng = np.random.default_rng(0); res = []
    with tempfile.TemporaryDirectory() as td:
        need = sorted(set(targets) | set(pool))
        for k, i in enumerate(need):
            if i not in sep:
                try:
                    sep[i] = separate(getsrc(i, td), td)
                except Exception:  # noqa: BLE001
                    sep[i] = None
            if k % 100 == 0:
                np.savez_compressed(cache, d=np.array(sep, dtype=object)); print(name, "separated", k, "/", len(need), flush=True)
        np.savez_compressed(cache, d=np.array(sep, dtype=object))
        for t_i, i in enumerate(targets):
            if sep.get(i) is None:
                continue
            v, a = sep[i]; rr = np.sqrt(np.mean(a ** 2)) + 1e-8
            cand = [j for j in pool if j != i and sep.get(j) is not None and cluster[j] != cluster[i]]
            opp = [j for j in cand if valence[j] == -valence[i]]; same = [j for j in cand if label[j] == label[i]]
            if len(opp) < 3 or len(same) < 3:
                continue
            o = rng.choice(opp, 3, replace=False); s_ = rng.choice(same, 3, replace=False)
            p_self = predict(wavlm(remix(v, a, rr)))[0]; p_voc = predict(wavlm(v))[0]
            p_o = [predict(wavlm(remix(v, sep[j][1], rr)))[0] for j in o]; p_s = [predict(wavlm(remix(v, sep[j][1], rr)))[0] for j in s_]
            res.append(dict(id=i, cl=cluster[i], y=label[i], val=valence[i], self=p_self, voc=p_voc, opp=p_o, same=p_s, opp_val=[valence[j] for j in o]))
            if t_i % 50 == 0:
                print(name, "swapped", t_i, "/", len(targets), flush=True)
    return res


def report(name, res, valmap):
    say(f"\n## {name} (targets with music and valenced label: {len(res)})")
    if len(res) < 2:
        say("- too few targets with enough donors"); return False
    fo = np.array([np.mean([p != r["self"] for p in r["opp"]]) for r in res]); fs = np.array([np.mean([p != r["self"] for p in r["same"]]) for r in res])
    d = 100 * (fo - fs); cl = np.array([r["cl"] for r in res]); uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    rng = np.random.default_rng(1); bs = [d[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); mde = 2.8 * np.std(bs); ok = d.mean() >= 5 and lo > 0
    say(f"- flip rate OPP {100 * fo.mean():.1f}% vs SAME {100 * fs.mean():.1f}% -> delta_flip = {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] (MDE {mde:.1f}; clusters {len(uc)}) -> {'supports' if ok else 'no'}")
    y = np.array([r["y"] for r in res])
    acc = lambda key: 100 * np.mean([np.mean([p == r["y"] for p in (r[key] if isinstance(r[key], list) else [r[key]])]) for r in res])  # noqa: E731
    say(f"- accuracy: SELF {acc('self'):.1f} | SAME {acc('same'):.1f} | OPP {acc('opp'):.1f} | VOC (vocals only) {acc('voc'):.1f}")
    tow = [valmap(p) == ov for r in res for p, ov in zip(r["opp"], r["opp_val"]) if p != r["self"]]
    if tow:
        say(f"- share of OPP flips landing on the donor's valence: {100 * np.mean(tow):.1f}% (n={len(tow)})")
    return ok


# ------------------------------------------------------------------------------------------------ CH-SIMS
import pandas as pd
Lb = pd.read_csv(SL, dtype={"clip_id": str, "video_id": str}); z = zipfile.ZipFile(SZ); names = set(z.namelist())
P0 = np.load(SP0, allow_pickle=True); P6 = np.load(SP6, allow_pickle=True); labs6 = list(P6["labels"]); mu6 = [i for i, l in enumerate(labs6) if MU.search(l)]
key = lambda v, c: f"{v}/{c}"  # noqa: E731
tagmap = {key(v, c): P6["tags"][i] for i, (v, c) in enumerate(zip(P6["video_id"], P6["clip_id"])) if P6["ok"][i]}
okp = P0["ok"].astype(bool) & (P0["label"].astype(float) != 0)
ids0 = [key(v, c) for v, c in zip(P0["video_id"], P0["clip_id"])]
tr = okp & (P0["mode"] == "train"); X0 = P0["a_wavlm"].astype(float); y0 = (P0["label"].astype(float) > 0).astype(int)
va = okp & (P0["mode"] == "valid"); pred_s = fit(X0[tr], y0[tr], X0[va], y0[va])
ev = [ids0[i] for i in np.where(okp & np.isin(P0["mode"], ["valid", "test"]))[0]]
lab_s = {ids0[i]: int(y0[i]) for i in np.where(okp)[0]}; val_s = {k: (1 if v == 1 else -1) for k, v in lab_s.items()}
clu_s = {k: k.split("/")[0] for k in lab_s}
mus = [k for k in ev if k in tagmap and tagmap[k][mu6].max() >= 0.2]


def src_sims(k, td):
    v, c = k.split("/"); p = next(f"Raw/{v}/{cc}.mp4" for cc in (c, c.zfill(4)) if f"Raw/{v}/{cc}.mp4" in names)
    f = os.path.join(td, "c.mp4"); open(f, "wb").write(z.read(p)); return f


say("# P9: music-swap test (branch E2)")
res_s = run_swaps("chsims", mus, mus, src_sims, lab_s, val_s, clu_s, pred_s)
ok_s = report("CH-SIMS (official split; eval = valid+test)", res_s, lambda p: 1 if p == 1 else -1)

# ------------------------------------------------------------------------------------------------ M3ED
A = json.load(open(os.path.join(MREPO, "annotation.json")))
spl = {s: [l.strip() for l in open(os.path.join(MREPO, f"splitInfo/movie_list_{s}.txt")) if l.strip()] for s in ("train", "val", "test")}
CL = ["Neutral", "Happy", "Sad", "Anger", "Disgust", "Fear", "Surprise"]; VAL = {"Happy": 1, "Sad": -1, "Anger": -1, "Disgust": -1, "Fear": -1}
rows = {s: [] for s in spl}
for s, mvs in spl.items():
    for mv in mvs:
        for dia in A.get(mv, {}).values():
            for uid, u in dia["Dialog"].items():
                f = os.path.join(MWAV, f"{u['Speaker']}_{uid}.wav")
                if os.path.exists(f):
                    rows[s].append((uid, f, u["EmoAnnotation"]["final_main_emo"], mv))
rng = np.random.default_rng(0)


def sample(rs, n):
    rs = list(rs); idx = rng.choice(len(rs), min(n, len(rs)), replace=False); return [rs[i] for i in sorted(idx)]


feat_cache = os.path.join(W, "m3ed_trainval_wavlm.npz")
NS = (60, 30, 80) if os.environ.get("P9_LIMIT") else (5000, 1000, 1000)
trs, vas, tes = sample(rows["train"], NS[0]), sample(rows["val"], NS[1]), sample(rows["test"], NS[2])
if os.path.exists(feat_cache):
    F = np.load(feat_cache, allow_pickle=True); Xtr, Xva = F["Xtr"], F["Xva"]
else:
    with tempfile.TemporaryDirectory() as td:
        Xtr = np.stack([wavlm(load16(f, td)) for _, f, _, _ in trs]); print("m3ed train feats done", flush=True)
        Xva = np.stack([wavlm(load16(f, td)) for _, f, _, _ in vas])
    np.savez_compressed(feat_cache, Xtr=Xtr, Xva=Xva)
pred_m = fit(Xtr, np.array([CL.index(e) for _, _, e, _ in trs]), Xva, np.array([CL.index(e) for _, _, e, _ in vas]))
ast_fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"); ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
mu_ast = [i for i in range(ast.config.num_labels) if MU.search(ast.config.id2label[i])]
src_m = {uid: f for uid, f, _, _ in tes}; lab_m = {uid: CL.index(e) for uid, _, e, _ in tes}; clu_m = {uid: mv for uid, _, _, mv in tes}
val_m = {uid: VAL.get(e, 0) for uid, _, e, _ in tes}
tagc = os.path.join(W, "m3ed_test_music.json")
if os.path.exists(tagc):
    music = json.load(open(tagc))
else:
    music = {}
    with tempfile.TemporaryDirectory() as td, torch.no_grad():
        for uid, f, _, _ in tes:
            x = load16(f, td); wins = [x[i:i + 160000] for i in range(0, min(len(x), 480000), 160000)]; wins = [w for w in wins if len(w) >= 1600] or [np.pad(x, (0, 1600))]
            music[uid] = float(torch.sigmoid(ast(**ast_fe(wins, sampling_rate=16000, return_tensors="pt")).logits).max(0).values[mu_ast].max())
    json.dump(music, open(tagc, "w"))
tg_m = [uid for uid, _, e, _ in tes if music[uid] >= 0.2 and VAL.get(e, 0) != 0]
pool_m = [uid for uid, _, e, _ in tes if music[uid] >= 0.2 and VAL.get(e, 0) != 0]
say(f"\n- M3ED evaluation sample: {len(tes)} test clips, music cue {100 * np.mean([music[u] >= 0.2 for u, _, _, _ in tes]):.1f}%")
res_m = run_swaps("m3ed", tg_m, pool_m, lambda u, td: src_m[u], lab_m, val_m, clu_m, pred_m)
ok_m = report("M3ED (official TV-independent split; eval = test sample)", res_m, lambda p: VAL.get(CL[p], 0))
say(f"\n**E2 premise {'HOLDS' if ok_s and ok_m else 'DOES NOT HOLD -> branch E closed for good'}** (CH-SIMS {'yes' if ok_s else 'no'}, M3ED {'yes' if ok_m else 'no'})")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
