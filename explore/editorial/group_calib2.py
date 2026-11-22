"""E' grouping calibration v2 on CH-SIMS (true video_id known). Run 1 (CLIP-L middle frame, max-ARI threshold) gave ARI 0.266,
purity 0.633: too weak to control source leakage. v2 chooses the procedure by the quantity that matters:
  LEAKAGE = fraction of clips whose true video also appears in the training part of a 5-fold grouped CV (mean over 3 seeds).
Candidates: agglomerative clustering (cosine, average linkage) on
  (spk) WavLM-base-plus-sv x-vector of the audio, (clip) CLIP ViT-B/32 middle frame, (both) concat of the two L2-normalised;
  distance thresholds 0.10 ... 0.70. Constraint: largest group <= 20% of clips and >= 25 groups.
Rule FIXED BEFORE MER2024: pick the candidate with minimum LEAKAGE under the constraint; ties -> fewer groups.
Usage: python -I group_calib2.py <sims Raw.zip> <label.csv> <out.md> <emb_cache.npz>
"""
import os, subprocess, sys, tempfile, zipfile
import numpy as np, pandas as pd, soundfile as sf, torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, completeness_score

torch.set_num_threads(4)
ZIP, LAB, OUT, CACHE = sys.argv[1:5]


def embed():
    from transformers import AutoFeatureExtractor, WavLMForXVector, CLIPModel, CLIPProcessor
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    z = zipfile.ZipFile(ZIP); names = set(z.namelist()); L = pd.read_csv(LAB, dtype={"clip_id": str, "video_id": str})
    S, C, V = [], [], []
    with tempfile.TemporaryDirectory() as td:
        for k, (v, c) in enumerate(zip(L.video_id, L.clip_id)):
            p = next((f"Raw/{v}/{cc}.mp4" for cc in (c, c.zfill(4)) if f"Raw/{v}/{cc}.mp4" in names), None)
            if p is None:
                continue
            f = os.path.join(td, "c.mp4"); open(f, "wb").write(z.read(p)); w = os.path.join(td, "a.wav"); img = os.path.join(td, "f.jpg")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f, "-ac", "1", "-ar", "16000", w], check=False)
            dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f], capture_output=True, text=True).stdout.strip()
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{(float(dur) / 2 if dur else 0.5):.2f}", "-i", f, "-frames:v", "1", img], check=False)
            if not (os.path.exists(w) and os.path.exists(img)):
                continue
            x, _ = sf.read(w, dtype="float32"); x = x[: 16000 * 20]
            with torch.no_grad():
                s = sv(**fe(x, sampling_rate=16000, return_tensors="pt")).embeddings[0].numpy()
                e = cm.get_image_features(**cp(images=Image.open(img).convert("RGB"), return_tensors="pt"))[0].numpy()
            S.append(s / np.linalg.norm(s)); C.append(e / np.linalg.norm(e)); V.append(v); os.remove(img); os.remove(w)
            if k % 300 == 0:
                print(k, flush=True)
    np.savez_compressed(CACHE, spk=np.stack(S), clip=np.stack(C), vid=np.array(V))


def grouped_folds(g, k, rng):
    groups = rng.permutation(np.unique(g)); sz = {v: (g == v).sum() for v in groups}; load = np.zeros(k); fold_of = {}
    for v in sorted(groups, key=lambda v: -sz[v] + rng.random() * 1e-3):
        f = int(np.argmin(load)); fold_of[v] = f; load[f] += sz[v]
    return np.array([fold_of[v] for v in g])


def leakage(g, vid):
    out = []
    for seed in range(3):
        fid = grouped_folds(g, 5, np.random.default_rng(seed)); leak = 0
        for f in range(5):
            tr_v = set(vid[fid != f]); leak += np.isin(vid[fid == f], list(tr_v)).sum()
        out.append(leak / len(g))
    return float(np.mean(out))


if not os.path.exists(CACHE):
    embed()
D = np.load(CACHE, allow_pickle=True); vid = D["vid"]
feats = {"spk": D["spk"], "clip": D["clip"], "both": np.concatenate([D["spk"], D["clip"]], 1) / np.sqrt(2)}
rows = ["# Grouping calibration v2 on CH-SIMS (n=%d clips, %d true videos)" % (len(vid), len(set(vid))),
        "| features | threshold | #groups | largest group share | ARI | completeness | LEAKAGE (5-fold grouped CV) |", "|---|---|---|---|---|---|---|"]
best = None
for fn, X in feats.items():
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        g = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=th).fit_predict(X)
        ng = len(np.unique(g)); share = np.bincount(g).max() / len(g)
        lk = leakage(g, vid) if ng >= 5 else float("nan")
        rows.append(f"| {fn} | {th:.2f} | {ng} | {share:.3f} | {adjusted_rand_score(vid, g):.3f} | {completeness_score(vid, g):.3f} | {lk:.3f} |")
        if ng >= 25 and share <= 0.20 and not np.isnan(lk) and (best is None or (lk, ng) < (best[0], best[3])):
            best = (lk, fn, th, ng)
rows.append(f"\nRandom 5-fold CV leakage for reference: {leakage(np.arange(len(vid)), vid):.3f}")
rows.append(f"Chosen (min leakage, largest group <= 20%, >= 25 groups): features={best[1]}, threshold={best[2]:.2f}, groups={best[3]}, leakage={best[0]:.3f}" if best else "No candidate satisfies the constraint")
open(OUT, "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
