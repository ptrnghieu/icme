"""P0 premise pilot: language-agnostic frozen features for MOSI (EN) / CH-SIMS (ZH).

Usage: python -I p0_extract.py --zip Raw.zip --labels label.csv --out feats.npz [--limit N]
Per clip: text = multilingual-e5-base mean-pooled; audio = eGeMAPSv02 functionals (88) + WavLM-base-plus
(mean over time of the layer-average); face = Haar face crop on 3 frames (25/50/75%) -> ViT FER model
(pooled embedding + 7 logits), averaged. Clips are read from the zip into a temp dir (untrusted data).
"""
import argparse, os, subprocess, tempfile, time, zipfile, json
import numpy as np, pandas as pd, torch, cv2

torch.set_num_threads(int(os.environ.get("NT","2")))
ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

L = pd.read_csv(a.labels, dtype={"clip_id": str, "video_id": str})
if a.limit:
    L = pd.concat([g.head(a.limit) for _, g in L.groupby("mode")])
z = zipfile.ZipFile(a.zip)
names = set(z.namelist())

from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, WavLMModel, AutoImageProcessor, AutoModelForImageClassification
import opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"); tm = AutoModel.from_pretrained("intfloat/multilingual-e5-base").eval()
wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); wm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
ip = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression"); vm = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").eval()
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def find(row):
    for c in (row.clip_id, str(int(row.clip_id)) if row.clip_id.isdigit() else row.clip_id, row.clip_id.zfill(4)):
        p = f"Raw/{row.video_id}/{c}.mp4"
        if p in names:
            return p
    return None


@torch.no_grad()
def text_emb(texts):
    out = []
    for i in range(0, len(texts), 32):
        b = tok(["query: " + str(t) for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = tm(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.concatenate(out)


rows, A_eg, A_wl, V_emb, V_log, nface, ok = [], [], [], [], [], [], []
t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    for k, row in enumerate(L.itertuples()):
        p = find(row)
        if p is None:
            ok.append(False); A_eg.append(np.zeros(88)); A_wl.append(np.zeros(768)); V_emb.append(np.zeros(768)); V_log.append(np.zeros(7)); nface.append(0); continue
        mp4 = os.path.join(td, "c.mp4"); open(mp4, "wb").write(z.read(p)); wav = os.path.join(td, "c.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4, "-ac", "1", "-ar", "16000", wav], check=False)
        try:
            eg = smile.process_file(wav).to_numpy()[0]
            import soundfile as sf
            x, sr = sf.read(wav, dtype="float32")
            x = x[: 16000 * 20]
            with torch.no_grad():
                hs = wm(**wfe(x, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
            wl = torch.stack(hs[1:]).mean(0)[0].mean(0).numpy()
        except Exception:
            eg, wl = np.zeros(88), np.zeros(768)
        cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        crops, nf = [], 0
        for frac in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * (n - 1))); r, fr = cap.read()
            if not r:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); fs = face.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
            if len(fs):
                x0, y0, w, h = max(fs, key=lambda f: f[2] * f[3]); nf += 1
                crop = fr[max(0, y0 - h // 8): y0 + h + h // 8, max(0, x0 - w // 8): x0 + w + w // 8]
            else:
                H, W = fr.shape[:2]; s = min(H, W); crop = fr[(H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        cap.release()
        if crops:
            with torch.no_grad():
                o = vm(**ip(crops, return_tensors="pt"), output_hidden_states=True)
            ve = o.hidden_states[-1][:, 0].mean(0).numpy(); vl = o.logits.mean(0).numpy()
        else:
            ve, vl = np.zeros(768), np.zeros(7)
        A_eg.append(eg); A_wl.append(wl); V_emb.append(ve); V_log.append(vl); nface.append(nf); ok.append(True)
        if k % 200 == 0:
            print(f"{k}/{len(L)} {time.time() - t0:.0f}s", flush=True)
T = text_emb(L["text"].tolist())
np.savez_compressed(a.out, video_id=L.video_id.values, clip_id=L.clip_id.values, label=L.label.values.astype(float),
                    mode=L["mode"].values, text=T, a_egemaps=np.array(A_eg), a_wavlm=np.array(A_wl), v_emb=np.array(V_emb),
                    v_logits=np.array(V_log), n_face=np.array(nface), ok=np.array(ok))
print(json.dumps({"clips": len(L), "ok": int(np.sum(ok)), "face_frames_mean": float(np.mean(nface)), "sec": round(time.time() - t0)}))
