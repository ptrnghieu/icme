"""P0b (posed -> spontaneous): same frozen features as P0, driven by a manifest CSV.

Manifest columns: id, video (path to video file), audio (path to wav, or empty -> taken from video), text (may be empty),
label (string), mode (train/valid/test), group (speaker/actor/dialogue id for cluster bootstrap).
Usage: NT=2 python -I p0b_extract.py manifest.csv out.npz
"""
import os, subprocess, sys, tempfile, time, json
import numpy as np, pandas as pd, torch, cv2, soundfile as sf

torch.set_num_threads(int(os.environ.get("NT", "2")))
M = pd.read_csv(sys.argv[1], dtype=str).fillna("")
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, WavLMModel, AutoImageProcessor, AutoModelForImageClassification
import opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"); tm = AutoModel.from_pretrained("intfloat/multilingual-e5-base").eval()
wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); wm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
ip = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression"); vm = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").eval()
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

A_eg, A_wl, V_emb, V_log, nface, ok = [], [], [], [], [], []
t0 = time.time()
with tempfile.TemporaryDirectory() as td:
    for k, r in enumerate(M.itertuples()):
        wav = r.audio
        try:
            if not wav:
                wav = os.path.join(td, "a.wav")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", r.video, "-ac", "1", "-ar", "16000", wav], check=True)
            x, sr = sf.read(wav, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                w2 = os.path.join(td, "b.wav")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-ac", "1", "-ar", "16000", w2], check=True)
                x, sr = sf.read(w2, dtype="float32"); wav = w2
            eg = smile.process_signal(x, 16000).to_numpy()[0]
            with torch.no_grad():
                hs = wm(**wfe(x[: 16000 * 20], sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
            wl = torch.stack(hs[1:]).mean(0)[0].mean(0).numpy(); good = True
        except Exception:
            eg, wl, good = np.zeros(88), np.zeros(768), False
        cap = cv2.VideoCapture(r.video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        crops, nf = [], 0
        for frac in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * (n - 1))); ret, fr = cap.read()
            if not ret:
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
            ve, vl = o.hidden_states[-1][:, 0].mean(0).numpy(), o.logits.mean(0).numpy()
        else:
            ve, vl, good = np.zeros(768), np.zeros(7), False
        A_eg.append(eg); A_wl.append(wl); V_emb.append(ve); V_log.append(vl); nface.append(nf); ok.append(good)
        if k % 200 == 0:
            print(f"{k}/{len(M)} {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def emb(texts):
    out = []
    for i in range(0, len(texts), 32):
        b = tok(["query: " + t for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = tm(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.concatenate(out)


np.savez_compressed(sys.argv[2], id=M["id"].values, label=M["label"].values, mode=M["mode"].values, group=M["group"].values,
                    text=emb(M["text"].tolist()), a_egemaps=np.array(A_eg), a_wavlm=np.array(A_wl), v_emb=np.array(V_emb),
                    v_logits=np.array(V_log), n_face=np.array(nface), ok=np.array(ok))
print(json.dumps({"clips": len(M), "ok": int(np.sum(ok)), "face_frames_mean": float(np.mean(nface)), "sec": round(time.time() - t0)}))
