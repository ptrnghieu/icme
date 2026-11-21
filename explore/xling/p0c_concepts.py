"""P0c: 'affect concept' features for idea H (semantic outputs of external emotion models, none trained on our eval sets).

audio concepts: superb/wav2vec2-base-superb-er logits (IEMOCAP-trained, 4 classes)
text concepts : cardiffnlp/twitter-xlm-roberta-base-sentiment logits (tweets, 3 classes, multilingual)
(face concepts = the 7 FER logits already in P0/P0b npz, from trpakov/vit-face-expression, FER2013-trained)

Usage:
  zip mode     : python -I p0c_concepts.py zip  Raw.zip label.csv out.npz      (MOSI / CH-SIMS, same keys as P0)
  manifest mode: python -I p0c_concepts.py man  manifest.csv out.npz          (CREMA-D / MELD, same ids as P0b)
"""
import os, subprocess, sys, tempfile, zipfile
import numpy as np, pandas as pd, torch, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(int(os.environ.get("NT", "2")))
mode = sys.argv[1]
if mode == "zip":
    src, lab, out = sys.argv[2:5]
else:
    src, out = sys.argv[2:4]; lab = None
fe = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
am = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-er").eval()
tt = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
tm = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").eval()


def audio_logits(wav):
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    with torch.no_grad():
        return am(**fe(x[: 16000 * 20], sampling_rate=16000, return_tensors="pt")).logits[0].numpy()


A, keys, texts = [], [], []
with tempfile.TemporaryDirectory() as td:
    wav = os.path.join(td, "a.wav")
    if mode == "zip":
        L = pd.read_csv(lab, dtype={"clip_id": str, "video_id": str}); z = zipfile.ZipFile(src); names = set(z.namelist())
        for k, r in enumerate(L.itertuples()):
            p = next((f"Raw/{r.video_id}/{c}.mp4" for c in (r.clip_id, r.clip_id.zfill(4), str(int(r.clip_id)) if r.clip_id.isdigit() else r.clip_id)
                      if f"Raw/{r.video_id}/{c}.mp4" in names), None)
            mp4 = os.path.join(td, "c.mp4"); open(mp4, "wb").write(z.read(p))
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4, "-ac", "1", "-ar", "16000", wav], check=False)
            try:
                A.append(audio_logits(wav))
            except Exception:
                A.append(np.full(4, np.nan))
            keys.append(f"{r.video_id}_{r.clip_id}"); texts.append(str(r.text))
            if k % 500 == 0:
                print(k, flush=True)
    else:
        M = pd.read_csv(src, dtype=str).fillna("")
        for k, r in enumerate(M.itertuples()):
            w = r.audio
            if not w:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", r.video, "-ac", "1", "-ar", "16000", wav], check=False); w = wav
            try:
                A.append(audio_logits(w))
            except Exception:
                A.append(np.full(4, np.nan))
            keys.append(r.id); texts.append(r.text)
            if k % 500 == 0:
                print(k, flush=True)
T = []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        b = tt([t if t else "." for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        T.append(tm(**b).logits.numpy())
np.savez_compressed(out, key=np.array(keys), a_concept=np.array(A), t_concept=np.concatenate(T),
                    a_labels=np.array([am.config.id2label[i] for i in range(4)]), t_labels=np.array([tm.config.id2label[i] for i in range(3)]))
print("done", len(keys), "nan audio:", int(np.isnan(np.array(A)).any(1).sum()))
