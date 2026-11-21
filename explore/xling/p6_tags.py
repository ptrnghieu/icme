"""P6 extraction: AudioSet tags (AST, MIT/ast-finetuned-audioset-10-10-0.4593) for every clip, to measure post-production
("editorial") audio cues: soundtrack music, audience laughter/applause.
Usage:
  python -I p6_tags.py zip  <Raw.zip> <label.csv> <out.npz>      # MOSI / CH-SIMS (Raw/<video_id>/<clip_id>.mp4)
  python -I p6_tags.py meld <manifest.csv> <out.npz>              # MELD (manifest with video paths)
Audio: ffmpeg -> 16 kHz mono; up to 3 non-overlapping 10 s windows, sigmoid probabilities max-pooled over windows.
Untrusted media: decoded by ffmpeg into a temp dir only.
"""
import os, subprocess, sys, tempfile, time, zipfile
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

torch.set_num_threads(int(os.environ.get("NT", "4")))
M = "MIT/ast-finetuned-audioset-10-10-0.4593"
fe = ASTFeatureExtractor.from_pretrained(M); model = ASTForAudioClassification.from_pretrained(M).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]


@torch.no_grad()
def tags(wav):
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < 1600:
        return None
    wins = [x[i:i + 160000] for i in range(0, min(len(x), 480000), 160000)]
    wins = [w for w in wins if len(w) >= 1600]
    p = torch.sigmoid(model(**fe(wins, sampling_rate=16000, return_tensors="pt")).logits).max(0).values
    return p.numpy()


def run(items, out, meta):
    P, ok = [], []; t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for k, get in enumerate(items):
            wav = os.path.join(td, "c.wav")
            try:
                src = get(td)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav], check=False, timeout=120)
                p = tags(wav)
            except Exception:  # noqa: BLE001
                p = None
            ok.append(p is not None); P.append(p if p is not None else np.zeros(len(labels), np.float32))
            if k % 200 == 0:
                print(k, len(items), f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, tags=np.stack(P), ok=np.array(ok), labels=np.array(labels), **meta)


if sys.argv[1] == "zip":
    z = zipfile.ZipFile(sys.argv[2]); names = set(z.namelist())
    L = pd.read_csv(sys.argv[3], dtype={"clip_id": str, "video_id": str})

    def mk(v, c):
        def get(td):
            for cc in (c, str(int(c)) if c.isdigit() else c, c.zfill(4)):
                p = f"Raw/{v}/{cc}.mp4"
                if p in names:
                    f = os.path.join(td, "c.mp4"); open(f, "wb").write(z.read(p)); return f
            raise FileNotFoundError
        return get
    run([mk(v, c) for v, c in zip(L.video_id, L.clip_id)], sys.argv[4],
        dict(video_id=L.video_id.values, clip_id=L.clip_id.values, label=L.label.values, mode=L["mode"].values))
else:
    L = pd.read_csv(sys.argv[2])
    run([(lambda td, p=p: p) for p in L.video], sys.argv[3], dict(id=L.id.values, label=L.label.values, mode=L["mode"].values, group=L.group.values))
