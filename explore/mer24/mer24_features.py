"""MER2024 Train&Val feature cache (one GPU session) for CPU-only screening of many ideas afterwards.
Runs on the USER's Kaggle. Writes per-clip features ONLY to the cache directory on Kaggle (never shared; MER licence).

Encoders follow the MER2024 baselines (2404.17113, Tab. 4 and links): audio TencentGameMate/chinese-hubert-large, visual
openai/clip-vit-large-patch14, text hfl/chinese-roberta-wwm-ext-large; plus WavLM-base-plus and multilingual-e5-base (same as our
open-dataset harness), AST AudioSet tags, a WavLM-SV speaker x-vector and an EasyOCR subtitle flag (middle frame).
Per clip:
  a_hub   chinese-hubert-large, mean over time of the layer average (1024)        a_hub_seg  same per 1-s segment, <= 20 (float16)
  a_wavlm WavLM-base-plus, mean over time of the layer average (768)              tags       AST sigmoid probs, 527 (float16)
  spk     WavLM-base-plus-sv x-vector (512)                                         dur        seconds
  v_clip  CLIP-L/14 image embeddings of 8 uniformly sampled frames (8 x 768, float16, L2-normalised); n_frames
  sub     1 if EasyOCR finds a box in the bottom 35% of the middle frame with width >= 15% of the frame
Text features are computed in the screen stage from label-transcription.csv (batching).
Usage: python mer24_features.py --videos DIR --labels label-disdim.csv --cache DIR [--hours 9] [--limit N]
"""
import argparse, glob, os, subprocess, tempfile, time
import numpy as np

PART = int(os.environ.get("E_PART", "250"))


def load(dev):
    import torch, easyocr
    from transformers import (AutoFeatureExtractor, AutoModel, ASTFeatureExtractor, ASTForAudioClassification, WavLMModel,
                              WavLMForXVector, CLIPModel, CLIPProcessor, Wav2Vec2FeatureExtractor)
    m = dict(torch=torch, dev=dev)
    m["hub_fe"] = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-large")
    m["hub"] = AutoModel.from_pretrained("TencentGameMate/chinese-hubert-large").to(dev).eval()
    m["wfe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"); m["wavlm"] = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    m["sv_fe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); m["sv"] = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    m["ast_fe"] = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    m["ast"] = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
    m["clip_p"] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"); m["clip"] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    m["ocr"] = easyocr.Reader(["ch_sim", "en"], gpu=(dev == "cuda"), verbose=False)
    return m


def hidden_avg(model, fe, x, m):
    torch = m["torch"]
    with torch.no_grad():
        inp = {k: v.to(m["dev"]) for k, v in fe(x, sampling_rate=16000, return_tensors="pt").items()}
        hs = model(**inp, output_hidden_states=True).hidden_states
    return torch.stack(hs[1:]).mean(0)[0].float().cpu().numpy()      # [T, D] at 50 Hz


def audio_feats(m, x):
    torch = m["torch"]; x = x[: 16000 * 20]
    if len(x) < 3200:
        x = np.pad(x, (0, 3200 - len(x)))
    h = hidden_avg(m["hub"], m["hub_fe"], x, m)
    seg = np.stack([h[i:i + 50].mean(0) for i in range(0, len(h), 50)])[:20]
    w = hidden_avg(m["wavlm"], m["wfe"], x, m).mean(0)
    with torch.no_grad():
        s = m["sv"](**{k: v.to(m["dev"]) for k, v in m["sv_fe"](x, sampling_rate=16000, return_tensors="pt").items()}).embeddings[0]
        wins = [x[i:i + 160000] for i in range(0, min(len(x), 480000), 160000)]
        wins = [z for z in wins if len(z) >= 1600] or [x]
        p = torch.sigmoid(m["ast"](**{k: v.to(m["dev"]) for k, v in m["ast_fe"](wins, sampling_rate=16000, return_tensors="pt").items()}).logits).max(0).values
    s = torch.nn.functional.normalize(s.float(), dim=-1).cpu().numpy()
    pad = np.zeros((20, seg.shape[1]), np.float32); pad[: len(seg)] = seg
    return h.mean(0), pad.astype(np.float16), len(seg), w, s, p.float().cpu().numpy().astype(np.float16)


def visual_feats(m, mp4):
    import cv2
    from PIL import Image
    torch = m["torch"]
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; frs = []
    for frac in np.linspace(0.05, 0.95, 8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * (n - 1))); r, fr = cap.read()
        if r:
            frs.append(fr)
    cap.release()
    out = np.zeros((8, 768), np.float16); sub = 0.0
    if frs:
        with torch.no_grad():
            e = m["clip"].get_image_features(**{k: v.to(m["dev"]) for k, v in m["clip_p"](images=[Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frs], return_tensors="pt").items()})
        e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy(); out[: len(e)] = e
        mid = frs[len(frs) // 2]; H, W = mid.shape[:2]; hl, fl = m["ocr"].detect(mid)
        boxes = [(b[0], b[1], b[2], b[3]) for b in (hl[0] if hl else [])]
        sub = float(any(((y0 + y1) / 2 > 0.65 * H) and (x1 - x0 >= 0.15 * W) for x0, x1, y0, y1 in boxes))
    return out, len(frs), sub


def main(a):
    import pandas as pd, soundfile as sf, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [str(n) for n in pd.read_csv(a.labels)["name"]][: (a.limit or None)]
    vids = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.videos, "**", "*.*"), recursive=True)
            if p.lower().endswith((".mp4", ".avi", ".mkv"))}
    os.makedirs(a.cache, exist_ok=True); m = load(dev); t0 = time.time()
    keys = ["name", "ok", "a_hub", "a_hub_seg", "n_seg", "a_wavlm", "spk", "tags", "v_clip", "n_frames", "sub", "dur"]
    for start in range(0, len(names), PART):
        part = os.path.join(a.cache, f"part_{start // PART:04d}.npz")
        if os.path.exists(part):
            continue
        if (time.time() - t0) / 3600 > a.hours:
            print("time budget reached; rerun to resume", flush=True); return
        rec = {k: [] for k in keys}
        with tempfile.TemporaryDirectory() as td:
            for nm in names[start:start + PART]:
                p = vids.get(nm); w = os.path.join(td, "a.wav")
                try:
                    if p is None:
                        raise FileNotFoundError
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-ac", "1", "-ar", "16000", w], check=False, timeout=120)
                    x, _ = sf.read(w, dtype="float32")
                    if x.ndim > 1:
                        x = x.mean(1)
                    ah, seg, ns, aw, sp, tg = audio_feats(m, x); vc, nf, sub = visual_feats(m, p)
                    vals = dict(ok=True, a_hub=ah, a_hub_seg=seg, n_seg=ns, a_wavlm=aw, spk=sp, tags=tg, v_clip=vc, n_frames=nf, sub=sub, dur=len(x) / 16000)
                except Exception as e:  # noqa: BLE001
                    print("fail", type(e).__name__, flush=True)
                    vals = dict(ok=False, a_hub=np.zeros(1024), a_hub_seg=np.zeros((20, 1024), np.float16), n_seg=0, a_wavlm=np.zeros(768),
                                spk=np.zeros(512), tags=np.zeros(527, np.float16), v_clip=np.zeros((8, 768), np.float16), n_frames=0, sub=0.0, dur=0.0)
                rec["name"].append(nm)
                for k in keys[1:]:
                    rec[k].append(vals[k])
        np.savez_compressed(part, **{k: np.array(v) for k, v in rec.items()})
        print(f"part {start // PART} done, {min(start + PART, len(names))}/{len(names)} clips, {time.time() - t0:.0f}s", flush=True)
    print("extraction complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--videos"); ap.add_argument("--labels"); ap.add_argument("--cache")
    ap.add_argument("--hours", type=float, default=9.0); ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
