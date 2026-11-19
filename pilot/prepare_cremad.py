"""Download CREMA-D and preprocess it exactly like OGM-GE.

Steps
1. `git clone` CREMA-D with LFS smudging disabled (gets CSVs + LFS pointer stubs).
2. Download real WAV/FLV content for each pointer from media.githubusercontent.com
   (verified working on 2026-11-19), checking the byte size stated in the pointer.
   If the clone already holds real files (e.g. you ran `git lfs pull`), they are used as is.
3. Audio  -> log-magnitude STFT, identical to OGM-GE CramedDataset.__getitem__
            (sr=22050, tile x3 and cut to 3 s, clip to [-1, 1], n_fft=512, hop=353).
            Saved as float16 .npy, shape (257, 188).
4. Video  -> frames kept like OGM-GE video_preprocessing.py `video2frame_update`
            (first frame of every second when --fps 1). Saved as JPEG.
5. Writes <root>/manifest.csv and prints a summary of failures.

Usage
    python prepare_cremad.py --root /data/cremad --workers 16
    # smoke test on a small subset:
    python prepare_cremad.py --root /tmp/cremad_smoke --max_actors 10 --max_clips_per_actor 12
"""
import argparse
import concurrent.futures as cf
import csv
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

from common import CLASS_DICT, actor_of, emo_of

REPO = "https://github.com/CheyneyComputerScience/CREMA-D.git"
MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"


def is_pointer(path):
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def pointer_size(path):
    with open(path) as f:
        for line in f:
            if line.startswith("size "):
                return int(line.split()[1])
    return None


def download(url, dst, expected, retries=6):
    if os.path.exists(dst) and (expected is None or os.path.getsize(dst) == expected):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if expected is not None and os.path.getsize(tmp) != expected:
                raise IOError(f"size mismatch {os.path.getsize(tmp)} != {expected}")
            os.replace(tmp, dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[retry {k+1}/{retries}] {url}: {e}", file=sys.stderr)
            time.sleep(2 * (k + 1))
    return False


def spectrogram(wav_path):
    import librosa

    samples, _ = librosa.load(wav_path, sr=22050)
    res = np.tile(samples, 3)[: 22050 * 3]
    res[res > 1.0] = 1.0
    res[res < -1.0] = -1.0
    spec = librosa.stft(res, n_fft=512, hop_length=353)
    return np.log(np.abs(spec) + 1e-7).astype(np.float16)


def extract_frames(video_path, out_dir, kept_per_second=1):
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    fps = int(round(vid.get(cv2.CAP_PROP_FPS))) or 30
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(fps / kept_per_second))
    count, frame_id, saved = 0, 0, 0
    while True:
        ok, img = vid.read()
        if not ok:
            break
        if count % fps == 0:
            frame_id = 0
        if frame_id < interval * kept_per_second and frame_id % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"{count:05d}.jpg"), img)
            saved += 1
        frame_id += 1
        count += 1
        if n_frames and count >= n_frames:
            break
    vid.release()
    return saved


def process_clip(args):
    clip, wav, flv, root, fps = args
    out = {"clip": clip, "spec_ok": 0, "n_frames": 0, "err": ""}
    try:
        sp = os.path.join(root, "spec", clip + ".npy")
        if not os.path.exists(sp):
            np.save(sp, spectrogram(wav))
        out["spec_ok"] = 1
    except Exception as e:  # noqa: BLE001
        out["err"] += f"audio:{e};"
    try:
        fd = os.path.join(root, f"frames_fps{fps}", clip)
        n = len(os.listdir(fd)) if os.path.isdir(fd) else 0
        if n == 0:
            n = extract_frames(flv, fd, fps)
        out["n_frames"] = n
    except Exception as e:  # noqa: BLE001
        out["err"] += f"video:{e};"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4, help="parallel preprocessing")
    ap.add_argument("--fps", type=int, default=1, help="frames kept per second (OGM-GE uses 1)")
    ap.add_argument("--max_actors", type=int, default=0, help="smoke test: limit #actors")
    ap.add_argument("--max_clips_per_actor", type=int, default=0, help="smoke test: limit clips/actor")
    ap.add_argument("--raw_dir", default="", help="where to put the CREMA-D clone and raw downloads "
                    "(default: <root>); use a scratch disk on Kaggle, e.g. /tmp/cremad_raw")
    ap.add_argument("--delete_raw", action="store_true", help="delete raw WAV/FLV downloads after preprocessing")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    raw_dir = os.path.abspath(args.raw_dir) if args.raw_dir else root
    repo = os.path.join(raw_dir, "CREMA-D")
    os.makedirs(os.path.join(root, "spec"), exist_ok=True)
    if not os.path.isdir(repo):
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, repo], env=env)

    clips = sorted(f[:-4] for f in os.listdir(os.path.join(repo, "AudioWAV")) if f.endswith(".wav"))
    vids = {f[:-4] for f in os.listdir(os.path.join(repo, "VideoFlash")) if f.endswith(".flv")}
    clips = [c for c in clips if c in vids and emo_of(c) in CLASS_DICT]
    if args.max_actors or args.max_clips_per_actor:
        by_actor = {}
        for c in clips:
            by_actor.setdefault(actor_of(c), []).append(c)
        actors = sorted(by_actor)[: args.max_actors or None]
        clips = [c for a in actors for c in by_actor[a][: args.max_clips_per_actor or None]]
    print(f"clips selected: {len(clips)}")

    # 1) resolve sources and download LFS content where needed
    jobs, src = [], {}
    for c in clips:
        for sub, ext in (("AudioWAV", ".wav"), ("VideoFlash", ".flv")):
            p = os.path.join(repo, sub, c + ext)
            if is_pointer(p):
                dst = os.path.join(raw_dir, "raw", sub, c + ext)
                jobs.append((f"{MEDIA}/{sub}/{c}{ext}", dst, pointer_size(p)))
                src[(c, sub)] = dst
            else:
                src[(c, sub)] = p
    print(f"files to download: {len(jobs)}")
    failed_dl = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download, u, d, s): u for u, d, s in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            if not fu.result():
                failed_dl.append(futs[fu])
            if i % 500 == 0:
                print(f"  downloaded {i}/{len(jobs)}")
    if failed_dl:
        print(f"WARNING: {len(failed_dl)} downloads failed, e.g. {failed_dl[:3]}")

    # 2) preprocess
    work = [(c, src[(c, "AudioWAV")], src[(c, "VideoFlash")], root, args.fps) for c in clips]
    results = []
    with cf.ProcessPoolExecutor(args.procs) as ex:
        for i, r in enumerate(ex.map(process_clip, work, chunksize=16), 1):
            results.append(r)
            if i % 500 == 0:
                print(f"  processed {i}/{len(work)}")

    man = os.path.join(root, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "label", "actor", "spec_ok", "n_frames", "err"])
        for r in results:
            w.writerow([r["clip"], emo_of(r["clip"]), actor_of(r["clip"]), r["spec_ok"], r["n_frames"], r["err"]])
    ok = sum(1 for r in results if r["spec_ok"] and r["n_frames"] > 0)
    bad = [r for r in results if not (r["spec_ok"] and r["n_frames"] > 0)]
    print(f"usable clips: {ok}/{len(results)} -> {man}")
    for r in bad[:20]:
        print("  unusable:", r)

    # keep the crowd ratings next to the processed data (needed by human_reliability.py)
    shutil.copy(os.path.join(repo, "finishedResponses.csv"), os.path.join(root, "finishedResponses.csv"))
    if args.delete_raw:
        shutil.rmtree(os.path.join(raw_dir, "raw"), ignore_errors=True)
        print("deleted raw downloads")


if __name__ == "__main__":
    main()
