"""Kaggle helpers for the CONSTRUCT tuning tiers 0-1 (notebooks 12_t0_smoke.ipynb and 13_t1_generate.ipynb).

The notebook process never imports torch: all GPU work runs in subprocesses, so a torch upgrade made by setup() takes effect
without restarting the kernel.
Data: MER2025-FG only (p17_prepare.py --sets MER2025-FG). OV-MERD+ media and labels are never extracted here.
Jobs: "CFG:i/n[:limit]" = sampling configuration CFG on shard i of n (optionally the first `limit` clips of that shard).
  S0 gets 32 samples and the greedy description (the baseline row and the N curve); S1-S3 get 16 samples and no greedy.
Outputs go to /kaggle/working (desc_<CFG>_<i>of<n>.jsonl, logs, t*_out.tar); big files stay in WORK (/tmp).
CLI (used by run_queues): python kaggle_t1.py queue --gpu G --jobs "S0:0/2,S1:0/2" --items I --kout K --deadline EPOCH
"""
import argparse, glob, json, os, shutil, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = {"S0": 32, "S1": 16, "S2": 16, "S3": 16}
GREEDY = {"S0": 1, "S1": 0, "S2": 0, "S3": 0}
OMNI = "Qwen/Qwen2.5-Omni-3B"; EXTRACTOR = "Qwen/Qwen2.5-7B-Instruct"


def sh(cmd, check=True):
    print("$", cmd if len(cmd) < 200 else cmd[:200] + " ...", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def out_name(job):
    cfg, shard = job.split(":")[:2]
    return f"desc_{cfg}_{shard.replace('/', 'of')}.jsonl"


def setup(work):
    """Python packages for the describe step (main environment) and version checks. Returns a dict of versions."""
    sh(f"{sys.executable} -m pip -q install -U 'transformers==5.24.0' accelerate 'qwen-omni-utils==0.0.9' librosa "
       f"soundfile av 'huggingface_hub>=1.31' pandas", check=False)
    sh(f"{sys.executable} -m pip -q install decord", check=False)  # as P17 (qwen-omni-utils[decord]); torchvision reader if absent
    v = subprocess.run([sys.executable, "-c", "import sys,torch;print(sys.version.split()[0], torch.__version__)"],
                       capture_output=True, text=True).stdout.split()
    py, tv = v[0], v[1]
    if tuple(int(x) for x in tv.split("+")[0].split(".")[:2]) < (2, 6):
        print("torch", tv, "< 2.6 (transformers 5.24 needs >= 2.6): installing torch 2.8 cu126 (supports the T4)", flush=True)
        sh(f"{sys.executable} -m pip -q install 'torch==2.8.*' 'torchvision==0.23.*' 'torchaudio==2.8.*' "
           f"--index-url https://download.pytorch.org/whl/cu126")
    chk = ("import sys,torch,transformers,qwen_omni_utils;"
           "from qwen_omni_utils.v2_5.vision_process import get_video_reader_backend as g;"
           "print('python',sys.version.split()[0],'| torch',torch.__version__,'cuda',torch.version.cuda,'| gpus',torch.cuda.device_count(),"
           "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'| transformers',transformers.__version__,"
           "'| video reader',g())")
    out = subprocess.run([sys.executable, "-c", chk], capture_output=True, text=True)
    print(out.stdout, out.stderr[-1500:], flush=True)
    if out.returncode:
        raise SystemExit("environment check failed (see above)")
    assert tuple(int(x) for x in py.split(".")[:2]) >= (3, 11), f"python {py}: transformers 5.24 needs >= 3.11"
    return out.stdout.strip()


def secrets():
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        raise SystemExit("HF_TOKEN secret is not attached to THIS notebook: Add-ons -> Secrets -> tick HF_TOKEN") from None
    try:
        os.environ["MER2026_ZIP_PW"] = UserSecretsClient().get_secret("MER2026_ZIP_PW")  # never printed
    except Exception:
        pass


def prepare(work, models=(OMNI,)):
    """MER2025-FG items (English subtitles) and model weights. Returns the items path."""
    os.environ["HF_HOME"] = f"{work}/hf"
    from huggingface_hub import hf_hub_download, snapshot_download
    aff = f"{work}/affectgpt"
    if not os.path.isdir(aff):
        sh(f"git clone -q --depth 1 https://github.com/zeroQiaoba/AffectGPT {aff}")
    z = hf_hub_download(repo_id="MERChallenge/MER2026", repo_type="dataset", filename="video_7z/video_track2_train_human/video_split.zip",
                        local_dir=f"{work}/MER2026", token=os.environ["HF_TOKEN"])
    items = f"{work}/items_fg.jsonl"
    sh(f"{sys.executable} {HERE}/p17_prepare.py --affectgpt {aff} --mer26zip {z} --work {work}/media --out {items} --sets MER2025-FG")
    n = sum(1 for _ in open(items))
    assert all(json.loads(l)["ds"] == "MER2025-FG" for l in open(items)), "items must hold MER2025-FG only"
    print("MER2025-FG items:", n, flush=True)
    for m in models:
        snapshot_download(m)
    return items


def restore(kout, pattern="desc_*.jsonl"):
    """Copy outputs of earlier sessions (attached as notebook inputs) into kout, keeping the longer file per name."""
    for p in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True):
        b = os.path.basename(p); dst = os.path.join(kout, b)
        if b.startswith("._"):
            continue
        n_src = sum(1 for _ in open(p)); n_dst = sum(1 for _ in open(dst)) if os.path.exists(dst) else -1
        if n_src > n_dst:
            shutil.copy(p, dst); print(f"restored {b}: {n_src} lines (from {p})", flush=True)


def snapshot(kout, name):
    tmp = os.path.join(kout, name + ".tmp")
    with tarfile.open(tmp, "w") as t:
        for p in sorted(glob.glob(f"{kout}/*.jsonl") + glob.glob(f"{kout}/*.md") + glob.glob(f"{kout}/log_*.txt")):
            t.add(p, arcname=os.path.basename(p))
    os.replace(tmp, os.path.join(kout, name))


def run_queues(jobs, items, kout, budget_h, hard_h, t0, tar_name, extra=""):
    """jobs: {gpu: [job, ...]}. One subprocess per GPU runs its jobs in order. Polls every 5 min, snapshots every 20 min,
    kills everything at hard_h hours after t0."""
    deadline = t0 + budget_h * 3600; procs = {}
    for g, js in jobs.items():
        if not js:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, os.path.abspath(__file__), "queue", "--gpu", str(g), "--jobs", ",".join(js), "--items", items,
               "--kout", kout, "--deadline", str(deadline)] + (extra.split() if extra else [])
        procs[g] = subprocess.Popen(cmd, env=env, stdout=open(f"{kout}/log_gpu{g}.txt", "a"), stderr=subprocess.STDOUT)
    last_snap = time.time()
    while any(p.poll() is None for p in procs.values()):
        time.sleep(300)
        for g in procs:
            L = [l.strip() for l in open(f"{kout}/log_gpu{g}.txt", errors="ignore") if any(k in l for k in ("clips,", "== job", "fail", "Error", "OOM"))]
            print(f"[{(time.time() - t0) / 3600:.2f} h] GPU{g}:", L[-1] if L else "(starting)", flush=True)
        if time.time() - last_snap > 1200:
            snapshot(kout, tar_name); last_snap = time.time()
        if time.time() - t0 > hard_h * 3600:
            print("hard stop: killing the queues (outputs so far are kept; rerun to resume)", flush=True)
            for p in procs.values():
                p.kill()
            break
    snapshot(kout, tar_name)
    print("queue return codes:", {g: p.returncode for g, p in procs.items()}, flush=True)


def queue(a):
    for job in a.jobs.split(","):
        parts = job.split(":"); cfg, shard = parts[0], parts[1]; limit = parts[2] if len(parts) > 2 else "0"
        left = (a.deadline - time.time()) / 3600 - 0.15  # margin: model loading happens before the describe clock starts
        if left <= 0.05:
            print("== job", job, "skipped: no time left", flush=True); continue
        print(f"== job {job} | {left:.2f} h left", flush=True)
        r = subprocess.run([sys.executable, f"{HERE}/t1_describe.py", "--items", a.items, "--out", os.path.join(a.kout, out_name(job)),
                            "--cfg", cfg, "--n_samples", str(N_SAMPLES[cfg]), "--greedy", str(GREEDY[cfg]), "--shard", shard,
                            "--limit", limit, "--hours", f"{left:.3f}", "--model", a.model, "--chunk", str(a.chunk)])
        print(f"== job {job} return code {r.returncode}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["queue"])
    ap.add_argument("--gpu"); ap.add_argument("--jobs"); ap.add_argument("--items"); ap.add_argument("--kout")
    ap.add_argument("--deadline", type=float); ap.add_argument("--model", default=OMNI); ap.add_argument("--chunk", type=int, default=8)
    queue(ap.parse_args())
