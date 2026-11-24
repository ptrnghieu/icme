"""P17 phase 1: emotion descriptions from Qwen2.5-Omni-7B (thinker only, bf16), zero-shot (runs on the user's rented GPU).

Per clip:
  - 1 greedy description;
  - n_seeds x 16 sampled descriptions (temperature 1.0, top_p 0.95; torch seed 1000 + s before the samples of seed s).
Prompt and inputs:
  - official OV-MER description prompt (p13_mbr.PROMPT_OFFICIAL) with the clip's subtitle;
  - official Qwen2.5-Omni system prompt;
  - video read by qwen-omni-utils with its default sampling;
  - audio taken from the video track (use_audio_in_video=True, the Qwen2.5-Omni recommended usage);
    --audio_mode wav instead passes a separately extracted 16 kHz track (fallback if the in-video path fails).
max_new_tokens 512. The raw text is stored; any cutting of invented dialogue turns happens at extraction (p17_extract_vllm.py).
Resume: the last ok record per key counts. A clip whose record has fewer than n_seeds x 16 samples gets the missing seeds.
Output lines: {key, ds, ok, greedy, samples, seeds, audio_mode, sec}.
Usage: python p17_describe.py --items items.jsonl --out desc.jsonl [--only_ds A,B] [--n_seeds 1] [--limit N] [--hours H]
        [--shard i/n] [--audio_mode video|wav]
"""
import argparse, json, os, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p13_mbr import PROMPT_OFFICIAL, SYSTEM  # noqa: E402

MODEL = "Qwen/Qwen2.5-Omni-7B"
N_PER_SEED, CHUNK = 16, int(os.environ.get("P17_CHUNK", "8"))


def load_done(path):
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("ok"):
                done[r["key"]] = r
    return done


def main(a):
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "needs a CUDA GPU with bf16"
    items = [json.loads(l) for l in open(a.items)]
    if a.only_ds:
        items = [it for it in items if it["ds"] in a.only_ds.split(",")]
    if a.shard:
        i, n = map(int, a.shard.split("/")); items = items[i::n]
    items = items[: (a.limit or None)]
    done = load_done(a.out); need = N_PER_SEED * a.n_seeds
    todo = [it for it in items if len(done.get(it["key"], {}).get("samples", [])) < need]
    print(f"clips {len(items)} | already complete {len(items) - len(todo)} | to do {len(todo)}", flush=True)
    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    in_video = a.audio_mode == "video"; gen = dict(max_new_tokens=512)
    t0 = time.time(); n = nfail = 0; tmp = tempfile.mkdtemp()
    with open(a.out, "a") as fo:
        for it in todo:
            if (time.time() - t0) / 3600 > a.hours:
                print("time budget reached; rerun to resume", flush=True); break
            t1 = time.time(); prev = done.get(it["key"])
            try:
                content = [{"type": "video", "video": it["video"]}]
                if not in_video:
                    wav = os.path.join(tmp, "a.wav")
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["video"], "-ac", "1", "-ar", "16000", wav], check=True, timeout=120)
                    content.append({"type": "audio", "audio": wav})
                content.append({"type": "text", "text": PROMPT_OFFICIAL.format(sub=it.get("sub", ""))})
                conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
                text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                aud, img, vid = process_mm_info(conv, use_audio_in_video=in_video)
                inp = proc(text=text, audio=aud, images=img, videos=vid, return_tensors="pt", padding=True, use_audio_in_video=in_video)
                inp = {k: (v.to("cuda").to(torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda")) for k, v in inp.items()}
                L0 = inp["input_ids"].shape[1]
                with torch.no_grad():
                    if prev:
                        greedy = prev["greedy"]; samples = list(prev["samples"]); seeds = list(prev.get("seeds", []))
                    else:
                        g = model.generate(**inp, use_audio_in_video=in_video, do_sample=False, **gen)
                        greedy = proc.batch_decode(g[:, L0:], skip_special_tokens=True)[0]; samples, seeds = [], []
                    for s in range(len(samples) // N_PER_SEED, a.n_seeds):
                        torch.manual_seed(1000 + s); got = []; chunk = CHUNK
                        while len(got) < N_PER_SEED:
                            k = min(chunk, N_PER_SEED - len(got))
                            try:
                                o = model.generate(**inp, use_audio_in_video=in_video, do_sample=True, temperature=1.0, top_p=0.95,
                                                   num_return_sequences=k, **gen)
                                got += proc.batch_decode(o[:, L0:], skip_special_tokens=True); del o
                            except torch.cuda.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                if chunk == 1:
                                    raise
                                chunk = max(1, chunk // 2)
                        samples += got; seeds.append(s)
                rec = dict(key=it["key"], ds=it["ds"], ok=True, greedy=greedy, samples=samples, seeds=seeds, audio_mode=a.audio_mode,
                           sec=round(time.time() - t1, 1), prompt_tokens=L0)
            except Exception as e:  # noqa: BLE001
                rec = dict(key=it["key"], ds=it["ds"], ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
                print("fail", it["key"], rec["err"], flush=True)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); n += 1; nfail += int(not rec["ok"])
            if n >= 5 and nfail == n:
                raise SystemExit("the first clips all failed: stopping (see the fail lines above)")
            torch.cuda.empty_cache()
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} clips, {el:.0f}s, {el / n:.1f} s/clip, fails {nfail}", flush=True)
    print("describe finished", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items"); ap.add_argument("--out"); ap.add_argument("--only_ds", default=""); ap.add_argument("--shard", default="")
    ap.add_argument("--n_seeds", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--hours", type=float, default=99)
    ap.add_argument("--audio_mode", default="video", choices=["video", "wav"])
    main(ap.parse_args())
