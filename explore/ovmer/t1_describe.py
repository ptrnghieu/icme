"""Tuning tier 1 (and the tier-0 smoke test): descriptions from Qwen2.5-Omni (thinker only) under one sampling configuration.

Same inputs and prompt as p17_describe.py (P17, 7B): official OV-MER description prompt with the English subtitle, official
Qwen2.5-Omni system prompt, frames read by qwen-omni-utils with its default sampling, audio of the video track interleaved with
the frames (use_audio_in_video), max_new_tokens 512.
Differences from p17_describe.py, all intended:
  - --model / --dtype (Kaggle T4: Qwen2.5-Omni-3B, float16; the T4 has no bfloat16);
  - --cfg picks the sampling configuration (CFGS below). transformers fills an unset top_k with 50
    (GenerationConfig._get_default_generation_params, read in the 5.24.0 wheel on 2026-11-28), so P17 sampled with
    temperature 1.0, top_p 0.95 AND top_k 50; S0 sets these three explicitly. top_k 0 disables top-k;
    top_p 1.0 disables top-p;
  - --stop adds stop strings at invented dialogue turns. Each one is a literal case of the extraction cut rule
    (p17_extract_vllm.TURN: newline, optional spaces, Human|User|Assistant, colon), so the text kept by the cut has the same
    distribution with or without them; they only save decoding time;
  - --greedy 0 skips the greedy description (it changes no CONSTRUCT selection: 0/532 OV-MERD+ clips, 2026-11-28);
  - per sample, the number of generated tokens before the end-of-turn token (n_tok) and whether the model ended its turn
    (ended) are stored;
  - eos_token_id = [<|im_end|>, <|endoftext|>] and pad_token_id = <|endoftext|> are passed to generate. The Omni
    generation_config.json sets no eos_token_id, so without them generate never stops at the end of the answer: every sequence runs
    to max_new_tokens and, after its own end token, writes invented dialogue turns (tier 0, 2026-11-28: every one of 864 samples held
    >= 1 <|endoftext|>; P17 used the same code path). With them, stop strings also take effect (finished rows are padded);
  - the batch of samples restarts at --chunk for every clip; a clip that runs out of memory at batch 1 is retried at lower video
    resolution (PIXEL_LEVELS), recorded as max_pixels.
Seeds: torch.manual_seed(1000 + s) before the s-th block of 16 samples, as in P17 (common random numbers across configs).
Resume: the last ok record per key counts; a clip with fewer than n_samples samples gets the missing blocks.
Output lines: {key, ds, ok, cfg, model, greedy, greedy_n_tok, greedy_ended, samples, n_tok, ended, seeds, sec, prompt_tokens,
              chunk, max_pixels}.
Usage: python t1_describe.py --items items.jsonl --out desc_S0_0.jsonl --cfg S0 --n_samples 32 --greedy 1
        [--shard i/n] [--limit N] [--hours H] [--model ...] [--dtype float16]
"""
import argparse, json, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p13_mbr import PROMPT_OFFICIAL, SYSTEM  # noqa: E402
from p17_describe import load_audio, load_done  # noqa: E402

CFGS = {
    "S0": dict(temperature=1.0, top_p=0.95, top_k=50),                 # P17 as it ran
    "S1": dict(temperature=0.7, top_p=0.95, top_k=50),                 # lower temperature only
    "S2": dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05),       # min-p sampling alone
    "S3": dict(temperature=1.0, top_p=1.0, top_k=0, epsilon_cutoff=3e-4),  # epsilon sampling alone
}
END_TOKENS = ["<|im_end|>", "<|endoftext|>"]  # end-of-turn ids passed to generate (the Omni generation_config.json sets none)
# None = qwen-omni-utils default frame size; smaller levels only for clips that run out of memory at batch 1 (same frames and
# timestamps, lower resolution; the level used is stored per clip so that every configuration can be compared on the same input)
PIXEL_LEVELS = [None, 256 * 28 * 28, 128 * 28 * 28]
STOPS = ["\nHuman:", "\nUser:", "\nAssistant:", "\n用户:", "\n助手:", "\n用户：", "\n助手："]
BLOCK = 16
MAX_NEW = 512


def main(a):
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    dt = getattr(torch, a.dtype)
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    items = [json.loads(l) for l in open(a.items)]
    if a.only_ds:
        items = [it for it in items if it["ds"] in a.only_ds.split(",")]
    if a.shard:
        i, n = map(int, a.shard.split("/")); items = items[i::n]
    items = items[: (a.limit or None)]
    done = load_done(a.out)
    todo = [it for it in items if len(done.get(it["key"], {}).get("samples", [])) < a.n_samples
            or (a.greedy and done.get(it["key"], {}).get("greedy") is None)]
    print(f"cfg {a.cfg} {CFGS[a.cfg]} | clips {len(items)} | already complete {len(items) - len(todo)} | to do {len(todo)}", flush=True)
    if not todo:
        return
    proc = Qwen2_5OmniProcessor.from_pretrained(a.model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(a.model, dtype=dt, attn_implementation="sdpa").to("cuda").eval()
    tk = proc.tokenizer
    eos = [tk.convert_tokens_to_ids(t) for t in END_TOKENS]; pad = tk.convert_tokens_to_ids("<|endoftext|>")
    assert all(isinstance(x, int) and x >= 0 for x in eos + [pad]), (eos, pad)
    gen = dict(max_new_tokens=MAX_NEW, eos_token_id=eos, pad_token_id=pad)
    if a.stop:
        gen.update(stop_strings=STOPS, tokenizer=tk)
    print("end-of-turn ids", eos, "| pad id", pad, flush=True)
    t0 = time.time(); n = nfail = 0; tmp = tempfile.mkdtemp()

    def ntok(rows):  # generated tokens before the first end-of-turn token; ended = the model ended its turn
        out = []
        for r in rows:
            r = [int(x) for x in r]; idx = [i for i, x in enumerate(r) if x in eos]
            out.append((idx[0] if idx else len(r), bool(idx)))
        return out

    def describe(it, prev, max_pixels):
        video = {"type": "video", "video": it["video"]}
        if max_pixels:
            video["max_pixels"] = max_pixels
        content = [video, {"type": "text", "text": PROMPT_OFFICIAL.format(sub=it.get("sub", ""))}]
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        _, img, vid = process_mm_info(conv, use_audio_in_video=False); aud = [load_audio(it["video"], tmp)]
        inp = proc(text=text, audio=aud, images=img, videos=vid, return_tensors="pt", padding=True,
                   load_audio_from_video=True, audio_from_video_indices=[0])
        inp = {k: (v.to("cuda").to(dt) if torch.is_floating_point(v) else v.to("cuda")) for k, v in inp.items()}
        L0 = inp["input_ids"].shape[1]; chunk = a.chunk
        greedy, greedy_n, greedy_end = prev.get("greedy"), prev.get("greedy_n_tok"), prev.get("greedy_ended")
        samples = list(prev.get("samples", [])); nt = list(prev.get("n_tok", [])); ended = list(prev.get("ended", []))
        seeds = list(prev.get("seeds", []))
        with torch.no_grad():
            if a.greedy and greedy is None:
                g = model.generate(**inp, use_audio_in_video=True, do_sample=False, **gen)
                greedy = proc.batch_decode(g[:, L0:], skip_special_tokens=True)[0]; greedy_n, greedy_end = ntok(g[:, L0:])[0]
            for s in range(len(samples) // BLOCK, a.n_samples // BLOCK):
                torch.manual_seed(1000 + s); got, gn = [], []
                while len(got) < BLOCK:
                    k = min(chunk, BLOCK - len(got))
                    try:
                        o = model.generate(**inp, use_audio_in_video=True, do_sample=True, num_return_sequences=k, **CFGS[a.cfg], **gen)
                        got += proc.batch_decode(o[:, L0:], skip_special_tokens=True); gn += ntok(o[:, L0:]); del o
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if chunk == 1:
                            raise
                        chunk = max(1, chunk // 2); print("OOM: chunk ->", chunk, it["key"], flush=True)
                samples += got; nt += [x for x, _ in gn]; ended += [e for _, e in gn]; seeds.append(s)
        return dict(key=it["key"], ds=it["ds"], ok=True, cfg=a.cfg, model=a.model, greedy=greedy, greedy_n_tok=greedy_n,
                    greedy_ended=greedy_end, samples=samples, n_tok=nt, ended=ended, seeds=seeds, prompt_tokens=L0, chunk=chunk,
                    max_pixels=max_pixels)

    with open(a.out, "a") as fo:
        for it in todo:
            if (time.time() - t0) / 3600 > a.hours:
                print("time budget reached; rerun to resume", flush=True); break
            t1 = time.time(); prev = done.get(it["key"]) or {}; rec = None
            levels = [prev["max_pixels"]] if prev.get("max_pixels") else PIXEL_LEVELS  # a resumed clip keeps its input size
            for mp in levels:
                try:
                    rec = describe(it, prev if mp == prev.get("max_pixels") else {}, mp); break
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache(); print("OOM at batch 1:", it["key"], "max_pixels", mp, "-> next level", flush=True)
                    rec = dict(key=it["key"], ds=it["ds"], ok=False, cfg=a.cfg, err=f"OutOfMemoryError: {str(e)[:200]}")
                except Exception as e:  # noqa: BLE001
                    rec = dict(key=it["key"], ds=it["ds"], ok=False, cfg=a.cfg, err=f"{type(e).__name__}: {str(e)[:300]}"); break
            rec["sec"] = round(time.time() - t1, 1)
            if not rec["ok"]:
                print("fail", it["key"], rec["err"], flush=True)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); n += 1; nfail += int(not rec["ok"])
            if n >= 5 and nfail == n:
                raise SystemExit("the first clips all failed: stopping (see the fail lines above)")
            torch.cuda.empty_cache()
            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)} clips, {el:.0f}s, {el / n:.1f} s/clip, fails {nfail}", flush=True)
    print("describe finished", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items"); ap.add_argument("--out"); ap.add_argument("--cfg", choices=sorted(CFGS), default="S0")
    ap.add_argument("--n_samples", type=int, default=16); ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1); ap.add_argument("--only_ds", default=""); ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--hours", type=float, default=99)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B"); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    assert a.n_samples % BLOCK == 0, "n_samples must be a multiple of 16"
    main(a)
