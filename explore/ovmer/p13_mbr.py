"""P13: pilot for idea F1 (ideas/cards_W_MERFG_2026-11-23.md): metric-aware, training-free decoding for open-vocabulary MER.
Runs on the USER's Kaggle (GPU). Writes only model outputs (emotion words) and aggregate scores; no media, no annotations.

PRE-REGISTERED (2026-11-23, before any result):
  Backbone: Qwen/Qwen2.5-Omni-3B thinker (zero-shot; video frames + audio track + subtitle text), fp16 on one T4.
  Data: (a) OV-MERD: the 332 clips of AffectGPT-RL/dataset/ovmerdplus-emotion.csv whose names are in MER2024 video-labeled.zip
        (Chinese TV; subtitles from AffectGPT-RL/dataset/subtitle_chieng.csv, Chinese column); GT = the released open-set labels.
        (b) EmoS-MELD (ACL 2026, github.com/NLP2CT/EmoS): MELD train utterances with GoEmotions soft labels; GT = labels with
        annotator share >= 0.5, excluding 'neutral' and 'unsure'; clips with an empty GT are excluded; 400 clips sampled (seed 0).
  Prompt (fixed): see PROMPT below. Output parsed as a comma/semicolon/newline-separated list, lower-cased, max 8 labels.
  Decoders (same model call per clip):
    GREEDY: greedy decoding (max 40 new tokens).
    VOTE  : N = 16 samples (temperature 1.0, top_p 0.95); keep labels present in >= 50% of samples (self-consistency voting;
            fallback: the most frequent label).
    MBR   : candidates = the 16 samples + the greedy output; pick the candidate c maximising mean_s EW_sample(gt=s, pred=c)
            over the 16 samples s (EW_sample = per-sample emotion-wheel F, level1, mean over 5 wheels; explore/ovmer/ew_metric.py).
  Metric: official corpus EW-F (level1, mean of 5 wheels; ew_metric.EW.corpus); level2 reported as a fine-grained guard.
  Statistics: paired bootstrap over clips (2,000 draws); MDE = 2.8 x SE.
  F1 premise HOLDS iff on BOTH datasets: MBR - GREEDY >= +3.0 (level1) with CI lower bound > 0, AND MBR - VOTE >= +1.0 with CI
  lower bound > 0, AND MBR - GREEDY at level2 >= 0 (point estimate). Otherwise the MBR form of F1 is closed (no variants).
Usage (Kaggle): python p13_mbr.py run   --items items.jsonl --out out.jsonl [--hours 9] [--limit N]
                python p13_mbr.py score --items items.jsonl --out out.jsonl --mapping wheel_mapping.npz --report p13_report.md
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROMPT = ("Watch the video, listen to the audio, and read the subtitle: \"{sub}\". Which emotions does the speaker in the video "
          "express? Answer with a comma-separated list of English emotion words (for example: worried, slightly angry, "
          "impatient). Output only the list.")
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, "
          "as well as generating text and speech.")  # official Qwen2.5-Omni system prompt
N_SAMPLES, MAX_LABELS = int(os.environ.get("P13_N", "16")), 8  # P13_N only for smoke tests


def parse(text):
    text = text.strip().split("\n")[0]  # first line only: the model may continue with an invented dialogue turn
    out = []
    for x in re.split(r"[,;\n、，]+", text):
        x = re.sub(r"^[\s\-\*\d\.\)]+", "", x).strip().strip(".").strip().lower()
        if x and len(x) < 40 and x not in out:
            out.append(x)
    return out[:MAX_LABELS]


def cmd_run(a):
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    items = [json.loads(l) for l in open(a.items)][: (a.limit or None)]
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = torch.float16 if dev == "cuda" else torch.bfloat16
    mid = "Qwen/Qwen2.5-Omni-3B"
    proc = Qwen2_5OmniProcessor.from_pretrained(mid)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(mid, dtype=dt, attn_implementation="sdpa").to(dev).eval()
    t0 = time.time(); n = 0; tmp = tempfile.mkdtemp()
    with open(a.out, "a") as fo:
        for it in items:
            if it["key"] in done:
                continue
            if (time.time() - t0) / 3600 > a.hours:
                print("time budget reached; rerun to resume", flush=True); break
            try:
                wav = os.path.join(tmp, "a.wav")  # audio extracted with ffmpeg (qwen_omni_utils' in-video audio path fails with recent librosa)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["video"], "-ac", "1", "-ar", "16000", "-t", "30", wav], check=True, timeout=120)
                conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": "video", "video": it["video"], "fps": 1.0, "min_pixels": 32 * 28 * 28, "max_pixels": 64 * 28 * 28, "max_frames": 8},
                                                     {"type": "audio", "audio": wav},
                                                     {"type": "text", "text": PROMPT.format(sub=it.get("sub", "")[:300])}]}]
                text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                aud, img, vid = process_mm_info(conv, use_audio_in_video=False)
                inp = proc(text=text, audio=aud, images=img, videos=vid, return_tensors="pt", padding=True, use_audio_in_video=False)
                inp = {k: (v.to(dev).to(dt) if torch.is_floating_point(v) else v.to(dev)) for k, v in inp.items()}
                L0 = inp["input_ids"].shape[1]
                with torch.no_grad():
                    g = model.generate(**inp, use_audio_in_video=False, max_new_tokens=40, do_sample=False)
                    s = model.generate(**inp, use_audio_in_video=False, max_new_tokens=40, do_sample=True, temperature=1.0, top_p=0.95,
                                       num_return_sequences=N_SAMPLES)
                greedy = proc.batch_decode(g[:, L0:], skip_special_tokens=True)[0]
                samples = proc.batch_decode(s[:, L0:], skip_special_tokens=True)
                rec = dict(key=it["key"], ds=it["ds"], greedy=greedy, samples=samples, ok=True)
                if n == 0 and not greedy.strip():
                    print("WARNING: empty greedy output on the first clip (check fp16 overflow)", flush=True)
            except Exception as e:  # noqa: BLE001
                rec = dict(key=it["key"], ds=it["ds"], ok=False, err=f"{type(e).__name__}: {str(e)[:200]}")
                print("fail", it["key"], rec["err"], flush=True)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); n += 1
            if n % 20 == 0:
                print(f"{n} clips, {time.time() - t0:.0f}s", flush=True)
    print("run finished", n, flush=True)


def decode(rec, ew):
    G = parse(rec["greedy"]); S = [parse(x) for x in rec["samples"]]
    cnt = {}
    for s in S:
        for x in set(s):
            cnt[x] = cnt.get(x, 0) + 1
    vote = [x for x, c in sorted(cnt.items(), key=lambda z: -z[1]) if c >= len(S) / 2] or ([max(cnt, key=cnt.get)] if cnt else [])
    cands = S + [G]
    util = [np.mean([ew.sample_f(s, c) for s in S]) if c else -1 for c in cands]
    mbr = cands[int(np.argmax(util))]
    return {"GREEDY": G, "VOTE": vote[:MAX_LABELS], "MBR": mbr}


def cmd_score(a):
    from ew_metric import EW
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    recs = [json.loads(l) for l in open(a.out)]; ok = [r for r in recs if r.get("ok")]
    L = [f"# P13: metric-aware decoding for OV-MER (zero-shot Qwen2.5-Omni-3B); aggregate scores only",
         f"- clips run {len(recs)}, ok {len(ok)}"]
    verdict = []
    for ds in sorted({r["ds"] for r in ok}):
        R = [r for r in ok if r["ds"] == ds]; gts = [items[r["key"]]["gt"] for r in R]; D = [decode(r, ew) for r in R]
        L.append(f"\n## {ds} (n={len(R)}; mean GT labels {np.mean([len(g) for g in gts]):.2f})")
        L.append("| decoder | EW-F level1 | P | R | EW-F level2 | mean #labels | share in wheel vocab |"); L.append("|---|---|---|---|---|---|---|")
        sc = {}
        for k in ("GREEDY", "VOTE", "MBR"):
            P = [d[k] for d in D]; f1 = ew.corpus(gts, P); f2 = ew.corpus(gts, P, "level2"); sc[k] = P
            voc = np.mean([ew.in_vocab(x) for p in P for x in p]) if any(P) else 0
            L.append(f"| {k} | {f1[0]:.2f} | {f1[1]:.2f} | {f1[2]:.2f} | {f2[0]:.2f} | {np.mean([len(p) for p in P]):.2f} | {100 * voc:.1f}% |")
        rng = np.random.default_rng(1); n = len(R)

        def boot(a_, b_, level="level1"):
            pt = ew.corpus(gts, sc[a_], level)[0] - ew.corpus(gts, sc[b_], level)[0]; bs = []
            for _ in range(2000):
                ix = rng.integers(0, n, n); g = [gts[i] for i in ix]
                bs.append(ew.corpus(g, [sc[a_][i] for i in ix], level)[0] - ew.corpus(g, [sc[b_][i] for i in ix], level)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi, 2.8 * np.std(bs)
        d1 = boot("MBR", "GREEDY"); d2 = boot("MBR", "VOTE"); d3 = ew.corpus(gts, sc["MBR"], "level2")[0] - ew.corpus(gts, sc["GREEDY"], "level2")[0]
        L.append(f"- MBR - GREEDY = {d1[0]:+.2f} [{d1[1]:+.2f}, {d1[2]:+.2f}] (MDE {d1[3]:.2f}); MBR - VOTE = {d2[0]:+.2f} [{d2[1]:+.2f}, {d2[2]:+.2f}]; level2 MBR - GREEDY = {d3:+.2f}")
        ok_ = d1[0] >= 3.0 and d1[1] > 0 and d2[0] >= 1.0 and d2[1] > 0 and d3 >= 0
        verdict.append(ok_); L.append(f"- {ds}: {'supports' if ok_ else 'does not support'} the F1 premise")
    L.append(f"\n**P13 verdict: F1 (MBR form) {'HOLDS' if verdict and all(verdict) and len(verdict) >= 2 else 'DOES NOT HOLD'}**")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--items"); ap.add_argument("--out"); ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--mapping"); ap.add_argument("--report", default="p13_report.md")
    a = ap.parse_args(); {"run": cmd_run, "score": cmd_score}[a.cmd](a)
