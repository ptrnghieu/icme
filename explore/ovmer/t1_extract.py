"""Tuning tiers 0 and 2: open-set label extraction with the official extractor (Qwen2.5-7B-Instruct, official AffectGPT-RL prompt and
post-processing, imported from p17_extract_vllm.py), on Kaggle T4s.

Engines:
  vllm : as P17 but float16 (the T4 has no bfloat16) and --tp GPUs (tensor parallel; 7B float16 does not fit one 16 GB T4);
  hf   : transformers fallback, float16, layers spread over the visible GPUs (device_map auto), left-padded batches. Sampling set to
         match vLLM: temperature 0.7, top_p 0.8, repetition_penalty 1.05, top_k 0 (vLLM SamplingParams default top_k is 0, read in
         the 0.34.0 wheel on 2026-11-28; transformers would otherwise fill top_k 50).
Modes (tuning knob X):
  official : one draw per description, temperature 0.7 (X0, the official setting);
  greedy   : temperature 0 (X1);
  k        : --k draws per description (X2); stored per sample as a list of K label strings.
Cut rule: p17_extract_vllm.cut, mode turn (as pre-registered for P17).
Output lines: {key, ds, ok, cfg, greedy, samples (label strings), samples_k (mode k only), extractor}.
Usage: python t1_extract.py --descs desc_S0_0.jsonl[,desc_S0_1.jsonl] --out labels_S0_X0.jsonl --engine vllm --tp 2 --mode official
        [--limit_clips N] [--hours H]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p17_extract_vllm import PROMPT_RL, cut, post_official, to_list  # noqa: E402

MID = "Qwen/Qwen2.5-7B-Instruct"


def load_descs(paths, limit):
    recs = {}
    for p in paths.split(","):
        for l in open(p):
            r = json.loads(l)
            if r.get("ok"):
                recs[r["key"]] = r
    keys = sorted(recs)[: (limit or None)]
    return [recs[k] for k in keys]


def prompts_of(r, tok):
    texts = ([r["greedy"]] if r.get("greedy") is not None else []) + r["samples"]
    return [tok.apply_chat_template([{"role": "user", "content": PROMPT_RL.format(reason=cut(t, "turn"))}], tokenize=False,
                                    add_generation_prompt=True) for t in texts]


def run_vllm(prompts, a):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MID, dtype="float16", tensor_parallel_size=a.tp, seed=a.seed, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enforce_eager=bool(a.eager), disable_custom_all_reduce=True)
    if a.mode == "greedy":
        sp = SamplingParams(temperature=0.0, max_tokens=512)
    else:
        sp = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512, seed=a.seed,
                            n=(a.k if a.mode == "k" else 1))

    def gen(batch):
        return [[c.text for c in o.outputs] for o in llm.generate(batch, sp)]
    return gen


def run_hf(prompts, a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MID); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="auto").eval()
    torch.manual_seed(a.seed)

    def gen(batch):
        out = []
        for b in range(0, len(batch), a.hf_batch):
            enc = tok(batch[b: b + a.hf_batch], return_tensors="pt", padding=True).to(model.device)
            kw = dict(do_sample=False) if a.mode == "greedy" else dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=0,
                                                                         repetition_penalty=1.05)
            n = a.k if a.mode == "k" else 1
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=512, num_return_sequences=n, pad_token_id=tok.pad_token_id, **kw)
            txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [txt[i * n:(i + 1) * n] for i in range(len(txt) // n)]
        return out
    return gen


def main(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MID)
    recs = load_descs(a.descs, a.limit_clips)
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [r for r in recs if r["key"] not in done]
    print(f"descriptions {len(recs)} | already extracted {len(done)} | to do {len(todo)} | engine {a.engine} mode {a.mode}", flush=True)
    if not todo:
        return
    gen = (run_vllm if a.engine == "vllm" else run_hf)(None, a)
    tag = dict(model=MID, engine=a.engine, dtype="float16", mode=a.mode, k=a.k, seed=a.seed, cut="turn", prompt="rl")
    t0 = time.time(); n_prompts = 0
    with open(a.out, "a") as fo:
        for b in range(0, len(todo), a.batch_clips):
            if (time.time() - t0) / 3600 > a.hours:
                print("time budget reached; rerun to resume", flush=True); break
            chunk = todo[b: b + a.batch_clips]; P, owner = [], []
            for r in chunk:
                ps = prompts_of(r, tok); P += ps; owner += [(r["key"], j) for j in range(len(ps))]
            outs = gen(P); n_prompts += len(P); lab = {}
            for (k, j), o in zip(owner, outs):
                lab.setdefault(k, []).append([", ".join(to_list(post_official(t))) for t in o])
            for r in chunk:
                L = lab[r["key"]]; hasg = r.get("greedy") is not None
                S = L[1:] if hasg else L
                rec = dict(key=r["key"], ds=r["ds"], ok=True, cfg=r.get("cfg"), greedy=(L[0][0] if hasg else None),
                           samples=[s[0] for s in S], extractor=tag)
                if a.mode == "k":
                    rec["samples_k"] = S
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush(); el = time.time() - t0
            print(f"extracted {min(b + a.batch_clips, len(todo))}/{len(todo)} clips | {n_prompts / el:.1f} prompts/s", flush=True)
    print("extraction finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--descs"); ap.add_argument("--out"); ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--mode", choices=["official", "greedy", "k"], default="official"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.85); ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--eager", type=int, default=1, help="skip CUDA-graph capture (faster start, fewer failure points on T4)")
    ap.add_argument("--hf_batch", type=int, default=24); ap.add_argument("--batch_clips", type=int, default=50)
    ap.add_argument("--limit_clips", type=int, default=0); ap.add_argument("--hours", type=float, default=99)
    main(ap.parse_args())
