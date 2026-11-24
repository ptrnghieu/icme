"""P14' official-protocol arm (for the comparison with published OV-MERD+ numbers). Runs on the USER's Kaggle.

Pipeline as in the official OV-MER / AffectGPT evaluation (zeroQiaoba/AffectGPT, read 2026-11-23):
  1. the MLLM writes a free-text emotion description (official prompt; p13_mbr.py --protocol official): greedy + 16 samples;
  2. an LLM extracts the open-set labels from each description with the official extraction prompt (AffectGPT/toolkit/utils/qwen.py,
     reason_to_openset_qwen), extractor Qwen/Qwen2.5-7B-Instruct (official: vLLM, temperature 0.7, top_p 0.8, repetition_penalty
     1.05); here 4-bit NF4 on a T4 with the same sampling parameters and a fixed seed (deviation: quantisation);
  3. GREEDY / TOPK / MBR / CONSTRUCT are applied to the extracted label lists; EW-F as in ew_metric.py.
PRE-REGISTERED (2026-11-23), descriptive arm, Qwen2.5-Omni-3B zero-shot on OV-MERD+ (532):
  - protocol effect: EW-F(GREEDY, official) vs EW-F(GREEDY, direct list) from the main P14' run of the same backbone;
  - the gain of CONSTRUCT under the official protocol is claimed only if CONSTRUCT - GREEDY >= +3.0 with bootstrap CI lower bound > 0;
  - the table places these numbers next to published OV-MERD+ results (AffectGPT 62.52; AffectGPT-R1 / AffectGPT-RL 68.39).
AMENDMENT 1 (2026-11-24, after the first P14' run; decision criteria in p14o_score.py unchanged; the original-pipeline numbers
  stay in the record, explore/ovmer/p14_report_2026-11-24.md). Pipeline-validity fixes, identical for every decoder:
  (a) Qwen2.5-Omni-3B does not emit EOS under the official prompt: 81% of descriptions continue with invented dialogue turns
      ("Human: ...", unrelated text). The description is cut at its first non-empty line (= generation with a stop at "\n",
      which leaves the distribution of the first line unchanged, so the stored samples are reused).
  (b) extraction call as in the official vLLM path (AffectGPT-RL/toolkit/utils/qwen.py, get_completion_qwen_bacth): user message
      only, official post-processing (func_postprocess_qwen) and list parsing (toolkit/utils/functions.py, string_to_list);
      max_new_tokens 128 (official 512; label lists are short).
  (c) extracted labels written in Chinese (19.5% of greedy extractions in run 1, outside the English wheel vocabulary) are
      translated with the official prompt (translate_chi2eng_qwen, as in get_chi2eng_label_mapping).
Usage: python p14_official.py extract --descs d.jsonl --out labels.jsonl [--shard 0/2] [--amend 1]
       python p14_official.py score --items items.jsonl --labels labels.jsonl --direct out_3b.jsonl --mapping wheel_mapping.npz --report r.md
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXTRACT = ("Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the "
           "characters. Based on the provided clues, please identify the emotional states of the main character. The main character is "
           "the one with the most detailed clues. Please separate different emotional categories with commas and output only the clearly "
           "identifiable emotional categories in a list format. If none are identified, please output an empty list. Input: We cannot "
           "recognize his emotional state; Output: [] Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] "
           "Input: {reason}; Output: ")


TRANSLATE = ("Please translate the Chinese input into English. Please ensure the translated results does not contain any Chinese words.\n"
             "Input: 高兴; Output: happy Input: 生气; Output: angry Input: {reason}; Output: ")
CJK = re.compile(r"[\u4e00-\u9fff]")


def first_para(t):
    for line in t.strip().split("\n"):
        if line.strip():
            return line.strip()
    return ""


def post_official(resp):  # AffectGPT-RL/toolkit/utils/qwen.py func_postprocess_qwen
    r = resp.strip()
    for p in ("输入", "输出", "翻译", "让我们来翻译一下：", "output", "Output", "input", "Input"):
        if r.startswith(p):
            r = r[len(p):]
    r = r.strip()
    for p in (":", "："):
        if r.startswith(p):
            r = r[len(p):]
    return r.strip().replace("\n", "").strip()


def to_list(r):  # AffectGPT-RL/toolkit/utils/functions.py string_to_list
    if not r:
        return []
    if r[0] == "[":
        r = r[1:]
    if r and r[-1] == "]":
        r = r[:-1]
    return [x.strip() for x in re.split("['\",]", r) if x.strip() not in ("", ",")]


def clean(resp):
    resp = resp.strip().split("\n")[0]
    m = re.search(r"\[(.*?)\]", resp)
    return (m.group(1) if m else resp).strip()


def cmd_extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.manual_seed(0)
    mid = "Qwen/Qwen2.5-7B-Instruct"; tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, device_map={"": 0}, dtype=torch.float16).eval()
    recs = [json.loads(l) for l in open(a.descs)]; recs = [r for r in recs if r.get("ok")]
    if a.amend == "1":  # primary sets first, so a time-out leaves the decision sets complete
        order = {"EmoS-MELD-fresh": 0, "MER2025-FG": 1, "OV-MERD+": 2}; recs.sort(key=lambda r: order.get(r["ds"], 3))
    if a.shard:
        i, n = map(int, a.shard.split("/")); recs = recs[i::n]
    import time; t0 = time.time(); hours = float(a.hours or 99)
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    amend = a.amend == "1"; cache = {}

    def gen_batch(user_texts, max_new):
        msgs = [[{"role": "user", "content": t}] if amend else [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": t}]
                for t in user_texts]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]; outs = []
        for b in range(0, len(prompts), 17 if amend else 9):
            enc = tok(prompts[b:b + (17 if amend else 9)], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.05)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def translate(labels):
        need = sorted({x for x in labels if CJK.search(x) and x not in cache})
        if need:
            for x, y in zip(need, gen_batch([TRANSLATE.format(reason=x) for x in need], 32)):
                cache[x] = post_official(y)
        return [cache.get(x, x) for x in labels]

    st = dict(n=0, empty=0, cjk_before=0, cjk_after=0)
    with open(a.out, "a") as fo:
        for k, r in enumerate(recs):
            if r["key"] in done:
                continue
            if (time.time() - t0) / 3600 > hours:
                print("time budget reached; rerun to resume", flush=True); break
            texts = [r["greedy"]] + r["samples"]
            if amend:
                outs = gen_batch([EXTRACT.format(reason=first_para(t)[:2000]) for t in texts], 128); res = []
                for o in outs:
                    L = to_list(post_official(o)); st["n"] += 1; st["empty"] += not L; st["cjk_before"] += any(CJK.search(x) for x in L)
                    L = translate(L); st["cjk_after"] += any(CJK.search(x) for x in L); res.append(", ".join(L))
                fo.write(json.dumps(dict(key=r["key"], ds=r["ds"], ok=True, greedy=res[0], samples=res[1:], amend=1), ensure_ascii=False) + "\n"); fo.flush()
                if k % 20 == 0:
                    print("extracted", k, "/", len(recs), {x: round(v / max(st["n"], 1), 3) if x != "n" else v for x, v in st.items()}, flush=True)
                continue
            prompts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": EXTRACT.format(reason=t.strip()[:2000])}], tokenize=False, add_generation_prompt=True)
                       for t in texts]
            outs = []
            for b in range(0, len(prompts), 9):
                enc = tok(prompts[b:b + 9], return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.05)
                outs += [clean(x) for x in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
            fo.write(json.dumps(dict(key=r["key"], ds=r["ds"], ok=True, greedy=outs[0], samples=outs[1:]), ensure_ascii=False) + "\n"); fo.flush()
            if k % 20 == 0:
                print("extracted", k, "/", len(recs), flush=True)
    print("extraction finished", flush=True)


def cmd_score(a):
    from ew_metric import EW
    from p13_mbr import decode
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    lab = [json.loads(l) for l in open(a.labels)]; lab = [r for r in lab if r["ds"] == "OV-MERD+"]
    G = [items[r["key"]]["gt"] for r in lab]; D = [decode(r, ew) for r in lab]
    L = ["# P14' official-protocol arm: Qwen2.5-Omni-3B zero-shot on OV-MERD+ (description -> LLM label extraction)",
         f"- clips {len(lab)} of {sum(1 for it in items.values() if it['ds'] == 'OV-MERD+')}",
         "| decoder | EW-F L1 | P | R | EW-F L2 | mean #labels |", "|---|---|---|---|---|---|"]
    sc = {}
    for k in ("GREEDY", "TOPK", "MBR", "CONSTRUCT"):
        f = ew.corpus(G, [d[k] for d in D]); f2 = ew.corpus(G, [d[k] for d in D], "level2")[0]; sc[k] = f[0]
        L.append(f"| {k} | {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} | {f2:.2f} | {np.mean([len(d[k]) for d in D]):.2f} |")
    n = len(lab); rng = np.random.default_rng(1); bs = []
    for _ in range(2000):
        ix = rng.integers(0, n, n); g = [G[i] for i in ix]
        bs.append(ew.corpus(g, [D[i]["CONSTRUCT"] for i in ix])[0] - ew.corpus(g, [D[i]["GREEDY"] for i in ix])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]); d = sc["CONSTRUCT"] - sc["GREEDY"]
    L.append(f"- CONSTRUCT - GREEDY (official protocol) = {d:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {'gain claimed' if d >= 3.0 and lo > 0 else 'gain NOT claimed'}")
    if a.direct and os.path.exists(a.direct):
        dr = [json.loads(l) for l in open(a.direct)]; dr = [r for r in dr if r.get("ok") and r["ds"] == "OV-MERD+"]
        Gd = [items[r["key"]]["gt"] for r in dr]; Dd = [decode(r, ew, r["samples"][:16]) for r in dr]
        L.append(f"- protocol effect (GREEDY): official {sc['GREEDY']:.2f} vs direct list {ew.corpus(Gd, [x['GREEDY'] for x in Dd])[0]:.2f}; "
                 f"CONSTRUCT: official {sc['CONSTRUCT']:.2f} vs direct list (seed 1) {ew.corpus(Gd, [x['CONSTRUCT'] for x in Dd])[0]:.2f}")
    L.append("- published OV-MERD+ EW-F (official pipeline): AffectGPT 62.52 (SFT 7B); AffectGPT-R1 / AffectGPT-RL 68.39 (SFT + RL 7B)")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    for k in ("descs", "out", "shard", "items", "labels", "direct", "mapping", "report", "amend", "hours"):
        ap.add_argument("--" + k, default="")
    a = ap.parse_args(); {"extract": cmd_extract, "score": cmd_score}[a.cmd](a)
