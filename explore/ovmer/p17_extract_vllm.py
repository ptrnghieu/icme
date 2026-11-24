"""P17 phase 2: open-set label extraction exactly as the official OV-MERD+ evaluation (runs on the user's rented GPU).

Official reference: zeroQiaoba/AffectGPT, AffectGPT-RL (read 2026-11-24):
  - statistic.py: calculate_ov_zeroshot -> extract_openset_batchcalling -> wheel_metric_calculation (level 1);
  - my_affectgpt/evaluation/ew_metric.py extract_openset_batchcalling: inputtype 'answer' keeps the text inside
    <answer>...</answer> when present;
  - toolkit/utils/qwen.py reason_to_openset_qwen (prompt below, with the "at most four labels" sentence) and
    get_completion_qwen_bacth: user message only, vLLM, func_postprocess_qwen;
  - extractor 'Qwen25' = Qwen2.5-7B-Instruct (config.py), vLLM default dtype (bfloat16 in the model config);
  - SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512); no Chinese-to-English translation.
Deviations, all stated:
  - a fixed vLLM seed (--seed), for reproducibility;
  - --cut, which removes dialogue turns invented after the description (decided from the timing-run outputs before the full run):
      none      = the official input;
      turn      = cut at the first invented turn marker ("Human:", "User:", "Assistant:", "<|im_start|>");
      firstline = keep the first non-empty line.
--prompt old uses the AffectGPT (non-RL) prompt without the four-label sentence (sensitivity analysis).
Output lines: {key, ds, ok, greedy, samples (label strings, comma-joined), extractor}.
Usage: python p17_extract_vllm.py --descs desc.jsonl --out labels.jsonl [--cut turn] [--prompt rl] [--seed 0]
"""
import argparse, json, os, re

PROMPT_RL = ("Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the "
             "characters. Based on the provided clues, please identify the emotional states of the main character. The main character is "
             "the one with the most detailed clues. Please separate different emotional categories with commas and output only the clearly "
             "identifiable emotional categories in a list format. Please remember that you can only ouput at most four labels for each "
             "input. If none are identified, please output an empty list. Input: We cannot recognize his emotional state; Output: [] "
             "Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] Input: {reason}; Output: ")
PROMPT_OLD = PROMPT_RL.replace("Please remember that you can only ouput at most four labels for each input. ", "")
TURN = re.compile(r"(\n\s*(Human|User|Assistant|用户|助手)\s*[:：])|(<\|im_start\|>)")


def cut(text, mode):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)  # official inputtype 'answer'
    if m:
        text = m.group(1)
    if mode == "turn":
        m = TURN.search(text)
        return (text[: m.start()] if m else text).strip()
    if mode == "firstline":
        for line in text.strip().split("\n"):
            if line.strip():
                return line.strip()
        return ""
    return text


def post_official(resp):  # toolkit/utils/qwen.py func_postprocess_qwen
    r = resp.strip()
    for p in ("输入", "输出", "翻译", "让我们来翻译一下：", "output", "Output", "input", "Input"):
        if r.startswith(p):
            r = r[len(p):]
    r = r.strip()
    for p in (":", "："):
        if r.startswith(p):
            r = r[len(p):]
    return r.strip().replace("\n", "").strip()


def to_list(r):  # toolkit/utils/functions.py string_to_list
    if not r:
        return []
    if r[0] == "[":
        r = r[1:]
    if r and r[-1] == "]":
        r = r[:-1]
    return [x.strip() for x in re.split("['\",]", r) if x.strip() not in ("", ",")]


def main(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    mid = "Qwen/Qwen2.5-7B-Instruct"; tmpl = PROMPT_RL if a.prompt == "rl" else PROMPT_OLD
    recs = {}
    for l in open(a.descs):
        r = json.loads(l)
        if r.get("ok"):
            recs[r["key"]] = r  # last ok record per key
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [r for k, r in recs.items() if k not in done and (not a.only_ds or r["ds"] in a.only_ds.split(","))]
    print(f"descriptions {len(recs)} | already extracted {len(done)} | to do {len(todo)}", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(mid)
    llm = LLM(model=mid, seed=a.seed, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512, seed=a.seed)
    tag = dict(model=mid, prompt=a.prompt, cut=a.cut, seed=a.seed)
    with open(a.out, "a") as fo:
        for b in range(0, len(todo), a.batch_clips):
            chunk = todo[b: b + a.batch_clips]; prompts, owner = [], []
            for r in chunk:
                for j, t in enumerate([r["greedy"]] + r["samples"]):
                    prompts.append(tok.apply_chat_template([{"role": "user", "content": tmpl.format(reason=cut(t, a.cut))}],
                                                           tokenize=False, add_generation_prompt=True))
                    owner.append((r["key"], j))
            outs = llm.generate(prompts, sp)
            lab = {}
            for (k, j), o in zip(owner, outs):
                lab.setdefault(k, {})[j] = ", ".join(to_list(post_official(o.outputs[0].text)))
            for r in chunk:
                L = lab[r["key"]]
                fo.write(json.dumps(dict(key=r["key"], ds=r["ds"], ok=True, greedy=L[0], samples=[L[j] for j in range(1, len(L))],
                                         extractor=tag), ensure_ascii=False) + "\n")
            fo.flush(); print(f"extracted {min(b + a.batch_clips, len(todo))}/{len(todo)}", flush=True)
    print("extraction finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--descs"); ap.add_argument("--out"); ap.add_argument("--cut", default="turn", choices=["none", "turn", "firstline"])
    ap.add_argument("--prompt", default="rl", choices=["rl", "old"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_ds", default="")
    ap.add_argument("--batch_clips", type=int, default=200); ap.add_argument("--gpu_mem", type=float, default=0.85)
    main(ap.parse_args())
