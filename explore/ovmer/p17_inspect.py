"""Aggregate-only summary of P17 descriptions: speed and the STRUCTURE of the texts (no text is printed, no scores).
Used after the timing run to decide the --cut rule before the full extraction, without looking at any metric.
Usage: python p17_inspect.py desc.jsonl
"""
import json, re, sys
from collections import defaultdict

TURN = re.compile(r"(\n\s*(Human|User|Assistant|用户|助手)\s*[:：])|(<\|im_start\|>)")
CJK = re.compile(r"[一-鿿]")

recs = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    recs[r["key"]] = r
by = defaultdict(list)
for r in recs.values():
    by[r["ds"]].append(r)
for ds, R in by.items():
    ok = [r for r in R if r.get("ok")]
    T = [t for r in ok for t in [r["greedy"]] + r["samples"]]
    if not T:
        print(ds, "no ok clips;", len(R) - len(ok), "failed:", [r.get("err", "")[:120] for r in R[:3]]); continue
    n = len(T); mean = lambda xs: sum(xs) / max(len(xs), 1)
    print(f"== {ds}: clips ok {len(ok)}/{len(R)} | s/clip {mean([r['sec'] for r in ok]):.1f} | prompt tokens {mean([r.get('prompt_tokens', 0) for r in ok]):.0f}"
          f" | samples/clip {mean([len(r['samples']) for r in ok]):.1f}")
    print(f"   texts {n} | with an invented turn marker {mean([bool(TURN.search(t)) for t in T]):.3f}"
          f" | multi-line {mean([len([x for x in t.strip().split(chr(10)) if x.strip()]) > 1 for t in T]):.3f}"
          f" | chars mean {mean([len(t) for t in T]):.0f} | first-line share of chars {mean([len(t.strip().split(chr(10))[0]) / max(len(t.strip()), 1) for t in T]):.2f}"
          f" | contains CJK {mean([bool(CJK.search(t)) for t in T]):.3f} | empty {mean([not t.strip() for t in T]):.3f}")
    fails = [r.get("err", "")[:120] for r in R if not r.get("ok")]
    if fails:
        print("   failures:", len(fails), fails[:3])

# ---- cut-rule statistics (structure only): what each candidate --cut rule would keep
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p17_extract_vllm import cut  # noqa: E402
from collections import Counter
MARK = re.compile(r"\n\s*(Human|User|Assistant|用户|助手)\s*[:：]|<\|im_start\|>")
for ds, R in by.items():
    T = [t for r in R if r.get("ok") for t in [r["greedy"]] + r["samples"]]
    if not T:
        continue
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    kinds = Counter((m.group(1) or "<|im_start|>") for t in T for m in [MARK.search(t)] if m)
    print(f"-- {ds}: first marker types {dict(kinds)} | texts >= 1,900 chars (near the 512-token cap) {mean([len(t) >= 1900 for t in T]):.3f}")
    for mode in ("none", "turn", "firstline"):
        K = [cut(t, mode) for t in T]
        print(f"   cut={mode:9s} kept chars mean {mean([len(k) for k in K]):.0f} | CJK {mean([bool(CJK.search(k)) for k in K]):.3f}"
              f" | mostly CJK (>30% of chars) {mean([len(CJK.findall(k)) > 0.3 * max(len(k), 1) for k in K]):.3f} | empty {mean([not k.strip() for k in K]):.3f}"
              f" | still multi-line {mean([len([x for x in k.split(chr(10)) if x.strip()]) > 1 for k in K]):.3f}")
