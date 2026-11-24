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
