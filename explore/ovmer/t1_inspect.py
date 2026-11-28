"""Label-free statistics for tuning tiers 0-1 (no ground truth is read).

Per sampling configuration, from the description files:
  clips ok / failed, seconds per clip, generated tokens per sample, share that hit max_new_tokens (512), share with text after an
  invented dialogue turn (about 0 when the stop strings work; the marker itself stays at the end), share lacking terminal punctuation after the cut rule.
Per label file (optional): labels per sample, share of empty samples, share of samples with only out-of-vocabulary labels,
level-1 groups per sample on wheel 1, and the mean pairwise per-clip utility between samples (diversity: 1 = all samples agree).
Projection: hours for --n_clips clips on --gpus GPUs from the measured seconds per clip.
Usage: python t1_inspect.py --descs d1.jsonl,d2.jsonl [--labels l1.jsonl,...] --mapping wheel_mapping.npz [--n_clips 1000 --gpus 2]
        [--report out.md]
"""
import argparse, collections, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p17_extract_vllm import TURN, cut  # noqa: E402

END = re.compile(r'[.!?。！？"”)\]]\s*$')


def after_turn(t):
    """True if text follows an invented-turn marker (a stop string leaves the marker itself at the end, which is fine)."""
    m = TURN.search(t)
    return bool(m and t[m.end():].strip())


def read(paths, by="cfg"):
    out = collections.defaultdict(dict)
    for p in paths.split(","):
        if p and os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                name = os.path.basename(p) if by == "file" else (r.get("cfg") or os.path.basename(p))
                out[name][r["key"]] = r  # last record per key
    return out


def desc_stats(recs, a):
    ok = [r for r in recs.values() if r.get("ok")]; bad = len(recs) - len(ok)
    if not ok:
        return dict(clips=0, failed=bad)
    S = [t for r in ok for t in r["samples"]]; nt = [x for r in ok for x in r.get("n_tok", [])]
    sec = np.mean([r["sec"] for r in ok])
    return dict(clips=len(ok), failed=bad, samples_per_clip=round(len(S) / len(ok), 1), sec_per_clip=round(sec, 1),
                proj_hours=round(sec * a.n_clips / a.gpus / 3600, 2),
                tok_per_sample=round(float(np.mean(nt)), 1) if nt else None,
                hit_max_pct=round(100 * float(np.mean([x >= 512 for x in nt])), 1) if nt else None,
                ended_pct=(round(100 * float(np.mean([e for r in ok for e in r["ended"]])), 1) if all("ended" in r for r in ok) else None),
                text_after_turn_pct=round(100 * float(np.mean([after_turn(t) for t in S])), 1),
                no_end_punct_pct=round(100 * float(np.mean([not END.search(cut(t, "turn")) for t in S])), 1),
                empty_text_pct=round(100 * float(np.mean([not cut(t, "turn") for t in S])), 1),
                errors=collections.Counter(r.get("err", "")[:60] for r in recs.values() if not r.get("ok")).most_common(3))


def label_stats(recs, ew, parse):
    L = [[parse(x) for x in r["samples"]] for r in recs.values() if r.get("ok")]
    flat = [s for c in L for s in c]
    if not flat:
        return {}
    iv = lambda s: [y for y in s if ew.in_vocab(y)]
    div = []
    for c in L:
        c = [s for s in c if iv(s)]
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                div.append(ew.sample_f(c[i], c[j]))
    return dict(clips=len(L), labels_per_sample=round(float(np.mean([len(s) for s in flat])), 2),
                empty_pct=round(100 * float(np.mean([not s for s in flat])), 1),
                oov_only_pct=round(100 * float(np.mean([bool(s) and not iv(s) for s in flat])), 1),
                groups_w1=round(float(np.mean([len(ew.groups(s, "wheel1")) for s in flat if s])), 2),
                pairwise_utility=round(float(np.mean(div)), 3) if div else None)


def main(a):
    D = read(a.descs); lines = ["## Descriptions", "", "| cfg | " + " | ".join(desc_stats(next(iter(D.values())), a).keys()) + " |"] if D else []
    if D:
        lines.append("|" + "---|" * (len(lines[-1].split("|")) - 2))
        for cfg in sorted(D):
            st = desc_stats(D[cfg], a); lines.append(f"| {cfg} | " + " | ".join(str(v) for v in st.values()) + " |")
    if a.labels:
        from ew_metric import EW
        from p13_mbr import parse
        ew = EW(a.mapping); L = read(a.labels, by="file"); lines += ["", "## Labels", ""]
        rows = [(name, label_stats(recs, ew, parse)) for name, recs in sorted(L.items())]
        rows = [(n, s) for n, s in rows if s]
        if rows:
            lines.append("| file | " + " | ".join(rows[0][1].keys()) + " |"); lines.append("|" + "---|" * (len(rows[0][1]) + 1))
            for n, s in rows:
                lines.append(f"| {n} | " + " | ".join(str(v) for v in s.values()) + " |")
    txt = "\n".join(lines); print(txt)
    if a.report:
        open(a.report, "a").write(txt + "\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--descs", default=""); ap.add_argument("--labels", default=""); ap.add_argument("--mapping", default="")
    ap.add_argument("--n_clips", type=int, default=1000); ap.add_argument("--gpus", type=int, default=2); ap.add_argument("--report", default="")
    main(ap.parse_args())
