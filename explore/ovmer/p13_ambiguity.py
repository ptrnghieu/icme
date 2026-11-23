"""Descriptive analysis on P13 outputs (EmoS-MELD only; not a pass/fail test): relation between the diversity of the 16 sampled
label lists and human annotator disagreement, and whether the MBR gain over GREEDY differs by disagreement level.
Human disagreement = normalised entropy of the EmoS GoEmotions annotator shares (excluding 'unsure'); sample diversity = mean
pairwise (1 - EW_sample F) between the 16 parsed samples. Reports Spearman rho (bootstrap CI) and per-tertile EW-F of GREEDY/MBR.
Usage: python p13_ambiguity.py --items items.jsonl --out out.jsonl --mapping wheel_mapping.npz --emos EMOS_DIR --report amb.md
"""
import argparse, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ew_metric import EW  # noqa: E402
from p13_mbr import decode, parse  # noqa: E402


def main(a):
    import pandas as pd
    from scipy.stats import spearmanr
    ew = EW(a.mapping); items = {json.loads(l)["key"]: json.loads(l) for l in open(a.items)}
    d = pd.concat([pd.read_excel(os.path.join(a.emos, f"datasets/data-final/English/goemotion/goemotion_soft_labels_{s}.xlsx")) for s in ("first3000", "last2000")])
    cols = [c for c in d.columns if c.startswith("goemotion_prob_") and not c.endswith("unsure")]
    H = {}
    for _, r in d.iterrows():
        p = np.array([float(r[c] or 0) for c in cols]); p = p / p.sum() if p.sum() else p
        H[f"meld:{int(r['Dialogue_ID'])}_{int(r['Utterance_ID'])}"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(cols)))
    R = [json.loads(l) for l in open(a.out)]; R = [r for r in R if r.get("ok") and r["ds"] == "EmoS-MELD" and r["key"] in H]
    div = []
    for r in R:
        S = [parse(x) for x in r["samples"]]
        div.append(np.mean([1 - ew.sample_f(x, y) for x, y in itertools.combinations(S, 2)]) if len(S) > 1 else 0.0)
    hum = np.array([H[r["key"]] for r in R]); div = np.array(div); rho = spearmanr(hum, div).correlation
    rng = np.random.default_rng(0); bs = [spearmanr(hum[i], div[i]).correlation for i in (rng.integers(0, len(R), len(R)) for _ in range(2000))]
    L = [f"# P13 ambiguity analysis (EmoS-MELD, n={len(R)}; descriptive)",
         f"- Spearman(human disagreement entropy, sample diversity) = {rho:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"]
    q = np.quantile(hum, [1 / 3, 2 / 3]); t = np.digitize(hum, q); D = [decode(r, ew) for r in R]; G = [items[r["key"]]["gt"] for r in R]
    for k in range(3):
        m = np.where(t == k)[0]
        g = [G[i] for i in m]; f = {dec: ew.corpus(g, [D[i][dec] for i in m])[0] for dec in ("GREEDY", "VOTE", "MBR")}
        L.append(f"- disagreement tertile {k + 1} (n={len(m)}): GREEDY {f['GREEDY']:.2f}, VOTE {f['VOTE']:.2f}, MBR {f['MBR']:.2f}")
    open(a.report, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("items", "out", "mapping", "emos", "report"):
        ap.add_argument("--" + k)
    main(ap.parse_args())
