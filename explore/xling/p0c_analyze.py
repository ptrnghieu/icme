"""P0c analysis for idea H (affect-concept features transfer better than raw embeddings). Criterion fixed BEFORE results
(2026-11-21): H premise HOLDS iff, in >= 2 of the 3 directions {MOSI->SIMS, SIMS->MOSI, CREMA-D->MELD}, TR(concept) > TR(embedding)
for BOTH audio (superb-ER logits vs WavLM) and face (FER logits vs ViT-FER embedding), each with group-bootstrap 95% CI of the
paired difference excluding 0, AND concept-fusion cross bacc >= embedding-fusion cross bacc in those directions.
A direction/modality whose target oracle is within 3 points of chance is 'undefined' and counts as not supporting H.

Usage: python -I p0c_analyze.py <dir with mosi_p0.npz sims_p0.npz cremad_p0b.npz meld_p0b.npz and *_p0c.npz> [out.md]
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(0)
L = []; say = lambda s="": (print(s), L.append(s))  # noqa: E731


def merge(base, conc, keyfn):
    b = np.load(os.path.join(R, base), allow_pickle=True); c = np.load(os.path.join(R, conc), allow_pickle=True)
    D = {k: b[k] for k in b.files}
    idx = {k: i for i, k in enumerate(c["key"])}
    keys = keyfn(D)
    sel = np.array([idx[k] for k in keys])
    D["a_concept"], D["t_concept"] = c["a_concept"][sel], c["t_concept"][sel]
    return D


def binary(D):
    keep = D["ok"].astype(bool) & (D["label"].astype(float) != 0) & ~np.isnan(D["a_concept"]).any(1)
    E = {k: v[keep] for k, v in D.items()}; E["y"] = (E["label"].astype(float) > 0).astype(int).astype(str); E["grp"] = E["video_id"]
    return E


def multi(D, classes):
    keep = D["ok"].astype(bool) & np.isin(D["label"], classes) & ~np.isnan(D["a_concept"]).any(1)
    E = {k: v[keep] for k, v in D.items()}; E["y"] = E["label"]; E["grp"] = E["group"]
    return E


def fit(D, keys):
    tr, va = D["mode"] == "train", D["mode"] == "valid"
    Xf = lambda m: np.concatenate([D[k][m] for k in keys], 1)  # noqa: E731
    sc = StandardScaler().fit(Xf(tr)); best = None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(sc.transform(Xf(tr)), D["y"][tr])
        s = balanced_accuracy_score(D["y"][va], m.predict(sc.transform(Xf(va))))
        if best is None or s > best[0]:
            best = (s, m)
    return lambda E: best[1].predict(sc.transform(np.concatenate([E[k][E["mode"] == "test"] for k in keys], 1)))


def run(S, T, nS, nT, sets, chance):
    say(f"\n## {nS} -> {nT}")
    say("| features | in-domain | cross | oracle | TR |"); say("|---|---|---|---|---|")
    res = {}
    yT, gT = T["y"][T["mode"] == "test"], T["grp"][T["mode"] == "test"]
    for name, keys in sets.items():
        pS = fit(S, keys); pT = fit(T, keys)
        ins = 100 * balanced_accuracy_score(S["y"][S["mode"] == "test"], pS(S))
        pc, po = pS(T), pT(T)
        c, o = 100 * balanced_accuracy_score(yT, pc), 100 * balanced_accuracy_score(yT, po)
        tr = (c - chance) / (o - chance) if o - chance >= 3 else np.nan
        res[name] = (pc, po, c, o)
        say(f"| {name} | {ins:.1f} | {c:.1f} | {o:.1f} | {'undefined' if np.isnan(tr) else f'{tr:+.2f}'} |")
    vids = np.unique(gT); ix = {v: np.where(gT == v)[0] for v in vids}
    out = {}
    for mod, (cn, en) in {"audio": ("audio(concept)", "audio(WavLM)"), "face": ("face(concept=FER logits)", "face(ViT-FER emb)")}.items():
        pcC, poC, _, oC = res[cn]; pcE, poE, _, oE = res[en]
        if oC - chance < 3 or oE - chance < 3:
            say(f"- {mod}: undefined (target oracle near chance)"); out[mod] = False; continue
        d = []
        for _ in range(1000):
            s = np.concatenate([ix[v] for v in rng.choice(vids, len(vids))])
            a = 100 * balanced_accuracy_score(yT[s], poC[s]) - chance; b = 100 * balanced_accuracy_score(yT[s], poE[s]) - chance
            if a > 1 and b > 1:
                d.append((100 * balanced_accuracy_score(yT[s], pcC[s]) - chance) / a - (100 * balanced_accuracy_score(yT[s], pcE[s]) - chance) / b)
        lo, hi = np.percentile(d, [2.5, 97.5])
        say(f"- {mod}: TR(concept) - TR(embedding) = {np.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]"); out[mod] = lo > 0
    fus_ok = res["fusion(concepts)"][2] >= res["fusion(embeddings)"][2]
    say(f"- concept-fusion cross bacc {res['fusion(concepts)'][2]:.1f} vs embedding-fusion {res['fusion(embeddings)'][2]:.1f}")
    return out["audio"] and out["face"] and fus_ok


kmosi = lambda D: [f"{v}_{c}" for v, c in zip(D["video_id"], D["clip_id"])]  # noqa: E731
M = binary(merge("mosi_p0.npz", "mosi_p0c.npz", kmosi)); Sm = binary(merge("sims_p0.npz", "sims_p0c.npz", kmosi))
HAVE_CM = os.path.exists(os.path.join(R, "cremad_p0c.npz")) and os.path.exists(os.path.join(R, "meld_p0c.npz"))
if HAVE_CM:
    Cr = merge("cremad_p0b.npz", "cremad_p0c.npz", lambda D: list(D["id"])); Me = merge("meld_p0b.npz", "meld_p0c.npz", lambda D: list(D["id"]))
    classes = sorted(set(Cr["label"]) & set(Me["label"])); Cr, Me = multi(Cr, classes), multi(Me, classes)
SET3 = {"text(e5)": ["text"], "text(concept)": ["t_concept"], "audio(WavLM)": ["a_wavlm"], "audio(concept)": ["a_concept"],
        "face(ViT-FER emb)": ["v_emb"], "face(concept=FER logits)": ["v_logits"],
        "fusion(embeddings)": ["text", "a_wavlm", "v_emb"], "fusion(concepts)": ["t_concept", "a_concept", "v_logits"]}
SET2 = {k: v for k, v in SET3.items() if not k.startswith("text")}
SET2["fusion(embeddings)"] = ["a_wavlm", "v_emb"]; SET2["fusion(concepts)"] = ["a_concept", "v_logits"]
votes = [run(M, Sm, "MOSI", "SIMS", SET3, 50.0), run(Sm, M, "SIMS", "MOSI", SET3, 50.0)]
if HAVE_CM:
    votes.append(run(Cr, Me, "CREMA-D", "MELD", SET2, 100.0 / len(classes)))
    say(f"\nDirections supporting H: {sum(votes)}/3 -> H premise {'HOLDS' if sum(votes) >= 2 else 'DOES NOT HOLD'}")
else:
    say(f"\nPARTIAL (CREMA-D->MELD pending): directions supporting H so far {sum(votes)}/2; no decision yet")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
