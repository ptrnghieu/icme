"""Protocol v3, step 1 (CLAUDE.md §4.7), SYMMETRIC version: the same headroom map on several datasets, each with its official
protocol, to keep only failure types that RECUR across sources (Chinese TV: MER2024, M3ED, CH-SIMS; English vlogs: MOSI;
US sitcom: MELD). Descriptive; the gate is applied on the joint table:
  keep a slice type iff its oracle ceiling >= 2 x MDE on >= 3 datasets (>= 2 if only 2 datasets have that slice).
Harness (same for all): frozen features -> multinomial logistic regression, C in {0.01, 0.1, 1} by validation WF1; early fusion =
concatenation. Metric WF1 (weighted F1) everywhere so that ceilings are comparable (MER2024's WAF is the same quantity).
  MER2024: 5-fold CV on Train&Val (fold seed 0), 6 classes; A chinese-hubert-large, V CLIP-L/14 mean of 8 frames, T roberta-wwm-large.
  M3ED: official TV-independent split (train = the 6,000-clip sample of P11, val 1,500, test all 4,201), 7 classes; A chinese-hubert
        mean pool (P11 cache), T multilingual-e5-base; no video (not released).
  CH-SIMS, CMU-MOSI: official splits, binary polarity on non-zero labels (Acc-2/F1 convention); A WavLM-base-plus, V face ViT-FER
        embedding, T multilingual-e5-base (P0 caches).
  MELD: official train/dev/test, 7 classes; same encoders as CH-SIMS/MOSI (added when its train features exist).
Ceiling of a slice = WF1 after setting predictions of that slice's error clips to the true label, minus baseline WF1.
MDE = 2.8 x (cluster-)bootstrap SE of WF1(early, all modalities) - WF1(early without the modality whose removal changes WF1 least)
  (the e06 definition, AVT vs AV; first draft used late fusion, which is weak on the split datasets and inflated the MDE).
NOTE: this is a harness, not the strongest published model; slices that pass the gate must be re-checked on a real baseline (step 2).
Usage: python -I d1_headroom.py <scratch dir> <dataset,...> [out.md]
"""
import csv, json, os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SP = sys.argv[1]; WANT = sys.argv[2].split(","); OUT = sys.argv[3] if len(sys.argv) > 3 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
wf1 = lambda y, p: 100 * f1_score(y, p, average="weighted")  # noqa: E731


# ------------------------------------------------------------------------------------------------ loaders
def e5(texts, cache):
    if os.path.exists(cache):
        return np.load(cache)["E"]
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"); m = AutoModel.from_pretrained("intfloat/multilingual-e5-base").eval(); E = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(["query: " + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = m(**b).last_hidden_state; mk = b["attention_mask"].unsqueeze(-1).float(); E.append(((h * mk).sum(1) / mk.sum(1)).numpy())
    E = np.concatenate(E).astype(np.float32); np.savez(cache, E=E); return E


def load_mer24():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mer24")); import mer24_screen as M
    c = os.path.join(SP, "mer24_cache"); D = M.load_cache(c); y, val, cls = M.labels(D, os.path.join(c, "label-disdim.csv")); ok = D["ok"].astype(bool)
    V = D["v_clip"].astype(np.float32).sum(1) / np.maximum(D["n_frames"], 1)[:, None]
    from sklearn.cluster import KMeans
    ident = KMeans(150, n_init=4, random_state=0).fit_predict(D["spk"][ok].astype(np.float32))
    return dict(F={"A": D["a_hub"][ok], "V": V[ok], "T": D["t_rob"][ok]}, y=y[ok], cls=cls, cv=5, clusters=np.arange(ok.sum()),
                ident=ident, tlen=D["t_len"][ok].astype(float), dur=D["dur"][ok], amb=None)


def load_m3ed():
    MREPO = os.path.join(SP, "dl/m3ed/repo"); MWAV = os.path.join(SP, "dl/m3ed/audio/wav/modality_speech")
    A = json.load(open(os.path.join(MREPO, "annotation.json"))); CL = ["Neutral", "Happy", "Sad", "Anger", "Disgust", "Fear", "Surprise"]
    spl = {s: [l.strip() for l in open(os.path.join(MREPO, f"splitInfo/movie_list_{s}.txt")) if l.strip()] for s in ("train", "val", "test")}
    rows = {s: [] for s in spl}
    for s, mvs in spl.items():
        for mv in mvs:
            for did, dia in A.get(mv, {}).items():
                for uid, u in dia["Dialog"].items():
                    f = os.path.join(MWAV, f"{u['Speaker']}_{uid}.wav")
                    if os.path.exists(f):
                        ea = u["EmoAnnotation"]; fin = ea["final_main_emo"]
                        agree = sum(ea.get(f"EmoAnnotator{k}") == fin for k in (1, 2, 3))
                        rows[s].append(dict(uid=uid, y=CL.index(fin), dia=did, text=u.get("Text", ""), agree=agree,
                                            who=mv + ":" + dia["SpeakerInfo"].get(u["Speaker"], {}).get("Name", u["Speaker"])))
    rng = np.random.default_rng(0)  # identical sampling to explore/tone/p11_tone.py
    pick = lambda rs, n: [rs[i] for i in sorted(rng.choice(len(rs), min(n, len(rs)), replace=False))]  # noqa: E731
    S = {"train": pick(rows["train"], 6000), "val": pick(rows["val"], 1500), "test": rows["test"]}
    items = S["train"] + S["val"] + S["test"]; mode = np.array(["train"] * len(S["train"]) + ["valid"] * len(S["val"]) + ["test"] * len(S["test"]))
    Aud = []
    for k, v in S.items():
        d = np.load(os.path.join(SP, "p11", f"hubert_{len(v)}_{v[0]['uid']}.npz")); X, off = d["X"], d["off"]
        Aud += [X[a:b].astype(np.float32).mean(0) for a, b in zip(off[:-1], off[1:])]
    T = e5([it["text"] for it in items], os.path.join(SP, "d1", "m3ed_e5.npz"))
    return dict(F={"A": np.stack(Aud), "T": T}, y=np.array([it["y"] for it in items]), cls=CL, mode=mode,
                clusters=np.array([it["dia"] for it in items]), ident=np.array([it["who"] for it in items]),
                tlen=np.array([len(it["text"]) for it in items], float), dur=None, amb=np.array([it["agree"] < 3 for it in items]))


def load_p0(name):
    d = np.load(os.path.join(SP, "xling/run", f"{name}_p0.npz"), allow_pickle=True); ok = d["ok"].astype(bool) & (d["label"] != 0)
    lab = {}
    nk = lambda c: str(c).lstrip("0") or "0"  # noqa: E731
    with open(os.path.join(SP, "xling", {"sims": "sims_label_h.csv", "mosi": "mosi_label.csv"}[name]), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab[(r["video_id"], nk(r["clip_id"]))] = r
    keys = [(str(v), nk(c)) for v, c in zip(d["video_id"], d["clip_id"])]
    print(name, "label rows matched", sum(k in lab for k in keys), "/", len(keys), flush=True)
    tlen = np.array([len(lab.get(k, {}).get("text", "")) for k in keys], float)
    amb = None
    if name == "sims":
        s = lambda x: np.sign(float(x)) if x not in ("", None) else 0  # noqa: E731
        amb = np.array([any(s(lab[k][c]) != np.sign(float(lab[k]["label"])) for c in ("label_T", "label_A", "label_V")) if k in lab else False for k in keys])
    y = (d["label"] > 0).astype(int)
    out = dict(F={"A": d["a_wavlm"], "V": d["v_emb"], "T": d["text"]}, y=y, cls=["negative", "positive"], mode=d["mode"].astype(str),
               clusters=d["video_id"].astype(str), ident=d["video_id"].astype(str), tlen=tlen, dur=None, amb=amb)
    for k in ("F",):
        out[k] = {m: v[ok].astype(np.float32) for m, v in out[k].items()}
    for k in ("y", "mode", "clusters", "ident", "tlen", "amb"):
        if out[k] is not None:
            out[k] = out[k][ok]
    return out


def load_meld():
    """Official MELD train/dev/test (13,707 utterances with video). Features: P0b extractor; earlier dev/test cache + new files."""
    man = os.path.join(SP, "d1", "meld_full_manifest.csv"); parts = [os.path.join(SP, "xling/run/meld_p0b.npz")] + [os.path.join(SP, "d1", f"meld_new_{h}.npz") for h in "ab"]
    if not all(os.path.exists(p) for p in [man] + parts):
        return None
    feat = {}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for i, k in enumerate(d["id"]):
            if d["ok"][i]:
                feat[str(k)] = (d["a_wavlm"][i], d["v_emb"][i], d["text"][i])
    rows = [r for r in csv.DictReader(open(man, encoding="utf-8")) if r["id"] in feat]; CL = sorted({r["label"] for r in rows})
    print("meld rows with features", len(rows), flush=True)
    return dict(F={m: np.stack([feat[r["id"]][j] for r in rows]).astype(np.float32) for j, m in enumerate("AVT")},
                y=np.array([CL.index(r["label"]) for r in rows]), cls=[{"NEU": "neutral"}.get(c, c) for c in CL], mode=np.array([r["mode"] for r in rows]),
                clusters=np.array([r["group"] for r in rows]), ident=np.array([r["speaker"] for r in rows]),
                tlen=np.array([len(r["text"]) for r in rows], float), dur=None, amb=None)


LOAD = {"mer24": load_mer24, "m3ed": load_m3ed, "sims": lambda: load_p0("sims"), "mosi": lambda: load_p0("mosi"), "meld": load_meld}


# ------------------------------------------------------------------------------------------------ models
def fit(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr); best = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr); s = wf1(yva, m.predict(sc.transform(Xva)))
        if best is None or s > best[0]:
            best = (s, m)
    return lambda X: best[1].predict_proba(sc.transform(X))


def probs(d):
    """Returns eval index, unimodal + early probs on eval, and late-fusion predictions on eval."""
    F, y = d["F"], d["y"]; mods = list(F); K = len(d["cls"])
    Fx = dict(F); Fx["early"] = np.concatenate([F[m] for m in mods], 1)
    for m in mods:
        if len(mods) > 2:
            Fx[f"early-{m}"] = np.concatenate([F[o] for o in mods if o != m], 1)
    if d.get("cv"):
        from sklearn.model_selection import StratifiedKFold
        P = {k: np.zeros((len(y), K)) for k in Fx}; rng = np.random.default_rng(0)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F[mods[0]], y):
            tr = rng.permutation(tr); nv = len(tr) // 10; va, tr2 = tr[:nv], tr[nv:]
            for k, X in Fx.items():
                P[k][te] = fit(X[tr2], y[tr2], X[va], y[va])(X[te])
        Z = np.concatenate([np.log(P[m] + 1e-6) for m in mods], 1); late = np.zeros(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(Z, y):
            nv = len(tr) // 10; late[te] = fit(Z[tr[nv:]], y[tr[nv:]], Z[tr[:nv]], y[tr[:nv]])(Z[te]).argmax(1)
        return np.arange(len(y)), P, late
    tr, va, te = (d["mode"] == "train"), (d["mode"] == "valid"), (d["mode"] == "test")
    P, Pva = {}, {}
    for k, X in Fx.items():
        f = fit(X[tr], y[tr], X[va], y[va]); P[k] = f(X[te]); Pva[k] = f(X[va])
    Zva = np.concatenate([np.log(Pva[m] + 1e-6) for m in mods], 1); Zte = np.concatenate([np.log(P[m] + 1e-6) for m in mods], 1)
    h = len(Zva) // 2; late = fit(Zva[:h], y[va][:h], Zva[h:], y[va][h:])(Zte).argmax(1)
    return np.where(te)[0], P, late


# ------------------------------------------------------------------------------------------------ map
def headroom(name, d):
    idx, P, late = probs(d); y = d["y"][idx]; mods = [m for m in d["F"]]; pe = P["early"].argmax(1); base = wf1(y, pe); err = pe != y; ne = err.sum()
    cl = d["clusters"][idx]; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(1); bs = []
    for _ in range(1000):
        s = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(wf1(y[s], pe[s]) - wf1(y[s], late[s]))
    near = [k for k in P if k.startswith("early-")]
    alt = min(near, key=lambda k: abs(wf1(y, P[k].argmax(1)) - base)) if near else None
    pa = P[alt].argmax(1) if alt else late; bs = []
    for _ in range(1000):
        s = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(wf1(y[s], pe[s]) - wf1(y[s], pa[s]))
    mde = 2.8 * np.std(bs)
    say(f"\n## {name}: eval n={len(y)}, classes {len(d['cls'])}, modalities {'+'.join(mods)}")
    say(f"- early fusion WF1 {base:.2f}; late fusion {wf1(y, late):.2f}; unimodal " + ", ".join(f"{m} {wf1(y, P[m].argmax(1)):.2f}" for m in mods))
    say(f"- MDE reference pair: early vs {alt or 'late'} ({wf1(y, pa):.2f})")
    say(f"- MDE {mde:.2f} -> gate 2 x MDE = {2 * mde:.2f}; errors {ne} ({100 * ne / len(y):.1f}%)")
    R = {}

    def ceil(key, mask, label):
        p = pe.copy(); m = mask & err; p[m] = y[m]; g = wf1(y, p) - base; R[key] = (g, m.sum(), 2 * mde)
        say(f"| {label} | {mask.sum()} | {m.sum()} | {100 * m.sum() / ne:.1f} | {g:+.2f} | {'PASS' if g >= 2 * mde else ''} |")
    say("\n| slice | clips | errors in slice | % of errors | ceiling WF1 | >= 2 x MDE |"); say("|---|---|---|---|---|---|")
    c = {m: P[m].argmax(1) == y for m in mods}; anyc = np.any([c[m] for m in mods], 0)
    ceil("fusion_failure", anyc, "fusion failure (some unimodal correct)"); ceil("none_correct", ~anyc, "no unimodal correct")
    for m in mods:
        only = c[m] & np.all([~c[o] for o in mods if o != m], 0); ceil(f"only_{m}", only, f"  only {m} correct")
    conf = np.stack([P[m].max(1) for m in mods], 1); sel = np.stack([P[m].argmax(1) for m in mods], 1)[np.arange(len(y)), conf.argmax(1)]
    ff = anyc & err; R["conf_selectable"] = 100 * (sel[ff] == y[ff]).mean()
    for i, cn in enumerate(d["cls"]):
        ceil(f"class:{cn.lower()}", y == i, f"true class {cn}")
    neu = [i for i, cn in enumerate(d["cls"]) if cn.lower().startswith("neu")]
    if neu:
        ceil("to_neutral", (y != neu[0]) & (pe == neu[0]), "emotional -> predicted neutral")
    if d.get("tlen") is not None:
        t = d["tlen"][idx]; q = np.quantile(t, 0.25); ceil("short_text", t <= q, f"shortest text quartile (<= {q:.0f} chars)")
    if d.get("dur") is not None:
        t = d["dur"][idx]; q = np.quantile(t, 0.25); ceil("short_clip", t <= q, f"shortest duration quartile (<= {q:.1f} s)")
    if d.get("amb") is not None:
        ceil("ambiguous", d["amb"][idx], "ambiguous label (annotator / unimodal-label disagreement)")
        say(f"- error rate: ambiguous {100 * err[d['amb'][idx]].mean():.1f}% vs unambiguous {100 * err[~d['amb'][idx]].mean():.1f}%")
    ide = d["ident"][idx]; ui, inv = np.unique(ide, return_inverse=True); cnt = np.bincount(inv)
    big = cnt[inv] >= 5; er = np.bincount(inv, err) / np.maximum(cnt, 1)
    sd_obs = np.std(er[cnt >= 5]); sd_perm = [np.std((np.bincount(inv, rng.permutation(err)) / np.maximum(cnt, 1))[cnt >= 5]) for _ in range(200)]
    worst = np.argsort(-np.where(cnt >= 5, er, -1))[: max(1, int(0.1 * (cnt >= 5).sum()))]
    ceil("worst_identities", np.isin(inv, worst) & big, "10% worst identities (>= 5 clips)")
    say(f"- identity concentration: sd of per-identity error rate {sd_obs:.3f} vs permutation {np.mean(sd_perm):.3f} (97.5% {np.percentile(sd_perm, 97.5):.3f}); identities with >= 5 clips: {(cnt >= 5).sum()}")
    say(f"- among fusion failures, the most confident unimodal is correct in {R['conf_selectable']:.1f}%; confidence-selection rule WF1 {wf1(y, sel):.2f} vs early {base:.2f}")
    return R


res = {}
for nm in WANT:
    d = LOAD[nm]()
    if d is None:
        say(f"\n## {nm}: features not available yet; skipped"); continue
    res[nm] = headroom(nm, d)
say("\n## Joint gate (slice types passing 2 x MDE)")
keys = sorted({k for r in res.values() for k in r if k != "conf_selectable"})
say("| slice type | " + " | ".join(res) + " | datasets passing |"); say("|---|" + "---|" * (len(res) + 1))
for k in keys:
    cells, npass, nhave = [], 0, 0
    for nm in res:
        if k in res[nm]:
            g, n, gate = res[nm][k]; nhave += 1; npass += g >= gate; cells.append(f"{g:+.2f}{' ✔' if g >= gate else ''}")
        else:
            cells.append("—")
    say(f"| {k} | " + " | ".join(cells) + f" | {npass}/{nhave} |")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
