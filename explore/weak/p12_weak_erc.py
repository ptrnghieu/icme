"""P12: premise test for direction W (ideas/round6_newtasks_2026-11-23.md): multimodal ERC trained only with CONVERSATION-LEVEL
emotion sets (the DERC setting of AAAI 2025, which was text-only). CPU, frozen features, official splits.

PRE-REGISTERED (2026-11-23, before any result):
  Data. M3ED: official TV-independent split; utterances = the P11 sample (train 6,000 / val 1,500 / test all 4,201); modalities
    A (chinese-hubert mean) and T (multilingual-e5); 7 classes. MELD: official train/dev/test (13,707 utterances with video);
    A (WavLM), V (face ViT-FER embedding), T (e5); 7 classes (run when its features exist).
  Weak labels (derived automatically, C5(d), exactly as DERC: "adding the utterance-level emotions into the corresponding
    conversation-level emotion sets"): each training utterance gets as candidate set the emotion set of its WHOLE dialogue
    (all annotated utterances of the dialogue, also those not in the feature sample).
  Models (identical architecture for all conditions): linear softmax on standardised features (Adam lr 1e-2, 150 epochs, full
    batch), weight decay in {1e-4, 1e-3, 1e-2}; 3 seeds; test prediction = mean probability over seeds.
    FULL: cross-entropy on utterance labels; selection by val WF1 on utterance labels.
    WEAK: partial-label learning (PRODEN: loss -sum_{y in S} w_y log p(y|x), weights re-estimated each epoch as the model's
      probabilities renormalised over S, uniform start); selection by val SET ACCURACY (predicted label in the dialogue set), i.e.
      no utterance label is used anywhere in training or selection.
  Metric: test utterance-level WF1. Cluster bootstrap over test dialogues (2,000 draws).
  PRIMARY (per dataset): (i) headroom gap_M = WF1(FULL, M) - WF1(WEAK, M) for the multimodal input M (AT for M3ED, AVT for
    MELD); (ii) D = gap_T - gap_M (multimodal narrows the weak-supervision gap more than text alone).
  W premise HOLDS iff on BOTH datasets gap_M >= 2.0 AND D >= +1.0 with bootstrap CI lower bound > 0. MDE of D = 2.8 x SE.
  If M3ED fails, the branch is closed without waiting for MELD.
Usage: python -I p12_weak_erc.py <scratch dir> <m3ed|meld|both> [out.md]
"""
import csv, json, os, sys, warnings
import numpy as np, torch
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore"); torch.set_num_threads(4)
SP = sys.argv[1]; WANT = sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
wf1 = lambda y, p: 100 * f1_score(y, p, average="weighted")  # noqa: E731


def load_m3ed():
    MREPO = os.path.join(SP, "dl/m3ed/repo"); MWAV = os.path.join(SP, "dl/m3ed/audio/wav/modality_speech")
    A = json.load(open(os.path.join(MREPO, "annotation.json"))); CL = ["Neutral", "Happy", "Sad", "Anger", "Disgust", "Fear", "Surprise"]
    spl = {s: [l.strip() for l in open(os.path.join(MREPO, f"splitInfo/movie_list_{s}.txt")) if l.strip()] for s in ("train", "val", "test")}
    rows = {s: [] for s in spl}; dset = {}
    for s, mvs in spl.items():
        for mv in mvs:
            for did, dia in A.get(mv, {}).items():
                dset[did] = sorted({CL.index(u["EmoAnnotation"]["final_main_emo"]) for u in dia["Dialog"].values()})
                for uid, u in dia["Dialog"].items():
                    if os.path.exists(os.path.join(MWAV, f"{u['Speaker']}_{uid}.wav")):
                        rows[s].append(dict(uid=uid, y=CL.index(u["EmoAnnotation"]["final_main_emo"]), dia=did))
    rng = np.random.default_rng(0)  # identical sampling to explore/tone/p11_tone.py
    pick = lambda rs, n: [rs[i] for i in sorted(rng.choice(len(rs), min(n, len(rs)), replace=False))]  # noqa: E731
    S = {"train": pick(rows["train"], 6000), "val": pick(rows["val"], 1500), "test": rows["test"]}
    items = S["train"] + S["val"] + S["test"]; mode = np.array(["train"] * 6000 + ["valid"] * len(S["val"]) + ["test"] * len(S["test"]))
    Aud = []
    for k, v in S.items():
        d = np.load(os.path.join(SP, "p11", f"hubert_{len(v)}_{v[0]['uid']}.npz")); X, off = d["X"], d["off"]
        Aud += [X[a:b].astype(np.float32).mean(0) for a, b in zip(off[:-1], off[1:])]
    T = np.load(os.path.join(SP, "d1", "m3ed_e5.npz"))["E"]
    return dict(name="M3ED", F={"A": np.stack(Aud), "T": T}, multi="AT", y=np.array([it["y"] for it in items]), K=7, mode=mode,
                dia=np.array([it["dia"] for it in items]), sets=[dset[it["dia"]] for it in items])


def load_meld():
    man = os.path.join(SP, "d1", "meld_full_manifest.csv"); parts = [os.path.join(SP, "xling/run/meld_p0b.npz")] + [os.path.join(SP, "d1", f"meld_new_{h}.npz") for h in "ab"]
    if not all(os.path.exists(p) for p in [man] + parts):
        return None
    feat = {}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for i, k in enumerate(d["id"]):
            if d["ok"][i]:
                feat[str(k)] = (d["a_wavlm"][i], d["v_emb"][i], d["text"][i])
    allrows = list(csv.DictReader(open(man, encoding="utf-8"))); CL = sorted({r["label"] for r in allrows})
    dset = {}
    for r in allrows:
        dset.setdefault(r["group"], set()).add(CL.index(r["label"]))
    rows = [r for r in allrows if r["id"] in feat]
    return dict(name="MELD", F={m: np.stack([feat[r["id"]][j] for r in rows]).astype(np.float32) for j, m in enumerate("AVT")}, multi="AVT",
                y=np.array([CL.index(r["label"]) for r in rows]), K=len(CL), mode=np.array([r["mode"] for r in rows]),
                dia=np.array([r["group"] for r in rows]), sets=[sorted(dset[r["group"]]) for r in rows])


def train(Xtr, ytr, Mtr, wd, seed, weak):
    torch.manual_seed(seed); X = torch.tensor(Xtr); K = Mtr.shape[1]; lin = torch.nn.Linear(X.shape[1], K)
    opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=wd); Mt = torch.tensor(Mtr, dtype=torch.float32)
    W = Mt / Mt.sum(1, keepdim=True); yt = torch.tensor(ytr)
    for ep in range(150):
        opt.zero_grad(); lp = torch.log_softmax(lin(X), 1)
        loss = -(W * lp).sum(1).mean() if weak else torch.nn.functional.nll_loss(lp, yt)
        loss.backward(); opt.step()
        if weak:
            with torch.no_grad():
                p = torch.softmax(lin(X), 1) * Mt; W = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    return lambda Z: torch.softmax(lin(torch.tensor(Z)), 1).detach().numpy()


def run(d):
    y, mode, K = d["y"], d["mode"], d["K"]; tr, va, te = mode == "train", mode == "valid", mode == "test"
    M = np.zeros((len(y), K)); [M.__setitem__((i, s), 1) for i, ss in enumerate(d["sets"]) for s in ss]
    assert M[np.arange(len(y)), y].all(), "true label must be in the dialogue set"
    sz = M[tr].sum(1); say(f"\n## {d['name']} (train {tr.sum()}, val {va.sum()}, test {te.sum()}; {K} classes)")
    say(f"- dialogue-set size on training utterances: mean {sz.mean():.2f}, median {np.median(sz):.0f}, share with |S|=1 {100 * (sz == 1).mean():.1f}%")
    feats = {"T": d["F"]["T"], d["multi"]: np.concatenate([d["F"][m] for m in d["multi"]], 1)}
    P = {}
    for fk, X in feats.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = ((X - mu) / sd).astype(np.float32)
        for weak in (False, True):
            best = None
            for wd in (1e-4, 1e-3, 1e-2):
                fs = [train(Z[tr], y[tr], M[tr], wd, s, weak) for s in range(3)]; pv = np.mean([f(Z[va]) for f in fs], 0).argmax(1)
                score = M[va][np.arange(va.sum()), pv].mean() if weak else wf1(y[va], pv)
                if best is None or score > best[0]:
                    best = (score, wd, fs)
            P[(fk, weak)] = np.mean([f(Z[te]) for f in best[2]], 0).argmax(1)
            say(f"- {fk} {'WEAK' if weak else 'FULL'}: selected wd {best[1]:g}; test WF1 {wf1(y[te], P[(fk, weak)]):.2f}")
    yt = y[te]; dia = d["dia"][te]; ud = np.unique(dia); ix = {u: np.where(dia == u)[0] for u in ud}; rng = np.random.default_rng(1)
    m = d["multi"]
    gap = lambda f, s: wf1(yt[s], P[(f, False)][s]) - wf1(yt[s], P[(f, True)][s])  # noqa: E731
    full = np.arange(len(yt)); gT, gM = gap("T", full), gap(m, full); D = gT - gM; bs = []
    for _ in range(2000):
        s = np.concatenate([ix[u] for u in rng.choice(ud, len(ud))]); bs.append(gap("T", s) - gap(m, s))
    lo, hi = np.percentile(bs, [2.5, 97.5]); mde = 2.8 * np.std(bs)
    holds = gM >= 2.0 and D >= 1.0 and lo > 0
    say(f"- gap_T = {gT:.2f}; gap_{m} = {gM:.2f} (headroom); D = gap_T - gap_{m} = {D:+.2f} [{lo:+.2f}, {hi:+.2f}], MDE {mde:.2f}; test dialogues {len(ud)}")
    say(f"- {d['name']}: {'supports' if holds else 'does not support'} the W premise")
    return holds


say("# P12: weak (conversation-level) supervision for multimodal ERC: premise test for W")
ok = []
for nm in (["m3ed", "meld"] if WANT == "both" else [WANT]):
    d = load_m3ed() if nm == "m3ed" else load_meld()
    if d is None:
        say(f"\n## {nm}: features not ready; skipped"); continue
    ok.append(run(d))
    if nm == "m3ed" and not ok[-1]:
        say("\n**M3ED does not support the premise -> W closed (pre-registered rule)**"); break
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
