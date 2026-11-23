"""P11: decisive test for direction T (ideas/round5_2026-11-23.md): is lexical tone a NUISANCE for audio emotion recognition in
Mandarin, for the Mandarin-pretrained encoder used by the MER2024 baseline (chinese-hubert-large)? CPU, frozen encoders.

PRE-REGISTERED (2026-11-23, before any result; user's objection: chinese-hubert is pretrained on Mandarin and may already handle tone):
  Data: M3ED (2205.10237), official TV-independent split (splitInfo/movie_list_*.txt), 7 classes (final_main_emo).
    train = 6,000 clips sampled from official train (seed 0); val = 1,500 sampled from official val; test = ALL official test clips.
    Metric: weighted F1 (WF1), as in 2205.10237 §5.3 (Tab. 5: utterance-level audio test WF1 46.09, for reference only).
  Tone targets (derived automatically, C5(d)): transcript `Text` -> pypinyin TONE3 with neutral tone = 5; tokens without a tone
    digit (punctuation, Latin) dropped. Clips with no tone token are kept for emotion but not used by the probe.
  Frames: layer-average of all transformer layers (50 Hz), capped at 6 s, standardised with train-frame mean/sd (z-space).
  Tone subspace: iterative nullspace projection (INLP) of a LINEAR CTC probe (z -> {blank, tone1..5}); probe trained on frames of
    2,000 train clips, evaluated on 500 val clips by tone error rate (TER, edit distance / reference length, greedy decoding);
    probe trained to convergence (Adam 3e-3, <= 40 epochs, early stopping on held-out TER, patience 4) [amended 2026-11-23 after
    the 120-clip smoke run showed a 6-epoch probe had not learned (TER 1.6); amended before any real result].
    Iterate (<= 10 times; each iteration adds the orthonormalised 6 probe rows) until held-out TER >= 0.9 x chance TER, where
    chance TER = 1 - (frequency of the most common tone in train targets).
  AMENDMENT 2 (2026-11-23, after the 800-clip SMOKE run, before any emotion result): the CTC criterion is confounded: TER mixes
    syllable segmentation with tone identity, and the "chance TER" assumed known lengths (smoke: TER 0.647 vs chance 0.686).
    PRIMARY tone subspace is therefore found where the emotion head looks: on the standardised pooled vectors [mean(z), std(z)],
    ridge (alpha 100) predicts the clip's tone histogram (proportions of tones 1-5; clips with >= 2 tone tokens). Tone composition is
    DECODABLE iff mean val R^2 >= 0.10; INLP then adds the 5 orthonormalised ridge directions per iteration until val R^2 < 0.02
    (<= 30 iterations). Projection is applied to the pooled vectors. Variance-matched random subspaces are drawn in the same pooled
    space. If not decodable -> T closed. Statistics and decision rule below are unchanged.
    The frame-level CTC probe is kept as DESCRIPTIVE only, against a probe trained on tone sequences shuffled across clips.
  AMENDMENT 3 (2026-11-23, after the 800-clip smoke run; STRICTER only): the smoke showed mean-only pooling beating mean+std and
    random-subspace removal also helping (regularisation). HOLDS now additionally requires
    D3 = WF1(TONE) - max(WF1(ORIG), WF1(ORIG_mean)) >= +0.5 (point estimate): the method must beat the best baseline pooling.
    The smoke used the first 800 test clips; the full run uses all test clips; no other change.
  [Original frame-level design, superseded:] Conditions (projection applied to every frame, z' = z - (zQ)Q^T, then pooling [mean(z'), std(z')] -> multinomial logistic
    regression, C in {0.01, 0.1, 1} by val WF1):
    ORIG (no projection); TONE (INLP subspace, rank r); RAND_1..5 (rank-r random subspaces inside the span of the top-K principal
    components of z, K chosen from {r, 2r, 4r, ..., D} so that the removed variance is closest to TONE's: variance-matched control);
    ISO (one isotropic random rank-r subspace, descriptive).
  Statistics: cluster bootstrap over dialogues (2,000 draws). D1 = WF1(TONE) - mean_k WF1(RAND_k); D2 = WF1(TONE) - WF1(ORIG).
    MDE = 2.8 x bootstrap SE of D1.
  DECISION (chinese-hubert only): if MDE > 1.5 -> UNDERPOWERED -> T closed (power rule). Else T HOLDS iff D1 >= +1.0 with CI
    lower bound > 0 AND D2 >= +0.5 (point estimate); otherwise T closed, no variants.
  Descriptive: the same pipeline on WavLM-base-plus (English-pretrained), to answer whether Mandarin pretraining removes the
    effect; mean-only pooling for ORIG and TONE.
Usage: python -I p11_tone.py <workdir> <m3ed repo dir> <m3ed wav dir> [out.md]      (P11_LIMIT=N for a smoke run)
"""
import json, os, sys, warnings
import numpy as np, soundfile as sf, torch
from pypinyin import pinyin, Style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore"); torch.set_num_threads(4)
W, MREPO, MWAV = sys.argv[1:4]; OUT = sys.argv[4] if len(sys.argv) > 4 else None; os.makedirs(W, exist_ok=True)
LIM = int(os.environ.get("P11_LIMIT", "0"))
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731
CL = ["Neutral", "Happy", "Sad", "Anger", "Disgust", "Fear", "Surprise"]

# ------------------------------------------------------------------------------------------------ data
A = json.load(open(os.path.join(MREPO, "annotation.json")))
spl = {s: [l.strip() for l in open(os.path.join(MREPO, f"splitInfo/movie_list_{s}.txt")) if l.strip()] for s in ("train", "val", "test")}
rows = {s: [] for s in spl}
for s, mvs in spl.items():
    for mv in mvs:
        for did, dia in A.get(mv, {}).items():
            for uid, u in dia["Dialog"].items():
                f = os.path.join(MWAV, f"{u['Speaker']}_{uid}.wav")
                if os.path.exists(f):
                    rows[s].append(dict(uid=uid, f=f, y=CL.index(u["EmoAnnotation"]["final_main_emo"]), dia=did, text=u.get("Text", "")))
rng = np.random.default_rng(0)
pick = lambda rs, n: [rs[i] for i in sorted(rng.choice(len(rs), min(n, len(rs)), replace=False))]  # noqa: E731
ntr, nva = (LIM, LIM // 2) if LIM else (6000, 1500)
S = {"train": pick(rows["train"], ntr), "val": pick(rows["val"], nva), "test": rows["test"][:LIM] if LIM else rows["test"]}
say("# P11: is lexical tone a nuisance for Mandarin audio emotion recognition? (M3ED)")
say(f"- official split sizes found: " + ", ".join(f"{k} {len(v)}" for k, v in rows.items()) + "; used: " + ", ".join(f"{k} {len(v)}" for k, v in S.items()))


def tones(t):
    out = []
    for (p,) in pinyin(t, style=Style.TONE3, neutral_tone_with_five=True):
        if p and p[-1] in "12345":
            out.append(int(p[-1]))
    return out


# ------------------------------------------------------------------------------------------------ features
def frames(enc, items):
    cache = os.path.join(W, f"{enc}_{len(items)}_{items[0]['uid']}.npz")
    if os.path.exists(cache):
        d = np.load(cache); X, off = d["X"], d["off"]  # load once: d["X"] re-reads the whole array on every access
        return [X[a:b] for a, b in zip(off[:-1], off[1:])]
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    mid = {"hubert": "TencentGameMate/chinese-hubert-large", "wavlm": "microsoft/wavlm-base-plus"}[enc]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(mid); m = AutoModel.from_pretrained(mid).eval(); out = []
    with torch.no_grad():
        for k, it in enumerate(items):
            x, sr = sf.read(it["f"], dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            x = x[: 16000 * 6]
            if len(x) < 8000:
                x = np.pad(x, (0, 8000 - len(x)))
            hs = m(**fe(x, sampling_rate=16000, return_tensors="pt"), output_hidden_states=True).hidden_states
            out.append(torch.stack(hs[1:]).mean(0)[0].numpy().astype(np.float16))
            if k % 1000 == 0:
                print(enc, k, len(items), flush=True)
    off = np.cumsum([0] + [len(o) for o in out]); np.savez(cache, X=np.concatenate(out), off=off); return out


# ------------------------------------------------------------------------------------------------ tone probe (CTC) + INLP
def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def train_probe(Ztr, Ttr, Zev, Tev, Q, seed=0):
    """Linear CTC probe trained to convergence: Adam lr 3e-3, <= 40 epochs, early stopping on held-out TER (patience 4)."""
    torch.manual_seed(seed); D = Ztr[0].shape[1]; lin = torch.nn.Linear(D, 6); opt = torch.optim.Adam(lin.parameters(), 3e-3)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True); Qt = torch.tensor(Q, dtype=torch.float32) if Q is not None else None
    proj = lambda z: z - (z @ Qt) @ Qt.T if Qt is not None else z  # noqa: E731
    Ttr_ = [torch.tensor(t) for t in Ttr]; Zt = [proj(torch.tensor(z, dtype=torch.float32)) for z in Ztr]; Ze = [proj(torch.tensor(z, dtype=torch.float32)) for z in Zev]

    def ter():
        errs = refs = 0
        with torch.no_grad():
            for z, t in zip(Ze, Tev):
                p = lin(z).argmax(-1).numpy(); seq = [int(c) for k, c in enumerate(p) if c != 0 and (k == 0 or c != p[k - 1])]
                errs += edit(seq, t); refs += len(t)
        return errs / max(refs, 1)
    idx = np.arange(len(Zt)); g = np.random.default_rng(seed); best, bw, bad = 9.0, None, 0
    for ep in range(40):
        g.shuffle(idx)
        for b in range(0, len(idx), 32):
            bb = idx[b:b + 32]; lp = torch.nn.utils.rnn.pad_sequence([lin(Zt[i]) for i in bb]).log_softmax(-1)
            tg = torch.cat([Ttr_[i] for i in bb]); il = torch.tensor([len(Zt[i]) for i in bb]); tl = torch.tensor([len(Ttr[i]) for i in bb])
            opt.zero_grad(); ctc(lp, tg, il, tl).backward(); opt.step()
        e = ter()
        if e < best - 1e-4:
            best, bw, bad = e, lin.weight.detach().clone().numpy(), 0
        else:
            bad += 1
            if bad >= 4:
                break
    return bw, best


def orth(M):
    q, _ = np.linalg.qr(M); return q


# ------------------------------------------------------------------------------------------------ emotion
def pool(Z, Q, mean_only=False):
    out = []
    for z in Z:
        z = z.astype(np.float32)
        if Q is not None:
            z = z - (z @ Q) @ Q.T
        out.append(z.mean(0) if mean_only else np.r_[z.mean(0), z.std(0)])
    return np.stack(out)


def fit_predict(Ftr, ytr, Fva, yva, Fte):
    sc = StandardScaler().fit(Ftr); best = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Ftr), ytr); s = f1_score(yva, m.predict(sc.transform(Fva)), average="weighted")
        if best is None or s > best[0]:
            best = (s, m)
    return best[1].predict(sc.transform(Fte))


def wf1(y, p):
    return 100 * f1_score(y, p, average="weighted")


def run(enc):
    say(f"\n## {enc}")
    Fr = {k: frames(enc, v) for k, v in S.items()}
    allf = np.concatenate(Fr["train"]).astype(np.float32); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    Z = {k: [(f.astype(np.float32) - mu) / sd for f in v] for k, v in Fr.items()}
    T = {k: [tones(it["text"]) for it in v] for k, v in S.items()}
    ptr = [i for i in range(min(2000, len(Z["train"]))) if T["train"][i] and len(T["train"][i]) <= len(Z["train"][i])]
    pev = [i for i in range(min(500, len(Z["val"]))) if T["val"][i] and len(T["val"][i]) <= len(Z["val"][i])]
    Ztr, Ttr = [Z["train"][i] for i in ptr], [T["train"][i] for i in ptr]; Zev, Tev = [Z["val"][i] for i in pev], [T["val"][i] for i in pev]
    sh = np.random.default_rng(3).permutation(len(Ttr)); Tsh = [Ttr[i] for i in sh]
    keep = [k for k in range(len(Ztr)) if len(Tsh[k]) <= len(Ztr[k])]
    _, ter = train_probe(Ztr, Ttr, Zev, Tev, None); _, ter_sh = train_probe([Ztr[k] for k in keep], [Tsh[k] for k in keep], Zev, Tev, None)
    say(f"- descriptive frame-level CTC probe: held-out TER {ter:.3f} vs shuffled-target probe {ter_sh:.3f}")
    # primary: tone composition in the pooled space
    F = {k: pool(v, None) for k, v in Z.items()}; scp = StandardScaler().fit(F["train"]); F = {k: scp.transform(v) for k, v in F.items()}
    H = {k: np.array([np.bincount(t, minlength=6)[1:] / max(len(t), 1) for t in T[k]]) for k in T}
    mk = {k: np.array([len(t) >= 2 for t in T[k]]) for k in T}
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def r2(Q):
        Xtr = F["train"][mk["train"]]; Xva = F["val"][mk["val"]]
        if Q is not None:
            Xtr = Xtr - (Xtr @ Q) @ Q.T; Xva = Xva - (Xva @ Q) @ Q.T
        m = Ridge(alpha=100).fit(Xtr, H["train"][mk["train"]]); return r2_score(H["val"][mk["val"]], m.predict(Xva)), m.coef_.T
    r0, Wc = r2(None); say(f"- tone-composition decodability from pooled features: val R^2 = {r0:.3f} (decodable iff >= 0.10)")
    if r0 < 0.10:
        say("- tone composition not decodable: nothing to remove"); return None
    Q = orth(Wc); hist = [r0]
    for it in range(30):
        rr, Wc = r2(Q); hist.append(rr)
        if rr < 0.02:
            break
        Q = orth(np.c_[Q, Wc - Q @ (Q.T @ Wc)])
    r = Q.shape[1]; say(f"- INLP val R^2 by iteration: {[round(h, 3) for h in hist]} -> tone subspace rank r = {r}")
    C = np.cov(F["train"].T); tot = np.trace(C); vt = np.trace(Q.T @ C @ Q) / tot; ev, U = np.linalg.eigh(C); U = U[:, ::-1]
    D = C.shape[0]; Ks = sorted({min(D, r * 2 ** j) for j in range(12)})

    def rand_q(K, seed):
        return orth(U[:, :K] @ np.random.default_rng(seed).normal(size=(K, r)))
    vK = {K: np.mean([np.trace(rand_q(K, s_).T @ C @ rand_q(K, s_)) / tot for s_ in range(3)]) for K in Ks}
    K = min(Ks, key=lambda k: abs(vK[k] - vt))
    say(f"- removed variance (pooled space): TONE {100 * vt:.2f}%; variance-matched random (K={K}) {100 * vK[K]:.2f}%; isotropic {100 * vK[D]:.2f}%")
    conds = {"ORIG": None, "TONE": Q, **{f"RAND_{k}": rand_q(K, 100 + k) for k in range(1, 6)}, "ISO": rand_q(D, 99)}
    y = {k: np.array([it["y"] for it in v]) for k, v in S.items()}; P = {}
    prj = lambda X, q: X if q is None else X - (X @ q) @ q.T  # noqa: E731
    for c, q in conds.items():
        P[c] = fit_predict(prj(F["train"], q), y["train"], prj(F["val"], q), y["val"], prj(F["test"], q))
    half = F["train"].shape[1] // 2
    for c, q in (("ORIG_mean", None), ("TONE_mean", Q)):
        P[c] = fit_predict(prj(F["train"], q)[:, :half], y["train"], prj(F["val"], q)[:, :half], y["val"], prj(F["test"], q)[:, :half])
    yt = y["test"]; say("\n| condition | test WF1 |"); say("|---|---|")
    for c in P:
        say(f"| {c} | {wf1(yt, P[c]):.2f} |")
    dia = np.array([it["dia"] for it in S["test"]]); ud = np.unique(dia); ix = {d: np.where(dia == d)[0] for d in ud}; bg = np.random.default_rng(1)
    R = [f"RAND_{k}" for k in range(1, 6)]
    d1f = lambda s: wf1(yt[s], P["TONE"][s]) - np.mean([wf1(yt[s], P[c][s]) for c in R])  # noqa: E731
    d2f = lambda s: wf1(yt[s], P["TONE"][s]) - wf1(yt[s], P["ORIG"][s])  # noqa: E731
    full = np.arange(len(yt)); b1, b2 = [], []
    for _ in range(2000):
        s = np.concatenate([ix[d] for d in bg.choice(ud, len(ud))]); b1.append(d1f(s)); b2.append(d2f(s))
    d1, d2 = d1f(full), d2f(full); lo1, hi1 = np.percentile(b1, [2.5, 97.5]); lo2, hi2 = np.percentile(b2, [2.5, 97.5]); mde = 2.8 * np.std(b1)
    say(f"\n- D1 = WF1(TONE) - mean WF1(RAND) = {d1:+.2f} [{lo1:+.2f}, {hi1:+.2f}]; MDE {mde:.2f} (dialogue clusters {len(ud)})")
    say(f"- D2 = WF1(TONE) - WF1(ORIG) = {d2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
    d3 = wf1(yt, P["TONE"]) - max(wf1(yt, P["ORIG"]), wf1(yt, P["ORIG_mean"]))
    say(f"- D3 = WF1(TONE) - best baseline pooling = {d3:+.2f}")
    return dict(d1=d1, lo1=lo1, d2=d2, d3=d3, mde=mde)


res = run("hubert")
if res is None:
    verdict = "tone not decodable -> T closed"
elif res["mde"] > 1.5:
    verdict = f"UNDERPOWERED (MDE {res['mde']:.2f} > 1.5) -> T closed (power rule)"
elif res["d1"] >= 1.0 and res["lo1"] > 0 and res["d2"] >= 0.5 and res["d3"] >= 0.5:
    verdict = "HOLDS"
else:
    verdict = "DOES NOT HOLD -> T closed"
say(f"\n**P11 verdict (chinese-hubert, pre-registered): {verdict}**")
say("\n_Descriptive: WavLM-base-plus (English-pretrained)_")
run("wavlm")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
