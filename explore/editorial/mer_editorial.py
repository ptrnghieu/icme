"""E' : post-production ("editorial") cues in MER2024, final pilot of branch E (CLAUDE.md §4.6; ideas/round4_original_ideas_2026-11-21.md).

Runs on the USER's Kaggle (GPU). Prints/writes ONLY aggregate statistics (MER licence: no redistribution of data/annotations).

PRE-REGISTERED DESIGN (fixed 2026-11-22, before any MER2024 result):
  Data: MER2024 Train&Val (label-disdim.csv, 5,030 clips, 6 classes), official protocol = 5-fold CV, metric WAF (2404.17113 §3, Tab. 4).
  Features (frozen): audio WavLM-base-plus layer-avg mean on MIX / VOCALS / ACCOMPANIMENT (Demucs htdemucs, accompaniment = sum of
    non-vocal stems); AudioSet tags (AST) on MIX; visual CLIP ViT-L/14 image embedding (mean of frames at 25/50/75%) on RAW frames and
    on frames with all EasyOCR text boxes blurred (MASKED); text multilingual-e5-base on the Chinese transcript.
  Cues: music cue = max AST prob over MUSIC classes >= 0.2 (same rule as P6); subtitle cue = any EasyOCR box with centre in the
    bottom 35% of the frame and width >= 15% of frame width, in >= 1 of the 3 frames.
  Source groups (leakage control; MER2024 has no source-video id): agglomerative clustering (cosine, average linkage) of the RAW
    middle-frame CLIP embedding with the distance threshold calibrated on CH-SIMS against true video ids (GROUP_THRESHOLD below).
  Classifier: multinomial logistic regression (L2, full-batch L-BFGS in torch) on standardised features, C in {0.01, 0.1, 1} chosen by WAF on an inner 10% split of each
    training fold. Out-of-fold predictions for all clips; 3 fold seeds; scheme G = 5 grouped folds on source groups, random size-balanced group-to-fold assignment per seed (PRIMARY),
    scheme R = StratifiedKFold(5) (official-style, secondary).
  Tests (WAF, mean over 3 seeds; group-bootstrap over source groups, 1000 draws; 98.3% CI = Bonferroni over 3 tests):
    (A) stream separation : WAF(A_voc+acc) - max(WAF(A_mix), WAF(A_mix+voc))    needs music-cue prevalence >= 15%
    (V) subtitle masking   : WAF(F_mask) - WAF(F_raw), F = T + A_mix + V            needs subtitle prevalence >= 15%
    (M) both               : WAF(T + A_voc+acc + V_mask) - WAF(F_raw)
  A test PASSES iff, under scheme G, delta >= +2.0 WAF points and the 98.3% CI excludes 0, its prevalence condition holds, and the
  point estimate under scheme R is > 0.  E' HOLDS iff >= 1 test passes; otherwise branch E is CLOSED (no further pilots).
  Guard: if the largest derived group holds > 20% of clips, scheme G is invalid and the verdict is INCONCLUSIVE.
  Power: MDE = 2.8 x bootstrap SE of each delta is printed next to it (reported, not used to change the decision).
  Descriptive only: E-only (AST music+audience tags) WAF under R vs G; V_raw vs V_mask; A_mix accuracy when the E-only prediction
  agrees / disagrees with the label, split by music cue; overlap of MER2024 Train&Val names with MER2025 track1 names.

Usage:
  python mer_editorial.py extract --videos DIR --labels label-disdim.csv --trans label-transcription.csv --cache DIR [--hours 9.5] [--limit N]
  python mer_editorial.py analyze --labels label-disdim.csv --cache DIR --out e_report.md [--mer25 track1_train_disdim.csv] [--mer25test track1_test_dis.csv]
"""
import argparse, glob, os, re, subprocess, sys, tempfile, time
import numpy as np

GROUP_THRESHOLD = None  # set from CH-SIMS calibration (explore/editorial/group_calibration_chsims.md) before running
if os.environ.get("E_TEST_GROUP_THRESHOLD"):  # local smoke tests only; never set in the Kaggle notebook
    GROUP_THRESHOLD = float(os.environ["E_TEST_GROUP_THRESHOLD"])
MUSIC = re.compile(r"(?i)\bmusic\b|instrument|orchestra|song|piano|guitar|violin|drum|synthesizer")
AUDIENCE = re.compile(r"(?i)laugh|giggle|chuckle|snicker|applause|clapping|cheering|crowd")
PART = int(os.environ.get("E_PART", "250"))


# ----------------------------------------------------------------------------------------------------------------- extract
def load_models(dev):
    import torch, easyocr
    from demucs.pretrained import get_model
    from transformers import (ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor, WavLMModel, CLIPModel,
                              CLIPProcessor)
    m = dict(torch=torch, dev=dev)
    m["demucs"] = get_model("htdemucs").to(dev).eval()
    m["ast_fe"] = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    m["ast"] = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
    m["wfe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    m["wavlm"] = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    m["clip_p"] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    m["clip"] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    m["ocr"] = easyocr.Reader(["ch_sim", "en"], gpu=(dev == "cuda"), verbose=False)
    m["ast_labels"] = [m["ast"].config.id2label[i] for i in range(m["ast"].config.num_labels)]
    return m


def ffmpeg_wav(src, dst, sr, ch):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", str(ch), "-ar", str(sr), dst], check=False, timeout=120)
    return os.path.exists(dst)


def resample(x, sr_in, sr_out):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(x).float()[None, None]
    n = int(round(len(x) * sr_out / sr_in))
    return F.interpolate(t, size=n, mode="linear", align_corners=False)[0, 0].numpy()


def separate(m, wav44):
    """returns (vocals, accompaniment) mono float32 at 16 kHz."""
    import soundfile as sf
    from demucs.apply import apply_model
    torch = m["torch"]
    x, sr = sf.read(wav44, dtype="float32", always_2d=True)          # [T, 2]
    w = torch.from_numpy(x.T.copy())
    ref = w.mean(0); mu, sd = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        s = apply_model(m["demucs"], ((w - mu) / sd)[None].to(m["dev"]), device=m["dev"], split=True, overlap=0.25, progress=False)[0]
    s = (s * sd + mu).cpu().numpy()                                 # [sources, 2, T]
    names = list(m["demucs"].sources); vi = names.index("vocals")
    voc = s[vi].mean(0); acc = s[[i for i in range(len(names)) if i != vi]].sum(0).mean(0)
    return resample(voc, sr, 16000), resample(acc, sr, 16000)


def wavlm(m, x):
    torch = m["torch"]
    x = x[: 16000 * 20]
    if len(x) < 1600:
        return np.zeros(768, np.float32)
    with torch.no_grad():
        hs = m["wavlm"](**{k: v.to(m["dev"]) for k, v in m["wfe"](x, sampling_rate=16000, return_tensors="pt").items()},
                        output_hidden_states=True).hidden_states
    return torch.stack(hs[1:]).mean(0)[0].mean(0).float().cpu().numpy()


def ast_tags(m, x):
    torch = m["torch"]
    wins = [x[i:i + 160000] for i in range(0, min(len(x), 480000), 160000)]
    wins = [w for w in wins if len(w) >= 1600] or [np.pad(x, (0, max(0, 1600 - len(x))))]
    with torch.no_grad():
        p = torch.sigmoid(m["ast"](**{k: v.to(m["dev"]) for k, v in m["ast_fe"](wins, sampling_rate=16000, return_tensors="pt").items()}).logits)
    return p.max(0).values.float().cpu().numpy()


def frames(mp4):
    import cv2
    cap = cv2.VideoCapture(mp4); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; out = []
    for frac in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * (n - 1))); r, fr = cap.read()
        if r:
            out.append(fr)
    cap.release()
    return out


def text_boxes(m, fr):
    hl, fl = m["ocr"].detect(fr)
    boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in (hl[0] if hl else [])]   # x_min, x_max, y_min, y_max
    for poly in (fl[0] if fl else []):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]; boxes.append((int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))))
    return boxes


def mask(fr, boxes):
    import cv2
    out = fr.copy(); H, W = fr.shape[:2]
    blur = cv2.GaussianBlur(fr, (0, 0), sigmaX=max(H, W) / 40)
    for x0, x1, y0, y1 in boxes:
        dx, dy = int(0.1 * (x1 - x0)) + 2, int(0.1 * (y1 - y0)) + 2
        a, b, c, d = max(0, x0 - dx), min(W, x1 + dx), max(0, y0 - dy), min(H, y1 + dy)
        out[c:d, a:b] = blur[c:d, a:b]
    return out


def clip_emb(m, imgs):
    import cv2
    from PIL import Image
    torch = m["torch"]
    if not imgs:
        return np.zeros(768, np.float32), np.zeros(768, np.float32)
    pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in imgs]
    with torch.no_grad():
        e = m["clip"].get_image_features(**{k: v.to(m["dev"]) for k, v in m["clip_p"](images=pil, return_tensors="pt").items()})
    e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return e.mean(0), e[len(e) // 2]


def cmd_extract(a):
    import pandas as pd, soundfile as sf, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = pd.read_csv(a.labels); names = list(L["name"])
    if a.limit:
        names = names[: a.limit]
    vids = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.videos, "**", "*.*"), recursive=True)
            if p.lower().endswith((".mp4", ".avi", ".mkv"))}
    os.makedirs(a.cache, exist_ok=True); m = load_models(dev); t0 = time.time()
    mu_idx = [i for i, l in enumerate(m["ast_labels"]) if MUSIC.search(l)]; au_idx = [i for i, l in enumerate(m["ast_labels"]) if AUDIENCE.search(l)]
    for start in range(0, len(names), PART):
        part = os.path.join(a.cache, f"part_{start // PART:04d}.npz")
        if os.path.exists(part):
            continue
        if (time.time() - t0) / 3600 > a.hours:
            print("time budget reached; rerun to resume", flush=True); return
        rec = {k: [] for k in ["name", "ok", "a_mix", "a_voc", "a_acc", "rms", "tags_mu", "tags_au", "v_raw", "v_mask", "v_mid",
                               "sub_any", "text_area", "n_frames"]}
        with tempfile.TemporaryDirectory() as td:
            for nm in names[start:start + PART]:
                rec["name"].append(nm); p = vids.get(str(nm)); ok = False
                try:
                    if p is None:
                        raise FileNotFoundError
                    w16, w44 = os.path.join(td, "a16.wav"), os.path.join(td, "a44.wav")
                    if not (ffmpeg_wav(p, w16, 16000, 1) and ffmpeg_wav(p, w44, 44100, 2)):
                        raise RuntimeError("no audio")
                    mix, _ = sf.read(w16, dtype="float32")
                    voc, acc = separate(m, w44)
                    tg = ast_tags(m, mix)
                    fr = frames(p); bx = [text_boxes(m, f) for f in fr]
                    sub = any(any(((y0 + y1) / 2 > 0.65 * f.shape[0]) and (x1 - x0 >= 0.15 * f.shape[1]) for x0, x1, y0, y1 in b)
                              for f, b in zip(fr, bx))
                    area = np.mean([sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in b) / (f.shape[0] * f.shape[1]) for f, b in zip(fr, bx)]) if fr else 0.0
                    v_raw, v_mid = clip_emb(m, fr); v_mask, _ = clip_emb(m, [mask(f, b) for f, b in zip(fr, bx)])
                    vals = dict(a_mix=wavlm(m, mix), a_voc=wavlm(m, voc), a_acc=wavlm(m, acc),
                                rms=np.array([np.sqrt(np.mean(z ** 2)) for z in (mix, voc, acc)], np.float32),
                                tags_mu=tg[mu_idx], tags_au=tg[au_idx], v_raw=v_raw, v_mask=v_mask, v_mid=v_mid,
                                sub_any=float(sub), text_area=float(area), n_frames=len(fr))
                    ok = True
                except Exception as e:  # noqa: BLE001
                    vals = dict(a_mix=np.zeros(768), a_voc=np.zeros(768), a_acc=np.zeros(768), rms=np.zeros(3), tags_mu=np.zeros(len(mu_idx)),
                                tags_au=np.zeros(len(au_idx)), v_raw=np.zeros(768), v_mask=np.zeros(768), v_mid=np.zeros(768),
                                sub_any=0.0, text_area=0.0, n_frames=0)
                    print("fail", type(e).__name__, flush=True)
                rec["ok"].append(ok)
                for k, v in vals.items():
                    rec[k].append(np.asarray(v, np.float32))
        np.savez_compressed(part, **{k: np.array(v) for k, v in rec.items()})
        print(f"part {start // PART} done, {start + PART}/{len(names)} clips, {time.time() - t0:.0f}s", flush=True)
    print("extraction complete", flush=True)


# ----------------------------------------------------------------------------------------------------------------- analyze
def text_features(texts, dev):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"); tm = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(["query: " + str(t) for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            h = tm(**b).last_hidden_state; mk = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mk).sum(1) / mk.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def grouped_folds(g, k, rng):
    """random group-to-fold assignment, greedy size balancing (varies with the seed, unlike sklearn GroupKFold)."""
    groups = rng.permutation(np.unique(g)); sz = {v: (g == v).sum() for v in groups}; load = np.zeros(k); fold_of = {}
    for v in sorted(groups, key=lambda v: -sz[v] + rng.random() * 1e-3):
        f = int(np.argmin(load)); fold_of[v] = f; load[f] += sz[v]
    fid = np.array([fold_of[v] for v in g])
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)]


def fit_lr(X, y, C, n_cls, dev):
    """multinomial logistic regression (L2 = 1/(C n), as sklearn), full-batch L-BFGS in torch (GPU when available)."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(X.shape[1], n_cls, device=dev, requires_grad=True); b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + 0.5 * lam * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    Wd, bd = W.detach(), b.detach()
    return lambda Z: (torch.as_tensor(Z, dtype=torch.float32, device=dev) @ Wd + bd).argmax(1).cpu().numpy()


def oof(X, y, folds, rng, dev="cpu"):
    from sklearn.metrics import f1_score
    pred = np.zeros_like(y); K = int(y.max()) + 1
    for tr, te in folds:
        tr = rng.permutation(tr); nv = max(1, len(tr) // 10); va, tr2 = tr[:nv], tr[nv:]
        mu, sd = X[tr2].mean(0), X[tr2].std(0) + 1e-6; best = None
        for C in (0.01, 0.1, 1.0):
            f = fit_lr((X[tr2] - mu) / sd, y[tr2], C, K, dev)
            s_ = f1_score(y[va], f((X[va] - mu) / sd), average="weighted")
            if best is None or s_ > best[0]:
                best = (s_, C)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        pred[te] = fit_lr((X[tr] - mu) / sd, y[tr], best[1], K, dev)((X[te] - mu) / sd)
    return pred


def cmd_analyze(a):
    import pandas as pd, torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import StratifiedKFold
    assert GROUP_THRESHOLD is not None, "set GROUP_THRESHOLD from the CH-SIMS calibration first"
    out = []; say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    parts = sorted(glob.glob(os.path.join(a.cache, "part_*.npz"))); P = [np.load(p, allow_pickle=True) for p in parts]
    D = {k: np.concatenate([p[k] for p in P]) for k in P[0].files}
    L = pd.read_csv(a.labels).set_index("name"); ok = D["ok"].astype(bool)
    names = D["name"][ok]; D = {k: v[ok] for k, v in D.items()}
    cls = sorted(L["discrete"].unique()); y = np.array([cls.index(L.loc[n, "discrete"]) for n in names])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_df = pd.read_csv(a.trans).set_index("name") if a.trans else None
    T = text_features([tr_df.loc[n, "chinese"] if tr_df is not None and n in tr_df.index else "" for n in names], dev)
    say("# E′ report: post-production cues in MER2024 (aggregate statistics only)")
    say(f"- clips with features: {ok.sum()} / {len(ok)}; classes {cls}")
    mu = D["tags_mu"].max(1) >= 0.2; au = D["tags_au"].max(1) >= 0.2; sub = D["sub_any"] > 0.5
    say(f"- prevalence: music cue {100 * mu.mean():.1f}%, audience cue {100 * au.mean():.1f}%, subtitle cue {100 * sub.mean():.1f}%; mean text-box area {100 * D['text_area'].mean():.2f}% of frame")
    g = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=GROUP_THRESHOLD).fit_predict(D["v_mid"])
    sizes = np.bincount(g)
    say(f"- derived source groups: {len(sizes)} (threshold {GROUP_THRESHOLD}); size median {int(np.median(sizes))}, max {sizes.max()}, singletons {(sizes == 1).sum()}")
    degenerate = sizes.max() > 0.2 * len(g)  # pre-registered guard: one group > 20% of clips makes scheme G meaningless
    if degenerate:
        say("- **grouping degenerate (largest group > 20% of clips): scheme G invalid -> decision INCONCLUSIVE**")
    if a.mer25:
        n25 = set(pd.read_csv(a.mer25)["name"]); n25t = set(pd.read_csv(a.mer25test)["name"]) if a.mer25test else set()
        say(f"- overlap of MER2024 Train&Val names with MER2025 track1 train: {len(set(names) & n25)}; with MER2025 track1 test: {len(set(names) & n25t)}")
    E = np.concatenate([D["tags_mu"], D["tags_au"]], 1)
    feats = {"E_only": E, "A_mix": D["a_mix"], "A_voc": D["a_voc"], "A_acc": D["a_acc"],
             "A_voc+acc": np.concatenate([D["a_voc"], D["a_acc"]], 1), "A_mix+voc": np.concatenate([D["a_mix"], D["a_voc"]], 1),
             "V_raw": D["v_raw"], "V_mask": D["v_mask"], "T": T,
             "F_raw": np.concatenate([T, D["a_mix"], D["v_raw"]], 1), "F_mask": np.concatenate([T, D["a_mix"], D["v_mask"]], 1),
             "F_full": np.concatenate([T, D["a_voc"], D["a_acc"], D["v_mask"]], 1)}
    preds = {s: {k: [] for k in feats} for s in ("G", "R")}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        fG = grouped_folds(g, 5, rng); fR = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y))
        for k, X in feats.items():
            preds["G"][k].append(oof(X, y, fG, np.random.default_rng(seed), dev)); preds["R"][k].append(oof(X, y, fR, np.random.default_rng(seed), dev))
    waf = lambda yy, pp: 100 * f1_score(yy, pp, average="weighted")  # noqa: E731
    say("\n| features | WAF scheme G (source-grouped, primary) | WAF scheme R (random, official-style) | ACC G | ACC R |")
    say("|---|---|---|---|---|")
    for k in feats:
        say(f"| {k} | {np.mean([waf(y, p) for p in preds['G'][k]]):.2f} | {np.mean([waf(y, p) for p in preds['R'][k]]):.2f} | "
            f"{np.mean([100 * accuracy_score(y, p) for p in preds['G'][k]]):.2f} | {np.mean([100 * accuracy_score(y, p) for p in preds['R'][k]]):.2f} |")
    ix = {v: np.where(g == v)[0] for v in np.unique(g)}; ug = np.array(list(ix)); brng = np.random.default_rng(123)
    boots = [np.concatenate([ix[v] for v in brng.choice(ug, len(ug))]) for _ in range(1000)]

    def delta(scheme, new, base_list):
        def d_of(s):
            return np.mean([waf(y[s], preds[scheme][new][i][s]) - max(waf(y[s], preds[scheme][b][i][s]) for b in base_list) for i in range(3)])
        point = d_of(np.arange(len(y))); bs = np.array([d_of(s) for s in boots])
        lo, hi = np.percentile(bs, [0.85, 99.15]); return point, lo, hi, 2.8 * bs.std()

    tests = [("A stream separation", "A_voc+acc", ["A_mix", "A_mix+voc"], mu.mean() >= 0.15),
             ("V subtitle masking", "F_mask", ["F_raw"], sub.mean() >= 0.15),
             ("M both", "F_full", ["F_raw"], True)]
    say("\n## Pre-registered tests (WAF points; 98.3% group-bootstrap CI)")
    passed = 0
    for nm, new, base, prev_ok in tests:
        pG, loG, hiG, mde = delta("G", new, base); pR = delta("R", new, base)[0]
        ok_t = pG >= 2.0 and loG > 0 and prev_ok and pR > 0; passed += ok_t
        say(f"- {nm}: delta_G = {pG:+.2f} [{loG:+.2f}, {hiG:+.2f}] (MDE {mde:.2f}); delta_R = {pR:+.2f}; prevalence ok: {prev_ok} -> {'PASS' if ok_t else 'fail'}")
    verdict = "INCONCLUSIVE (degenerate grouping)" if degenerate else ("HOLDS" if passed else "DOES NOT HOLD -> branch E CLOSED")
    say(f"\n**E′ {verdict}** ({passed}/3 tests pass)")
    say("\n## Descriptive (not part of the decision)")
    for s in ("G", "R"):
        say(f"- E-only WAF scheme {s}: {np.mean([waf(y, p) for p in preds[s]['E_only']]):.2f}; majority-class WAF {waf(y, np.full_like(y, np.bincount(y).argmax())):.2f}")
    pe, pa = preds["G"]["E_only"][0], preds["G"]["A_mix"][0]
    for lab, msk in [("music cue, E-pred agrees", mu & (pe == y)), ("music cue, E-pred disagrees", mu & (pe != y)),
                     ("no music cue, E-pred agrees", ~mu & (pe == y)), ("no music cue, E-pred disagrees", ~mu & (pe != y))]:
        if msk.sum() >= 20:
            say(f"- A_mix accuracy (scheme G, seed 0), {lab}: {100 * (pa[msk] == y[msk]).mean():.1f} (n={msk.sum()})")
    for lab, msk in [("subtitle cue present", sub), ("no subtitle cue", ~sub)]:
        if msk.sum() >= 20:
            say(f"- V_raw vs V_mask accuracy (scheme G, seed 0), {lab}: {100 * (preds['G']['V_raw'][0][msk] == y[msk]).mean():.1f} vs "
                f"{100 * (preds['G']['V_mask'][0][msk] == y[msk]).mean():.1f} (n={msk.sum()})")
    open(a.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    e = sp.add_parser("extract"); e.add_argument("--videos"); e.add_argument("--labels"); e.add_argument("--trans", default="")
    e.add_argument("--cache"); e.add_argument("--hours", type=float, default=9.5); e.add_argument("--limit", type=int, default=0)
    n = sp.add_parser("analyze"); n.add_argument("--labels"); n.add_argument("--trans", default=""); n.add_argument("--cache"); n.add_argument("--out")
    n.add_argument("--mer25", default=""); n.add_argument("--mer25test", default="")
    a = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[a.cmd](a)
