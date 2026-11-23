"""P10b: does learned attention pooling already do what peak-end pooling does? (amendment to P10, written 2026-11-23 while P10
was still extracting features, i.e. BEFORE any P10 result was printed; prompted by the user's objection that attention pooling is
flexible enough to attend to emotionally rich segments.)

PRE-REGISTERED (2026-11-23):
  Same data, splits, segments, target and test metric as P10 (reads the P10 caches <workdir>/{sims,mosi}_seg.npz).
  ATT   = additive attention pooling over the 1-s segments, a_i = softmax_i(v . tanh(W x_i + b)), W: 768 -> 128; regression head
          linear on [mean, sum_i a_i x_i]. ATT+P adds a learned position embedding for "index from the end" (0..19) inside the
          scorer, so attention CAN learn recency. Inputs standardised with train-segment statistics. Adam lr 1e-3, weight decay in
          {1e-4, 1e-2}, dropout 0.1, <= 200 epochs, early stopping on valid r (patience 20); variant and weight decay chosen on
          valid r. 5 seeds; prediction = mean over seeds (an ensemble: this favours ATT, conservative for peak-end).
  MPE   = P10's ridge on [mean, peak, end] (recomputed here identically).
  METHOD GATE (decides whether peak-end can be a METHOD contribution, C3): on BOTH datasets r(MPE) - r(ATT*) >= +0.02 with 95%
          video-cluster bootstrap CI excluding 0, where ATT* is the valid-selected attention model. Otherwise peak-end is not a
          method contribution and the branch is closed even if the P10 premise holds.
  Descriptive (answers "does attention attend to peaks/ends?"): mean attention mass on the P10 peak segment and on the last
          segment versus the uniform share 1/n, for test clips with n >= 3 segments.
Usage: python -I p10b_attention.py <p10 workdir> [out.md]
"""
import os, sys, warnings
import numpy as np, torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore"); torch.set_num_threads(2)
W = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
L = []; say = lambda s="": (print(s, flush=True), L.append(s))  # noqa: E731


class Att(torch.nn.Module):
    def __init__(self, d, pos):
        super().__init__(); self.W = torch.nn.Linear(d, 128); self.v = torch.nn.Linear(128, 1, bias=False)
        self.pos = torch.nn.Embedding(20, 128) if pos else None; self.drop = torch.nn.Dropout(0.1); self.head = torch.nn.Linear(2 * d, 1)

    def forward(self, X, M, R):  # X [B,T,D], M mask [B,T], R index-from-end [B,T]
        h = self.W(self.drop(X))
        if self.pos is not None:
            h = h + self.pos(R)
        s = self.v(torch.tanh(h)).squeeze(-1).masked_fill(~M, -1e9); a = torch.softmax(s, 1)
        mean = (X * M[..., None]).sum(1) / M.sum(1, keepdim=True)
        return self.head(self.drop(torch.cat([mean, (a[..., None] * X).sum(1)], 1))).squeeze(-1), a


def tens(S, sd, mu):
    n = np.array([len(s) for s in S]); T = 20; X = np.zeros((len(S), T, S[0].shape[1]), np.float32)
    for i, s in enumerate(S):
        X[i, :len(s)] = (s[:T] - mu) / sd
    M = np.arange(T)[None] < n[:, None]; R = np.clip(n[:, None] - 1 - np.arange(T)[None], 0, T - 1)
    return torch.tensor(X), torch.tensor(M), torch.tensor(R)


def train(D, y, idx, pos, wd, seed):
    torch.manual_seed(seed); tr, va = idx; X, M, R = D; m = Att(X.shape[2], pos); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=wd)
    yt = torch.tensor(y, dtype=torch.float32); best, state, bad = -2, None, 0; g = torch.Generator().manual_seed(seed)
    for ep in range(200):
        m.train()
        for b in torch.randperm(len(tr), generator=g).split(64):
            j = tr[b]; opt.zero_grad(); p, _ = m(X[j], M[j], R[j]); torch.nn.functional.mse_loss(p, yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            r = pearsonr(m(X[va], M[va], R[va])[0].numpy(), y[va])[0]
        if r > best:
            best, state, bad = r, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    m.load_state_dict(state); m.eval(); return m, best


def run(name, path):
    d = np.load(path, allow_pickle=True); ok = d["ok"].astype(bool)
    S = d["seg"][ok]; y = d["label"][ok].astype(float); mode = d["mode"][ok]; vid = d["vid"][ok]; dur = d["dur"][ok]
    tr = np.where(mode == "train")[0]; va = np.where(mode == "valid")[0]; te = np.where(mode == "test")[0]
    allseg = np.concatenate(list(S[tr])); mu, sd = allseg.mean(0), allseg.std(0) + 1e-6
    mean = np.stack([s.mean(0) for s in S]); end = np.stack([s[-1] for s in S])
    pk_idx = np.array([np.argmax(np.linalg.norm((s - s.mean(0)) / sd, axis=1)) for s in S]); peak = np.stack([s[i] for s, i in zip(S, pk_idx)])
    X = np.c_[mean, peak, end]; sc = StandardScaler().fit(X[tr]); best = None
    for a in (1, 10, 100, 1000):
        m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr]); r = pearsonr(m.predict(sc.transform(X[va])), y[va])[0]
        if best is None or r > best[0]:
            best = (r, m)
    p_mpe = best[1].predict(sc.transform(X[te]))
    D = tens(S, sd, mu); idx = (torch.tensor(tr), torch.tensor(va)); res = {}
    for pos in (False, True):
        for wd in (1e-4, 1e-2):
            ms = [train(D, y, idx, pos, wd, s) for s in range(5)]; res[(pos, wd)] = ms
            say(f"  {name} ATT{'+P' if pos else ''} wd={wd:g}: valid r per seed " + ", ".join(f"{b:.3f}" for _, b in ms))
    key = max(res, key=lambda k: np.mean([b for _, b in res[k]])); ms = res[key]
    Xt, Mt, Rt = (t[torch.tensor(te)] for t in D)
    with torch.no_grad():
        outs = [m(Xt, Mt, Rt) for m, _ in ms]
    p_att = np.mean([o[0].numpy() for o in outs], 0); A = np.mean([o[1].numpy() for o in outs], 0)
    yt, vt = y[te], vid[te]; uv = np.unique(vt); ix = {v: np.where(vt == v)[0] for v in uv}; rng = np.random.default_rng(1)
    pt = pearsonr(p_mpe, yt)[0] - pearsonr(p_att, yt)[0]; bs = []
    for _ in range(2000):
        s = np.concatenate([ix[v] for v in rng.choice(uv, len(uv))]); bs.append(pearsonr(p_mpe[s], yt[s])[0] - pearsonr(p_att[s], yt[s])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]); gate = pt >= 0.02 and lo > 0
    n = np.array([len(s) for s in S])[te]; k3 = n >= 3
    mass_pk = A[np.arange(len(te)), pk_idx[te]][k3].mean(); mass_end = A[np.arange(len(te)), n - 1][k3].mean(); uni = (1 / n[k3]).mean()
    say(f"\n## {name}: test n={len(te)}; selected ATT{'+P' if key[0] else ''} wd={key[1]:g}")
    say(f"- r(MPE, ridge) = {pearsonr(p_mpe, yt)[0]:.3f}; r(ATT*, 5-seed ensemble) = {pearsonr(p_att, yt)[0]:.3f}")
    say(f"- METHOD GATE r(MPE) - r(ATT*) = {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'passes' if gate else 'fails'}")
    say(f"- attention mass (test clips with >= 3 segments, n={k3.sum()}): on peak segment {mass_pk:.3f}, on last segment {mass_end:.3f}, uniform share {uni:.3f}")
    return gate


say("# P10b: peak-end pooling versus learned attention pooling")
g1 = run("CH-SIMS", os.path.join(W, "sims_seg.npz")); g2 = run("CMU-MOSI", os.path.join(W, "mosi_seg.npz"))
say(f"\n**Method gate {'PASSES' if g1 and g2 else 'FAILS'}** (CH-SIMS {'yes' if g1 else 'no'}, MOSI {'yes' if g2 else 'no'})")
if OUT:
    open(OUT, "w").write("\n".join(L) + "\n")
