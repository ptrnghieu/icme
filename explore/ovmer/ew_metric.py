"""Emotion-wheel (EW) metric for open-vocabulary MER, re-implemented from the official code
(github.com/zeroQiaoba/AffectGPT, AffectGPT-RL/my_affectgpt/evaluation/{wheel.py, ew_metric.py}; read 2026-11-23).
Uses the official pre-computed mapping file AffectGPT-RL/emotion_wheel/wheel_mapping.npz (not redistributed here).
Official corpus metric (level1): for each of 5 wheels, precision = mean_i |G(Y_i) & G(P_i)| / |G(P_i)|, recall = mean_i |..| / |G(Y_i)|,
F = 2PR/(P+R) of the MEANS; report the mean of (F, P, R) over the wheels. Samples whose mapped GT is empty are skipped; an
empty mapped prediction counts as P = R = 0. Labels are lower-cased and stripped; unmapped labels are dropped.
sample_f() is the per-sample analogue (mean over wheels of the per-sample F), used as the MBR utility.
"""
import numpy as np

WHEELS = ["wheel1", "wheel2", "wheel3", "wheel4", "wheel5"]


class EW:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.fmt = d["format_mapping"].tolist(); self.raw = d["raw_mapping"].tolist(); self.wm = d["wheel_map_whole"].tolist()

    def _back(self, label, wheel_map):  # func_backward_case3
        if label not in self.fmt:
            return ""
        lv1 = [r for f in self.fmt[label] for r in self.raw[f]]
        for l in sorted(lv1):
            if l in wheel_map:
                return wheel_map[l]
        return ""

    def groups(self, labels, wheel, level="level1"):
        wmap = self.wm[wheel][level]; out = set()
        for x in labels:
            g = self._back(str(x).lower().strip(), wmap)
            if g:
                out.add(g)
        return out

    def corpus(self, gts, preds, level="level1"):
        """gts, preds: lists of label lists (aligned). Returns (F, P, R) averaged over the 5 wheels, in percent."""
        res = []
        for w in WHEELS:
            P, R = [], []
            for g, p in zip(gts, preds):
                G = self.groups(g, w, level); Q = self.groups(p, w, level)
                if not G:
                    continue
                if not Q:
                    P.append(0.0); R.append(0.0)
                else:
                    P.append(len(G & Q) / len(Q)); R.append(len(G & Q) / len(G))
            p, r = (np.mean(P) if P else 0.0), (np.mean(R) if R else 0.0)
            res.append([2 * p * r / (p + r) if p + r else 0.0, p, r])
        return (100 * np.mean(res, 0)).tolist()

    def sample_f(self, g, p, level="level1"):
        fs = []
        for w in WHEELS:
            G = self.groups(g, w, level); Q = self.groups(p, w, level)
            if not G:
                continue
            i = len(G & Q); fs.append(0.0 if not Q or not i else 2 * i / (len(G) + len(Q)))
        return float(np.mean(fs)) if fs else 0.0

    def in_vocab(self, label):
        return str(label).lower().strip() in self.fmt
