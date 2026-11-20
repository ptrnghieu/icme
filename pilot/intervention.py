"""Per-clip visual-emphasis weights for the H intervention (see INTERVENTION.md).

s_i in (0, 1) weights the visual auxiliary loss of training clip i (the audio one gets 1 - s_i).
All four conditions share the same marginal distribution of weights except `uniform` (constant 0.5,
same mean). Weights are computed from the TRAINING clips only.
"""
import os

import numpy as np
import pandas as pd

KINDS = ("human", "shuffled", "anti", "uniform")


def human_weights(root, train_clips, kind, seed):
    if kind not in KINDS:
        raise ValueError(f"unknown weight kind {kind!r}; choose from {KINDS}")
    h = pd.read_csv(os.path.join(root, "human_reliability.csv")).set_index("clip")
    missing = [c for c in train_clips if c not in h.index]
    if missing:
        raise KeyError(f"{len(missing)} training clips have no human ratings, e.g. {missing[:3]}")
    d = h.loc[list(train_clips), "d_human"].to_numpy(dtype=float)  # r_face - r_voice, training clips only
    # Ordinal ranks with ties broken by a FIXED random order (independent of the run seed): ranks are a permutation
    # of 1..N, so s takes exactly the values (k - 0.5) / N. This makes human, shuffled and anti (= 1 - s) share the
    # same multiset of weights. Average ranks would not: with tied d values (about 10 raters per modality),
    # 1 - s would have a different distribution from s.
    tie_break = np.random.RandomState(12345).permutation(len(d))
    order = np.lexsort((tie_break, d))  # sort by d, ties by the fixed random key
    r = np.empty(len(d))
    r[order] = np.arange(1, len(d) + 1)
    s = (r - 0.5) / len(d)
    if kind == "human":
        return s
    if kind == "anti":
        return 1.0 - s
    if kind == "shuffled":
        return s[np.random.RandomState(1000 + seed).permutation(len(s))]
    return np.full(len(s), 0.5)
