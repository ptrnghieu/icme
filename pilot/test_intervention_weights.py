"""Checks for the H intervention weights (INTERVENTION.md).

1. human / shuffled / anti have exactly the same multiset of values (same balancing strength), mean 0.5;
   uniform is constant 0.5.
2. human is a monotone function of d_human = r_face - r_voice; anti = 1 - human; shuffled breaks the
   clip correspondence.
3. Only training clips are used; train, val and test actors are disjoint.
Run: python test_intervention_weights.py <root with human_reliability.csv and splits/>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import actor_of, read_split
from intervention import human_weights


def main(root):
    sd = os.path.join(root, "splits")
    tr = [c for c, _ in read_split(os.path.join(sd, "actor0_train.csv"))]
    va = [c for c, _ in read_split(os.path.join(sd, "actor0_val.csv"))]
    te = [c for c, _ in read_split(os.path.join(sd, "actor0_test.csv"))]
    A = lambda cs: {actor_of(c) for c in cs}  # noqa: E731
    assert not (A(tr) & A(va)) and not (A(tr) & A(te)) and not (A(va) & A(te)), "actor overlap between splits"
    w = {k: human_weights(root, tr, k, seed=0) for k in ("human", "shuffled", "anti", "uniform")}
    for k in ("shuffled", "anti"):
        assert np.allclose(np.sort(w[k]), np.sort(w["human"])), f"{k} must have the same values as human"
    for k, v in w.items():
        assert abs(v.mean() - 0.5) < 1e-9, (k, v.mean())
    assert np.allclose(w["uniform"], 0.5)
    assert np.allclose(w["anti"], 1 - w["human"])
    d = pd.read_csv(os.path.join(root, "human_reliability.csv")).set_index("clip").loc[tr, "d_human"].to_numpy()
    rho_h = spearmanr(w["human"], d).correlation
    rho_s = spearmanr(w["shuffled"], d).correlation
    o = np.argsort(w["human"])
    assert np.all(np.diff(d[o]) >= 0), "human weights must be a monotone (non-decreasing) function of d_human"
    assert np.allclose(np.sort(w["human"]), (np.arange(1, len(tr) + 1) - 0.5) / len(tr))
    assert not np.allclose(w["shuffled"], w["human"])
    w2 = human_weights(root, tr, "shuffled", seed=1)
    assert not np.allclose(w2, w["shuffled"]), "different seeds must give different permutations"
    try:
        human_weights(root, tr + ["not_a_clip"], "human", 0)
        raise SystemExit("expected a KeyError for an unknown clip")
    except KeyError:
        pass
    print(f"train clips {len(tr)} (actors {len(A(tr))}), val actors {len(A(va))}, test actors {len(A(te))}: disjoint")
    print(f"same multiset for human/shuffled/anti; means = {[round(v.mean(), 6) for v in w.values()]}")
    print(f"Spearman(weights, d_human): human {rho_h:.3f}, shuffled {rho_s:.3f}, anti "
          f"{spearmanr(w['anti'], d).correlation:.3f}")
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
