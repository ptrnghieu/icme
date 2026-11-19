"""Build the two evaluation protocols of the pilot.

(S) standard : the OGM-GE CREMA-D split (6,698 train / 744 test; downloaded from the MIT-licensed
               GeWu-Lab/OGM-GE_CVPR2022 repo). All 91 actors appear in both train and test.
               We hold out 10% of the training clips (random, seed 0) as a validation set so that
               checkpoints are selected on validation, not on test. Per-epoch test accuracy is still
               logged, so the OGM-GE-style "best test epoch" number can be reported for comparison.
(A) actor_k  : 5 actor-disjoint folds (91 actors shuffled with seed 0). For fold k the test actors are
               fold k; 10% of the remaining actors (seed k) form the validation set (also unseen actors).

Only clips present and usable in <root>/manifest.csv are kept (relevant for smoke tests).
"""
import argparse
import csv
import os
import random
import urllib.request

from common import actor_of, read_split, write_split

OGM = "https://raw.githubusercontent.com/GeWu-Lab/OGM-GE_CVPR2022/main/data/CREMAD/{}.csv"


def usable_clips(root):
    with open(os.path.join(root, "manifest.csv")) as f:
        return {r["clip"]: r["label"] for r in csv.DictReader(f) if r["spec_ok"] == "1" and int(r["n_frames"]) > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()
    sd = os.path.join(args.root, "splits")
    os.makedirs(sd, exist_ok=True)
    ok = usable_clips(args.root)

    # ---- (S) standard OGM-GE split
    for part in ("train", "test"):
        dst = os.path.join(sd, f"ogmge_{part}.csv")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(OGM.format(part), dst)
    tr = [r for r in read_split(os.path.join(sd, "ogmge_train.csv")) if r[0] in ok]
    te = [r for r in read_split(os.path.join(sd, "ogmge_test.csv")) if r[0] in ok]
    rng = random.Random(0)
    tr_sh = tr[:]
    rng.shuffle(tr_sh)
    n_val = int(round(len(tr_sh) * args.val_frac))
    write_split(os.path.join(sd, "standard_train.csv"), sorted(tr_sh[n_val:]))
    write_split(os.path.join(sd, "standard_val.csv"), sorted(tr_sh[:n_val]))
    write_split(os.path.join(sd, "standard_test.csv"), sorted(te))
    a_tr, a_te = {actor_of(c) for c, _ in tr}, {actor_of(c) for c, _ in te}
    print(f"[standard] train={len(tr_sh)-n_val} val={n_val} test={len(te)} | actors train={len(a_tr)} "
          f"test={len(a_te)} overlap={len(a_tr & a_te)}")

    # ---- (A) actor-disjoint folds
    actors = sorted({actor_of(c) for c in ok})
    rng = random.Random(0)
    rng.shuffle(actors)
    folds = [actors[i:: args.n_folds] for i in range(args.n_folds)]
    for k, test_actors in enumerate(folds):
        rest = [a for a in actors if a not in test_actors]
        r2 = random.Random(k)
        r2.shuffle(rest)
        n_va = max(1, int(round(len(rest) * args.val_frac)))
        val_actors, train_actors = set(rest[:n_va]), set(rest[n_va:])
        rows = sorted(ok.items())
        sel = lambda S: [(c, l) for c, l in rows if actor_of(c) in S]  # noqa: E731
        write_split(os.path.join(sd, f"actor{k}_train.csv"), sel(train_actors))
        write_split(os.path.join(sd, f"actor{k}_val.csv"), sel(val_actors))
        write_split(os.path.join(sd, f"actor{k}_test.csv"), sel(set(test_actors)))
        print(f"[actor{k}] actors train/val/test = {len(train_actors)}/{len(val_actors)}/{len(test_actors)} | "
              f"clips = {len(sel(train_actors))}/{len(sel(val_actors))}/{len(sel(set(test_actors)))}")


if __name__ == "__main__":
    main()
