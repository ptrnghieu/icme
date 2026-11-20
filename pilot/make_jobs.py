"""Write a job list (one train.py argument string per line).

--grid intervention (32 runs, see INTERVENTION.md):
  method human_aux, aux_weights in {human, shuffled, anti, uniform}, aux_lambda 1.0
  x actor folds {0,1,2,3} x seeds {0,1}

--grid pilot (default, 49 runs):
  standard split : joint, ogm_ge alpha in {0.1, 0.3, 0.5, 0.8}, audio_only, visual_only  x seeds {0,1,2}
  actor folds 0,1: same 7 methods                                                       x seeds {0,1}
"""
import argparse

METHODS = [("joint", None), ("ogm_ge", 0.1), ("ogm_ge", 0.3), ("ogm_ge", 0.5), ("ogm_ge", 0.8),
           ("audio_only", None), ("visual_only", None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--jobs", default="jobs.txt")
    ap.add_argument("--std_seeds", default="0,1,2")
    ap.add_argument("--actor_folds", default="0,1")
    ap.add_argument("--actor_seeds", default="0,1")
    ap.add_argument("--extra", default="--amp", help="extra args appended to every job")
    ap.add_argument("--grid", default="pilot", choices=["pilot", "intervention"])
    ap.add_argument("--int_folds", default="0,1,2,3")
    ap.add_argument("--int_seeds", default="0,1")
    ap.add_argument("--int_lambda", default="1.0")
    a = ap.parse_args()
    lines = []
    if a.grid == "intervention":
        folds = a.int_folds.split(",")
        groups = [folds[: len(folds) // 2], folds[len(folds) // 2:]]  # with 2 shards: shard 0 <- first half of folds
        per_group = []
        for g in groups:
            jl = []
            for f in g:
                for seed in a.int_seeds.split(","):
                    for w in ("human", "shuffled", "anti", "uniform"):
                        jl.append(f"--root {a.root} --out {a.out} --split actor{f} --method human_aux --aux_weights {w} "
                                  f"--aux_lambda {a.int_lambda} --seed {seed} {a.extra}")
            per_group.append(jl)
        # interleave so that launch.py --n_shards 2 (index % 2) gives each shard complete (fold, seed) sets
        for x, y in zip(*per_group):
            lines += [x, y]
        open(a.jobs, "w").write("\n".join(lines) + "\n")
        print(f"{len(lines)} jobs -> {a.jobs}")
        return
    plan = [("standard", s) for s in a.std_seeds.split(",")]
    plan += [(f"actor{f}", s) for f in a.actor_folds.split(",") for s in a.actor_seeds.split(",")]
    for split, seed in plan:
        for m, alpha in METHODS:
            al = f" --alpha {alpha}" if alpha is not None else ""
            lines.append(f"--root {a.root} --out {a.out} --split {split} --method {m}{al} --seed {seed} {a.extra}")
    open(a.jobs, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {a.jobs}")


if __name__ == "__main__":
    main()
