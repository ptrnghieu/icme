"""Write the pilot job list (one train.py argument string per line).

Default grid (49 runs):
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
    a = ap.parse_args()
    lines = []
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
