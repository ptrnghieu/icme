"""Synthetic check of analyze_intervention.py's decision logic.

Builds 8 (fold, seed) pairs of fake human_aux runs:
  case "effect": human gets +2 points accuracy and higher alignment than shuffled -> expect "H SUPPORTED (causal)"
  case "null"  : alignment differs (manipulation works) but accuracy does not       -> expect "H NOT SUPPORTED"
  case "nomanip": neither alignment nor accuracy differs                             -> expect "INCONCLUSIVE"
Run: python test_analyze_intervention.py <scratch dir>
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def make(root, case, n_clips=300):
    rng = np.random.default_rng(0)
    clips = [f"{1000 + i // 30}_X_ANG_XX_{i}" for i in range(n_clips)]
    d = rng.uniform(-1, 1, n_clips)
    os.makedirs(root, exist_ok=True)
    pd.DataFrame({"clip": clips, "d_human": d}).to_csv(os.path.join(root, "human_reliability.csv"), index=False)
    runs = os.path.join(root, "runs")
    for f in range(4):
        for s in range(2):
            base = 50 + rng.normal(0, 1)
            for cond in ["human", "shuffled", "anti", "uniform"]:
                gain = {"effect": 2.0, "null": 0.0, "nomanip": 0.0}[case] if cond == "human" else 0.0
                strength = 0.6 if (cond == "human" and case != "nomanip") else 0.0
                rel = strength * d + rng.normal(0, 0.5, n_clips)
                acc = (base + gain + rng.normal(0, 0.3)) / 100
                od = os.path.join(runs, f"actor{f}_haux_{cond}_l1_s{s}_f1")
                os.makedirs(od, exist_ok=True)
                json.dump({"method": "human_aux", "split": f"actor{f}", "seed": s, "aux_weights": cond,
                           "last10_test_acc": acc, "test_acc": acc}, open(os.path.join(od, "metrics.json"), "w"))
                correct = (rng.uniform(size=n_clips) < acc).astype(int)
                kv, ka = np.exp(rel) * 0.1, np.full(n_clips, 0.1)
                pd.DataFrame({"clip": clips, "correct": correct, "rel_lf": rel, "kl_drop_v": kv,
                              "kl_drop_a": ka}).to_csv(os.path.join(od, "preds_test.csv"), index=False)
    return root


def main(scratch):
    here = os.path.dirname(os.path.abspath(__file__))
    expect = {"effect": "H SUPPORTED (causal)", "null": "H NOT SUPPORTED", "nomanip": "INCONCLUSIVE"}
    for case, want in expect.items():
        root = make(os.path.join(scratch, case), case)
        out = subprocess.run([sys.executable, os.path.join(here, "analyze_intervention.py"), "--root", root, "--runs",
                              os.path.join(root, "runs"), "--out", os.path.join(root, "r.md"), "--boot", "200"],
                             capture_output=True, text=True, check=True).stdout
        line = [l for l in out.splitlines() if l.startswith("- Outcome:")][0]
        assert want in line, (case, line)
        print(f"{case:8s} -> {line}")
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
