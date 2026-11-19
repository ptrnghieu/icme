"""Per-clip human modality reliability from CREMA-D crowd ratings.

Input : <root>/CREMA-D/finishedResponses.csv (queryType 1 = voice only, 2 = face only, 3 = audio-visual;
        repeated and practice responses already removed by the dataset authors, see CREMA-D README).
Output: <root>/human_reliability.csv with, for each clip and modality m in {voice, face, av}:
        n_m        number of ratings
        r_m        fraction of ratings equal to the intended (acted) emotion
        dist_m_X   fraction of ratings for emotion letter X (X in N,H,S,F,D,A)
        agree_m    fraction of ratings equal to the majority emotion of that condition
        d_human  = r_face - r_voice   (positive: humans recover the intended emotion better from the face)
"""
import argparse
import os

import pandas as pd

from common import LETTER, LETTERS

QT = {1: "voice", 2: "face", 3: "av"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    path = os.path.join(args.root, "finishedResponses.csv")  # copied here by prepare_cremad.py
    if not os.path.exists(path):
        path = os.path.join(args.root, "CREMA-D", "finishedResponses.csv")
    r = pd.read_csv(path, low_memory=False)
    r["intended"] = r["clipName"].str.split("_").str[2].map(LETTER)
    r["correct"] = (r["respEmo"] == r["intended"]).astype(float)
    out = None
    for q, name in QT.items():
        g = r[r["queryType"] == q].groupby("clipName")
        df = pd.DataFrame({f"n_{name}": g.size(), f"r_{name}": g["correct"].mean()})
        dist = pd.crosstab(r.loc[r.queryType == q, "clipName"], r.loc[r.queryType == q, "respEmo"], normalize="index")
        for x in LETTERS:
            df[f"dist_{name}_{x}"] = dist[x] if x in dist.columns else 0.0
        df[f"agree_{name}"] = dist.max(axis=1)
        out = df if out is None else out.join(df, how="outer")
    out.index.name = "clip"
    out["d_human"] = out["r_face"] - out["r_voice"]
    dst = os.path.join(args.root, "human_reliability.csv")
    out.reset_index().to_csv(dst, index=False)
    print(f"clips: {len(out)} -> {dst}")
    print(out[["r_voice", "r_face", "r_av", "d_human"]].describe().round(3))


if __name__ == "__main__":
    main()
