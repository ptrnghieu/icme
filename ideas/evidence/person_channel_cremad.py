import sys,glob,json,os
import numpy as np, pandas as pd
H=pd.read_csv(sys.argv[1]); runs=sys.argv[2]
H["actor"]=H["clip"].str[:4]
print("human_reliability columns:",list(H.columns)[:12])
A=H.groupby("actor").agg(rv=("r_voice","mean"),rf=("r_face","mean"),d=("d_human","mean"),n=("clip","size"))
print(f"actors {len(A)}; clips/actor {A.n.min()}-{A.n.max()}")
print(f"per-actor mean d_human (face-voice): mean {A.d.mean():.3f}, SD {A.d.std():.3f}, min {A.d.min():.3f}, max {A.d.max():.3f}")
print(f"actors with d<0 (voice more recognizable than face): {(A.d<0).sum()}/{len(A)}")
# variance decomposition (one-way ANOVA ICC(1)) of clip-level d_human by actor
g=H.groupby("actor")["d_human"]; k=g.size().mean(); grand=H["d_human"].mean()
msb=(g.size()*(g.mean()-grand)**2).sum()/(g.ngroups-1); msw=((H["d_human"]-g.transform("mean"))**2).sum()/(len(H)-g.ngroups)
icc=(msb-msw)/(msb+(k-1)*msw); print(f"ICC(1) of d_human by actor = {icc:.3f}  (MSB {msb:.4f}, MSW {msw:.4f})")
# split-half reliability of per-actor d (odd/even clips)
H["half"]=H.groupby("actor").cumcount()%2
P=H.pivot_table(index="actor",columns="half",values="d_human",aggfunc="mean")
r=np.corrcoef(P[0],P[1])[0,1]; print(f"split-half r of per-actor d = {r:.3f}; Spearman-Brown = {2*r/(1+r):.3f}")
# models: per test-actor accuracy of audio-only vs visual-only (pilot actor folds)
rows=[]
for m in glob.glob(f"{runs}/actor*_*_only_*/metrics.json"):
    r_=json.load(open(m)); d=os.path.dirname(m); p=pd.read_csv(d+"/preds_test.csv")
    p["actor"]=p["clip"].str[:4]; p["ok"]=(p["pred"]==p["label"]) if "label" in p else p["correct"]
    for a,gg in p.groupby("actor"): rows.append(dict(split=r_["split"],seed=r_["seed"],method=r_["method"],actor=a,acc=gg.ok.mean(),n=len(gg)))
R=pd.DataFrame(rows); W=R.pivot_table(index=["split","seed","actor"],columns="method",values="acc").dropna().reset_index()
W["dm"]=W.visual_only-W.audio_only; W=W.merge(A[["d"]],left_on="actor",right_index=True)
print(f"model per-actor (n={len(W)} actor x run): visual-audio acc diff mean {W.dm.mean():.3f}, SD {W.dm.std():.3f}; actors where audio-only beats visual-only: {(W.dm<0).mean():.2%}")
from scipy.stats import spearmanr
print(f"Spearman(model per-actor visual-audio diff, human per-actor d) = {spearmanr(W.dm,W.d).correlation:.3f}")
for (s,sd),gg in W.groupby(["split","seed"]):
    glob_best=max(gg.audio_only.mean(),gg.visual_only.mean()); orac=np.maximum(gg.audio_only,gg.visual_only).mean()
    print(f"  {s} seed{sd}: best single modality (global) {100*glob_best:.2f} | per-actor oracle modality {100*orac:.2f} | headroom {100*(orac-glob_best):+.2f}")
