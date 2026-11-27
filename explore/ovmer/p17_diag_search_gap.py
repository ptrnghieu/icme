# Label-free diagnostics on OV-MERD+ P17 outputs (no ground truth is read).
import json,sys,itertools,collections,numpy as np
sys.path.insert(0,'/home/user/icme/explore/ovmer')
from ew_metric import EW, WHEELS; from p13_mbr import parse, construct
ew=EW('deep/affectgpt/AffectGPT-RL/emotion_wheel/wheel_mapping.npz')
L=lambda f:{(r:=json.loads(l))['key']:r for l in open(f)}
A,B,O=L('p17r/labels.jsonl'),L('p17r/labels_seed1.jsonl'),L('p17r/labels_oldprompt.jsonl')
keys=sorted(A)
def U(a,b):  # symmetric per-clip EW-F between two label lists (a as reference, as in sample_f)
    return ew.sample_f(a,b) if a and b else np.nan
# 3. search gap: greedy CONSTRUCT vs exhaustive argmax of the same objective over the same 12-label pool, |Y|<=6
def gid(lbl): return [next(iter(ew.groups([lbl],w)),None) for w in WHEELS]
chk=[];gap=[];hit=0;sizes_g=[];sizes_x=[];nclips=0
for k in keys:
    S=[parse(x) for x in A[k]['samples']]; S=[s for s in S if s]
    if not S: continue
    G=parse(A[k]['greedy'])
    cntr=collections.Counter(x for s in S for x in set(s))
    for x in G: cntr.setdefault(x,0)
    pool=[x for x,_ in sorted(cntr.items(),key=lambda z:(-z[1],z[0])) if ew.in_vocab(x)][:12]
    Yg=construct([parse(x) for x in A[k]['samples']],G,ew)
    obj=lambda Y: np.mean([ew.sample_f(s,Y) for s in S])
    # exhaustive with wheel-group bitmasks
    allg=sorted({(w,g) for x in set(pool+[y for s in S for y in s]) for w in range(5) for g in ew.groups([x],WHEELS[w])})
    idx={};cw=collections.Counter()
    for (w,g) in allg: idx[(w,g)]=cw[w]; cw[w]+=1
    assert max(cw.values())<63
    def mask(lbls,w):
        m=0
        for x in lbls:
            gg=ew.groups([x],WHEELS[w])
            for g in gg: m|=1<<idx[(w,g)]
        return m
    lm=np.array([[mask([x],w) for w in range(5)] for x in pool],dtype=np.int64)
    sm=np.array([[mask(s,w) for w in range(5)] for s in S],dtype=np.int64)
    subs=[c for r in range(1,min(6,len(pool))+1) for c in itertools.combinations(range(len(pool)),r)]
    Ym=np.zeros((len(subs),5),dtype=np.int64)
    for i,c in enumerate(subs):
        for j in c: Ym[i]|=lm[j]
    inter=np.bitwise_count(Ym[:,None,:]&sm[None,:,:]); ny=np.bitwise_count(Ym)[:,None,:]; ns=np.bitwise_count(sm)[None,:,:]
    valid=(ns>0)
    f=np.where(valid,np.where((inter>0),2*inter/(ny+ns),0.0),0.0)
    per=(f.sum(2)/np.maximum(valid.sum(2),1))  # mean over wheels with reference groups (as sample_f)
    per=np.where(valid.sum(2)>0,per,0.0)
    val=np.nanmean(per,1); best=val.max(); og=obj(Yg) if Yg else 0
    if Yg and all(y in pool for y in Yg):
        chk.append(abs(val[subs.index(tuple(sorted(pool.index(y) for y in Yg)))]-og))
    gap.append(best-og); hit+=best-og<1e-9; sizes_g.append(len(Yg)); sizes_x.append(len(subs[int(val.argmax())])); nclips+=1
gap=np.array(gap)
print('3) clips %d | greedy reaches the exhaustive optimum of R-hat in %.1f%% | mean R-hat gap %.4f | max %.3f | sizes greedy %.2f vs exact %.2f'%(nclips,100*hit/nclips,gap.mean(),gap.max(),np.mean(sizes_g),np.mean(sizes_x)))

print('check |bitmask R-hat - sample_f R-hat| on greedy sets: max %.2e over %d clips'%(max(chk),len(chk)))
import numpy as _n;print('gap quantiles 50/90/99:',_n.quantile(gap,[.5,.9,.99]).round(4))
