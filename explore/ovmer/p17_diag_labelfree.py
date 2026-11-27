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
# 1. extractor vs description variability
same=[];diff=[];exact=[]
rng=np.random.default_rng(0)
for k in keys:
    sa=[parse(x) for x in A[k]['samples']]; sb=[parse(x) for x in B[k]['samples']]
    for j in range(16):
        if sa[j] and sb[j]:
            same.append(U(sa[j],sb[j])); exact.append(set(sa[j])==set(sb[j]))
        j2=(j+1+rng.integers(15))%16
        if sa[j] and sa[j2]: diff.append(U(sa[j],sa[j2]))
print('1) utility between two extractions of the SAME description: %.3f (identical label set %.1f%%)'%(np.nanmean(same),100*np.mean(exact)))
print('   utility between extractions of two DIFFERENT descriptions of the same clip: %.3f'%np.nanmean(diff))
print('   -> share of disagreement due to the extractor: %.1f%%'%(100*(1-np.nanmean(same))/(1-np.nanmean(diff))))
# 2. label count per sample by prompt
def cnt(D):
    c=[len(parse(x)) for k in keys for x in D[k]['samples'] if parse(x)]
    iv=[len([y for y in parse(x) if ew.in_vocab(y)]) for k in keys for x in D[k]['samples'] if parse(x)]
    return np.mean(c),np.mean(np.array(c)>=4),np.mean(np.array(c)>4),np.mean(iv),collections.Counter(min(x,7) for x in c)
for n,D in [('rl (at most four)',A),('old (no cap)',O)]:
    m,ge4,gt4,iv,h=cnt(D); print('2) %-18s labels/sample %.2f | in-vocab %.2f | >=4: %.1f%% | >4: %.1f%% | hist %s'%(n,m,iv,100*ge4,100*gt4,dict(sorted(h.items()))))
# groups per sample on wheel1 (what the utility actually sees)
for n,D in [('rl',A),('old',O)]:
    g=[len(ew.groups(parse(x),'wheel1')) for k in keys for x in D[k]['samples'] if parse(x)]
    print('   %s: level-1 groups per sample (wheel1) %.2f'%(n,np.mean(g)))
# 3. search gap: greedy CONSTRUCT vs exhaustive argmax of the same objective over the same 12-label pool, |Y|<=6
def gid(lbl): return [next(iter(ew.groups([lbl],w)),None) for w in WHEELS]
gap=[];hit=0;sizes_g=[];sizes_x=[];nclips=0
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
    allg=sorted({g for x in pool+[y for s in S for y in s] for g in [(w,ew.groups([x],w) and next(iter(ew.groups([x],w)))) for w in range(5)] if g[1]})
    idx={g:i for i,g in enumerate(allg)}
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
    per=np.where(valid.sum(2)>0,per,np.nan)
    val=np.nanmean(per,1); best=val.max(); og=obj(Yg) if Yg else 0
    gap.append(best-og); hit+=best-og<1e-9; sizes_g.append(len(Yg)); sizes_x.append(len(subs[int(val.argmax())])); nclips+=1
gap=np.array(gap)
print('3) clips %d | greedy reaches the exhaustive optimum of R-hat in %.1f%% | mean R-hat gap %.4f | max %.3f | sizes greedy %.2f vs exact %.2f'%(nclips,100*hit/nclips,gap.mean(),gap.max(),np.mean(sizes_g),np.mean(sizes_x)))
