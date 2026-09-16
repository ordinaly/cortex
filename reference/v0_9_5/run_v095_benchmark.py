from __future__ import annotations
import sys,time,math,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/cortex_v095_compute')
sys.path[:0]=[str(ROOT),str(ROOT/'dependencies'),'/mnt/data/cortex_v094_rank_refinement']
from cortex_v095_compute_optimized import ComputeOptimizedRefinementReasoner
from cortex_v094_rank_refinement import RankRefinementReasoner
from cortex_v093_tensor_activation import BurstTensorPlasticityReasoner
from run_v094_benchmark import (
    LOADERS,GATED,fixture_kwargs,stream_update,stream_split,stream_novel,stream_merge
)
from run_v092_benchmark import make_xy,kwargs,structural_equal

def run_fixture(Cls,fn,T,seed,extra=None):
    extra=extra or {}
    r=Cls(**fixture_kwargs(),**GATED,**extra); rng=np.random.default_rng(seed)
    Y=[];P=[];split_t=merge_t=novel_t=None
    for t in range(T):
        x,p=fn(rng,t); y=int(rng.binomial(1,p));Y.append(y);P.append(r.step(x,y).prediction)
        if getattr(r,'split_promotions',0) and split_t is None: split_t=t
        if getattr(r,'merge_promotions',0) and merge_t is None: merge_t=t
        if r.discovery_count>=2 and novel_t is None: novel_t=t
    Y=np.asarray(Y,float);P=np.asarray(P,float)
    return dict(brier=float(np.mean((Y-P)**2)),split_t=split_t,merge_t=merge_t,novel_t=novel_t,
                splits=getattr(r,'split_promotions',0),merges=getattr(r,'merge_promotions',0),
                proposals=getattr(r,'split_proposals',0),discoveries=r.discovery_count,stored=len(r.prototypes),
                curvature_updates=getattr(r,'curvature_updates',0),rank_checks=getattr(r,'rank_checks',0))

def behavioral():
    rows=[]
    specs=[('update',stream_update,3200,range(100,108)),('split',stream_split,4500,range(100,108)),
           ('novel',stream_novel,3200,range(100,108)),('merge',stream_merge,9000,range(300,305))]
    for kind,fn,T,seeds in specs:
        for seed in seeds:
            for label,Cls,extra in [('v0.9.4',RankRefinementReasoner,{}),('v0.9.5-alpha',ComputeOptimizedRefinementReasoner,{'curvature_stride':2})]:
                rec=run_fixture(Cls,fn,T,seed,extra);rec.update(fixture=kind,seed=seed,T=T,model=label);rows.append(rec)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'CORTEX_V095_BEHAVIORAL.csv',index=False);return df

def real_stream(repeats=7):
    rows=[];checks=[]
    classes=[('v0.9.3',BurstTensorPlasticityReasoner,{}),('v0.9.4',RankRefinementReasoner,{}),
             ('v0.9.5-alpha',ComputeOptimizedRefinementReasoner,{'curvature_stride':2})]
    for L in LOADERS:
        name,X,Y,_=make_xy(L)
        a=RankRefinementReasoner(**kwargs(),**GATED)
        b=ComputeOptimizedRefinementReasoner(**kwargs(),**GATED,curvature_stride=3)
        mism=[];pdiff=0
        for t,(x,y) in enumerate(zip(X,Y)):
            ra=a.step(x,int(y));rb=b.step(x,int(y));ok,where=structural_equal(a,b)
            if not ok:mism.append((t,where))
            if abs(ra.prediction-rb.prediction)>1e-12:pdiff+=1
        checks.append(dict(dataset=name,steps=len(Y),structural_mismatches=len(mism),prediction_mismatches=pdiff,
                           v094_splits=a.split_promotions,v095_splits=b.split_promotions,
                           v094_proposals=a.split_proposals,v095_proposals=b.split_proposals,
                           v094_curvature_updates=a.curvature_updates,v095_curvature_updates=b.curvature_updates))
        for rep in range(repeats):
            order=classes[rep%3:]+classes[:rep%3]
            for label,Cls,extra in order:
                m=Cls(**kwargs(),**GATED,**extra);ps=[];t0=time.perf_counter_ns()
                for x,y in zip(X,Y):ps.append(m.step(x,int(y)).prediction)
                us=(time.perf_counter_ns()-t0)/1000/max(1,len(Y));ps=np.asarray(ps,float)
                rows.append(dict(dataset=name,n=len(Y),rep=rep,model=label,us_per_step=float(us),
                                 brier=float(np.mean((Y-ps)**2)),stored=len(m.prototypes),
                                 proposals=getattr(m,'split_proposals',0),splits=getattr(m,'split_promotions',0),
                                 curvature_updates=getattr(m,'curvature_updates',0)))
    raw=pd.DataFrame(rows);raw.to_csv(ROOT/'CORTEX_V095_REAL_TIMING_RAW.csv',index=False)
    checks=pd.DataFrame(checks);checks.to_csv(ROOT/'CORTEX_V095_NONINTERFERENCE.csv',index=False)
    agg=raw.groupby(['dataset','n','model']).agg(us_per_step=('us_per_step','median'),brier=('brier','mean'),stored=('stored','first'),
          proposals=('proposals','max'),splits=('splits','max'),curvature_updates=('curvature_updates','max')).reset_index()
    agg.to_csv(ROOT/'CORTEX_V095_REAL_RESULTS.csv',index=False)
    return agg,checks

if __name__=='__main__':
    b=behavioral();a,c=real_stream()
    # Compact summary
    good={}
    for fixture in ['update','split','novel','merge']:
        d=b[(b.fixture==fixture)&(b.model=='v0.9.5-alpha')]
        if fixture=='update': good[fixture]=int((d.splits==0).sum())
        elif fixture=='split': good[fixture]=int((d.splits==1).sum())
        elif fixture=='novel': good[fixture]=int(((d.discoveries>=2)&(d.splits==0)).sum())
        else: good[fixture]=int(((d.splits>=1)&(d.merges>=1)&(d.stored==1)).sum())
    weighted={}
    for model in ['v0.9.3','v0.9.4','v0.9.5-alpha']:
        d=a[a.model==model];weighted[model]=float(np.average(d.us_per_step,weights=d.n))
    summary={'contracts':good,'weighted_us_per_step':weighted,
             'v095_vs_v094_speedup':weighted['v0.9.4']/weighted['v0.9.5-alpha'],
             'v095_vs_v093_speedup':weighted['v0.9.3']/weighted['v0.9.5-alpha'],
             'real_structural_mismatches':int(c.structural_mismatches.sum()),
             'real_prediction_mismatches':int(c.prediction_mismatches.sum()),
             'curvature_update_reduction':float(1-a[a.model=='v0.9.5-alpha'].curvature_updates.sum()/a[a.model=='v0.9.4'].curvature_updates.sum())}
    (ROOT/'CORTEX_V095_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print(a.to_string(index=False));print(c.to_string(index=False))
