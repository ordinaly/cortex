#!/usr/bin/env python3
from __future__ import annotations
import gzip, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "v0_9_5"
sys.path.insert(0, str(REF))
from cortex_v095_compute_optimized import ComputeOptimizedRefinementReasoner

BASE = dict(
    dim=12,budget=24,tau=.20,alpha_cut=.10,stay_tolerance=.28,recurrence_tolerance=.38,update_tolerance=.22,
    redundancy_tolerance=.14,centroid_lr_cap=.08,decay=.9995,prior_strength=2.0,recompress_interval=32,
    recurrence_threshold=.75,evidence_decay=.85,hazard_lr=.35,novelty_threshold=3.5,
)
GATED = dict(initial_burst=24,post_ready_burst=8,refresh_interval=144,refresh_burst=4,
             degradation_ratio=1.0,degradation_patience=6)
FIXTURE = dict(BASE, budget=6, stay_tolerance=.16, recurrence_tolerance=.28,
               update_tolerance=.12, centroid_lr_cap=.02)


def stream_update(rng,t):
    x=rng.normal(0,.08,12); p=1/(1+math.exp(-7*x[0])); return x,p

def stream_split(rng,t):
    x=rng.normal(0,.07,12); sub=1 if (t//160)%2==0 else -1; x[2]+=.26*sub
    p=1/(1+math.exp(-(9*x[0]+2.0*sub))); return x,p

def stream_novel(rng,t):
    x=rng.normal(0,.07,12)
    if t>=1500: x[:4]+=0.9; p=.82
    else: p=.22
    return x,p

def stream_merge(rng,t):
    x=rng.normal(0,.07,12); sub=1 if (t//160)%2==0 else -1; x[2]+=.26*sub
    b=2.0 if t<4500 else 0.0; p=1/(1+math.exp(-(9*x[0]+b*sub))); return x,p


def state(m):
    return {
        "t":m.t,"current_id":m.current_id,"pending_kind":m.pending_kind,"pending_id":m.pending_id,
        "pending_count":m.pending_count,"pending_score":float(m.pending_score),"sigma2":float(m.sigma2),"hazard":float(m.hazard),
        "revision_count":m.revision_count,"reactivation_count":m.reactivation_count,"discovery_count":m.discovery_count,
        "unresolved_count":m.unresolved_count,"budget_pressure_count":m.budget_pressure_count,"recompression_count":m.recompression_count,
        "split_proposals":m.split_proposals,"split_rejections":m.split_rejections,"split_promotions":m.split_promotions,
        "merge_promotions":m.merge_promotions,"rank_checks":m.rank_checks,"curvature_updates":m.curvature_updates,
        "prototypes":[{
            "centroid":[float(v) for v in p.centroid],"count":float(p.count),"successes":float(p.successes),"failures":float(p.failures),
            "utility":float(p.utility),"last_used":int(p.last_used),"created_t":int(p.created_t),
            "lineage_id":getattr(p,"lineage_id",None),"split_created_t":getattr(p,"split_created_t",None),
            "recent_mean":float(getattr(p,"recent_mean",p.mean)),"recent_obs":int(getattr(p,"recent_obs",0)),
            "refinement_depth":int(getattr(p,"refinement_depth",0)),
        } for p in m.prototypes],
    }


def generate(name, fn, T, seed):
    out = ROOT / "tests" / "fixtures" / f"{name}.jsonl.gz"
    meta = ROOT / "tests" / "fixtures" / f"{name}.final.json"
    rng=np.random.default_rng(seed)
    m=ComputeOptimizedRefinementReasoner(**FIXTURE,**GATED,curvature_stride=2)
    with gzip.open(out,"wt",encoding="utf8") as f:
        for t in range(T):
            x,p=fn(rng,t); y=int(rng.binomial(1,p)); r=m.step(x,y)
            rec={"x":[float(v) for v in x],"y":y,"prediction":float(r.prediction),
                 "current_id":r.current_id,"nearest_id":r.nearest_id,"nearest_dist":float(r.nearest_dist),
                 "revision":bool(r.revision),"reactivated":bool(r.reactivated),"discovered":bool(r.discovered),
                 "unresolved":bool(r.unresolved),"budget_pressure":bool(r.budget_pressure),"stored":int(r.stored)}
            f.write(json.dumps(rec,separators=(",",":"))+"\n")
    meta.write_text(json.dumps({"name":name,"T":T,"seed":seed,"config":{**FIXTURE,**GATED,"curvature_stride":2},"final":state(m)},indent=2))
    print(name, out.stat().st_size, state(m)["split_promotions"], state(m)["merge_promotions"], state(m)["discovery_count"])

if __name__ == "__main__":
    generate("update_seed100",stream_update,3200,100)
    generate("split_seed100",stream_split,4500,100)
    generate("novel_seed100",stream_novel,3200,100)
    generate("merge_seed300",stream_merge,9000,300)
