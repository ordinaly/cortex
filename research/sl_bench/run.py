"""Run: PYTHONPATH=python:. python -m research.sl_bench.run --output PATH."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from cortex.predictive_memory import PredictiveMemory, PredictiveMemoryPolicy, PredictiveEvidenceAdapter
from cortex.closed_articulation_loop import EvidenceAtom
from .worlds import exact_worlds, stream
from .oracle import exact_quotient, frontier, stable_memory, audit_partition

ROOT=Path(__file__).resolve().parents[2]
BASE=PredictiveMemoryPolicy(epsilon=.25, max_contexts=32, window=128, rebuild_every=64,min_samples=8)
POLICIES={
    'guarded':BASE,
    'empirical_fold':replace(BASE,confidence=False),
    'no_fold':replace(BASE,folding=False),
    'chain_ablation':replace(BASE,confidence=False,complete_link=False),
    'no_reopen':replace(BASE,confidence=False,reopen=False),
    'long_memory':replace(BASE,confidence=False,window=10000),
}
KINDS=('stationary','chain','drift_recurrence','rare_witness','budget_pressure')


def loss(p,y):
    return -(y*math.log(max(p,1e-12))+(1-y)*math.log(max(1-p,1e-12)))


def synthetic(kind,seed,policy,length):
    # Generate outside the timed learner run, preserve paired samples across policies.
    rows=list(stream(kind,random.Random(seed),length))
    m=PredictiveMemory(policy=policy)
    start=time.perf_counter()
    score=oracle_score=brier=unsafe=audits=cell_sum=sample_sum=0
    learner_seconds=0.0
    transition_score=transition_count=0
    truth={}
    for t,(context,y,p) in enumerate(rows):
        tick=time.perf_counter()
        prediction=m.predict(context)
        m.observe(context,y)
        learner_seconds+=time.perf_counter()-tick
        score+=loss(prediction,y)
        oracle_score+=loss(p,y)
        brier+=(prediction-y)**2
        if kind=='drift_recurrence' and any(s <= t < s+300 for s in (length//3,2*length//3)):
            transition_score+=loss(prediction,y);transition_count+=1
        truth[context]=p
        if kind == 'drift_recurrence':
            rates=[.1,.1,.9,.9,.5,.5]
            if length//3 <= t < 2*length//3:rates[0],rates[2]=.9,.1
            truth={c:rates[c[1]] for c in truth}
        # Evaluator owns probabilities, never supplies them to learner.
        if (t+1)%64==0:
            bad=any(max(truth[x] for x in c)-min(truth[x] for x in c)>policy.epsilon+1e-12 for c in m.cells)
            unsafe+=bad;audits+=1
            cell_sum+=len(m.cells);sample_sum+=m.snapshot['retained_samples']
    duration=time.perf_counter()-start
    return dict(log_loss=score/length,oracle_log_loss=oracle_score/length,
                excess_log_loss=(score-oracle_score)/length,brier=brier/length,
                transition_log_loss=transition_score/max(1,transition_count),
                unsafe_fold_fraction=unsafe/max(1,audits),mean_cells=cell_sum/max(1,audits),
                mean_retained_samples=sample_sum/max(1,audits),seconds=duration,
                microseconds_per_event=learner_seconds*1e6/length,**m.snapshot)


def real_stream():
    path=ROOT/'research/data/sunspots.csv'
    with path.open() as f: rows=list(csv.DictReader(f))
    values=[float(r['SUNACTIVITY']) for r in rows]
    # Fixed feature map chosen before inspecting scores: two lagged direction bits.
    directions=[int(b>a) for a,b in zip(values,values[1:])]
    output={}
    policies={k:POLICIES[k] for k in ('guarded','empirical_fold','no_fold','long_memory')}
    for name,policy in policies.items():
        adapter=PredictiveEvidenceAdapter(('down','up'),policy)
        score=brier=hits=count=0
        start=time.perf_counter()
        for i in range(2,len(directions)):
            context=tuple(str(x) for x in directions[i-2:i])
            year=int(rows[i+1]['YEAR'])
            y=directions[i]
            p=adapter.predict(context)
            if year>=1800:
                score+=loss(p,y);brier+=(p-y)**2;hits+=int((p>=.5)==y);count+=1
            adapter.observe(EvidenceAtom(context,('down','up')[y],year,year,'sunspot-archive'))
        output[name]=dict(log_loss=score/count,brier=brier/count,accuracy=hits/count,n=count,
                          seconds=time.perf_counter()-start,**adapter.memory.snapshot)
    # Prequential expanding Bernoulli and smoothed persistence baselines.
    for name in ('expanding_rate','persistence'):
        score=brier=hits=count=0
        for i in range(2,len(directions)):
            p=(sum(directions[:i])+.5)/(i+1) if name=='expanding_rate' else (.9 if directions[i-1] else .1)
            y=directions[i];year=int(rows[i+1]['YEAR'])
            if year>=1800:
                score+=loss(p,y);brier+=(p-y)**2;hits+=int((p>=.5)==y);count+=1
        output[name]=dict(log_loss=score/count,brier=brier/count,accuracy=hits/count,n=count)
    return dict(source='statsmodels annual sunspots 1700–2008; public domain NGDC series',
                source_url='https://github.com/statsmodels/statsmodels/blob/main/statsmodels/datasets/sunspots/sunspots.csv',
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                protocol='Chronological archival replay, not live feed; predict annual increase from two previous direction bits; warm-up before 1800; score 1800–2008 then update.',results=output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--length',type=int,default=6000)
    args=parser.parse_args()
    if args.length<600:raise ValueError('Need at least 600 events')
    started=datetime.now(timezone.utc).isoformat();start=time.perf_counter()
    exact={}
    for name,w in exact_worlds().items():
        profiles=w.profiles(3)
        exact[name]=dict(horizon=3,quotient=exact_quotient(profiles),stable_memory=stable_memory(w),
                        frontier={k:dict(distortion=str(v),partition=p) for k,(v,p) in frontier(profiles).items()},
                        h1_candidate_audit=audit_partition(profiles,exact_quotient(w.profiles(1))))
    raw=[]
    for split,seeds in (('development',range(4)),('heldout',range(100,108))):
        for kind in KINDS:
            for seed in seeds:
                for name,policy in POLICIES.items():
                    raw.append(dict(split=split,world=kind,seed=seed,method=name,**synthetic(kind,seed,policy,args.length)))
            print(f'{split}: {kind} complete',flush=True)
    summary={}
    for kind in KINDS:
        summary[kind]={}
        for name in POLICIES:
            rows=[r for r in raw if r['split']=='heldout' and r['world']==kind and r['method']==name]
            summary[kind][name]={key:statistics.mean(r[key] for r in rows) for key in ('log_loss','excess_log_loss','brier','unsafe_fold_fraction','mean_cells','mean_retained_samples','microseconds_per_event','transition_log_loss','reopenings')}
    paired={}
    for name in POLICIES:
        if name=='no_fold':continue
        differences=[]
        for seed in range(100,108):
            candidate=statistics.mean(r['log_loss'] for r in raw if r['split']=='heldout' and r['seed']==seed and r['method']==name)
            baseline=statistics.mean(r['log_loss'] for r in raw if r['split']=='heldout' and r['seed']==seed and r['method']=='no_fold')
            differences.append(candidate-baseline)
        mean=statistics.mean(differences);se=statistics.stdev(differences)/math.sqrt(len(differences))
        paired[name]=dict(mean_log_loss_delta=mean,seed_deltas=differences,descriptive_95_t_interval=[mean-2.365*se,mean+2.365*se])
    output=dict(contract='SL-Bench v0.1 finite predictive subset',base_commit='01f0fdeb1cf7f6ac353c3eb7894146fa1fd96e65',
                policies={k:asdict(v) for k,v in POLICIES.items()},length=args.length,exact=exact,raw=raw,heldout=summary,paired=paired,real=real_stream())
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2)+'\n')
    inputs=[*sorted((ROOT/'research/sl_bench').glob('*.py')),ROOT/'python/cortex/predictive_memory.py',ROOT/'tests/test_sl_oracle.py',ROOT/'research/data/sunspots.csv']
    manifest=dict(schema_version=1,claim_id='sl-bench-v0.1-finite-characterization',repository=dict(commit=output['base_commit'],dirty=True),
        command=f'PYTHONPATH=python:. python -m research.sl_bench.run --output {args.output} --length {args.length}',
        environment=dict(software=[platform.python_version()],hardware=platform.platform()+' '+platform.machine()),
        mathematics=dict(assertion_tested='Exact finite profiles/partition frontiers plus sampled ablations; not a universal theory test',
            coefficient_domain='Fraction rational oracle; Python binary64 learner and scores',conventions='Current-state emission followed by controlled transition; all open-loop probes length 1..3; binary streaming targets',
            inputs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
            bounds=dict(exact_states_max=4,exact_horizon=3,development_seeds=list(range(4)),heldout_seeds=list(range(100,108)),events_per_run=args.length,hard_memory_cap='none; algorithmic caps in policies'),
            non_claims=['No whole-theory validation','No general online consistency','No native Rust integration','No universal predictive or causal adequacy','No live deployment']),
        randomness=dict(used=True,generator='Python random.Random MT19937',seed=100),
        run=dict(started_at=started,runtime_seconds=time.perf_counter()-start,exit_status=0),
        outputs=[dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())],
        checks=['Exact path enumeration compared with independent state-path summation in test_sl_oracle.py','Exhaustive Bell partition counts and lumpable minimality','Paired streams with evaluator-only probabilities','Chronological test-then-update real replay'],
        result='Finite experiment completed; interpret numeric results with documented scope',residual_risks=['Small synthetic family','Finite H','Context grammar supplied','Dependent real stream invalidates IID confidence guarantees','Total wall time includes evaluator instrumentation; per-event timing isolates predict/update'])
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(paired=paired,real=output['real']),indent=2))

if __name__=='__main__':main()
