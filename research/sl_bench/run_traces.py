"""Frozen-train/held-out trial experiment for finite joint profiles."""
import argparse,hashlib,json,math,platform,random,statistics,time,sys
from collections import deque
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
from cortex.trace_memory import TracePredictiveMemory
from .trace_worlds import controlled_world,mixture_profiles,trial
from .oracle import FiniteWorld,audit_partition

ROOT=Path(__file__).resolve().parents[2]


def deep_size(value,seen=None):
    # CPython retained object graph estimate; not allocator RSS or peak memory.
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));total=sys.getsizeof(value)
    if isinstance(value,dict):
        total+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
    elif isinstance(value,(tuple,list,set,deque)):
        total+=sum(deep_size(v,seen) for v in value)
    elif hasattr(value,'__dict__'):total+=deep_size(vars(value),seen)
    return total


def factorize(joint,words):
    marginals=[sum(p for p,w in zip(joint,words) if w[i]==1) for i in range(len(words[0]))]
    return tuple(math.prod(marginals[i] if y else 1-marginals[i] for i,y in enumerate(w)) for w in words)


def run(seed,family,train=4096,test=1024):
    w,m=controlled_world(delay=family=='delayed_action',noisy=family=='noisy_correlation')
    if family=='null':w=FiniteWorld(w.kernel,tuple(('1/2','1/2') for _ in w.emission))
    exact=mixture_profiles(w,m)
    models={'joint_action':TracePredictiveMemory(), 'joint_blind':TracePredictiveMemory(action_conditioned=False), 'joint_action_budget':TracePredictiveMemory(window=48)}
    rng=random.Random(seed)
    probes=list(product((0,1),repeat=3))
    train_rows=[]
    for _ in range(train):
        c=rng.randrange(4);probe=rng.choice(probes)
        train_rows.append((c,probe,trial(rng,w,m[c],probe)))
    fit={}
    for name,model in models.items():
        tick=time.perf_counter()
        for c,p,y in train_rows:model.observe(('opaque',c),p,y)
        fit[name]=time.perf_counter()-tick
    # Independent holdout RNG; no updates or hyperparameter selection on holdout.
    rng=random.Random(100000+seed)
    losses={k:[] for k in ('joint_action','joint_blind','joint_action_budget','factorized_action','oracle','uniform')}
    for _ in range(test):
        c=rng.randrange(4);p=rng.choice(probes);y=trial(rng,w,m[c],p)
        words=models['joint_action'].words[3];index=words.index(y)
        joint=models['joint_action'].predict(('opaque',c),p)
        predictions={'joint_action':joint,'joint_blind':models['joint_blind'].predict(('opaque',c),p),
                     'joint_action_budget':models['joint_action_budget'].predict(('opaque',c),p),
                     'factorized_action':factorize(joint,words),'oracle':exact[c][p],'uniform':(1/8,)*8}
        for name,pred in predictions.items():losses[name].append(-math.log(max(float(pred[index]),1e-15)))
    output={}
    for name in losses:
        row={'log_loss':statistics.mean(losses[name])}
        if name not in ('oracle','uniform'):
            errors=[]
            model=models[name if name in models else 'joint_action']
            for c in range(4):
                for p,true in exact[c].items():
                    pred=model.predict(('opaque',c),p)
                    if name=='factorized_action':pred=factorize(pred,model.words[len(p)])
                    errors.append(sum(abs(float(a)-b) for a,b in zip(true,pred))/2)
            row.update(mean_profile_tv=statistics.mean(errors),max_profile_tv=max(errors),fit_seconds=fit[name if name in models else 'joint_action'],retained_object_bytes=deep_size(model),**model.snapshot)
        output[name]=row
    cells,unresolved=models['joint_action'].empirical_partition(epsilon=.25,min_samples=32)
    partition=tuple(tuple(c[1] for c in cell) for cell in cells)
    return dict(seed=seed,family=family,methods=output,partition_audit=audit_partition(exact,partition,'1/4'),unresolved=len(unresolved))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    files=[ROOT/'python/cortex/trace_memory.py',ROOT/'research/sl_bench/run_traces.py',ROOT/'research/sl_bench/trace_worlds.py',ROOT/'research/sl_bench/oracle.py',ROOT/'tests/test_sl_trace_memory.py']
    hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before=hashes();start=time.perf_counter();date=datetime.now(timezone.utc).isoformat()
    rows=[]
    for family in ('correlation','delayed_action','noisy_correlation','null'):
        for seed in range(200,208):rows.append(run(seed,family))
        print(family,flush=True)
    summary={}
    for family in ('correlation','delayed_action','noisy_correlation','null'):
        group=[r for r in rows if r['family']==family]
        summary[family]={name:{key:statistics.mean(r['methods'][name][key] for r in group) for key in group[0]['methods'][name]} for name in group[0]['methods']}
        summary[family]['partition_passes']=sum(r['partition_audit']['passed'] for r in group)
        summary[family]['exact_minimum_cell_count_hits']=sum(r['partition_audit']['passed'] and r['partition_audit']['excess_cells']==0 for r in group)
    assert before==hashes(),'Inputs changed during run'
    target=Path(args.output);target.parent.mkdir(exist_ok=True,parents=True)
    output=dict(version='sl-bench-traces-v0.2',train_trials=4096,test_trials=1024,seeds=list(range(200,208)),summary=summary,runs=rows,
                interpretation='Frozen fitted joint distributions; observed reset contexts; no learned state discovery; exact finite profiles score proposals; predictions are not pooled')
    target.write_text(json.dumps(output,indent=2)+'\n')
    manifest=dict(schema_version=1,claim_id='sl-traces-v0.2',repository=dict(commit='11c2d8e50ca37dab07f25741919b2f699e6aba67',dirty=True),
        command=f'PYTHONPATH=python:. python -m research.sl_bench.run_traces --output {args.output}',
        environment=dict(software=[platform.python_version()],hardware=platform.platform()),
        mathematics=dict(assertion_tested='Finite learned joint profiles distinguish correlation and delayed/action witnesses',coefficient_domain='rational oracle; binary64 estimator',conventions='Current emission then transition; independent reset trials; horizon 3',inputs=[dict(path=p,sha256=h) for p,h in before.items()],bounds=dict(states=8,contexts=4,horizon=3,seeds=list(range(200,208)),training=4096,heldout=1024,window_per_probe=256),non_claims=['No state discovery','No compression promotion','No universal dynamic sufficiency','No real-data validation of this extension']),
        randomness=dict(used=True,generator='random.Random MT19937; holdout seed=100000+seed',seed=200),run=dict(started_at=date,runtime_seconds=time.perf_counter()-start,exit_status=0),
        outputs=[dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())],checks=['Source hashes unchanged before/after','Heldout is prediction-only','Exact mixture-profile audit'],result='Completed finite sampled campaign',residual_risks=['Known reset contexts','Small fixed world family','No process-enforced resource caps','Categorical counts scale exponentially in horizon'])
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
