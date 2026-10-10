"""Physical folding campaign; fixed parameters, fresh seeds, paired streams."""
import argparse,json,hashlib,platform,random,statistics,time,math,subprocess
from dataclasses import replace,asdict
from datetime import datetime,timezone
from pathlib import Path
from cortex.accordion_memory import AccordionMemory,AccordionPolicy
from .run_traces import deep_size
ROOT=Path(__file__).resolve().parents[2]
POLICY=AccordionPolicy(max_contexts=32)


def stream(family,seed,length):
    rng=random.Random(seed)
    for t in range(length):
        n=96 if family=='pressure' else 16
        c=rng.randrange(n)
        p=.1 if c%2==0 else .9
        if family=='fair':p=.5
        if family=='chain':p=.1+.05*(c%16)
        if family=='local_drift' and c==0 and t>=length//2:p=.9
        yield c,int(rng.random()<p),p


def run(family,seed,length):
    rows=list(stream(family,seed,length));results={}
    for name,policy in [('unfolded',replace(POLICY,folding=False)),('accordion',POLICY),('frozen',replace(POLICY,reopening=False))]:
        m=AccordionMemory(policy);loss=post=0;post_n=0;bytes_=[];payload=[];times=0
        recovery=None;streak=0
        for t,(c,y,true) in enumerate(rows):
            tick=time.perf_counter();p=m.predict(c);m.observe(c,y);times+=time.perf_counter()-tick
            l=-(y*math.log(p)+(1-y)*math.log(1-p));loss+=l
            if t>=8192:post+=l;post_n+=1
            if family=='local_drift' and c==0 and t>=length//2 and recovery is None:
                streak=streak+1 if abs(p-true)<=.15 else 0
                if streak>=5:recovery=t-length//2
            if t>=8192 and (t+1)%1024==0:bytes_.append(deep_size(m));payload.append(m.snapshot['payload_bytes'])
        results[name]=dict(log_loss=loss/length,postwarm_log_loss=post/post_n,mean_retained_bytes=statistics.mean(bytes_),mean_payload_bytes=statistics.mean(payload),
                           predict_update_us=times*1e6/length,recovery_events=recovery,**m.snapshot)
    return dict(family=family,seed=seed,methods=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    sources=[ROOT/'python/cortex/__init__.py',ROOT/'python/cortex/accordion_memory.py',Path(__file__),ROOT/'research/sl_bench/run_traces.py',ROOT/'tests/test_sl_accordion.py']
    hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    before=hashes();start=time.perf_counter();date=datetime.now(timezone.utc).isoformat()
    rows=[]
    for f in ['redundant','fair','chain','local_drift','pressure']:
        for seed in range(300,308):rows.append(run(f,seed,24000))
        print(f,flush=True)
    summary={}
    for f in ['redundant','fair','chain','local_drift','pressure']:
        group=[r for r in rows if r['family']==f];summary[f]={}
        for name in ['unfolded','accordion','frozen']:
            summary[f][name]={k:statistics.mean(r['methods'][name][k] for r in group) for k in ['log_loss','postwarm_log_loss','mean_retained_bytes','mean_payload_bytes','predict_update_us','folds','reopens']}
            recovery=[r['methods'][name]['recovery_events'] for r in group if r['methods'][name]['recovery_events'] is not None]
            summary[f][name]['recovered_runs']=len(recovery)
            summary[f][name]['mean_recovery_events']=statistics.mean(recovery) if recovery else None
    assert before==hashes()
    target=Path(args.output);target.parent.mkdir(exist_ok=True,parents=True)
    result=dict(policy=asdict(POLICY),length=24000,seeds=list(range(300,308)),warmup=8192,summary=summary,runs=rows,
                protocol='Prequential; predictions precede updates; binary observed contexts; physical fold deletes full windows; full object-graph bytes include witness/metadata costs; means across eight paired seeds')
    target.write_text(json.dumps(result,indent=2)+'\n')
    manifest=dict(schema_version=1,claim_id='sl-accordion-v0.3',repository=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())),command=f'PYTHONPATH=python:. python -m research.sl_bench.run_accordion --output {args.output}',environment=dict(software=[platform.python_version()],hardware=platform.platform()),
        mathematics=dict(assertion_tested='Retained storage and prediction tradeoff of physical binary folding',coefficient_domain='binary64 probabilities/loss; integer samples',conventions='Prequential, bounded witnesses, empirical complete-link eligibility; no statistical certificate',inputs=[dict(path=k,sha256=v) for k,v in before.items()],bounds=dict(seeds=list(range(300,308)),families=5,methods=3,events=24000,policy=asdict(POLICY)),non_claims=['No lossless recovery','No universal sufficiency','No native promotion','No streaming domain transfer']),
        randomness=dict(used=True,generator='random.Random MT19937',seed=300),run=dict(started_at=date,runtime_seconds=time.perf_counter()-start,exit_status=0),outputs=[dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())],checks=['Input hashes identical before/after','Paired streams','Total object graph includes witnesses and summaries'],result='Completed bounded campaign',residual_risks=['Fixed binary contexts','Small generator family','Empirical thresholds uncalibrated','Object size not RSS or peak','No enforced process caps'])
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
