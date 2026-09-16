#!/usr/bin/env python3
"""Replay frozen golden fixtures and report native throughput.
Run after `pip install -e .` builds the Rust extension.
"""
from __future__ import annotations
import gzip,json,time
from pathlib import Path
from cortex import Cortex
ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests'/'fixtures'
for path in sorted(FIX.glob('*.jsonl.gz')):
    meta=json.loads((path.with_suffix('').with_suffix('.final.json')).read_text())
    cfg=json.dumps(meta['config'])
    m=Cortex(dim=meta['config']['dim'],budget=meta['config']['budget'],config_json=cfg)
    n=0; se=0.0; t0=time.perf_counter_ns()
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            rec=json.loads(line); r=m.step(rec['x'],float(rec['y']))
            se+=(rec['y']-r.prediction)**2; n+=1
    us=(time.perf_counter_ns()-t0)/1000/max(1,n)
    print(f'{path.stem}: n={n}  {us:.3f} us/step  brier={se/max(1,n):.6f}  stored={m.stored}')
