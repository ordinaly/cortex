#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, sys

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "v0_9_5"

def check_freeze() -> list[str]:
    errors=[]
    for line in (REF/'SHA256SUMS.txt').read_text().splitlines():
        digest, rel = line.split('  ',1)
        p=ROOT/rel
        if not p.exists(): errors.append(f'missing frozen file: {rel}'); continue
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=digest: errors.append(f'frozen hash mismatch: {rel}')
    return errors

def check_paths() -> list[str]:
    errors=[]
    text_ext={'.py','.rs','.toml','.md','.yml','.yaml','.json'}
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix not in text_ext: continue
        if REF in p.parents or p == Path(__file__).resolve(): continue
        try: text=p.read_text()
        except UnicodeDecodeError: continue
        if '/mnt/data' in text: errors.append(f'nonportable path in {p.relative_to(ROOT)}')
    return errors

def main():
    errors=check_freeze()+check_paths()
    if errors:
        print('\n'.join(errors)); return 1
    print('repo audit: PASS')
    return 0
if __name__=='__main__': raise SystemExit(main())
