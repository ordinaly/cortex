"""Campaign for semantic-revision-v1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from random import Random

from semantic_grounding import NeuralLabelEvidence, StructuralSemanticObservation
from semantic_grounding_campaign import FAMILY_EDGES, RELATIONS, build_world
from semantic_revision import CortexSemanticRevision

PROTOCOL="semantic-revision-v1"
DEVELOPMENT_SEEDS=tuple(range(20))
OFFICIAL_SEEDS=tuple(range(400,420))

def swapped_mapping(truth: dict[str,str], concepts: list[str], seed:int):
    rng=Random(seed*1777+31); order=list(concepts); rng.shuffle(order)
    a,b=order[:2]; m=dict(truth); m[a],m[b]=m[b],m[a]
    return m,(a,b)

def neural_for(labels, mapping, confidence=0.88):
    out=[]
    rest=(1-confidence)/(len(labels)-1)
    for c,l in mapping.items():
        p={x:rest for x in labels}; p[l]=confidence
        out.append(NeuralLabelEvidence(c,p))
    return out

def structure_for(concepts, mapping, semantic_edges):
    return [
        StructuralSemanticObservation(a,r,b,(mapping[a],r,mapping[b]) in semantic_edges)
        for a in concepts for b in concepts if a!=b for r in RELATIONS
    ]

def event_epoch(model, concept, kind, after):
    vals=[e.epoch for e in model.events if e.concept==concept and e.kind==kind and e.epoch>=after]
    return vals[0] if vals else None

def run_case(family,seed):
    concepts,labels,_ce,semantic_edges,truth=build_world(family,seed)
    concepts=list(concepts); labels=list(labels)
    alt,swapped=swapped_mapping(truth,concepts,seed)
    model=CortexSemanticRevision(concepts,labels,RELATIONS,semantic_edges)

    epoch=0
    # establish incumbent
    for _ in range(3):
        model.advance(epoch,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,semantic_edges)); epoch+=1
    initial={c:model.read(c).accepted_label for c in concepts}

    # one contradictory perceptual shock only
    shock_epoch=epoch
    model.advance(epoch,neural=neural_for(labels,alt)); epoch+=1
    shock_preserved=all(model.read(c).accepted_label==truth[c] for c in concepts)
    model.advance(epoch,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,semantic_edges)); epoch+=1

    shift_start=epoch
    for _ in range(8):
        model.advance(epoch,neural=neural_for(labels,alt),structural=structure_for(concepts,alt,semantic_edges)); epoch+=1
        if all(model.read(c).accepted_label==alt[c] for c in concepts): break
    shift_end=epoch-1
    shifted=all(model.read(c).accepted_label==alt[c] for c in concepts)

    recovery_start=epoch
    for _ in range(8):
        model.advance(epoch,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,semantic_edges)); epoch+=1
        if all(model.read(c).accepted_label==truth[c] for c in concepts): break
    recovery_end=epoch-1
    recovered=all(model.read(c).accepted_label==truth[c] for c in concepts)

    unaffected=[c for c in concepts if c not in swapped]
    unaffected_preserved=all(
        not any(e.kind in {"withdrawn","revised"} for e in model.events if e.concept==c)
        for c in unaffected
    )
    paths=[]
    for c in swapped:
        ch=event_epoch(model,c,"challenged",shift_start)
        wd=event_epoch(model,c,"withdrawn",shift_start)
        rv=event_epoch(model,c,"revised",shift_start)
        paths.append(ch is not None and wd is not None and rv is not None and ch<=wd<rv)

    return {
      "protocol":PROTOCOL,"family":family,"seed":seed,
      "initial_correct":all(initial[c]==truth[c] for c in concepts),
      "shock_preserved":shock_preserved,
      "shifted":shifted,
      "recovered":recovered,
      "shift_latency":shift_end-shift_start+1,
      "recovery_latency":recovery_end-recovery_start+1,
      "revision_path_valid":all(paths),
      "unaffected_preserved":unaffected_preserved,
      "neural_history_bounded":model.neural_history_size<=model.evidence_horizon*len(concepts),
      "structural_history_bounded":model.structural_history_size<=model.evidence_horizon*len(concepts)*(len(concepts)-1)*len(RELATIONS),
    }

def campaign(seeds):
    return [run_case(f,s) for s in seeds for f in FAMILY_EDGES]

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group()
    g.add_argument("--development",action="store_true"); g.add_argument("--official",action="store_true")
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    rows=campaign(OFFICIAL_SEEDS if a.official else DEVELOPMENT_SEEDS)
    a.output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
if __name__=="__main__": main()
