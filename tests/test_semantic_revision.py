from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))

from semantic_revision import CortexSemanticRevision
from semantic_revision_campaign import build_world, neural_for, structure_for, swapped_mapping, run_case, RELATIONS

def test_single_semantic_shock_does_not_flip_incumbent():
    concepts,labels,_ce,se,truth=build_world("asymmetric-chain",3)
    concepts=list(concepts); labels=list(labels)
    alt,_=swapped_mapping(truth,concepts,3)
    m=CortexSemanticRevision(concepts,labels,RELATIONS,se)
    for e in range(3):
        m.advance(e,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,se))
    before=m.accepted_mapping()
    m.advance(3,neural=neural_for(labels,alt))
    assert m.accepted_mapping()==before

def test_sustained_conflict_revises_through_unresolved():
    concepts,labels,_ce,se,truth=build_world("asymmetric-branch",4)
    concepts=list(concepts); labels=list(labels)
    alt,swapped=swapped_mapping(truth,concepts,4)
    m=CortexSemanticRevision(concepts,labels,RELATIONS,se)
    for e in range(3):
        m.advance(e,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,se))
    for e in range(3,11):
        m.advance(e,neural=neural_for(labels,alt),structural=structure_for(concepts,alt,se))
        if all(m.read(c).accepted_label==alt[c] for c in concepts): break
    for c in swapped:
        kinds=[x.kind for x in m.events if x.concept==c]
        assert "challenged" in kinds and "withdrawn" in kinds and "revised" in kinds
        assert kinds.index("challenged") < kinds.index("withdrawn") < kinds.index("revised")

def test_revision_is_local_to_changed_labels():
    row=run_case("asymmetric-weave",5)
    assert row["shifted"] and row["recovered"]
    assert row["unaffected_preserved"]

def test_history_is_bounded():
    concepts,labels,_ce,se,truth=build_world("asymmetric-chain",6)
    concepts=list(concepts); labels=list(labels)
    m=CortexSemanticRevision(concepts,labels,RELATIONS,se,evidence_horizon=5)
    for e in range(20):
        m.advance(e,neural=neural_for(labels,truth),structural=structure_for(concepts,truth,se))
    assert m.neural_history_size <= 5*len(concepts)
    assert m.structural_history_size <= 5*len(concepts)*(len(concepts)-1)*len(RELATIONS)

def test_development_case_passes_core_behaviors():
    row=run_case("asymmetric-chain",7)
    assert row["initial_correct"]
    assert row["shock_preserved"]
    assert row["shifted"]
    assert row["recovered"]
    assert row["revision_path_valid"]
