from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from compositional_language import (Symbol, Entity, Variable, Atom, Schema, Fact, MotifExample,
    LanguageModel, invent_motif, RuleExample, propose_rule, validate_rule, from_hyperedge)

R = Symbol('opaque-r', 1, ('entity', 'entity'))
S = Symbol('opaque-s', 1, ('entity', 'entity'))
T = Symbol('opaque-t', 1, ('entity', 'entity'))

def edge(symbol, *names, polarity=1):
    return Atom(symbol, tuple(Entity(n) for n in names), polarity)

def motif(prefix):
    return (edge(R, prefix+'a', prefix+'b'), edge(S, prefix+'b', prefix+'c'))

def definition():
    return invent_motif([MotifExample('train1', motif('one')), MotifExample('train2', tuple(reversed(motif('two'))))])

def test_invented_symbol_ignores_entity_names_and_edge_order_but_preserves_roles():
    a = definition()
    b = invent_motif([MotifExample('new1', motif('x')), MotifExample('new2', motif('y'))])
    assert a.head == b.head and a.body == b.body
    with pytest.raises(ValueError):
        invent_motif([MotifExample('a', motif('x')), MotifExample('b', (edge(R,'ya','yb'), edge(S,'yc','yb')))])
    with pytest.raises(ValueError):
        invent_motif([MotifExample('a', motif('x')), MotifExample('b', (edge(R,'ya','yb'), edge(S,'yb','ya')))])


def test_nested_composition_transfers_and_tracks_dependencies():
    d = definition()
    x, y, z, u, v = [Variable(c) for c in 'xyzuv']
    # Canonical R then S yields exposed roles source, intermediate, target.
    composed = Symbol('composed', 1, ('entity', 'entity'))
    rule = Schema('compose', (Atom(d.head.symbol,(x,y,z)), Atom(d.head.symbol,(z,u,v))), Atom(composed,(x,v)), 'definition')
    facts = [Fact('a','episode',edge(R,'A','B')), Fact('b','episode',edge(S,'B','C')),
             Fact('c','episode',edge(R,'C','D')), Fact('d','episode',edge(S,'D','E'))]
    model = LanguageModel(facts, [d,rule])
    explanation = model.explain(edge(composed,'A','E'),'episode')
    assert explanation.state == 1
    proof = next(n for n in explanation.nodes if n.proof_id == explanation.selected_positive_id)
    assert set(proof.fact_dependencies) == {'a','b','c','d'}
    assert {model.origins[r][0] for r in proof.rule_dependencies} == {d.rule_id, 'compose'}
    model.retract_fact('c')
    assert model.explain(edge(composed,'A','E'),'episode').reason == 'unproven'
    model.restore_fact('c')
    assert model.explain(edge(composed,'A','E'),'episode').state == 1
    model.set_rule_active(d.rule_id, False)
    assert model.explain(edge(composed,'A','E'),'episode').state == 0
    model.set_rule_active(d.rule_id, True)
    assert model.explain(edge(composed,'A','E'),'episode').state == 1


def test_scope_does_not_join_across_contexts():
    model = LanguageModel([Fact('a','one',edge(R,'a','b')), Fact('b','two',edge(S,'b','c'))], [definition()])
    assert model.explain(edge(definition().head.symbol,'a','b','c'),'one').reason == 'unproven'


def test_explicit_negation_conflict_and_no_explosion():
    x,y=Variable('x'),Variable('y')
    rule=Schema('r', (Atom(R,(x,y)),), Atom(T,(x,y)))
    model=LanguageModel([Fact('p','c',edge(R,'a','b')),Fact('n','c',edge(T,'a','b',polarity=-1))],[rule])
    assert model.explain(edge(T,'a','b'),'c').reason == 'conflict'
    assert model.explain(edge(S,'a','b'),'c').reason == 'unproven'
    model.retract_fact('p')
    assert model.explain(edge(T,'a','b'),'c').reason == 'refuted'


def test_redundant_evidence_survives_retraction():
    d=definition()
    facts=[Fact('a','c',motif('')[0]),Fact('a2','c',motif('')[0]),Fact('b','c',motif('')[1])]
    model=LanguageModel(facts,[d])
    q=edge(d.head.symbol,'a','b','c')
    assert len(model.explain(q,'c').positive_proof_ids)==2
    model.retract_fact('a')
    assert model.explain(q,'c').state==1


def example(prefix, polarity=1):
    return RuleExample(prefix, motif(prefix), edge(T,prefix+'a',prefix+'c',polarity=polarity))

def test_empirical_rule_is_proposed_then_tested_on_fresh_entities():
    schema=propose_rule([example('train1'),example('train2')])
    result=validate_rule(schema,[example('test1'),example('test2')])
    assert result.accepted and result.supported==2
    assert not validate_rule(schema,[example('test1'),example('counter',-1)]).accepted
    with pytest.raises(ValueError):
        validate_rule(schema,[example('train1')])
    missing=RuleExample('unknown',(edge(R,'a','b'),),edge(T,'a','c'))
    result=validate_rule(schema,[missing])
    assert result.untested==1 and result.refuted==0


def test_higher_arity_and_type_checks():
    gives=Symbol('g',1,('person','object','person'))
    owns=Symbol('o',1,('person','object'))
    x,obj,y=Variable('x','person'),Variable('obj','object'),Variable('y','person')
    rule=Schema('r',(Atom(gives,(x,obj,y)),),Atom(owns,(y,obj)))
    fact=Fact('e','c',Atom(gives,(Entity('A','person'),Entity('cup','object'),Entity('B','person'))))
    m=LanguageModel([fact],[rule])
    assert m.explain(Atom(owns,(Entity('B','person'),Entity('cup','object'))),'c').state==1
    with pytest.raises(ValueError):
        Atom(gives,(Entity('cup','object'),Entity('A','person'),Entity('B','person')))


def test_unsafe_head_cycles_and_budgets_are_explicit_errors():
    x,y,z=[Variable(c) for c in 'xyz']
    with pytest.raises(ValueError):
        Schema('unsafe',(Atom(R,(x,y)),),Atom(T,(x,z)))
    with pytest.raises(ValueError,match='acyclic'):
        LanguageModel([Fact('f','c',edge(R,'a','b'))],[Schema('loop',(Atom(R,(x,y)),),Atom(R,(x,y)))])
    with pytest.raises(RuntimeError,match='grounding budget'):
        LanguageModel([Fact('f','c',edge(R,'a','b'))],[Schema('r',(Atom(R,(x,y)),),Atom(T,(x,y)))],max_groundings=1)
    with pytest.raises(RuntimeError,match='proof bound'):
        LanguageModel([Fact('f','c',edge(R,'a','b')),Fact('f2','c',edge(R,'a','b'))],
                      [Schema('r',(Atom(R,(x,y)),),Atom(T,(x,y)))],max_proofs=1)


def test_versions_and_signature_collisions():
    r2=Symbol(R.name,2,R.roles)
    x,y=Variable('x'),Variable('y')
    model=LanguageModel([Fact('f','c',edge(r2,'a','b'))],[Schema('r',(Atom(R,(x,y)),),Atom(T,(x,y)))])
    assert model.explain(edge(T,'a','b'),'c').state==0
    bad=Symbol(R.name,1,('entity',))
    with pytest.raises(ValueError,match='incompatible signature'):
        LanguageModel([Fact('f','c',edge(R,'a','b')),Fact('g','c',edge(bad,'a'))],[])


def test_existing_hypergraph_adapter_preserves_roles_without_claiming_confidence():
    from cortex.hypergraph_geometry import RelationalHyperedge
    e=RelationalHyperedge('edge',1,2,3,'alice','holds','cup',.3)
    fact=from_hyperedge(e,'episode')
    assert fact.atom.symbol.name=='basin:2'
    assert tuple(t.name for t in fact.atom.terms)==('basin:1','basin:3')
    assert fact.fact_id=='edge'


def test_abstraction_preserves_distinct_entities_and_repeated_roles():
    d = definition()
    m = LanguageModel([Fact('r','c',edge(R,'a','b')), Fact('s','c',edge(S,'b','a'))],[d])
    assert m.explain(edge(d.head.symbol,'a','b','a'),'c').reason == 'unproven'
    repeated = invent_motif([MotifExample('one',(edge(R,'a','a'),)),
                             MotifExample('two',(edge(R,'b','b'),))])
    m = LanguageModel([Fact('r','c',edge(R,'a','b'))],[repeated])
    assert m.explain(edge(repeated.head.symbol,'a'),'c').reason == 'unproven'
