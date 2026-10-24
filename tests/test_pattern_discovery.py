from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from compositional_language import Atom, Entity, Symbol, Fact
from pattern_discovery import Episode, discover, select, evaluate

R, S, T, N = [Symbol(n, 1, ('entity', 'entity')) for n in ('R', 'S', 'T', 'noise')]

def atom(symbol, a, b, polarity=1):
    return Atom(symbol, (Entity(a), Entity(b)), polarity)

def episode(name, polarity=1, reverse=False, noise=True):
    a, b, c = [name + suffix for suffix in 'abc']
    atoms = [atom(R, a, b), atom(S, c, b) if reverse else atom(S, b, c)]
    if noise:
        atoms.append(atom(N, name+'u', name+'v'))
    return Episode(name, tuple(Fact(name+str(i), name, x) for i, x in enumerate(atoms)),
                   (atom(T, a, c, polarity),))

def split(prefix, **kwargs):
    return [episode(prefix+str(i), **kwargs) for i in range(3)]

def test_mine_select_and_transfer_without_supplied_motifs():
    d = discover(split('train'))
    assert len(d.definitions) == len(d.hypotheses) == 1
    chosen = select(d, split('validation'))
    assert len(chosen.schemas) == 1
    result = chosen.test(split('test'))
    assert result.correct == 3 and result.wrong == 0
    assert result.loss == 0 and result.baseline_loss == 3
    assert chosen.complexity_cost == pytest.approx(.9)

def test_counterexamples_and_cost_reject():
    d = discover(split('train'))
    assert not select(d, split('negative', polarity=-1)).schemas
    assert not select(d, split('validation'), cost_per_unit=1).schemas
    assert evaluate(d.hypotheses, split('reversed', reverse=True)).abstained == 3

def test_no_cross_context_motifs_or_duplicate_support():
    e = episode('one')
    # Duplicate source evidence within one episode cannot count as repetition.
    duplicated = Episode('one', e.facts + (Fact('copy', 'one', e.facts[0].atom),))
    assert not discover([duplicated]).definitions
    with pytest.raises(ValueError, match='one context'):
        Episode('mixed', (e.facts[0], episode('two').facts[1]))
    isolated = [Episode('left', (e.facts[0],)), Episode('right', (e.facts[1],))]
    assert not discover(isolated).definitions

def test_order_rename_invariance_and_evidence_provenance():
    train = split('train')
    d = discover(train)
    shuffled = [Episode(e.episode_id, tuple(reversed(e.facts)), e.outcomes) for e in reversed(train)]
    other = discover(shuffled)
    assert d.definitions == other.definitions and d.hypotheses == other.hypotheses
    renamed = discover(split('renamed'))
    assert d.hypotheses[0].rule_id == renamed.hypotheses[0].rule_id
    assert d.definitions[0].evidence_ids == ('train0', 'train1', 'train2')

def test_leakage_and_budget_guards():
    train = split('train')
    d = discover(train)
    with pytest.raises(ValueError, match='fresh'):
        select(d, train)
    chosen = select(d, split('val'))
    with pytest.raises(ValueError, match='fresh'):
        chosen.test(train)
    with pytest.raises(ValueError, match='fresh'):
        chosen.test(split('val'))
    with pytest.raises(ValueError, match='withheld'):
        Episode('leak', train[0].facts, (train[0].facts[0].atom,))
    with pytest.raises(ValueError, match='duplicate episode'):
        discover([train[0], train[0]])
    with pytest.raises(RuntimeError, match='budget'):
        discover(train, max_examined=1)

def test_unobserved_outcomes_and_missing_patterns_are_unknown():
    train = split('train')
    unlabeled = [Episode(e.episode_id, e.facts) for e in train]
    assert discover(unlabeled).definitions
    assert not discover(unlabeled).hypotheses
    chosen = select(discover(train), split('val'))
    score = chosen.test([Episode('unknown', episode('unknown').facts)])
    assert score.correct == score.wrong == score.abstained == 0
    assert not select(discover(train), []).schemas

def test_redundant_larger_patterns_do_not_earn_extra_gain():
    def enriched(prefix):
        result = []
        for e in split(prefix, noise=False):
            extra = Fact(e.episode_id+'extra', e.episode_id,
                         atom(N, e.episode_id+'a', e.episode_id+'c'))
            result.append(Episode(e.episode_id, (*e.facts, extra), e.outcomes))
        return result
    d = discover(enriched('train'))
    assert len(d.hypotheses) > 1
    chosen = select(d, enriched('validation'))
    assert len(chosen.schemas) == 1
    assert len(chosen.schemas[0].body) == 2
    assert chosen.test(enriched('test')).correct == 3


def test_native_memory_adapter_to_discovered_rule_and_retraction():
    from cortex.hypergraph_geometry import RelationalHyperedge
    from compositional_language import from_hyperedge, LanguageModel
    def native(name, offset):
        edges = [RelationalHyperedge(name+'r', offset, 7, offset+1, 'a', 'r', 'b', .8),
                 RelationalHyperedge(name+'s', offset+1, 2, offset+2, 'b', 's', 'c', .8)]
        facts = tuple(from_hyperedge(e, name) for e in edges)
        target = Atom(Symbol('basin:9', 1, ('entity', 'entity')),
                      (Entity(f'basin:{offset}'), Entity(f'basin:{offset+2}')))
        return Episode(name, facts, (target,))
    d = discover([native('train1', 100), native('train2', 200)])
    chosen = select(d, [native('val1', 300), native('val2', 400)])
    test = native('fresh', 500)
    model = LanguageModel(test.facts, chosen.schemas)
    assert model.explain(test.outcomes[0], 'fresh').state == 1
    model.retract_fact('freshr')
    assert model.explain(test.outcomes[0], 'fresh').state == 0


def test_disconnected_candidates_still_consume_budget():
    facts = tuple(Fact(str(i), 'ctx', atom(N, str(2*i), str(2*i+1))) for i in range(8))
    with pytest.raises(RuntimeError, match='budget'):
        discover([Episode('isolated', facts)], max_examined=5)
