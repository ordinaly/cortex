from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from possibility_geometry import Candidate, FiniteModel, PossibilityFilter, Probe


def cup():
    return FiniteModel('cup-v1', tuple(Candidate(v, ('shared-cup',)) for v in 'ABC'),
                       ('drop', 'inspect'), ('broken', 'intact', 'hard', 'soft'), {
        ('A', 'drop'): {('A', 'broken'): 1},
        ('B', 'drop'): {('B', 'intact'): 1},
        ('C', 'drop'): {('C', 'intact'): 1},
        ('A', 'inspect'): {('A', 'hard'): 1},
        ('B', 'inspect'): {('B', 'hard'): 1},
        ('C', 'inspect'): {('C', 'soft'): 1},
    })


def test_cup_preserves_explanations_then_resolves():
    model = cup()
    f = PossibilityFilter(model, dict.fromkeys('ABC', F(1, 3)))
    prediction = f.predict('drop')
    assert prediction.outcomes['intact'] == F(2, 3)
    assert prediction.candidate_outcomes['B']['intact'] == 1
    result = f.assimilate('drop', 'intact')
    assert result.posterior == {'A': 0, 'B': F(1, 2), 'C': F(1, 2)}
    before = model.geometry([Probe(('drop',), frozenset({('intact',)}))])
    after = model.geometry([Probe(('drop',), frozenset({('intact',)})),
                            Probe(('inspect',), frozenset({('hard',)}))])
    assert before.contains(('B', 'C'))
    assert not after.contains(('B', 'C'))
    assert model.geometry([Probe(('drop',), frozenset({('intact',)}))]) == before
    f.predict('inspect')
    assert f.assimilate('inspect', 'hard').posterior == {'A': 0, 'B': 1, 'C': 0}
    assert prediction.prior == dict.fromkeys('ABC', F(1, 3))


def test_equal_likelihood_preserves_odds_and_rare_event_stays_soft():
    m = FiniteModel('noisy', (Candidate('a'), Candidate('b')), ('act',), ('rare', 'usual'), {
        (v, 'act'): {(v, 'rare'): F(1, 100), (v, 'usual'): F(99, 100)} for v in 'ab'})
    f = PossibilityFilter(m, {'a': F(1, 4), 'b': F(3, 4)})
    f.predict('act')
    result = f.assimilate('act', 'rare')
    assert result.status == 'updated'
    assert result.evidence == F(1, 100)
    assert result.posterior == {'a': F(1, 4), 'b': F(3, 4)}


@pytest.mark.parametrize('observation', ['soft', 'outside-alphabet'])
def test_failure_preserves_witness_but_has_no_posterior(observation):
    f = PossibilityFilter(cup(), {'A': 1, 'B': 0, 'C': 0})
    p = f.predict('drop')
    result = f.assimilate('drop', observation)
    assert result.posterior is None and result.evidence == 0
    assert result.status == 'model_failure'
    assert result.observation == observation and result.prediction is p
    assert f.belief == p.prior
    with pytest.raises(RuntimeError):
        f.predict('inspect')


def test_prequential_contract_and_immutable_records():
    m = cup()
    f = PossibilityFilter(m, {'A': 1, 'B': 0, 'C': 0})
    with pytest.raises(ValueError):
        f.assimilate('drop', 'broken')
    p = f.predict('drop')
    with pytest.raises(RuntimeError):
        f.predict('inspect')
    with pytest.raises(ValueError):
        f.assimilate('inspect', 'hard')
    with pytest.raises(TypeError):
        p.candidate_outcomes['A']['broken'] = 0
    with pytest.raises(TypeError):
        m.rows['A', 'drop']['A', 'broken'] = 0
    with pytest.raises(FrozenInstanceError):
        m.version = 'changed'
    f.assimilate('drop', 'broken')
    with pytest.raises(ValueError):
        f.assimilate('drop', 'broken')


def stochastic():
    return FiniteModel('joint', tuple(Candidate(v) for v in 'ab'), ('x', 'y'), ('0', '1'), {
        ('a', 'x'): {('a', '0'): F(1, 3), ('b', '1'): F(2, 3)},
        ('b', 'x'): {('a', '1'): F(1, 2), ('b', '0'): F(1, 2)},
        ('a', 'y'): {('b', '0'): 1},
        ('b', 'y'): {('a', '1'): 1},
    })


def test_multistep_against_independent_path_enumeration():
    m = stochastic()
    prior = {'a': F(2, 5), 'b': F(3, 5)}
    actions = ('x', 'y', 'x')
    expected = {}
    for states in product(m.names, repeat=4):
        for observations in product(m.observations, repeat=3):
            weight = prior[states[0]]
            for i, action in enumerate(actions):
                weight *= m.rows[states[i], action].get((states[i+1], observations[i]), 0)
            expected[observations] = expected.get(observations, F(0)) + weight
    actual = m.trace_distribution(prior, actions)
    assert all(actual.get(trace, 0) == weight for trace, weight in expected.items())
    assert sum(actual.values()) == 1
    for trace, weight in expected.items():
        if not weight:
            continue
        f = PossibilityFilter(m, prior)
        likelihood = F(1)
        for action, observation in zip(actions, trace):
            f.predict(action)
            result = f.assimilate(action, observation)
            likelihood *= result.evidence
        assert likelihood == weight


def test_joint_kernel_handles_predecessor_dependent_observation():
    m = FiniteModel('predecessor', tuple(Candidate(v) for v in 'ab'), ('act',), ('0', '1'), {
        ('a', 'act'): {('a', '0'): 1}, ('b', 'act'): {('a', '1'): 1}})
    f = PossibilityFilter(m, {'a': F(1, 3), 'b': F(2, 3)})
    p = f.predict('act')
    assert p.successor == {'a': 1, 'b': 0}
    assert p.outcomes == {'0': F(1, 3), '1': F(2, 3)}
    assert f.assimilate('act', '1').posterior == {'a': 1, 'b': 0}


def test_similarity_does_not_constrain_temporal_transport():
    m = stochastic()
    geometry = m.geometry([Probe(('y',), frozenset({('0',)}))])
    assert not geometry.contains(('a', 'b'))
    f = PossibilityFilter(m, {'a': 1, 'b': 0})
    assert f.predict('y').successor == {'a': 0, 'b': 1}


def test_similarity_chain_not_a_simplex_or_valid_merge():
    m = FiniteModel('chain', tuple(Candidate(v) for v in 'abc'), ('act',), ('0', '1'), {
        (v, 'act'): {(v, '1'): p, (v, '0'): 1-p}
        for v, p in zip('abc', [F(0), F(1, 2), F(1)])})
    g = m.geometry([Probe(('act',), frozenset({('1',)}))], F(1, 2))
    assert g.contains(('a', 'b')) and g.contains(('b', 'c'))
    assert not g.contains(('a', 'c')) and not g.contains(('a', 'b', 'c'))
    assert g.contains(()) and g.contains(('a',))
    assert not m.is_exact_partition([('a', 'b', 'c')])
    assert m.is_exact_partition([('a',), ('b',), ('c',)])


def test_multistep_probe_separates_identical_one_step_marginals():
    # a -> c or d; b -> e or f. First outputs agree, joint traces differ.
    rows = {
        ('a', 'act'): {('c', '0'): F(1, 2), ('d', '1'): F(1, 2)},
        ('b', 'act'): {('e', '0'): F(1, 2), ('f', '1'): F(1, 2)},
    }
    for v, o in [('c', '0'), ('d', '1'), ('e', '1'), ('f', '0')]:
        rows[v, 'act'] = {(v, o): 1}
    m = FiniteModel('traces', tuple(Candidate(v) for v in 'abcdef'), ('act',), ('0', '1'), rows)
    one = Probe(('act',), frozenset({('0',)}))
    two = Probe(('act', 'act'), frozenset({('0', '0'), ('1', '1')}))
    assert m.geometry([one]).contains(('a', 'b'))
    assert not m.geometry([one, two]).contains(('a', 'b'))
    assert not m.is_exact_partition([('a', 'b'), ('c',), ('d',), ('e',), ('f',)])


def test_exact_partition_positive_case_and_missing_observation():
    rows = {(v, 'act'): {(v, 'missing'): 1} for v in 'ab'}
    m = FiniteModel('missing', tuple(Candidate(v) for v in 'ab'), ('act',), ('missing',), rows)
    assert m.is_exact_partition([('a', 'b')])
    f = PossibilityFilter(m, {'a': F(1, 3), 'b': F(2, 3)})
    f.predict('act')
    assert f.assimilate('act', 'missing').posterior == {'a': F(1, 3), 'b': F(2, 3)}


@pytest.mark.parametrize('p', [-1, F(3, 2), F(1, 2), float('nan'), float('inf')])
def test_invalid_kernel_rejected(p):
    with pytest.raises((ValueError, OverflowError)):
        FiniteModel('bad', (Candidate('a'),), ('act',), ('0',), {('a', 'act'): {('a', '0'): p}})


def test_invalid_domains_and_empty_queries_rejected():
    m = cup()
    with pytest.raises(ValueError):
        PossibilityFilter(m, {'A': 1})
    with pytest.raises(ValueError):
        m.geometry([])
    with pytest.raises(ValueError):
        m.is_exact_partition([('A', 'B'), ('B', 'C')])
    with pytest.raises(ValueError):
        m.trace_distribution(dict.fromkeys('ABC', F(1, 3)), ['unknown'])
    with pytest.raises(ValueError):
        Probe(('drop',), frozenset({('intact', 'hard')}))
