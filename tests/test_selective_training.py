from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from selective_training import Acquisition, Counts, PartialEvidence, validate
from selective_training_campaign import train, run_case
from question_learning_campaign import Episode, episodes


def test_missing_answer_never_becomes_negative_evidence():
    counts = Counts()
    counts.update(PartialEvidence('a', 0, (None, 0), 1))
    s = counts.snapshot()
    assert s.n(0, 'probe_0') == 0
    assert s.n(0, 'probe_1') == 1
    assert s.root[0] == (1, 1)
    assert s.cells[0, 'probe_0', 0] == (0, 0)
    assert s.cells[0, 'probe_1', 0] == (1, 1)


def test_snapshot_is_immutable_and_not_changed_by_later_learning():
    counts = Counts()
    old = counts.snapshot()
    counts.update(PartialEvidence('a', 0, (1, None), 1))
    assert old.root[0] == (0, 0)
    with pytest.raises(TypeError):
        old.root[0] = (1, 1)
    with pytest.raises(ValueError):
        counts.update(PartialEvidence('a', 0, (1, None), 1))


@pytest.mark.parametrize('budget', [0, 1, 7, 64])
def test_all_policies_have_exact_equal_acquisition_cost_and_free_outcomes(budget):
    roots = []
    for policy in ('active', 'random', 'exhaustive'):
        s, history, cost = train(episodes(3, 'discovery', 64), policy, budget, 9)
        assert cost == budget == sum(len(r.questions) for r in history)
        assert sum(v[1] for v in s.root.values()) == 64
        assert sum(v[1] for v in s.cells.values()) == budget
        roots.append(s.root)
    assert roots[0] == roots[1] == roots[2]


def test_current_answer_and_outcome_do_not_leak_into_acquisition():
    def stream(last_answer, last_outcome):
        return [Episode(str(i), i%2, (i%2, (i+1)%2), i%2) for i in range(20)] + [
            Episode('20', 0, last_answer, last_outcome)]
    _, a, _ = train(stream((0, 0), 0), 'active', 21, 7)
    _, b, _ = train(stream((1, 1), 1), 'active', 21, 7)
    assert a == b
    with pytest.raises(FrozenInstanceError):
        a[0].questions = ()


def test_exploration_can_override_an_early_preference():
    counts = Counts()
    for i in range(100):
        counts.update(PartialEvidence(str(i), 1, (i%2, None), i%2))
    policy = Acquisition(7, epsilon=1)
    choices = [policy.choose(counts.snapshot(), 1) for _ in range(40)]
    assert {q for q, _ in choices} == {'probe_0', 'probe_1'}
    assert all(reason == 'explore' for _, reason in choices)


def balanced(prefix, signal=True):
    return [PartialEvidence(f'{prefix}:{c}:{a}:{i}', c, (a, None),
                            int(i < (18 if a else 2)) if signal and c else int(i < 10))
            for c in (0, 1) for a in (0, 1) for i in range(20)]


def test_frozen_proposal_requires_fresh_acquired_validation():
    counts = Counts()
    discovery = balanced('train')
    for row in discovery:
        counts.update(row)
    s = counts.snapshot()
    assert s.propose() == 'probe_0'
    assert validate(s, 'probe_0', balanced('validate')).adopted == 'probe_0'
    assert validate(s, 'probe_0', balanced('noise', False)).adopted is None
    with pytest.raises(ValueError):
        validate(s, 'probe_0', discovery)
    with pytest.raises(ValueError):
        validate(s, 'probe_1', balanced('missing'))
    assert s.predict(1, 'probe_0', 0) < F(1, 2) < s.predict(1, 'probe_0', 1)
    assert s.predict(1, 'probe_0') == F(1, 2)


def test_full_campaign_cost_includes_validation_and_evaluation():
    for policy in ('active', 'random', 'exhaustive'):
        r = run_case(3, policy, 64, True)
        assert r['total_queries'] == 64 + 64 + 512
        assert r['total_loss'] == pytest.approx(r['error_rate'] + .1*640/512)


def test_input_validation():
    with pytest.raises(ValueError):
        PartialEvidence('a', 0, (2, None), 1)
    with pytest.raises(ValueError):
        Acquisition(0, epsilon=2)
    with pytest.raises(ValueError):
        train([], 'random', 1, 0)
