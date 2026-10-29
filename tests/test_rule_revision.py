from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from pattern_discovery import Episode
from rule_revision import RevisingLanguage
from rule_revision_campaign import initial, sample, stream, T, U, run_one


def test_counterexample_quarantines_only_responsible_rule_with_provenance():
    base = initial()
    model = RevisingLanguage(base)
    episode = sample('failure',positive=False)
    result = model.step(episode)
    assert result.predictions == (1,1) and result.wrong == 1
    assert len(result.witnesses) == 1
    witness = result.witnesses[0]
    assert witness.observed.polarity == -1
    assert set(witness.fact_ids) <= {f.fact_id for f in episode.facts}
    assert len(witness.fact_ids) == 2
    assert {r.head.symbol for r in model.active.values()} == {U}
    assert model.step(sample('next',positive=False)).predictions == (0,1)


def test_scope_narrowing_is_validated_before_use_and_preserves_stable_rule():
    model = RevisingLanguage(initial())
    stable_id = next(r.rule_id for r in model.active.values() if r.head.symbol == U)
    for i in range(8):
        result = model.step(sample('repair'+str(i),hot=bool(i%2),positive=bool(i%2)))
        if i < 7:
            assert not result.admitted
        if i > 0:
            assert result.predictions == (0,1)
    assert result.admitted
    assert stable_id in model.active
    assert model.step(sample('cold',hot=False,positive=False)).predictions == (0,1)
    assert model.step(sample('hot',hot=True)).predictions == (1,1)
    assert len(next(r for r in model.active.values() if r.head.symbol == T).body) == 3


def test_single_noise_event_can_revalidate_same_rule_but_has_cost():
    model = RevisingLanguage(initial())
    target_id = next(r.rule_id for r in model.active.values() if r.head.symbol == T)
    model.step(sample('noise',positive=False))
    for i in range(7):
        result = model.step(sample('fresh'+str(i),hot=bool(i%2)))
        assert result.predictions == (0,1)
    assert target_id in model.active
    assert target_id not in model.quarantined
    assert model.step(sample('after')).predictions == (1,1)


def test_unknown_labels_do_not_trigger_or_validate_revision():
    model = RevisingLanguage(initial())
    unknown = sample('unknown')
    assert model.step(Episode('unknown',unknown.facts)).predictions == ()
    assert not model.pending
    model.step(sample('failure',positive=False))
    for i in range(10):
        e = sample('unlabeled'+str(i))
        model.step(Episode(e.episode_id,e.facts))
    assert model.searches == 0
    assert len(model.pending[T].proposal) == 1


def test_freshness_and_stream_bounds():
    model = RevisingLanguage(initial(),max_steps=1)
    with pytest.raises(ValueError,match='fresh'):
        model.step(sample('train0'))
    model.step(sample('one'))
    with pytest.raises(ValueError,match='fresh'):
        model.step(sample('one'))
    with pytest.raises(RuntimeError,match='budget'):
        model.step(sample('two'))
    assert len(model.steps) == 1


def test_failed_mining_is_atomic():
    model = RevisingLanguage(initial(),block_size=2,max_examined=1)
    model.step(sample('failure',positive=False))
    before = dict(model.active)
    with pytest.raises(RuntimeError,match='budget'):
        model.step(sample('over-budget'))
    assert model.active == before
    assert 'over-budget' not in model.used_ids
    assert len(model.pending[T].proposal) == 1 and len(model.steps) == 1


def test_campaign_exposes_both_benefit_and_false_alarm_cost():
    scope = run_one('scope',0)
    assert scope['targeted']['loss'] < scope['frozen']['loss']
    assert scope['targeted']['wrong'] < scope['periodic']['wrong']
    stationary = run_one('stationary',0)
    assert stationary['targeted']['loss'] == 0
    assert stationary['targeted']['searches'] == 0
    noise = run_one('isolated_noise',0)
    assert noise['targeted']['loss'] > noise['frozen']['loss']


def test_two_counterexamples_in_window_tolerates_isolated_noise():
    model = RevisingLanguage(initial(),counterexample_limit=2)
    first = model.step(sample('bad',positive=False))
    assert first.wrong == 1 and first.witnesses and not model.pending
    for i in range(4):
        assert model.step(sample('good'+str(i))).predictions == (1,1)
    model.step(sample('new-bad',positive=False))
    assert not model.pending  # The first error has expired.
    model.step(sample('between'))
    model.step(sample('second-bad',positive=False))
    assert T in model.pending and {r.head.symbol for r in model.active.values()} == {U}


def test_all_rules_supporting_false_prediction_are_quarantined():
    from dataclasses import replace
    from compositional_language import Atom
    from rule_revision_campaign import HOT
    base = initial()
    rule = next(r for r in base.schemas if r.head.symbol == T)
    alternative = replace(rule,rule_id='alternative',body=(*rule.body,Atom(HOT,(rule.head.terms[0],))))
    model = RevisingLanguage(replace(base,schemas=(*base.schemas,alternative)))
    result = model.step(sample('both',hot=True,positive=False))
    assert {w.rule_id for w in result.witnesses} == {rule.rule_id,'alternative'}
    assert {r.head.symbol for r in model.active.values()} == {U}
