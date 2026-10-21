from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from question_learning import Evidence, QUESTIONS, fit
from question_learning_campaign import Episode, collect, episodes, evaluate


def balanced(prefix, *, signal=True, swapped=False):
    rows = []
    for c in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                ones = 2 if c == 0 else (18 if (b if swapped else a) else 2) if signal else 10
                for i in range(20):
                    rows.append(Evidence(f'{prefix}:{c}:{a}:{b}:{i}', c, (a, b), int(i < ones)))
    return rows


@pytest.mark.parametrize('swapped', [False, True])
def test_fresh_evidence_adopts_relevant_distinction_and_selective_question(swapped):
    model = fit(balanced('train', swapped=swapped), balanced('validate', swapped=swapped))
    assert model.decision.adopted == QUESTIONS[int(swapped)]
    assert model.choose_question(0, F(1, 10)) is None
    assert model.choose_question(1, F(1, 10)) == QUESTIONS[int(swapped)]
    assert model.choose_question(1, 1) is None
    p0 = model.probability(1, model.decision.adopted, 0)
    p1 = model.probability(1, model.decision.adopted, 1)
    assert 0 < p0 < F(1, 2) < p1 < 1
    assert model.probability(1) == F(1, 2)


def test_unlearnable_variation_does_not_create_a_distinction():
    model = fit(balanced('train', signal=False), balanced('validate', signal=False))
    assert model.decision.adopted is None
    assert model.choose_question(1, F(1, 10)) is None
    assert model.probability(1) == F(1, 2)


def test_proposal_rejected_when_fresh_validation_does_not_support_it():
    model = fit(balanced('train'), balanced('validate', signal=False))
    assert model.decision.discovery_gain > model.decision.margin
    assert model.decision.adopted is None
    assert model.decision.validation_gain < 0


def test_reused_evidence_and_insufficient_cells_rejected():
    rows = balanced('same')
    with pytest.raises(ValueError):
        fit(rows, rows)
    assert fit(balanced('train'), balanced('validate'), min_cell=1000).decision.status == 'insufficient_support'


def test_oracle_charges_observations_and_refuses_hidden_or_repeated_queries():
    e = Episode('one', 0, (0, 1), 1)
    with pytest.raises(ValueError):
        e.ask('correct_hypothesis')
    assert e.query_count == 0
    assert e.ask('probe_0') == 0
    with pytest.raises(ValueError):
        e.ask('probe_0')
    assert e.query_count == 1
    assert e.reveal() == 1
    with pytest.raises(ValueError):
        e.ask('probe_1')
    with pytest.raises(ValueError):
        e.reveal()


def test_reference_preserves_alternatives_after_partial_answer():
    from possibility_geometry import PossibilityFilter
    model = fit(balanced('train'), balanced('validate'))
    finite, prior = model.finite_model(1)
    learner = PossibilityFilter(finite, prior)
    learner.predict('probe_0')
    posterior = learner.assimilate('probe_0', '1').posterior
    assert posterior == {'00': 0, '01': 0, '10': F(1, 2), '11': F(1, 2)}
    assert learner.predict('outcome').outcomes['1'] == model.probability(1, 'probe_0', 1)


def test_evaluation_is_frozen_and_random_has_identical_query_budget():
    model = fit(balanced('train'), balanced('validate'))
    counts = dict(model.root_counts)
    active, forecasts = evaluate(model, episodes(2, 'evaluation', 200), 'active', F(1, 10))
    random, _ = evaluate(model, episodes(2, 'evaluation', 200), 'random_matched', F(1, 10),
                          random_quota=active['queries'], seed=8)
    assert active['queries'] == random['queries']
    assert active['cost_per_episode'] == random['cost_per_episode']
    assert dict(model.root_counts) == counts
    with pytest.raises(FrozenInstanceError):
        forecasts[0].probability = F(1)
    # Evaluation outcomes are not inputs to the question policy or forecast.
    changed = [Episode(f.episode_id, f.context, (0, 1), 1) for f in forecasts]
    flipped = [Episode(f.episode_id, f.context, (0, 1), 0) for f in forecasts]
    _, a = evaluate(model, changed, 'active', F(1, 10))
    _, b = evaluate(model, flipped, 'active', F(1, 10))
    assert a == b


def test_collection_costs_and_unseen_context():
    records, queries = collect(episodes(7, 'discovery', 20))
    assert queries == 40 and len(records) == 20
    model = fit(balanced('train'), balanced('validate'))
    with pytest.raises(ValueError):
        model.probability(2)
    with pytest.raises(ValueError):
        model.choose_question(0, -1)


def test_evaluation_rejects_discovery_and_validation_reuse():
    discovery, validation = balanced('train'), balanced('validate')
    model = fit(discovery, validation)
    for row in (discovery[0], validation[0]):
        with pytest.raises(ValueError):
            evaluate(model, [Episode(row.episode_id, row.context, row.answers, row.outcome)],
                     'active', F(1, 10))
