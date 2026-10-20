import pytest

from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy
from cortex.probe_sampling import ProbeSamplingPolicy, ProbeSamplingScheduler


def _learner():
    return EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=1,
        max_contexts=4,
        window=16,
        policy=LearnedDistinctionPolicy(
            epsilon=0.25,
            min_samples=4,
            confidence=False,
        ),
    )


def _observe(learner, context, probe, outcome, count=16):
    for _ in range(count):
        learner.observe(context, probe, (outcome,))


def _phase_one(learner):
    _observe(learner, "a", (0,), 0)
    _observe(learner, "b", (0,), 1)
    _observe(learner, "a", (1,), 0)
    _observe(learner, "b", (1,), 0)


def _phase_two(learner):
    _observe(learner, "a", (0,), 0)
    _observe(learner, "b", (0,), 0)
    _observe(learner, "a", (1,), 0)
    _observe(learner, "b", (1,), 1)


def _population():
    learner = _learner()
    _phase_one(learner)
    economy = ProbeEconomy(learner, ("a", "b"))
    population = HystereticProbePopulation(economy)
    assert population.step() == ((0,),)
    return learner, economy, population


def test_active_probe_is_continuous_and_latent_probe_has_bounded_revisit_gap():
    _, economy, population = _population()
    scheduler = ProbeSamplingScheduler(
        economy,
        population,
        policy=ProbeSamplingPolicy(budget_per_tick=2, latent_max_gap=4),
    )

    latent = (1,)
    for _ in range(24):
        selected = scheduler.step()
        assert (0,) in selected
        assert len(selected) <= 2

    assert scheduler.max_observed_gap(latent) <= 4
    assert scheduler.snapshot["relative_query_saving"] >= 0.30
    assert scheduler.snapshot["active_queries"] == 24


def test_scheduler_tracks_a_changed_active_basis_without_resetting():
    learner, economy, population = _population()
    scheduler = ProbeSamplingScheduler(
        economy,
        population,
        policy=ProbeSamplingPolicy(budget_per_tick=2, latent_max_gap=4),
    )
    for _ in range(5):
        scheduler.step()

    _phase_two(learner)
    assert population.step() == ((1,),)

    selected = scheduler.step()
    assert (1,) in selected
    assert len(selected) <= 2
    assert scheduler.snapshot["active"] == 1
    assert scheduler.snapshot["latent"] == 1


def test_scheduler_rejects_a_budget_that_would_starve_latent_questions():
    _, economy, population = _population()
    with pytest.raises(ValueError, match="latent"):
        ProbeSamplingScheduler(
            economy,
            population,
            policy=ProbeSamplingPolicy(budget_per_tick=1, latent_max_gap=4),
        )
