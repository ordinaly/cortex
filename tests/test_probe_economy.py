from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from cortex.probe_economy import (
    HystereticProbePopulation,
    ProbeEconomy,
    ProbeEconomyPolicy,
)


def _two_probe_learner():
    return EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=1,
        max_contexts=4,
        window=32,
        policy=LearnedDistinctionPolicy(
            epsilon=0.25,
            min_samples=8,
            confidence=False,
        ),
    )


def _observe_binary(learner, context, probe, outcome, count=32):
    for _ in range(count):
        learner.observe(context, probe, (outcome,))


def _phase_one(learner):
    # Probe (0,) is the only useful separator; probe (1,) is redundant.
    _observe_binary(learner, "a", (0,), 0)
    _observe_binary(learner, "b", (0,), 1)
    _observe_binary(learner, "a", (1,), 0)
    _observe_binary(learner, "b", (1,), 0)


def _phase_two(learner):
    # Flush the finite windows so the useful distinction migrates to probe (1,).
    _observe_binary(learner, "a", (0,), 0)
    _observe_binary(learner, "b", (0,), 0)
    _observe_binary(learner, "a", (1,), 0)
    _observe_binary(learner, "b", (1,), 1)


def test_exact_probe_cover_prunes_redundant_distinguishers():
    learner = _two_probe_learner()
    _phase_one(learner)
    economy = ProbeEconomy(learner, ("a", "b"))

    assert economy.required_edges == frozenset({(0, 1)})
    assert economy.exact_minimal_cover() == ((0,),)
    assert economy.coverage(((0,),)) == 1.0
    assert economy.propose_partition(((0,),)) == (("a",), ("b",))
    assert economy.marginal_utility((1,), ((0,), (1,))) < 0.0


def test_hysteretic_population_reactivates_a_latent_probe_after_regime_change():
    learner = _two_probe_learner()
    _phase_one(learner)
    economy = ProbeEconomy(
        learner,
        ("a", "b"),
        policy=ProbeEconomyPolicy(
            probe_cost=0.01,
            activation_threshold=0.05,
            retirement_threshold=0.0,
        ),
    )
    population = HystereticProbePopulation(economy)

    assert population.step() == ((0,),)
    assert population.snapshot["coverage"] == 1.0
    assert population.snapshot["latent"] == 1

    _phase_two(learner)

    assert economy.exact_minimal_cover(prefer=population.active) == ((1,),)
    assert population.step() == ((1,),)
    assert population.snapshot["coverage"] == 1.0
    assert population.snapshot["activations"] >= 1
    assert population.snapshot["retirements"] >= 2


def test_empty_distinction_relation_allows_all_probes_to_go_latent():
    learner = _two_probe_learner()
    for probe in learner.probes:
        _observe_binary(learner, "a", probe, 0)
        _observe_binary(learner, "b", probe, 0)
    economy = ProbeEconomy(learner, ("a", "b"))
    population = HystereticProbePopulation(economy)

    assert economy.required_edges == frozenset()
    assert economy.exact_minimal_cover() == ()
    assert population.step() == ()
    assert population.snapshot["coverage"] == 1.0
