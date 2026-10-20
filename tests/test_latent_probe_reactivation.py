from cortex.latent_probe_reactivation import (
    LatentProbeReactivationMonitor,
    LatentReactivationPolicy,
)
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


def _seed_phase_one(learner):
    _observe(learner, "a", (0,), 0)
    _observe(learner, "b", (0,), 1)
    _observe(learner, "a", (1,), 0)
    _observe(learner, "b", (1,), 0)


def _outcome(phase, context, probe):
    if phase == 1:
        return {
            (0,): {"a": 0, "b": 1},
            (1,): {"a": 0, "b": 0},
        }[probe][context]
    return {
        (0,): {"a": 0, "b": 0},
        (1,): {"a": 0, "b": 1},
    }[probe][context]


def _fixture():
    learner = _learner()
    _seed_phase_one(learner)
    economy = ProbeEconomy(learner, ("a", "b"))
    population = HystereticProbePopulation(economy)
    assert population.step() == ((0,),)
    scheduler = ProbeSamplingScheduler(
        economy,
        population,
        policy=ProbeSamplingPolicy(budget_per_tick=2, latent_max_gap=4),
    )
    monitor = LatentProbeReactivationMonitor(
        learner,
        economy,
        population,
        policy=LatentReactivationPolicy(
            sentinel_window=3,
            min_sentinels=3,
            trigger_threshold=0.50,
            escalation_samples=16,
            max_priority_probes=1,
        ),
    )
    return learner, economy, population, scheduler, monitor


def _tick(learner, population, scheduler, monitor, phase):
    priority = monitor.priority_probes()
    selected = scheduler.step(priority_latent=priority)
    newly_triggered = False
    for probe in selected:
        for context in ("a", "b"):
            outcome = _outcome(phase, context, probe)
            newly_triggered |= monitor.observe(probe, context, (outcome,))
            learner.observe(context, probe, (outcome,))
    active = population.step()
    monitor.after_tick(selected)
    return selected, active, newly_triggered


def test_stable_sentinels_do_not_trigger_false_reactivation():
    learner, _, population, scheduler, monitor = _fixture()
    for _ in range(20):
        _tick(learner, population, scheduler, monitor, phase=1)
    assert monitor.snapshot["triggers"] == 0
    assert tuple(sorted(population.active)) == ((0,),)


def test_latent_drift_escalates_sampling_and_earns_reactivation():
    learner, _, population, scheduler, monitor = _fixture()
    for _ in range(12):
        _tick(learner, population, scheduler, monitor, phase=1)
    assert monitor.snapshot["triggers"] == 0

    trigger_tick = None
    reactivation_tick = None
    for tick in range(40):
        _, active, triggered = _tick(
            learner, population, scheduler, monitor, phase=2
        )
        if triggered and trigger_tick is None:
            trigger_tick = tick
        if (1,) in active and reactivation_tick is None:
            reactivation_tick = tick
            break

    assert trigger_tick is not None
    assert trigger_tick <= 12
    assert reactivation_tick is not None
    assert reactivation_tick <= trigger_tick + 20
    assert tuple(sorted(population.active)) == ((1,),)
    assert monitor.snapshot["reactivations"] == 1
    assert scheduler.snapshot["priority_queries"] > 0


def test_priority_escalation_never_preempts_active_probe_or_exceeds_budget():
    learner, _, population, scheduler, monitor = _fixture()
    for _ in range(12):
        _tick(learner, population, scheduler, monitor, phase=1)

    saw_priority = False
    for _ in range(20):
        selected, active, _ = _tick(
            learner, population, scheduler, monitor, phase=2
        )
        assert len(selected) <= 2
        if scheduler.snapshot["priority_queries"]:
            saw_priority = True
        if (1,) in active:
            break
    assert saw_priority
