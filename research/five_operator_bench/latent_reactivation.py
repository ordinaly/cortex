"""Compare sparse-only and sentinel-escalated latent-probe reactivation.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.latent_reactivation
"""
from __future__ import annotations

import json

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


def _seed(learner):
    rows = {
        (0,): {"a": 0, "b": 1},
        (1,): {"a": 0, "b": 0},
    }
    for probe, contexts in rows.items():
        for context, outcome in contexts.items():
            for _ in range(16):
                learner.observe(context, probe, (outcome,))


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


def _setup():
    learner = _learner()
    _seed(learner)
    economy = ProbeEconomy(learner, ("a", "b"))
    population = HystereticProbePopulation(economy)
    population.step()
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
    return learner, population, scheduler, monitor


def _tick(learner, population, scheduler, monitor, *, phase, escalate):
    priority = monitor.priority_probes() if escalate else ()
    selected = scheduler.step(priority_latent=priority)
    triggered = False
    for probe in selected:
        for context in ("a", "b"):
            outcome = _outcome(phase, context, probe)
            triggered |= monitor.observe(probe, context, (outcome,))
            learner.observe(context, probe, (outcome,))
    active = population.step()
    monitor.after_tick(selected)
    return selected, active, triggered


def _run_arm(*, escalate: bool, stable_ticks: int = 16, max_change_ticks: int = 100):
    learner, population, scheduler, monitor = _setup()
    false_trigger = False
    for _ in range(stable_ticks):
        _, _, triggered = _tick(
            learner,
            population,
            scheduler,
            monitor,
            phase=1,
            escalate=escalate,
        )
        false_trigger |= triggered

    trigger_tick = None
    reactivation_tick = None
    for tick in range(max_change_ticks):
        _, active, triggered = _tick(
            learner,
            population,
            scheduler,
            monitor,
            phase=2,
            escalate=escalate,
        )
        if triggered and trigger_tick is None:
            trigger_tick = tick
        if (1,) in active:
            reactivation_tick = tick
            break

    return {
        "escalation_enabled": escalate,
        "false_trigger_during_stable": false_trigger,
        "trigger_tick_after_change": trigger_tick,
        "reactivation_tick_after_change": reactivation_tick,
        "final_active": tuple(sorted(population.active)),
        "monitor": monitor.snapshot,
        "sampling": scheduler.snapshot,
    }


def run_latent_reactivation_experiment():
    sparse = _run_arm(escalate=False)
    escalated = _run_arm(escalate=True)
    sparse_lag = sparse["reactivation_tick_after_change"]
    escalated_lag = escalated["reactivation_tick_after_change"]
    speedup = (
        sparse_lag / escalated_lag
        if sparse_lag is not None and escalated_lag not in (None, 0)
        else None
    )
    return {
        "version": "latent-probe-reactivation-v0.5",
        "sparse_only": sparse,
        "sentinel_escalation": escalated,
        "reactivation_speedup": speedup,
        "passed": (
            not escalated["false_trigger_during_stable"]
            and escalated["trigger_tick_after_change"] is not None
            and escalated_lag is not None
            and sparse_lag is not None
            and escalated_lag < sparse_lag
            and speedup is not None
            and speedup >= 1.5
            and escalated["final_active"] == ((1,),)
            and escalated["monitor"]["reactivations"] == 1
            and escalated["sampling"]["priority_queries"] > 0
            and escalated["sampling"]["relative_query_saving"] > 0.0
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_latent_reactivation_experiment(), indent=2, sort_keys=True, default=str))
