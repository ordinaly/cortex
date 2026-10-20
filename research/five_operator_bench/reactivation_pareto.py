"""Noisy Pareto calibration for latent-probe reactivation attention.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.reactivation_pareto

This is a development calibration, not a population theorem. Hidden phase labels
are used only by the evaluator to score false alarms and recovery delay.
"""
from __future__ import annotations

import json
import random
import statistics
from itertools import product

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


CONTEXTS = ("a", "b")
PROBES = ((0,), (1,))


def _rate(phase, context, probe):
    if phase == 1:
        return {
            (0,): {"a": 0.10, "b": 0.90},
            (1,): {"a": 0.20, "b": 0.20},
        }[probe][context]
    return {
        (0,): {"a": 0.20, "b": 0.20},
        (1,): {"a": 0.10, "b": 0.90},
    }[probe][context]


def _draw(rng, probability):
    return int(rng.random() < probability)


def _setup(seed, gap, monitor_policy):
    rng = random.Random(seed)
    learner = EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=1,
        max_contexts=4,
        window=32,
        policy=LearnedDistinctionPolicy(
            epsilon=0.35,
            min_samples=8,
            confidence=False,
        ),
    )
    # Learner-visible warmup. Only the last 32 samples per context/probe remain.
    for probe in PROBES:
        for context in CONTEXTS:
            for _ in range(64):
                y = _draw(rng, _rate(1, context, probe))
                learner.observe(context, probe, (y,))
    economy = ProbeEconomy(learner, CONTEXTS)
    population = HystereticProbePopulation(economy)
    population.step()
    scheduler = ProbeSamplingScheduler(
        economy,
        population,
        policy=ProbeSamplingPolicy(budget_per_tick=2, latent_max_gap=gap),
    )
    monitor = LatentProbeReactivationMonitor(
        learner,
        economy,
        population,
        policy=monitor_policy,
    )
    return rng, learner, population, scheduler, monitor


def _tick(rng, learner, population, scheduler, monitor, *, phase, escalate):
    priority = monitor.priority_probes() if escalate else ()
    selected = scheduler.step(priority_latent=priority)
    triggered = False
    for probe in selected:
        for context in CONTEXTS:
            y = _draw(rng, _rate(phase, context, probe))
            triggered |= monitor.observe(probe, context, (y,))
            learner.observe(context, probe, (y,))
    active = population.step()
    monitor.after_tick(selected)
    return active, triggered


def _run(seed, *, gap, window, threshold, escalation, escalate=True):
    policy = LatentReactivationPolicy(
        sentinel_window=window,
        min_sentinels=window,
        trigger_threshold=threshold,
        escalation_samples=escalation,
        max_priority_probes=1,
    )
    rng, learner, population, scheduler, monitor = _setup(seed, gap, policy)
    setup_ok = tuple(sorted(population.active)) == ((0,),)

    false_trigger = False
    false_activation = False
    for _ in range(60):
        active, triggered = _tick(
            rng, learner, population, scheduler, monitor,
            phase=1, escalate=escalate,
        )
        false_trigger |= triggered
        false_activation |= (1,) in active

    trigger_tick = None
    reactivation_tick = None
    for tick in range(100):
        active, triggered = _tick(
            rng, learner, population, scheduler, monitor,
            phase=2, escalate=escalate,
        )
        if triggered and trigger_tick is None:
            trigger_tick = tick
        if (1,) in active:
            reactivation_tick = tick
            break

    return {
        "setup_ok": setup_ok,
        "false_trigger": false_trigger,
        "false_activation": false_activation,
        "trigger_tick": trigger_tick,
        "reactivation_tick": reactivation_tick,
        "saving": scheduler.snapshot["relative_query_saving"],
        "priority_queries": scheduler.snapshot["priority_queries"],
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _summarize(config, rows, sparse_rows):
    valid = [row for row in rows if row["setup_ok"]]
    sparse_valid = [row for row in sparse_rows if row["setup_ok"]]
    recoveries = [row["reactivation_tick"] for row in valid if row["reactivation_tick"] is not None]
    sparse_recoveries = [row["reactivation_tick"] for row in sparse_valid if row["reactivation_tick"] is not None]
    mean_delay = _mean(recoveries)
    sparse_delay = _mean(sparse_recoveries)
    speedup = (
        sparse_delay / mean_delay
        if sparse_delay is not None and mean_delay not in (None, 0)
        else None
    )
    return {
        **config,
        "valid_runs": len(valid),
        "false_trigger_rate": sum(row["false_trigger"] for row in valid) / max(1, len(valid)),
        "false_activation_rate": sum(row["false_activation"] for row in valid) / max(1, len(valid)),
        "reactivation_rate": len(recoveries) / max(1, len(valid)),
        "mean_reactivation_delay": mean_delay,
        "sparse_mean_delay": sparse_delay,
        "mean_speedup_vs_sparse": speedup,
        "mean_query_saving": _mean(row["saving"] for row in valid),
        "mean_priority_queries": _mean(row["priority_queries"] for row in valid),
    }


def _dominates(a, b):
    if a["mean_reactivation_delay"] is None or b["mean_reactivation_delay"] is None:
        return False
    no_worse = (
        a["false_trigger_rate"] <= b["false_trigger_rate"]
        and a["false_activation_rate"] <= b["false_activation_rate"]
        and a["mean_reactivation_delay"] <= b["mean_reactivation_delay"]
        and a["mean_query_saving"] >= b["mean_query_saving"]
        and a["reactivation_rate"] >= b["reactivation_rate"]
    )
    strict = (
        a["false_trigger_rate"] < b["false_trigger_rate"]
        or a["false_activation_rate"] < b["false_activation_rate"]
        or a["mean_reactivation_delay"] < b["mean_reactivation_delay"]
        or a["mean_query_saving"] > b["mean_query_saving"]
        or a["reactivation_rate"] > b["reactivation_rate"]
    )
    return no_worse and strict


def run_reactivation_pareto(seeds=tuple(range(12))):
    summaries = []
    sparse_cache = {}
    for gap, window, threshold, escalation in product(
        (3, 5),
        (3, 5, 7),
        (0.40, 0.55, 0.70),
        (8, 16),
    ):
        key = (gap, window, threshold, escalation)
        config = {
            "gap": gap,
            "window": window,
            "threshold": threshold,
            "escalation": escalation,
        }
        rows = [
            _run(
                seed,
                gap=gap,
                window=window,
                threshold=threshold,
                escalation=escalation,
                escalate=True,
            )
            for seed in seeds
        ]
        sparse_key = gap
        if sparse_key not in sparse_cache:
            # Detection parameters do not affect sparse scheduling or the main
            # learner, so one sparse baseline per sentinel gap is sufficient.
            sparse_cache[sparse_key] = [
                _run(
                    seed,
                    gap=gap,
                    window=3,
                    threshold=0.50,
                    escalation=16,
                    escalate=False,
                )
                for seed in seeds
            ]
        summaries.append(_summarize(config, rows, sparse_cache[sparse_key]))

    eligible = [
        row for row in summaries
        if row["valid_runs"] >= max(1, len(seeds) - 1)
        and row["reactivation_rate"] >= 0.90
        and row["mean_reactivation_delay"] is not None
    ]
    frontier = [
        row for row in eligible
        if not any(_dominates(other, row) for other in eligible if other is not row)
    ]
    frontier.sort(
        key=lambda row: (
            row["false_trigger_rate"],
            row["mean_reactivation_delay"],
            -row["mean_query_saving"],
        )
    )
    safe = [
        row for row in eligible
        if row["false_trigger_rate"] <= 0.10
        and row["false_activation_rate"] <= 0.10
        and row["mean_speedup_vs_sparse"] is not None
        and row["mean_speedup_vs_sparse"] > 1.0
        and row["mean_query_saving"] > 0.0
    ]
    safe.sort(
        key=lambda row: (
            -row["mean_speedup_vs_sparse"],
            row["false_trigger_rate"],
            -row["mean_query_saving"],
        )
    )
    return {
        "version": "latent-reactivation-pareto-v0.6",
        "seeds": list(seeds),
        "candidates": len(summaries),
        "eligible": len(eligible),
        "frontier": frontier,
        "safe_candidates": safe,
        "recommended": safe[0] if safe else None,
        "passed": bool(frontier) and bool(safe),
    }


if __name__ == "__main__":
    print(json.dumps(run_reactivation_pareto(), indent=2, sort_keys=True))
