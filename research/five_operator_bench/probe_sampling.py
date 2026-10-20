"""Characterize budgeted question sampling after Probe Economy v0.3.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.probe_sampling
"""
from __future__ import annotations

import json

from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy
from cortex.probe_sampling import ProbeSamplingPolicy, ProbeSamplingScheduler
from research.five_operator_bench.learn_delta_noise import train_controlled


def run_probe_sampling_experiment(ticks: int = 120):
    _, mixtures, learner = train_controlled(
        delay=True,
        horizon=3,
        samples_per_probe=1024,
        seed=43,
    )
    contexts = tuple(range(len(mixtures)))
    economy = ProbeEconomy(learner, contexts)
    population = HystereticProbePopulation(economy)
    active = population.step()

    # v0.3 establishes that this fixture has 14 declared probes and an exact
    # empirical minimum of two active probes. One additional query slot per
    # tick is enough to revisit all twelve latent probes once every 12 ticks.
    scheduler = ProbeSamplingScheduler(
        economy,
        population,
        policy=ProbeSamplingPolicy(budget_per_tick=3, latent_max_gap=12),
    )

    active_hits = {probe: 0 for probe in active}
    latent_hits = {probe: 0 for probe in scheduler._ordered_latent()}
    for _ in range(ticks):
        selected = scheduler.step()
        for probe in active_hits:
            active_hits[probe] += int(probe in selected)
        for probe in latent_hits:
            latent_hits[probe] += int(probe in selected)

    snapshot = scheduler.snapshot
    active_continuous = all(count == ticks for count in active_hits.values())
    latent_seen = all(count > 0 for count in latent_hits.values())
    latent_gap_ok = all(
        scheduler.max_observed_gap(probe) <= scheduler.policy.latent_max_gap
        for probe in latent_hits
    )

    return {
        "version": "probe-sampling-v0.4",
        "ticks": ticks,
        "declared_probes": len(economy.probes),
        "active_probes": len(active),
        "latent_probes": len(latent_hits),
        "active_probe_indices": population.snapshot["active_probe_indices"],
        "snapshot": snapshot,
        "active_continuous": active_continuous,
        "latent_seen": latent_seen,
        "latent_gap_ok": latent_gap_ok,
        "passed": (
            len(economy.probes) == 14
            and len(active) == 2
            and active_continuous
            and latent_seen
            and latent_gap_ok
            and snapshot["total_queries"] <= ticks * 3
            and snapshot["relative_query_saving"] >= 0.75
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe_sampling_experiment(), indent=2, sort_keys=True, default=str))
