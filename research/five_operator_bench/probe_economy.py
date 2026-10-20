"""Evaluate empirical probe economy without exposing hidden state to the learner.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.probe_economy
"""
from __future__ import annotations

import json
from fractions import Fraction as F

from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy
from research.five_operator_bench.learn_delta import train_action_blindness
from research.five_operator_bench.learn_delta_noise import train_controlled
from research.sl_bench.oracle import audit_partition, exact_quotient
from research.sl_bench.trace_worlds import mixture_profiles


def _canonical(partition):
    return tuple(sorted(tuple(sorted(cell)) for cell in partition))


def _evaluate_action_blindness():
    world, learner = train_action_blindness(horizon=2, samples_per_probe=512, seed=17)
    contexts = tuple(range(world.n))
    economy = ProbeEconomy(learner, contexts)
    population = HystereticProbePopulation(economy)
    active = population.step()

    full = _canonical(learner.propose_partition(contexts))
    reduced = _canonical(economy.propose_partition(active))
    profiles = world.profiles(2)
    exact = exact_quotient(profiles)
    audit = audit_partition(profiles, reduced, epsilon=F(1, 4))
    minimum = economy.exact_minimal_cover()

    return {
        "case": "action_blindness_h2",
        "full_partition": full,
        "reduced_partition": reduced,
        "exact_partition": exact,
        "total_probes": len(economy.probes),
        "active_probes": len(active),
        "minimum_probes": len(minimum),
        "active_probe_indices": population.snapshot["active_probe_indices"],
        "coverage": population.snapshot["coverage"],
        "oracle_audit": audit,
        "passed": (
            full == exact
            and reduced == exact
            and audit["passed"]
            and audit["excess_cells"] == 0
            and population.snapshot["coverage"] == 1.0
            and len(active) == len(minimum)
            and len(active) < len(economy.probes)
        ),
    }


def _evaluate_noisy(*, delay: bool, horizon: int, seed: int):
    world, mixtures, learner = train_controlled(
        delay=delay,
        horizon=horizon,
        samples_per_probe=1024,
        seed=seed,
    )
    contexts = tuple(range(len(mixtures)))
    economy = ProbeEconomy(learner, contexts)
    population = HystereticProbePopulation(economy)
    active = population.step()

    profiles = mixture_profiles(world, mixtures, horizon=horizon)
    full = _canonical(learner.propose_partition(contexts))
    reduced = _canonical(economy.propose_partition(active))
    exact = exact_quotient(profiles)
    audit = audit_partition(profiles, reduced, epsilon=F(1, 4))
    minimum = economy.exact_minimal_cover()

    return {
        "case": f"noisy_{'delayed' if delay else 'immediate'}_h{horizon}",
        "full_partition": full,
        "reduced_partition": reduced,
        "exact_partition": exact,
        "total_probes": len(economy.probes),
        "active_probes": len(active),
        "minimum_probes": len(minimum),
        "active_probe_indices": population.snapshot["active_probe_indices"],
        "coverage": population.snapshot["coverage"],
        "oracle_audit": audit,
        "passed": (
            full == exact
            and reduced == exact
            and audit["passed"]
            and audit["excess_cells"] == 0
            and population.snapshot["coverage"] == 1.0
            and len(active) == len(minimum)
            and len(active) < len(economy.probes)
        ),
    }


def run_probe_economy_experiment():
    rows = (
        _evaluate_action_blindness(),
        _evaluate_noisy(delay=False, horizon=2, seed=32),
        _evaluate_noisy(delay=True, horizon=3, seed=43),
    )
    return {
        "version": "probe-economy-v0.3",
        "passed": all(row["passed"] for row in rows),
        "cases": list(rows),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe_economy_experiment(), indent=2, sort_keys=True, default=str))
