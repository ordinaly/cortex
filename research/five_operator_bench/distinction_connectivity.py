"""Benchmark learner-visible pairwise topology over surviving distinguishers.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.distinction_connectivity
"""
from __future__ import annotations

import json

from cortex.distinction_connectivity import (
    DistinctionConnectivity,
    HystereticDistinctionGraph,
)
from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy
from research.five_operator_bench.learn_delta_noise import train_controlled


def run_distinction_connectivity_experiment():
    # This is the same noisy delayed H=3 world where Probe Economy v0.3 proved
    # that 14 declared questions reduce to exactly two complementary active
    # probes while preserving the exact predictive quotient.
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

    connectivity = DistinctionConnectivity(economy)
    graph = HystereticDistinctionGraph(connectivity, population)
    edges = graph.step()

    if len(active) == 2:
        left, right = active
        score = connectivity.complementarity(left, right)
        left_unique = len(connectivity.unique_edges(left, right))
        right_unique = len(connectivity.unique_edges(right, left))
        joint_gain = connectivity.joint_coverage_gain(left, right)
    else:
        score = joint_gain = 0.0
        left_unique = right_unique = 0

    return {
        "version": "distinction-connectivity-v0.7",
        "declared_probes": len(economy.probes),
        "active_probes": len(active),
        "active_probe_indices": population.snapshot["active_probe_indices"],
        "required_context_edges": len(economy.required_edges),
        "graph_edges": len(edges),
        "graph_edge_indices": graph.snapshot["edge_indices"],
        "complementarity": score,
        "joint_coverage_gain": joint_gain,
        "left_unique_context_edges": left_unique,
        "right_unique_context_edges": right_unique,
        "passed": (
            len(economy.probes) == 14
            and len(active) == 2
            and economy.coverage(active) == 1.0
            and left_unique > 0
            and right_unique > 0
            and score > 0.0
            and abs(score - joint_gain) < 1.0e-12
            and len(edges) == 1
            and set(edges[0]) == set(active)
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_distinction_connectivity_experiment(), indent=2, sort_keys=True))
