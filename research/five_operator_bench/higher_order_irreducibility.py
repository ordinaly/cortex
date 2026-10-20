"""Benchmark genuine 3-way composition over a fixed learned K3 boundary.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.higher_order_irreducibility
"""
from __future__ import annotations

import json
import random
from statistics import mean

from cortex.distinction_connectivity import (
    DistinctionConnectivity,
    DistinctionConnectivityPolicy,
    HystereticDistinctionGraph,
)
from cortex.higher_order_irreducibility import (
    HigherOrderDistinctionPolicy,
    HystereticDistinctionComplex,
    JointDistinctionMemory,
)
from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from cortex.probe_economy import HystereticProbePopulation, ProbeEconomy


CONTEXTS = tuple(range(6))
BASE = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
}


def _sample_joint(rng: random.Random, context: int, *, synergistic: bool, noise: float) -> tuple[int, int, int]:
    if context in BASE:
        return BASE[context]

    if not synergistic:
        return rng.randrange(2), rng.randrange(2), rng.randrange(2)

    # Contexts 4 and 5 have opposite parity but identical one- and two-bit
    # marginals. Symmetric parity flips add noise without changing those faces.
    target_parity = context - 4
    parity = target_parity if rng.random() >= noise else 1 - target_parity
    first = rng.randrange(2)
    second = rng.randrange(2)
    third = first ^ second ^ parity
    return first, second, third


def build_case(
    seed: int,
    *,
    synergistic: bool,
    samples: int = 256,
    noise: float = 0.10,
):
    rng = random.Random(seed)
    learner = EmpiricalTraceDistinguishability(
        actions=(0, 1, 2),
        symbols=(0, 1),
        horizon=1,
        max_contexts=8,
        window=samples,
        policy=LearnedDistinctionPolicy(
            epsilon=0.20,
            min_samples=32,
            confidence=False,
        ),
    )

    rows: dict[int, list[tuple[int, int, int]]] = {context: [] for context in CONTEXTS}
    for context in CONTEXTS:
        for _ in range(samples):
            response = _sample_joint(rng, context, synergistic=synergistic, noise=noise)
            rows[context].append(response)
            for probe, bit in zip(learner.probes, response):
                learner.observe(context, probe, (bit,))

    economy = ProbeEconomy(learner, CONTEXTS)
    population = HystereticProbePopulation(economy)
    active = population.step()
    connectivity = DistinctionConnectivity(economy)
    graph = HystereticDistinctionGraph(
        connectivity,
        population,
        policy=DistinctionConnectivityPolicy(edge_on=0.10, edge_off=0.05),
    )
    graph.step()

    memory = JointDistinctionMemory(economy, window=samples, min_samples=32)
    for context, samples_for_context in rows.items():
        for response in samples_for_context:
            memory.observe(
                context,
                {probe: bit for probe, bit in zip(economy.probes, response)},
            )

    complex_ = HystereticDistinctionComplex(
        graph,
        memory,
        policy=HigherOrderDistinctionPolicy(
            face_tolerance=0.12,
            simplex_on=0.25,
            simplex_off=0.15,
        ),
    )
    triangles = complex_.step()
    triple = tuple(economy.probes)

    return {
        "learner": learner,
        "economy": economy,
        "population": population,
        "graph": graph,
        "memory": memory,
        "complex": complex_,
        "active": active,
        "triangles": triangles,
        "triple": triple,
        "score": complex_.score(triple),
        "joint_45": memory.distance(4, 5, triple),
        "boundary_45": memory.boundary_distance(4, 5, triple),
    }


def _feed_regime(case: dict, rng: random.Random, *, synergistic: bool, samples: int = 320, noise: float = 0.10) -> None:
    learner = case["learner"]
    memory = case["memory"]
    economy = case["economy"]
    for context in (4, 5):
        for _ in range(samples):
            response = _sample_joint(rng, context, synergistic=synergistic, noise=noise)
            for probe, bit in zip(learner.probes, response):
                learner.observe(context, probe, (bit,))
            memory.observe(context, {probe: bit for probe, bit in zip(economy.probes, response)})
    case["graph"].step()
    case["complex"].step()


def run_transition(seed: int = 91) -> dict:
    case = build_case(seed, synergistic=True)
    initial_edges = len(case["graph"].edges)
    initial_triangles = len(case["complex"].triangles)

    rng = random.Random(seed + 1000)
    _feed_regime(case, rng, synergistic=False)
    reduced_edges = len(case["graph"].edges)
    reduced_triangles = len(case["complex"].triangles)
    reduced_score = case["complex"].score(case["triple"])

    _feed_regime(case, rng, synergistic=True)
    restored_edges = len(case["graph"].edges)
    restored_triangles = len(case["complex"].triangles)
    restored_score = case["complex"].score(case["triple"])

    return {
        "initial_edges": initial_edges,
        "initial_triangles": initial_triangles,
        "reduced_edges": reduced_edges,
        "reduced_triangles": reduced_triangles,
        "reduced_score": reduced_score,
        "restored_edges": restored_edges,
        "restored_triangles": restored_triangles,
        "restored_score": restored_score,
        "passed": (
            initial_edges == reduced_edges == restored_edges == 3
            and initial_triangles == 1
            and reduced_triangles == 0
            and restored_triangles == 1
        ),
    }


def _aggregate(seeds, *, synergistic: bool) -> dict:
    rows = [build_case(seed, synergistic=synergistic) for seed in seeds]
    return {
        "runs": len(rows),
        "complete_boundary_rate": mean(len(row["graph"].edges) == 3 for row in rows),
        "three_probe_rate": mean(len(row["active"]) == 3 for row in rows),
        "simplex_rate": mean(len(row["triangles"]) == 1 for row in rows),
        "mean_score": mean(row["score"] for row in rows),
        "min_score": min(row["score"] for row in rows),
        "max_score": max(row["score"] for row in rows),
        "mean_joint_45": mean(row["joint_45"] for row in rows),
        "mean_boundary_45": mean(row["boundary_45"] for row in rows),
    }


def run_higher_order_irreducibility_experiment() -> dict:
    development = tuple(range(12))
    holdout = tuple(range(12, 24))
    positive_dev = _aggregate(development, synergistic=True)
    negative_dev = _aggregate(development, synergistic=False)
    positive_holdout = _aggregate(holdout, synergistic=True)
    negative_holdout = _aggregate(holdout, synergistic=False)
    transition = run_transition()

    passed = (
        positive_dev["complete_boundary_rate"] == 1.0
        and negative_dev["complete_boundary_rate"] == 1.0
        and positive_holdout["complete_boundary_rate"] == 1.0
        and negative_holdout["complete_boundary_rate"] == 1.0
        and positive_dev["three_probe_rate"] == 1.0
        and negative_dev["three_probe_rate"] == 1.0
        and positive_holdout["three_probe_rate"] == 1.0
        and negative_holdout["three_probe_rate"] == 1.0
        and positive_dev["simplex_rate"] == 1.0
        and positive_holdout["simplex_rate"] == 1.0
        and negative_dev["simplex_rate"] <= 0.05
        and negative_holdout["simplex_rate"] <= 0.05
        and positive_holdout["mean_score"] >= 0.50
        and negative_holdout["max_score"] < 0.25
        and transition["passed"]
    )

    return {
        "version": "higher-order-irreducibility-v0.9",
        "development_seeds": development,
        "holdout_seeds": holdout,
        "positive_development": positive_dev,
        "negative_development": negative_dev,
        "positive_holdout": positive_holdout,
        "negative_holdout": negative_holdout,
        "naive_clique_fill_false_positive_rate": negative_holdout["complete_boundary_rate"],
        "transition": transition,
        "passed": passed,
    }


if __name__ == "__main__":
    print(json.dumps(run_higher_order_irreducibility_experiment(), indent=2, sort_keys=True))
