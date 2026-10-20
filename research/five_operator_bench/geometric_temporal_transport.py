"""Benchmark temporal identity transport across higher-order rearticulation.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.geometric_temporal_transport
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import random
from math import log
from statistics import mean

from cortex.geometric_temporal_transport import GeometricTemporalTransport
from cortex.simplicial_integration import SimplicialIntegrationRead


PROBES = ((0,), (1,), (2,), (3,))


def _sample(rng: random.Random, kind: str, marker_p: float) -> tuple[int, int, int, int]:
    marker = int(rng.random() < marker_p)
    if kind == "uniform":
        bits = (rng.randrange(2), rng.randrange(2), rng.randrange(2))
    else:
        first = rng.randrange(2)
        second = rng.randrange(2)
        parity = 0 if kind == "even" else 1
        third = first ^ second ^ parity
        bits = (first, second, third)
    return marker, *bits


def _empirical_read(
    rng: random.Random,
    kind: str,
    marker_p: float,
    *,
    samples: int = 512,
    filled: bool,
    drop_last: bool = False,
) -> SimplicialIntegrationRead:
    probes = PROBES[:-1] if drop_last else PROBES
    counts = Counter()
    for _ in range(samples):
        row = _sample(rng, kind, marker_p)
        if drop_last:
            row = row[:-1]
        counts[row] += 1
    distribution = {state: count / samples for state, count in counts.items()}
    constraints = [(i,) for i in range(len(probes))]
    constraints.extend(combinations(range(len(probes)), 2))
    if filled and len(probes) == 4:
        constraints.append((1, 2, 3))
    entropy = -sum(p * log(p) for p in distribution.values() if p > 0.0)
    return SimplicialIntegrationRead(
        probes=probes,
        distribution=tuple(sorted(distribution.items())),
        residual=0.0,
        entropy=entropy,
        used_constraints=tuple(tuple(row) for row in constraints),
        max_dimension=2 if filled and len(probes) == 4 else 1,
        iterations=0,
    )


def _naive_best(transport, source, target_a, target_b) -> str:
    da = transport.compare(source, target_a).full_joint_distance
    db = transport.compare(source, target_b).full_joint_distance
    return "A" if da <= db else "B"


def _run_seed(seed: int, *, drop_last: bool = False) -> dict:
    rng = random.Random(seed)
    source_a = _empirical_read(rng, "even", 0.30, filled=True)
    source_b = _empirical_read(rng, "odd", 0.70, filled=True)
    target_a = _empirical_read(rng, "uniform", 0.30, filled=False, drop_last=drop_last)
    target_b = _empirical_read(rng, "even", 0.70, filled=not drop_last, drop_last=drop_last)

    transport = GeometricTemporalTransport()
    candidates = (("A", target_a), ("B", target_b))
    aa = transport.compare(source_a, target_a)
    ab = transport.compare(source_a, target_b)
    bb = transport.compare(source_b, target_b)
    ba = transport.compare(source_b, target_a)
    kernel_a = dict(transport.kernel(source_a, candidates))
    kernel_b = dict(transport.kernel(source_b, candidates))
    geometric_a = transport.best_successor(source_a, candidates)[0]
    geometric_b = transport.best_successor(source_b, candidates)[0]
    naive_a = _naive_best(transport, source_a, target_a, target_b)
    naive_b = _naive_best(transport, source_b, target_a, target_b)

    return {
        "geometric_accuracy": mean((geometric_a == "A", geometric_b == "B")),
        "naive_accuracy": mean((naive_a == "A", naive_b == "B")),
        "mean_correct_probability": mean((kernel_a["A"], kernel_b["B"])),
        "mean_distance_margin": mean((ab.distance - aa.distance, ba.distance - bb.distance)),
        "mean_correct_distance": mean((aa.distance, bb.distance)),
        "mean_cross_distance": mean((ab.distance, ba.distance)),
        "mean_correct_full_joint": mean((aa.full_joint_distance, bb.full_joint_distance)),
        "mean_cross_full_joint": mean((ab.full_joint_distance, ba.full_joint_distance)),
        "mean_topology_change": mean((aa.topology_change, bb.topology_change)),
        "mean_overlap": mean((aa.overlap, bb.overlap)),
    }


def _aggregate(seeds, *, drop_last: bool = False) -> dict:
    rows = [_run_seed(seed, drop_last=drop_last) for seed in seeds]
    return {
        "runs": len(rows),
        "geometric_identity_accuracy": mean(row["geometric_accuracy"] for row in rows),
        "naive_full_joint_accuracy": mean(row["naive_accuracy"] for row in rows),
        "mean_correct_kernel_probability": mean(row["mean_correct_probability"] for row in rows),
        "mean_distance_margin": mean(row["mean_distance_margin"] for row in rows),
        "min_seed_distance_margin": min(row["mean_distance_margin"] for row in rows),
        "mean_correct_distance": mean(row["mean_correct_distance"] for row in rows),
        "mean_cross_distance": mean(row["mean_cross_distance"] for row in rows),
        "mean_correct_full_joint": mean(row["mean_correct_full_joint"] for row in rows),
        "mean_cross_full_joint": mean(row["mean_cross_full_joint"] for row in rows),
        "mean_topology_change": mean(row["mean_topology_change"] for row in rows),
        "mean_overlap": mean(row["mean_overlap"] for row in rows),
    }


def run_geometric_temporal_transport_experiment() -> dict:
    development = tuple(range(24))
    holdout = tuple(range(24, 48))
    development_full = _aggregate(development)
    holdout_full = _aggregate(holdout)
    development_drop = _aggregate(development, drop_last=True)
    holdout_drop = _aggregate(holdout, drop_last=True)

    passed = (
        development_full["geometric_identity_accuracy"] == 1.0
        and holdout_full["geometric_identity_accuracy"] == 1.0
        and holdout_full["naive_full_joint_accuracy"] <= 0.10
        and holdout_full["mean_correct_kernel_probability"] >= 0.65
        and holdout_full["min_seed_distance_margin"] > 0.05
        and development_drop["geometric_identity_accuracy"] == 1.0
        and holdout_drop["geometric_identity_accuracy"] == 1.0
        and holdout_drop["mean_overlap"] == 0.75
    )
    return {
        "version": "geometric-temporal-transport-v1.1",
        "development_seeds": development,
        "holdout_seeds": holdout,
        "development_full": development_full,
        "holdout_full": holdout_full,
        "development_probe_drop": development_drop,
        "holdout_probe_drop": holdout_drop,
        "passed": passed,
    }


if __name__ == "__main__":
    print(json.dumps(run_geometric_temporal_transport_experiment(), indent=2, sort_keys=True))
