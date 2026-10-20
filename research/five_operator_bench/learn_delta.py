"""Learn predictive Delta from traces; use the finite oracle only for scoring.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.learn_delta
"""
from __future__ import annotations

import json
import random
from fractions import Fraction as F

from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from research.sl_bench.oracle import audit_partition, exact_quotient
from research.sl_bench.trace_worlds import trial
from research.sl_bench.worlds import exact_worlds


def _canonical(partition):
    return tuple(sorted(tuple(sorted(cell)) for cell in partition))


def _one_hot(size, index):
    return tuple(int(i == index) for i in range(size))


def train_action_blindness(*, horizon: int, samples_per_probe: int = 512, seed: int = 17):
    """Train only from public action/output trials for each opaque start context."""
    world = exact_worlds()["action_blindness"]
    learner = EmpiricalTraceDistinguishability(
        actions=tuple(range(len(world.kernel))),
        symbols=tuple(range(len(world.emission[0]))),
        horizon=horizon,
        max_contexts=8,
        window=1024,
        policy=LearnedDistinctionPolicy(
            epsilon=0.25,
            min_samples=64,
            delta=0.05,
            confidence=True,
        ),
    )
    rng = random.Random(seed)
    contexts = tuple(range(world.n))
    for context in contexts:
        mixture = _one_hot(world.n, context)
        for probe in learner.probes:
            for _ in range(samples_per_probe):
                learner.observe(context, probe, trial(rng, world, mixture, probe))
    return world, learner


def evaluate_horizon(horizon: int):
    world, learner = train_action_blindness(horizon=horizon)
    contexts = tuple(range(world.n))
    proposed = _canonical(learner.propose_partition(contexts))
    profiles = world.profiles(horizon)
    exact = exact_quotient(profiles)
    audit = audit_partition(profiles, proposed, epsilon=F(1, 4))
    bank = learner.build_bank()
    zero_pair = bank.compare(0, 2).aggregate
    witness_pair = learner.population_distance(0, 1)
    return {
        "horizon": horizon,
        "proposed": proposed,
        "exact": exact,
        "oracle_audit": audit,
        "bank_zero_pair_mean": zero_pair,
        "population_witness_distance": witness_pair,
        "snapshot": learner.snapshot,
        "passed": proposed == exact and audit["passed"] and audit["excess_cells"] == 0,
    }


def run_learning_experiment():
    h1 = evaluate_horizon(1)
    h2 = evaluate_horizon(2)
    # The world is deliberately aliased at one step and separable with the
    # additional temporal/action probe family at two steps.
    refined = h1["proposed"] == ((0, 1, 2), (3,)) and h2["proposed"] == ((0, 2), (1,), (3,))
    return {
        "version": "learned-distinguishability-v0.1",
        "passed": h1["passed"] and h2["passed"] and refined,
        "refinement_observed": refined,
        "horizon_1": h1,
        "horizon_2": h2,
    }


if __name__ == "__main__":
    print(json.dumps(run_learning_experiment(), indent=2, sort_keys=True, default=str))
