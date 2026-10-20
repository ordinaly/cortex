"""Characterize learned predictive distinguishability under stochastic emissions.

The learner sees only opaque context ids, declared probes, and sampled output
words. Hidden mixtures and exact profiles are evaluator-only.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.learn_delta_noise
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
from research.sl_bench.trace_worlds import controlled_world, mixture_profiles, trial


def _canonical(partition):
    return tuple(sorted(tuple(sorted(cell)) for cell in partition))


def train_controlled(
    *,
    delay: bool,
    horizon: int,
    samples_per_probe: int = 1024,
    seed: int = 31,
):
    world, mixtures = controlled_world(delay=delay, noisy=True)
    learner = EmpiricalTraceDistinguishability(
        actions=(0, 1),
        symbols=(0, 1),
        horizon=horizon,
        max_contexts=8,
        window=max(1024, samples_per_probe),
        policy=LearnedDistinctionPolicy(
            epsilon=0.25,
            min_samples=128,
            delta=0.05,
            confidence=True,
        ),
    )
    rng = random.Random(seed)
    for context, mixture in enumerate(mixtures):
        for probe in learner.probes:
            for _ in range(samples_per_probe):
                learner.observe(context, probe, trial(rng, world, mixture, probe))
    return world, mixtures, learner


def evaluate_controlled(*, delay: bool, horizon: int, seed: int = 31):
    world, mixtures, learner = train_controlled(
        delay=delay,
        horizon=horizon,
        seed=seed,
    )
    profiles = mixture_profiles(world, mixtures, horizon=horizon)
    exact = exact_quotient(profiles)
    proposed = _canonical(learner.propose_partition(range(len(mixtures))))
    audit = audit_partition(profiles, proposed, epsilon=F(1, 4))
    pairwise = {
        f"{left}-{right}": learner.population_distance(left, right)
        for left in range(len(mixtures))
        for right in range(left + 1, len(mixtures))
    }
    return {
        "delay": delay,
        "horizon": horizon,
        "seed": seed,
        "proposed": proposed,
        "exact": exact,
        "oracle_audit": audit,
        "pairwise_population_distance": pairwise,
        "snapshot": learner.snapshot,
        "passed": proposed == exact and audit["passed"] and audit["excess_cells"] == 0,
    }


def run_noise_experiment():
    immediate_h1 = evaluate_controlled(delay=False, horizon=1, seed=31)
    immediate_h2 = evaluate_controlled(delay=False, horizon=2, seed=32)
    delayed_h1 = evaluate_controlled(delay=True, horizon=1, seed=41)
    delayed_h2 = evaluate_controlled(delay=True, horizon=2, seed=42)
    delayed_h3 = evaluate_controlled(delay=True, horizon=3, seed=43)

    expected = (
        immediate_h1["proposed"] == ((0, 1, 2, 3),)
        and immediate_h2["proposed"] == ((0, 2), (1, 3))
        and delayed_h1["proposed"] == ((0, 1, 2, 3),)
        and delayed_h2["proposed"] == ((0, 1, 2, 3),)
        and delayed_h3["proposed"] == ((0,), (1,), (2,), (3,))
    )
    rows = (immediate_h1, immediate_h2, delayed_h1, delayed_h2, delayed_h3)
    return {
        "version": "learned-distinguishability-noise-v0.2",
        "passed": expected and all(row["passed"] for row in rows),
        "expected_refinement_ladder": expected,
        "cases": list(rows),
    }


if __name__ == "__main__":
    print(json.dumps(run_noise_experiment(), indent=2, sort_keys=True, default=str))
