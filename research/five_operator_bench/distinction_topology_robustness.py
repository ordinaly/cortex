"""Calibrate pairwise distinction-topology hysteresis under noisy regime change.

This benchmark keeps three distinction neurons alive over four opaque contexts.
The learner sees only stochastic probe outcomes. Ground-truth Bernoulli response
probabilities are used by the benchmark evaluator only to define the intended
pre/post topology; they are never passed to Probe Economy or connectivity.

The regime switch rewires the ideal graph while preserving all three vertices:

    before: (probe 0) ---- (probe 2)
    after:  (probe 1) ---- (probe 2)

The experiment therefore measures both stationary topology chatter and genuine
rewiring latency. It also exposes the expected hysteresis failure modes:
- edge_off too close to edge_on -> noisy chatter;
- edge_off too low -> stale-edge lock-in / plasticity loss.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.distinction_topology_robustness
"""
from __future__ import annotations

import json
from itertools import combinations
from math import inf
from random import Random
from statistics import mean

from cortex.distinction_connectivity import DistinctionConnectivity
from cortex.learned_distinguishability import (
    EmpiricalTraceDistinguishability,
    LearnedDistinctionPolicy,
)
from cortex.probe_economy import ProbeEconomy


CONTEXTS = ("a", "b", "c", "d")
PAIR_INDICES = tuple(combinations(range(3), 2))
CONTEXT_PAIRS = tuple(combinations(range(4), 2))

LOW = 0.25
HIGH = 0.75
EPSILON = 0.35
WINDOW = 64
STABLE_TICKS = 100
SWITCH_TICKS = 120
STABLE_IGNORE = 20
SETTLE_HOLD = 5
POST_TAIL = 40

# q0: a | bcd
# q1: ab | cd
# q2 before: abc | d
PRE_PROBABILITIES = (
    (LOW, HIGH, HIGH, HIGH),
    (LOW, LOW, HIGH, HIGH),
    (LOW, LOW, LOW, HIGH),
)

# q2 after: ac | bd. q0 and q1 stay unchanged.
POST_PROBABILITIES = (
    PRE_PROBABILITIES[0],
    PRE_PROBABILITIES[1],
    (LOW, HIGH, LOW, HIGH),
)

TARGET_PRE = frozenset({(0, 2)})
TARGET_POST = frozenset({(1, 2)})


def _learner() -> EmpiricalTraceDistinguishability:
    return EmpiricalTraceDistinguishability(
        actions=(0, 1, 2),
        symbols=(0, 1),
        horizon=1,
        max_contexts=len(CONTEXTS),
        window=WINDOW,
        policy=LearnedDistinctionPolicy(
            epsilon=EPSILON,
            min_samples=32,
            confidence=False,
        ),
    )


def _observe_round(learner, rng: Random, probabilities) -> None:
    assert learner.probes == ((0,), (1,), (2,))
    for probe_index, probe in enumerate(learner.probes):
        for context_index, context in enumerate(CONTEXTS):
            p_one = probabilities[probe_index][context_index]
            outcome = 1 if rng.random() < p_one else 0
            learner.observe(context, probe, (outcome,))


def _score_snapshot(connectivity: DistinctionConnectivity) -> tuple[float, ...]:
    probes = connectivity.economy.probes
    return tuple(
        connectivity.complementarity(probes[left], probes[right])
        for left, right in PAIR_INDICES
    )


def build_score_trace(seed: int) -> tuple[tuple[float, ...], ...]:
    """Generate one learner-visible complementarity trace."""
    rng = Random(seed)
    learner = _learner()

    # Fill the complete sliding window before measuring topology so lack of
    # support cannot masquerade as a structural transition.
    for _ in range(WINDOW):
        _observe_round(learner, rng, PRE_PROBABILITIES)

    economy = ProbeEconomy(learner, CONTEXTS)
    connectivity = DistinctionConnectivity(economy)
    trace: list[tuple[float, ...]] = []

    for _ in range(STABLE_TICKS):
        _observe_round(learner, rng, PRE_PROBABILITIES)
        trace.append(_score_snapshot(connectivity))

    for _ in range(SWITCH_TICKS):
        _observe_round(learner, rng, POST_PROBABILITIES)
        trace.append(_score_snapshot(connectivity))

    return tuple(trace)


def _true_veto_edges(probabilities, probe_index: int) -> frozenset[tuple[int, int]]:
    row = probabilities[probe_index]
    return frozenset(
        edge
        for edge in CONTEXT_PAIRS
        if abs(row[edge[0]] - row[edge[1]]) > EPSILON
    )


def ideal_complementarity(probabilities) -> dict[tuple[int, int], float]:
    """Evaluator-only complementarity from latent Bernoulli response laws."""
    veto = {
        probe: _true_veto_edges(probabilities, probe)
        for probe in range(len(probabilities))
    }
    required = frozenset().union(*veto.values())
    if not required:
        return {pair: 0.0 for pair in PAIR_INDICES}
    return {
        (left, right): min(
            len(veto[left] - veto[right]),
            len(veto[right] - veto[left]),
        )
        / len(required)
        for left, right in PAIR_INDICES
    }


def _run_policy(
    trace: tuple[tuple[float, ...], ...],
    *,
    edge_on: float,
    edge_off: float,
) -> dict:
    edges: set[tuple[int, int]] = set()
    history: list[frozenset[tuple[int, int]]] = []
    stable_errors = 0
    stable_evaluated = 0
    stable_flips = 0
    total_flips = 0

    for tick, scores in enumerate(trace):
        before = set(edges)
        for edge, score in zip(PAIR_INDICES, scores):
            if edge in edges:
                if score < edge_off:
                    edges.remove(edge)
            elif score >= edge_on:
                edges.add(edge)

        flips = len(before.symmetric_difference(edges))
        total_flips += flips
        if STABLE_IGNORE <= tick < STABLE_TICKS:
            stable_evaluated += 1
            stable_errors += edges != TARGET_PRE
            stable_flips += flips
        history.append(frozenset(edges))

    post = history[STABLE_TICKS:]
    rewire_delay = None
    for offset in range(0, len(post) - SETTLE_HOLD + 1):
        if all(post[offset + j] == TARGET_POST for j in range(SETTLE_HOLD)):
            rewire_delay = offset + 1
            break

    tail = post[-POST_TAIL:]
    return {
        "stable_error_rate": stable_errors / max(1, stable_evaluated),
        "stable_flips": stable_flips,
        "rewire_delay": rewire_delay,
        "post_tail_error_rate": sum(state != TARGET_POST for state in tail) / len(tail),
        "total_flips": total_flips,
    }


def _summarize(
    traces: tuple[tuple[tuple[float, ...], ...], ...],
    *,
    edge_on: float,
    edge_off: float,
) -> dict:
    rows = [
        _run_policy(trace, edge_on=edge_on, edge_off=edge_off)
        for trace in traces
    ]
    delays = [row["rewire_delay"] for row in rows if row["rewire_delay"] is not None]
    return {
        "edge_on": edge_on,
        "edge_off": edge_off,
        "runs": len(rows),
        "stable_error_rate": mean(row["stable_error_rate"] for row in rows),
        "mean_stable_flips": mean(row["stable_flips"] for row in rows),
        "rewire_rate": len(delays) / len(rows),
        "mean_rewire_delay": mean(delays) if delays else None,
        "post_tail_error_rate": mean(row["post_tail_error_rate"] for row in rows),
        "mean_total_flips": mean(row["total_flips"] for row in rows),
    }


def run_distinction_topology_robustness(
    *,
    development_seeds=range(12),
    holdout_seeds=range(12, 24),
) -> dict:
    development_seeds = tuple(development_seeds)
    holdout_seeds = tuple(holdout_seeds)
    development = tuple(build_score_trace(seed) for seed in development_seeds)
    holdout = tuple(build_score_trace(seed) for seed in holdout_seeds)

    candidates = []
    for edge_on in (0.20, 0.25, 0.30, 0.35):
        for edge_off in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            if edge_off > edge_on:
                continue
            candidates.append(
                _summarize(development, edge_on=edge_on, edge_off=edge_off)
            )

    safe = [
        row
        for row in candidates
        if row["stable_error_rate"] <= 0.10
        and row["mean_stable_flips"] <= 3.0
        and row["rewire_rate"] >= 0.95
        and row["mean_rewire_delay"] is not None
        and row["mean_rewire_delay"] <= WINDOW
        and row["post_tail_error_rate"] <= 0.15
    ]
    # Safety-first: stationary correctness, post-switch correctness, and churn
    # dominate latency. On exact ties, prefer the stricter activation threshold.
    safe.sort(
        key=lambda row: (
            row["stable_error_rate"],
            row["post_tail_error_rate"],
            row["mean_stable_flips"],
            row["mean_rewire_delay"] if row["mean_rewire_delay"] is not None else inf,
            -row["edge_on"],
            -row["edge_off"],
        )
    )
    recommended = safe[0] if safe else None

    holdout_result = None
    no_hysteresis = None
    sticky = None
    if recommended is not None:
        edge_on = recommended["edge_on"]
        edge_off = recommended["edge_off"]
        holdout_result = _summarize(holdout, edge_on=edge_on, edge_off=edge_off)
        no_hysteresis = _summarize(holdout, edge_on=edge_on, edge_off=edge_on)
        sticky = _summarize(holdout, edge_on=edge_on, edge_off=0.10)

    ideal_pre = ideal_complementarity(PRE_PROBABILITIES)
    ideal_post = ideal_complementarity(POST_PROBABILITIES)
    ideal_ok = (
        abs(ideal_pre[(0, 1)] - 1 / 6) < 1.0e-12
        and abs(ideal_pre[(0, 2)] - 1 / 3) < 1.0e-12
        and abs(ideal_pre[(1, 2)] - 1 / 6) < 1.0e-12
        and abs(ideal_post[(0, 1)] - 1 / 6) < 1.0e-12
        and abs(ideal_post[(0, 2)] - 1 / 6) < 1.0e-12
        and abs(ideal_post[(1, 2)] - 1 / 3) < 1.0e-12
    )

    passed = bool(
        ideal_ok
        and recommended is not None
        and recommended["edge_off"] < recommended["edge_on"]
        and holdout_result is not None
        and holdout_result["stable_error_rate"] <= 0.12
        and holdout_result["mean_stable_flips"] <= 3.0
        and holdout_result["rewire_rate"] >= 0.95
        and holdout_result["mean_rewire_delay"] is not None
        and holdout_result["mean_rewire_delay"] <= WINDOW
        and holdout_result["post_tail_error_rate"] <= 0.15
        and no_hysteresis is not None
        and holdout_result["post_tail_error_rate"] < no_hysteresis["post_tail_error_rate"]
        and sticky is not None
        and sticky["rewire_rate"] < holdout_result["rewire_rate"]
    )

    def _ideal_rows(values):
        return [
            {"edge": list(edge), "score": values[edge]}
            for edge in PAIR_INDICES
        ]

    return {
        "version": "distinction-topology-robustness-v0.8",
        "development_seeds": list(development_seeds),
        "holdout_seeds": list(holdout_seeds),
        "window": WINDOW,
        "stable_ticks": STABLE_TICKS,
        "switch_ticks": SWITCH_TICKS,
        "candidate_policies": len(candidates),
        "safe_development_policies": len(safe),
        "ideal_pre_scores": _ideal_rows(ideal_pre),
        "ideal_post_scores": _ideal_rows(ideal_post),
        "recommended_development": recommended,
        "recommended_holdout": holdout_result,
        "no_hysteresis_holdout": no_hysteresis,
        "sticky_holdout": sticky,
        "passed": passed,
    }


if __name__ == "__main__":
    print(json.dumps(run_distinction_topology_robustness(), indent=2, sort_keys=True))
