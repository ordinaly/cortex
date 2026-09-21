#!/usr/bin/env python3
"""Autonomy Robustness & Temporal Plasticity v1 — Phase B campaign.

Phase B attacks the two failures measured in Phase A:
- source-reliability blindness;
- stale exact-execution permission during missing observations.

The campaign compares source-weighting and freshness candidates, then validates
one selected policy while preserving the frozen full-trust behavior.
"""
from __future__ import annotations

import argparse
import json
from random import Random
import statistics

from cortex.articulation_controller import ArticulationState
from cortex.autonomy_robustness import (
    AutonomyTemporalMemory,
    TemporalObservation,
)
from cortex.autonomy_temporal_phase_b import (
    PhaseBTemporalMemory,
    TemporalRobustnessPolicy,
)


PROTOCOL = "autonomy-robustness-temporal-plasticity-v1-phase-b"
OLD = "OLD"
NEW = "NEW"
CTX = ("R0:forward", "R1:forward")
SELECTED_SOURCE_WEIGHTING = "odds"
SELECTED_FRESHNESS_GRACE = 1


def _bootstrap(memory, context=CTX, prior=32, relation=OLD):
    for _ in range(prior):
        memory.observe(
            TemporalObservation(
                context=context,
                relation=relation,
                source="childhood",
                source_reliability=1.0,
                true_relation=relation,
                bootstrap=True,
            )
        )


def _summarize_reads(reads):
    judged = [row for row in reads if row.unsafe_execute is not None]
    unsafe = sum(row.unsafe_execute is True for row in judged)
    safe = sum(row.safe_execute is True for row in judged)
    abstain = sum(not row.executable for row in judged)
    return {
        "steps": len(reads),
        "unsafe_execute_rate": unsafe / max(1, len(judged)),
        "safe_execute_rate": safe / max(1, len(judged)),
        "abstain_rate": abstain / max(1, len(judged)),
        "unsafe_execute_steps": unsafe,
    }


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.mean(values) if values else None


def _p95(rows, key):
    values = sorted(row[key] for row in rows if row[key] is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(0.95 * len(values)) - 1))
    return values[index]


def _selected_memory(
    *,
    source_weighting=SELECTED_SOURCE_WEIGHTING,
    freshness_grace=SELECTED_FRESHNESS_GRACE,
):
    return PhaseBTemporalMemory(
        TemporalRobustnessPolicy(
            source_weighting=source_weighting,
            freshness_missing_grace=freshness_grace,
        )
    )


def clean_shift(memory):
    _bootstrap(memory)
    reads = []
    start_observed = memory.observed_steps
    start_wall = memory.wall_steps
    switched_observed = None
    switched_wall = None
    for _ in range(80):
        read = memory.observe(
            TemporalObservation(
                CTX,
                NEW,
                source="world",
                source_reliability=1.0,
                true_relation=NEW,
            )
        )
        reads.append(read)
        if read.accepted_relation == NEW and read.state is ArticulationState.EXECUTE:
            switched_observed = memory.observed_steps - start_observed
            switched_wall = memory.wall_steps - start_wall
            break
    return {
        "switch_observed": switched_observed,
        "switch_wall": switched_wall,
        **_summarize_reads(reads),
    }


def missing_shift(memory, seed, missing_rate, *, truth=NEW, observed_relation=NEW):
    rng = Random(101003 + seed * 9176 + int(missing_rate * 1000))
    _bootstrap(memory)
    reads = []
    start_observed = memory.observed_steps
    start_wall = memory.wall_steps
    switched_observed = None
    switched_wall = None
    for _ in range(400):
        relation = None if rng.random() < missing_rate else observed_relation
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source="world",
                source_reliability=1.0,
                true_relation=truth,
            )
        )
        reads.append(read)
        if (
            truth == NEW
            and read.accepted_relation == NEW
            and read.state is ArticulationState.EXECUTE
        ):
            switched_observed = memory.observed_steps - start_observed
            switched_wall = memory.wall_steps - start_wall
            break
        if truth == OLD and len(reads) >= 300:
            break
    return {
        "missing_rate": missing_rate,
        "switch_observed": switched_observed,
        "switch_wall": switched_wall,
        **_summarize_reads(reads),
    }


def corrupted_shift(memory, seed, corruption_rate):
    rng = Random(202007 + seed * 10007 + int(corruption_rate * 1000))
    _bootstrap(memory)
    reads = []
    start_observed = memory.observed_steps
    switched = None
    for _ in range(300):
        relation = OLD if rng.random() < corruption_rate else NEW
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source="noisy-world",
                source_reliability=1.0,
                true_relation=NEW,
            )
        )
        reads.append(read)
        if read.accepted_relation == NEW and read.state is ArticulationState.EXECUTE:
            switched = memory.observed_steps - start_observed
            break
    return {
        "switch_observed": switched,
        **_summarize_reads(reads),
    }


def stationary_noise(memory, seed, noise_rate, steps):
    rng = Random(303011 + seed * 11003 + int(noise_rate * 1000))
    _bootstrap(memory)
    reads = []
    false_switch = False
    for _ in range(steps):
        relation = NEW if rng.random() < noise_rate else OLD
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source="stationary-noise",
                source_reliability=1.0,
                true_relation=OLD,
            )
        )
        reads.append(read)
        false_switch |= read.accepted_relation == NEW
    return {
        "false_switch": false_switch,
        **_summarize_reads(reads),
    }


def source_reliability_trap(seed, source_weighting, cycles=30):
    memory = _selected_memory(
        source_weighting=source_weighting,
        freshness_grace=SELECTED_FRESHNESS_GRACE,
    )
    _bootstrap(memory)
    reads = []
    weighted_new = 0.0
    weighted_old = 0.0
    switched_wall = None

    pattern = (
        (NEW, "trusted", 0.95),
        (OLD, "weak-a", 0.20),
        (NEW, "trusted", 0.95),
        (OLD, "weak-b", 0.20),
        (OLD, "weak-c", 0.20),
    )
    step = 0
    for _ in range(cycles):
        for relation, source, reliability in pattern:
            step += 1
            if relation == NEW:
                weighted_new += reliability
            else:
                weighted_old += reliability
            read = memory.observe(
                TemporalObservation(
                    CTX,
                    relation,
                    source=source,
                    source_reliability=reliability,
                    true_relation=NEW,
                )
            )
            reads.append(read)
            if (
                switched_wall is None
                and read.accepted_relation == NEW
                and read.state is ArticulationState.EXECUTE
            ):
                switched_wall = step

    final = memory.snapshot(CTX)
    return {
        "weighted_oracle": NEW if weighted_new > weighted_old else OLD,
        "final_accepted": final.accepted_relation,
        "final_state": final.state.value,
        "switch_wall": switched_wall,
        "history": dict(final.historical_evidence),
        **_summarize_reads(reads),
    }


def source_holdout(seed, source_weighting, cycles=24):
    """Second source pattern not used by the Phase-A diagnostic."""
    memory = _selected_memory(source_weighting=source_weighting)
    _bootstrap(memory)
    reads = []
    pattern = (
        (NEW, "trusted-a", 0.90),
        (OLD, "weak-a", 0.25),
        (OLD, "weak-b", 0.15),
        (NEW, "trusted-b", 0.90),
        (OLD, "weak-c", 0.25),
    )
    for _ in range(cycles):
        for relation, source, reliability in pattern:
            reads.append(
                memory.observe(
                    TemporalObservation(
                        CTX,
                        relation,
                        source=source,
                        source_reliability=reliability,
                        true_relation=NEW,
                    )
                )
            )
    final = memory.snapshot(CTX)
    return {
        "final_accepted": final.accepted_relation,
        "history": dict(final.historical_evidence),
        **_summarize_reads(reads),
    }


def weak_noise_control(seed, source_weighting, steps=300):
    """Trusted incumbent with frequent weak contradictory observations."""
    rng = Random(707021 + seed * 17011)
    memory = _selected_memory(source_weighting=source_weighting)
    _bootstrap(memory)
    false_switch = False
    reads = []
    for _ in range(steps):
        if rng.random() < 0.35:
            relation, source, reliability = NEW, "weak-noise", 0.20
        else:
            relation, source, reliability = OLD, "trusted", 0.95
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source=source,
                source_reliability=reliability,
                true_relation=OLD,
            )
        )
        reads.append(read)
        false_switch |= read.accepted_relation == NEW
    return {
        "false_switch": false_switch,
        **_summarize_reads(reads),
    }


def compound_locality(seed):
    memory = _selected_memory()
    shifted = ("A0", "A1")
    stable = ("B0", "B1", "B2", "B3")
    reverse = ("C0", "C1", "C2", "C3", "C4", "C5")
    for context, relation in ((shifted, OLD), (stable, "KEEP"), (reverse, OLD)):
        _bootstrap(memory, context=context, prior=32, relation=relation)

    for step in range(64):
        memory.observe(TemporalObservation(shifted, NEW, true_relation=NEW))
        memory.observe(TemporalObservation(stable, "KEEP", true_relation="KEEP"))
        relation = NEW if step < 32 else OLD
        memory.observe(
            TemporalObservation(reverse, relation, true_relation=relation)
        )

    a = memory.snapshot(shifted)
    b = memory.snapshot(stable)
    c = memory.snapshot(reverse)
    return {
        "locality_preserved": (
            a.accepted_relation == NEW
            and b.accepted_relation == "KEEP"
            and b.state is ArticulationState.EXECUTE
            and c.accepted_relation == OLD
        ),
        "active_cells": memory.active_cells,
    }


def fixed_vocabulary_memory(contexts=48, steps=6000):
    memory = _selected_memory()
    keys = [("M", str(index), "path") for index in range(contexts)]
    for key in keys:
        for _ in range(8):
            memory.observe(TemporalObservation(key, OLD, bootstrap=True))
        memory.observe(TemporalObservation(key, NEW))
        memory.observe(TemporalObservation(key, OLD))
    before = memory.active_cells
    for step in range(steps):
        key = keys[step % len(keys)]
        relation = NEW if step % 17 == 0 else OLD
        memory.observe(TemporalObservation(key, relation))
    return {
        "contexts": memory.context_count,
        "cells_before": before,
        "cells_after": memory.active_cells,
        "bounded_fixed_vocabulary": before == memory.active_cells,
    }


def campaign(seeds, stationary_steps):
    baseline_clean = [clean_shift(AutonomyTemporalMemory()) for _ in range(seeds)]
    selected_clean = [clean_shift(_selected_memory()) for _ in range(seeds)]

    baseline_missing = {
        str(rate): [
            missing_shift(AutonomyTemporalMemory(), seed, rate)
            for seed in range(seeds)
        ]
        for rate in (0.30, 0.60)
    }
    freshness_candidates = {}
    for grace in (0, 1, 2, 4):
        freshness_candidates[str(grace)] = {
            str(rate): [
                missing_shift(
                    _selected_memory(freshness_grace=grace),
                    seed,
                    rate,
                )
                for seed in range(seeds)
            ]
            for rate in (0.30, 0.60)
        }

    stable_missing = {
        str(rate): [
            missing_shift(
                _selected_memory(),
                seed,
                rate,
                truth=OLD,
                observed_relation=OLD,
            )
            for seed in range(seeds)
        ]
        for rate in (0.30, 0.60)
    }

    source_candidates = {
        mode: [source_reliability_trap(seed, mode) for seed in range(seeds)]
        for mode in ("unweighted", "linear", "odds")
    }
    source_holdout_rows = [
        source_holdout(seed, SELECTED_SOURCE_WEIGHTING)
        for seed in range(seeds)
    ]
    weak_noise = [
        weak_noise_control(seed, SELECTED_SOURCE_WEIGHTING)
        for seed in range(seeds)
    ]

    baseline_corrupted = {
        str(rate): [
            corrupted_shift(AutonomyTemporalMemory(), seed, rate)
            for seed in range(seeds)
        ]
        for rate in (0.10, 0.20)
    }
    selected_corrupted = {
        str(rate): [
            corrupted_shift(_selected_memory(), seed, rate)
            for seed in range(seeds)
        ]
        for rate in (0.10, 0.20)
    }
    baseline_stationary = {
        str(rate): [
            stationary_noise(
                AutonomyTemporalMemory(),
                seed,
                rate,
                stationary_steps,
            )
            for seed in range(seeds)
        ]
        for rate in (0.10, 0.20)
    }
    selected_stationary = {
        str(rate): [
            stationary_noise(
                _selected_memory(),
                seed,
                rate,
                stationary_steps,
            )
            for seed in range(seeds)
        ]
        for rate in (0.10, 0.20)
    }

    locality = [compound_locality(seed) for seed in range(seeds)]
    memory = fixed_vocabulary_memory()
    selected_missing = freshness_candidates[str(SELECTED_FRESHNESS_GRACE)]

    expected_trap_history = {OLD: 32 + 30 * 3, NEW: 30 * 2}
    expected_holdout_history = {OLD: 32 + 24 * 3, NEW: 24 * 2}

    gates = {
        "full_trust_clean_exactly_preserved": all(
            left["switch_observed"] == right["switch_observed"] == 9
            for left, right in zip(baseline_clean, selected_clean)
        ),
        "full_trust_corrupted_10_preserved": all(
            left["switch_observed"] == right["switch_observed"]
            for left, right in zip(
                baseline_corrupted["0.1"],
                selected_corrupted["0.1"],
            )
        ),
        "full_trust_corrupted_20_preserved": all(
            left["switch_observed"] == right["switch_observed"]
            for left, right in zip(
                baseline_corrupted["0.2"],
                selected_corrupted["0.2"],
            )
        ),
        "full_trust_stationary_10_preserved": all(
            left["false_switch"] == right["false_switch"] == False
            for left, right in zip(
                baseline_stationary["0.1"],
                selected_stationary["0.1"],
            )
        ),
        "full_trust_stationary_20_preserved": all(
            left["false_switch"] == right["false_switch"] == False
            for left, right in zip(
                baseline_stationary["0.2"],
                selected_stationary["0.2"],
            )
        ),
        "freshness_preserves_observed_latency_30": all(
            left["switch_observed"] == right["switch_observed"] == 9
            for left, right in zip(
                baseline_missing["0.3"],
                selected_missing["0.3"],
            )
        ),
        "freshness_preserves_observed_latency_60": all(
            left["switch_observed"] == right["switch_observed"] == 9
            for left, right in zip(
                baseline_missing["0.6"],
                selected_missing["0.6"],
            )
        ),
        "freshness_reduces_unsafe_30": (
            _mean(selected_missing["0.3"], "unsafe_execute_rate")
            < _mean(baseline_missing["0.3"], "unsafe_execute_rate")
        ),
        "freshness_reduces_unsafe_60": (
            _mean(selected_missing["0.6"], "unsafe_execute_rate")
            < _mean(baseline_missing["0.6"], "unsafe_execute_rate")
        ),
        "stable_missing_30_retains_coverage": (
            _mean(stable_missing["0.3"], "safe_execute_rate") >= 0.85
        ),
        "stable_missing_60_retains_coverage": (
            _mean(stable_missing["0.6"], "safe_execute_rate") >= 0.60
        ),
        "odds_source_trap_resolved": all(
            row["final_accepted"] == NEW
            for row in source_candidates["odds"]
        ),
        "source_holdout_resolved": all(
            row["final_accepted"] == NEW
            for row in source_holdout_rows
        ),
        "weak_source_noise_no_false_switch": not any(
            row["false_switch"] for row in weak_noise
        ),
        "source_history_exact": all(
            row["history"] == expected_trap_history
            for row in source_candidates["odds"]
        ),
        "source_holdout_history_exact": all(
            row["history"] == expected_holdout_history
            for row in source_holdout_rows
        ),
        "compound_locality": all(
            row["locality_preserved"] for row in locality
        ),
        "fixed_vocabulary_state_bounded": memory["bounded_fixed_vocabulary"],
    }

    summary = {
        "protocol": PROTOCOL,
        "phase": "B-source-aware-freshness",
        "seeds": seeds,
        "stationary_steps": stationary_steps,
        "selected_policy": {
            "source_weighting": SELECTED_SOURCE_WEIGHTING,
            "full_trust_reliability": 0.95,
            "freshness_missing_grace": SELECTED_FRESHNESS_GRACE,
        },
        "source_candidate_comparison": {
            mode: {
                "final_new": sum(row["final_accepted"] == NEW for row in rows),
                "final_old": sum(row["final_accepted"] == OLD for row in rows),
                "mean_switch_wall": _mean(rows, "switch_wall"),
                "mean_unsafe_execute_rate": _mean(
                    rows, "unsafe_execute_rate"
                ),
            }
            for mode, rows in source_candidates.items()
        },
        "source_holdout": {
            "final_new": sum(
                row["final_accepted"] == NEW
                for row in source_holdout_rows
            ),
            "mean_unsafe_execute_rate": _mean(
                source_holdout_rows, "unsafe_execute_rate"
            ),
        },
        "weak_source_noise": {
            "false_switches": sum(row["false_switch"] for row in weak_noise),
            "mean_abstain_rate": _mean(weak_noise, "abstain_rate"),
        },
        "freshness_candidate_comparison": {
            grace: {
                rate: {
                    "mean_switch_observed": _mean(
                        rows, "switch_observed"
                    ),
                    "mean_switch_wall": _mean(rows, "switch_wall"),
                    "mean_unsafe_execute_rate": _mean(
                        rows, "unsafe_execute_rate"
                    ),
                }
                for rate, rows in by_rate.items()
            }
            for grace, by_rate in freshness_candidates.items()
        },
        "baseline_missing": {
            rate: {
                "mean_switch_observed": _mean(rows, "switch_observed"),
                "mean_switch_wall": _mean(rows, "switch_wall"),
                "mean_unsafe_execute_rate": _mean(
                    rows, "unsafe_execute_rate"
                ),
            }
            for rate, rows in baseline_missing.items()
        },
        "selected_missing": {
            rate: {
                "mean_switch_observed": _mean(rows, "switch_observed"),
                "mean_switch_wall": _mean(rows, "switch_wall"),
                "p95_switch_wall": _p95(rows, "switch_wall"),
                "mean_unsafe_execute_rate": _mean(
                    rows, "unsafe_execute_rate"
                ),
            }
            for rate, rows in selected_missing.items()
        },
        "stable_missing": {
            rate: {
                "mean_safe_execute_rate": _mean(
                    rows, "safe_execute_rate"
                ),
                "mean_abstain_rate": _mean(rows, "abstain_rate"),
            }
            for rate, rows in stable_missing.items()
        },
        "full_trust_non_regression": {
            "clean_switch": _mean(selected_clean, "switch_observed"),
            "corrupted_10_p95": _p95(
                selected_corrupted["0.1"], "switch_observed"
            ),
            "corrupted_20_p95": _p95(
                selected_corrupted["0.2"], "switch_observed"
            ),
            "stationary_10_false_switches": sum(
                row["false_switch"]
                for row in selected_stationary["0.1"]
            ),
            "stationary_20_false_switches": sum(
                row["false_switch"]
                for row in selected_stationary["0.2"]
            ),
        },
        "compound_locality": {
            "passed": all(row["locality_preserved"] for row in locality),
            "mean_active_cells": _mean(locality, "active_cells"),
        },
        "fixed_vocabulary_memory": memory,
        "promotion_gates": gates,
        "passed": all(gates.values()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--stationary-steps", type=int, default=300)
    args = parser.parse_args()
    result = campaign(args.seeds, args.stationary_steps)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
