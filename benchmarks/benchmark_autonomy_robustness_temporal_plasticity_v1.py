#!/usr/bin/env python3
"""Autonomy Robustness & Temporal Plasticity v1 — Phase A baseline.

This campaign intentionally evaluates the frozen Relational Regime Plasticity
v0.1 policy before adding source reliability or new temporal mechanisms.

The purpose is diagnostic:
- preserve known clean/noisy-shift behavior;
- separate event-time from wall-time under missingness;
- measure downstream unsafe exact execution;
- expose source-reliability blindness if present;
- verify locality and fixed-vocabulary bounded state across many contexts.
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


PROTOCOL = "autonomy-robustness-temporal-plasticity-v1-phase-a"
OLD = "OLD"
NEW = "NEW"
CTX = ("R0:forward", "R1:forward")


def _bootstrap(memory: AutonomyTemporalMemory, context=CTX, prior=32):
    for _ in range(prior):
        memory.observe(
            TemporalObservation(
                context=context,
                relation=OLD,
                source="childhood",
                source_reliability=1.0,
                true_relation=OLD,
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


def clean_shift(seed: int):
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)
    reads = []
    switched_observed = None
    switched_wall = None
    start_observed = memory.observed_steps
    start_wall = memory.wall_steps
    for _ in range(80):
        read = memory.observe(
            TemporalObservation(
                CTX,
                NEW,
                source="world",
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


def missing_shift(seed: int, missing_rate: float):
    rng = Random(101003 + seed * 9176 + int(missing_rate * 1000))
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)
    reads = []
    start_observed = memory.observed_steps
    start_wall = memory.wall_steps
    switched_observed = None
    switched_wall = None
    for _ in range(400):
        relation = None if rng.random() < missing_rate else NEW
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source="world",
                true_relation=NEW,
            )
        )
        reads.append(read)
        if read.accepted_relation == NEW and read.state is ArticulationState.EXECUTE:
            switched_observed = memory.observed_steps - start_observed
            switched_wall = memory.wall_steps - start_wall
            break
    return {
        "missing_rate": missing_rate,
        "switch_observed": switched_observed,
        "switch_wall": switched_wall,
        **_summarize_reads(reads),
    }


def corrupted_shift(seed: int, corruption_rate: float):
    rng = Random(202007 + seed * 10007 + int(corruption_rate * 1000))
    memory = AutonomyTemporalMemory()
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
                true_relation=NEW,
            )
        )
        reads.append(read)
        if read.accepted_relation == NEW and read.state is ArticulationState.EXECUTE:
            switched = memory.observed_steps - start_observed
            break
    return {
        "corruption_rate": corruption_rate,
        "switch_observed": switched,
        **_summarize_reads(reads),
    }


def stationary_noise(seed: int, noise_rate: float, steps: int):
    rng = Random(303011 + seed * 11003 + int(noise_rate * 1000))
    memory = AutonomyTemporalMemory()
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
                true_relation=OLD,
            )
        )
        reads.append(read)
        false_switch |= read.accepted_relation == NEW
    return {
        "noise_rate": noise_rate,
        "false_switch": false_switch,
        **_summarize_reads(reads),
    }


def gradual_shift(seed: int, ramp_steps: int = 40):
    rng = Random(404017 + seed * 12007)
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)
    reads = []
    switch_step = None
    for step in range(1, 161):
        p_new = min(1.0, step / ramp_steps)
        relation = NEW if rng.random() < p_new else OLD
        truth = NEW if p_new >= 0.5 else OLD
        read = memory.observe(
            TemporalObservation(
                CTX,
                relation,
                source="gradual-world",
                true_relation=truth,
            )
        )
        reads.append(read)
        if (
            switch_step is None
            and read.accepted_relation == NEW
            and read.state is ArticulationState.EXECUTE
        ):
            switch_step = step
    return {
        "ramp_steps": ramp_steps,
        "switch_wall": switch_step,
        **_summarize_reads(reads),
    }


def source_reliability_trap(seed: int, cycles: int = 30):
    """Trusted minority supports NEW; weak raw-count majority supports OLD.

    Phase A intentionally ignores source_reliability, so this is diagnostic and
    is NOT part of the baseline pass/fail gates.
    """
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)
    reads = []
    weighted_new = 0.0
    weighted_old = 0.0

    pattern = (
        (NEW, "trusted", 0.95),
        (OLD, "weak-a", 0.20),
        (NEW, "trusted", 0.95),
        (OLD, "weak-b", 0.20),
        (OLD, "weak-c", 0.20),
    )
    for _ in range(cycles):
        for relation, source, reliability in pattern:
            if relation == NEW:
                weighted_new += reliability
            else:
                weighted_old += reliability
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
        "weighted_new_support": weighted_new,
        "weighted_old_support": weighted_old,
        "weighted_oracle": NEW if weighted_new > weighted_old else OLD,
        "final_accepted": final.accepted_relation,
        "final_state": final.state.value,
        "source_reliability_used_by_policy": False,
        **_summarize_reads(reads),
    }


def compound_locality(seed: int):
    memory = AutonomyTemporalMemory()
    shifted = ("A0", "A1")
    stable = ("B0", "B1", "B2", "B3")
    reverse = ("C0", "C1", "C2", "C3", "C4", "C5")
    for context, relation in ((shifted, OLD), (stable, "KEEP"), (reverse, OLD)):
        _bootstrap(memory, context=context, prior=32)
        if relation != OLD:
            # Replace the bootstrap label for the stable context.
            local = AutonomyTemporalMemory()
            for _ in range(32):
                local.observe(TemporalObservation(context, relation, bootstrap=True))
            # Feed stable context into the shared bank directly from scratch.
            memory._trackers[context] = local._trackers[context]

    for step in range(64):
        memory.observe(TemporalObservation(shifted, NEW, true_relation=NEW))
        memory.observe(TemporalObservation(stable, "KEEP", true_relation="KEEP"))
        relation = NEW if step < 32 else OLD
        memory.observe(TemporalObservation(reverse, relation, true_relation=relation))

    a = memory.snapshot(shifted)
    b = memory.snapshot(stable)
    c = memory.snapshot(reverse)
    return {
        "shifted_final": a.accepted_relation,
        "stable_final": b.accepted_relation,
        "stable_state": b.state.value,
        "reverse_final": c.accepted_relation,
        "locality_preserved": (
            a.accepted_relation == NEW
            and b.accepted_relation == "KEEP"
            and b.state is ArticulationState.EXECUTE
            and c.accepted_relation == OLD
        ),
        "contexts": memory.context_count,
        "active_cells": memory.active_cells,
    }


def fixed_vocabulary_memory(seed: int, contexts: int = 48, steps: int = 6000):
    memory = AutonomyTemporalMemory()
    keys = [("M", str(index), "path") for index in range(contexts)]
    for key in keys:
        for _ in range(8):
            memory.observe(TemporalObservation(key, OLD, bootstrap=True))
        # Ensure both relation cells exist before the growth measurement.
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
        "stream_steps": steps,
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


def campaign(seeds: int, stationary_steps: int):
    clean = [clean_shift(seed) for seed in range(seeds)]
    missing = {
        str(rate): [missing_shift(seed, rate) for seed in range(seeds)]
        for rate in (0.30, 0.60)
    }
    corrupted = {
        str(rate): [corrupted_shift(seed, rate) for seed in range(seeds)]
        for rate in (0.10, 0.20)
    }
    stationary = {
        str(rate): [
            stationary_noise(seed, rate, stationary_steps)
            for seed in range(seeds)
        ]
        for rate in (0.10, 0.20)
    }
    gradual = [gradual_shift(seed) for seed in range(seeds)]
    source = [source_reliability_trap(seed) for seed in range(seeds)]
    locality = [compound_locality(seed) for seed in range(seeds)]
    memory = fixed_vocabulary_memory(0)

    clean_switches = [row["switch_observed"] for row in clean]
    gates = {
        "clean_switch_bounded": all(
            value is not None and value <= 16 for value in clean_switches
        ),
        "missing_30_observed_latency_invariant": all(
            row["switch_observed"] == clean[index]["switch_observed"]
            for index, row in enumerate(missing["0.3"])
        ),
        "missing_60_observed_latency_invariant": all(
            row["switch_observed"] == clean[index]["switch_observed"]
            for index, row in enumerate(missing["0.6"])
        ),
        "corrupted_10_no_failures": all(
            row["switch_observed"] is not None for row in corrupted["0.1"]
        ),
        "corrupted_20_no_failures": all(
            row["switch_observed"] is not None for row in corrupted["0.2"]
        ),
        "stationary_10_no_false_switch": not any(
            row["false_switch"] for row in stationary["0.1"]
        ),
        "stationary_20_no_false_switch": not any(
            row["false_switch"] for row in stationary["0.2"]
        ),
        "compound_locality": all(row["locality_preserved"] for row in locality),
        "fixed_vocabulary_state_bounded": memory["bounded_fixed_vocabulary"],
    }

    summary = {
        "protocol": PROTOCOL,
        "phase": "A-frozen-policy-baseline",
        "seeds": seeds,
        "stationary_steps": stationary_steps,
        "policy_changes": "none",
        "source_reliability_consumed": False,
        "clean": {
            "mean_switch_observed": _mean(clean, "switch_observed"),
            "max_switch_observed": max(clean_switches),
            "mean_unsafe_execute_rate": _mean(clean, "unsafe_execute_rate"),
        },
        "missing": {
            rate: {
                "mean_switch_observed": _mean(rows, "switch_observed"),
                "mean_switch_wall": _mean(rows, "switch_wall"),
                "p95_switch_wall": _p95(rows, "switch_wall"),
                "mean_unsafe_execute_rate": _mean(rows, "unsafe_execute_rate"),
            }
            for rate, rows in missing.items()
        },
        "corrupted_shift": {
            rate: {
                "mean_switch_observed": _mean(rows, "switch_observed"),
                "p95_switch_observed": _p95(rows, "switch_observed"),
                "failures": sum(row["switch_observed"] is None for row in rows),
                "mean_unsafe_execute_rate": _mean(rows, "unsafe_execute_rate"),
            }
            for rate, rows in corrupted.items()
        },
        "stationary_noise": {
            rate: {
                "false_switches": sum(row["false_switch"] for row in rows),
                "mean_unsafe_execute_rate": _mean(rows, "unsafe_execute_rate"),
                "mean_abstain_rate": _mean(rows, "abstain_rate"),
            }
            for rate, rows in stationary.items()
        },
        "gradual_shift": {
            "mean_switch_wall": _mean(gradual, "switch_wall"),
            "p95_switch_wall": _p95(gradual, "switch_wall"),
            "mean_unsafe_execute_rate": _mean(
                gradual, "unsafe_execute_rate"
            ),
            "mean_abstain_rate": _mean(gradual, "abstain_rate"),
        },
        "source_reliability_trap": {
            "weighted_oracle": source[0]["weighted_oracle"],
            "final_acceptance_counts": {
                relation: sum(row["final_accepted"] == relation for row in source)
                for relation in (OLD, NEW, None)
            },
            "mean_unsafe_execute_rate": _mean(
                source, "unsafe_execute_rate"
            ),
            "diagnostic_only": True,
        },
        "compound_locality": {
            "passed": all(row["locality_preserved"] for row in locality),
            "mean_active_cells": _mean(locality, "active_cells"),
        },
        "fixed_vocabulary_memory": memory,
        "baseline_gates": gates,
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
