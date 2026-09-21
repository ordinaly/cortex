#!/usr/bin/env python3
"""Calibrate the Pareto frontier for Cortex relational regime plasticity v0.1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
import statistics

from cortex.articulation_controller import ArticulationState
from cortex.regime_plasticity import (
    RegimePlasticityPolicy,
    RelationalRegimeTracker,
    assess_fast_evidence,
)


PROTOCOL = "regime-plasticity-pareto-v0.1"
PRIOR_MASSES = (4, 8, 16, 32, 64)
BURSTS = (1, 2, 4, 8)
DECAYS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80)
PRESSURE_DECAYS = (0.85, 0.90, 0.95)
SUSPICION_RATIOS = (0.04, 0.05, 0.06)
CONFIRMATION_STEPS = (1, 2, 3, 4, 5)
MATURITY_CAP = 64
MAX_STEPS = 1_600


def _raw_state(weights: dict[str, float], relation: str) -> str:
    return assess_fast_evidence(relation, weights).state.value


def _cumulative_curve(prior_mass: int) -> dict:
    old_exit = None
    old_unresolved = None
    new_execute = None
    for new_count in range(24 * prior_mass + 1):
        weights = {"OLD": float(prior_mass)}
        if new_count:
            weights["NEW"] = float(new_count)
        old_state = _raw_state(weights, "OLD")
        new_state = _raw_state(weights, "NEW")
        if old_exit is None and old_state != ArticulationState.EXECUTE.value:
            old_exit = new_count
        if (
            old_unresolved is None
            and old_state == ArticulationState.UNRESOLVED.value
        ):
            old_unresolved = new_count
        if (
            new_execute is None
            and new_state == ArticulationState.EXECUTE.value
        ):
            new_execute = new_count
            break
    return {
        "prior_mass": prior_mass,
        "old_exit_execute": old_exit,
        "old_unresolved": old_unresolved,
        "new_execute": new_execute,
    }


def _bootstrap(policy: RegimePlasticityPolicy, prior_mass: int):
    tracker = RelationalRegimeTracker(policy)
    for _ in range(prior_mass):
        tracker.observe("OLD", bootstrap=True)
    return tracker


def _deterministic_shift(policy: RegimePlasticityPolicy, prior_mass: int) -> dict:
    tracker = _bootstrap(policy, prior_mass)
    detection = None
    switch = None
    trace = []
    for step in range(1, MAX_STEPS + 1):
        snap = tracker.observe("NEW")
        trace.append(
            {
                "step": step,
                "state": snap.state.value,
                "accepted": snap.accepted_relation,
                "pressure": snap.pressure,
                "threshold": snap.suspicion_threshold,
            }
        )
        if detection is None and snap.state is not ArticulationState.EXECUTE:
            detection = step
        if (
            snap.state is ArticulationState.EXECUTE
            and snap.accepted_relation == "NEW"
        ):
            switch = step
            break
    return {
        "prior_mass": prior_mass,
        "detection_lag": detection,
        "switch_lag": switch,
        "trace": trace,
    }


def _burst(policy: RegimePlasticityPolicy, burst: int) -> dict:
    tracker = _bootstrap(policy, 32)
    switched = False
    deepest = ArticulationState.EXECUTE
    order = {
        ArticulationState.EXECUTE: 0,
        ArticulationState.EXPLORE: 1,
        ArticulationState.UNRESOLVED: 2,
    }
    for _ in range(burst):
        snap = tracker.observe("NEW")
        switched = switched or snap.accepted_relation == "NEW"
        if order[snap.state] > order[deepest]:
            deepest = snap.state

    recovery = None
    for step in range(1, 257):
        snap = tracker.observe("OLD")
        if (
            snap.state is ArticulationState.EXECUTE
            and snap.accepted_relation == "OLD"
        ):
            recovery = step
            break
    return {
        "burst": burst,
        "false_switch": switched,
        "deepest_state": deepest.value,
        "recovery_support": recovery,
    }


def _reversal(policy: RegimePlasticityPolicy) -> dict:
    tracker = _bootstrap(policy, 32)
    forward = None
    for step in range(1, 257):
        snap = tracker.observe("NEW")
        if (
            snap.state is ArticulationState.EXECUTE
            and snap.accepted_relation == "NEW"
        ):
            forward = step
            break
    if forward is None:
        return {"forward_lag": None, "reverse_lag": None}

    for _ in range(32):
        tracker.observe("NEW")

    reverse = None
    for step in range(1, 257):
        snap = tracker.observe("OLD")
        if (
            snap.state is ArticulationState.EXECUTE
            and snap.accepted_relation == "OLD"
        ):
            reverse = step
            break
    return {"forward_lag": forward, "reverse_lag": reverse}


def _stationary_noise(
    policy: RegimePlasticityPolicy,
    *,
    rate: float,
    seeds: int,
    steps: int,
) -> dict:
    false_switch_steps = 0
    unresolved_steps = 0
    for seed in range(seeds):
        rng = Random(730_001 + seed * 1009 + int(rate * 10_000))
        tracker = _bootstrap(policy, 32)
        for _ in range(steps):
            observed = "NEW" if rng.random() < rate else "OLD"
            snap = tracker.observe(observed)
            if snap.accepted_relation != "OLD":
                false_switch_steps += 1
            if snap.state is not ArticulationState.EXECUTE:
                unresolved_steps += 1
    total = seeds * steps
    return {
        "rate": rate,
        "false_switch_steps": false_switch_steps,
        "unresolved_rate": unresolved_steps / total,
    }


def _noisy_shift(
    policy: RegimePlasticityPolicy,
    *,
    new_rate: float,
    seeds: int,
) -> dict:
    lags = []
    for seed in range(seeds):
        rng = Random(830_003 + seed * 1013 + int(new_rate * 10_000))
        tracker = _bootstrap(policy, 32)
        lag = None
        for step in range(1, 301):
            observed = "NEW" if rng.random() < new_rate else "OLD"
            snap = tracker.observe(observed)
            if (
                snap.state is ArticulationState.EXECUTE
                and snap.accepted_relation == "NEW"
            ):
                lag = step
                break
        lags.append(301 if lag is None else lag)
    ordered = sorted(lags)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "new_rate": new_rate,
        "median_lag": statistics.median(lags),
        "p95_lag": p95,
        "max_lag": max(lags),
        "failures": sum(lag > 300 for lag in lags),
    }


def _memory_and_history(policy: RegimePlasticityPolicy) -> dict:
    tracker = _bootstrap(policy, 64)
    initial_cells = None
    for index in range(10_000):
        snap = tracker.observe("NEW" if index % 7 == 0 else "OLD")
        if index == 100:
            initial_cells = snap.active_cells
    final = snap
    return {
        "initial_cells": initial_cells,
        "final_cells": final.active_cells,
        "bounded_for_fixed_vocabulary": initial_cells == final.active_cells,
        "history_total": sum(final.historical_evidence.values()),
        "history_exact": (
            sum(final.historical_evidence.values()) == 64 + 10_000
        ),
    }


def _locality(policy: RegimePlasticityPolicy) -> bool:
    primary = _bootstrap(policy, 32)
    control = RelationalRegimeTracker(policy)
    for _ in range(8):
        control.observe("CONTROL", bootstrap=True)
    before_key = (
        control.accepted_relation,
        control.state,
        tuple(sorted(control.fast_evidence.items())),
        tuple(sorted(control.historical_evidence.items())),
    )
    for _ in range(64):
        primary.observe("NEW")
    after_key = (
        control.accepted_relation,
        control.state,
        tuple(sorted(control.fast_evidence.items())),
        tuple(sorted(control.historical_evidence.items())),
    )
    return before_key == after_key


def _evaluate(
    policy: RegimePlasticityPolicy,
    *,
    baseline_detection: dict[int, int],
    noise_seeds: int,
    noise_steps: int,
) -> dict:
    shifts = [
        _deterministic_shift(policy, prior)
        for prior in PRIOR_MASSES
    ]
    bursts = [_burst(policy, burst) for burst in BURSTS]
    reversal = _reversal(policy)
    stationary = [
        _stationary_noise(
            policy,
            rate=rate,
            seeds=noise_seeds,
            steps=noise_steps,
        )
        for rate in (0.10, 0.20)
    ]
    noisy = [
        _noisy_shift(policy, new_rate=rate, seeds=noise_seeds)
        for rate in (0.90, 0.80)
    ]
    memory = _memory_and_history(policy)

    detection_excess = sum(
        max(
            0,
            (row["detection_lag"] or MAX_STEPS)
            - baseline_detection[row["prior_mass"]],
        )
        for row in shifts
    )
    switch_lags = [
        row["switch_lag"] if row["switch_lag"] is not None else MAX_STEPS
        for row in shifts
    ]
    recoveries = [
        row["recovery_support"] or 257
        for row in bursts
    ]
    false_switch_steps = sum(
        row["false_switch_steps"] for row in stationary
    )
    noisy80 = next(row for row in noisy if row["new_rate"] == 0.80)
    noisy90 = next(row for row in noisy if row["new_rate"] == 0.90)

    return {
        "policy": {
            "fast_decay": policy.fast_decay,
            "pressure_decay": policy.pressure_decay,
            "suspicion_ratio": policy.suspicion_ratio,
            "confirmation_steps": policy.confirmation_steps,
            "maturity_cap": policy.maturity_cap,
        },
        "deterministic_shifts": shifts,
        "max_shift_lag": max(switch_lags),
        "shift_lag_spread": max(switch_lags) - min(switch_lags),
        "detection_excess": detection_excess,
        "bursts": bursts,
        "burst_false_switches": sum(
            int(row["false_switch"]) for row in bursts
        ),
        "mean_recovery_support": statistics.mean(recoveries),
        "max_recovery_support": max(recoveries),
        "reversal": reversal,
        "stationary_noise": stationary,
        "stationary_false_switch_steps": false_switch_steps,
        "stationary_unresolved_rate": statistics.mean(
            row["unresolved_rate"] for row in stationary
        ),
        "noisy_shifts": noisy,
        "noisy_shift_p95_90": noisy90["p95_lag"],
        "noisy_shift_p95_80": noisy80["p95_lag"],
        "noisy_shift_failures": sum(row["failures"] for row in noisy),
        "locality_preserved": _locality(policy),
        "memory": memory,
        "history_retained": memory["history_exact"],
        "memory_bounded": memory["bounded_for_fixed_vocabulary"],
    }


def _feasible(row: dict) -> bool:
    return (
        row["detection_excess"] == 0
        and row["burst_false_switches"] == 0
        and row["stationary_false_switch_steps"] == 0
        and row["noisy_shift_failures"] == 0
        and row["locality_preserved"]
        and row["history_retained"]
        and row["memory_bounded"]
        and row["reversal"]["reverse_lag"] is not None
    )


OBJECTIVES = (
    "max_shift_lag",
    "noisy_shift_p95_80",
    "mean_recovery_support",
    "stationary_unresolved_rate",
    "reverse_lag",
)


def _objective(row: dict, name: str) -> float:
    if name == "reverse_lag":
        value = row["reversal"]["reverse_lag"]
        return float(MAX_STEPS if value is None else value)
    return float(row[name])


def _dominates(left: dict, right: dict) -> bool:
    lv = tuple(_objective(left, name) for name in OBJECTIVES)
    rv = tuple(_objective(right, name) for name in OBJECTIVES)
    return all(a <= b for a, b in zip(lv, rv)) and any(
        a < b for a, b in zip(lv, rv)
    )


def _pareto(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if not any(
            other is not row and _dominates(other, row)
            for other in rows
        )
    ]


def _knee(frontier: list[dict]) -> dict:
    ranges = {}
    for name in OBJECTIVES:
        values = [_objective(row, name) for row in frontier]
        ranges[name] = (min(values), max(values))

    def distance(row):
        total = 0.0
        for name in OBJECTIVES:
            value = _objective(row, name)
            low, high = ranges[name]
            normalized = 0.0 if high <= low else (value - low) / (high - low)
            total += normalized * normalized
        policy = row["policy"]
        tie = (
            row["max_shift_lag"],
            row["noisy_shift_p95_80"],
            row["mean_recovery_support"],
            row["stationary_unresolved_rate"],
            policy["fast_decay"],
            policy["confirmation_steps"],
        )
        return (math.sqrt(total), tie)

    return min(frontier, key=distance)


def _decay_only_summary(decay: float, *, preserve_history: bool) -> dict:
    switch_lags = []
    burst_switches = 0
    for prior in PRIOR_MASSES:
        weights = {"OLD": 0.0, "NEW": 0.0}
        for _ in range(prior):
            weights["OLD"] *= decay
            weights["NEW"] *= decay
            weights["OLD"] += 1.0
        switch = None
        for step in range(1, 301):
            weights["OLD"] *= decay
            weights["NEW"] *= decay
            weights["NEW"] += 1.0
            if _raw_state(weights, "NEW") == ArticulationState.EXECUTE.value:
                switch = step
                break
        switch_lags.append(301 if switch is None else switch)

    for burst in BURSTS:
        weights = {"OLD": 0.0, "NEW": 0.0}
        for _ in range(32):
            weights["OLD"] *= decay
            weights["NEW"] *= decay
            weights["OLD"] += 1.0
        switched = False
        for _ in range(burst):
            weights["OLD"] *= decay
            weights["NEW"] *= decay
            weights["NEW"] += 1.0
            switched = switched or (
                _raw_state(weights, "NEW")
                == ArticulationState.EXECUTE.value
            )
        burst_switches += int(switched)

    return {
        "decay": decay,
        "history_retained": preserve_history,
        "max_shift_lag": max(switch_lags),
        "shift_lags": switch_lags,
        "burst_false_switches": burst_switches,
    }


def campaign(noise_seeds: int, noise_steps: int) -> dict:
    cumulative = [_cumulative_curve(prior) for prior in PRIOR_MASSES]
    baseline_detection = {
        row["prior_mass"]: int(row["old_exit_execute"])
        for row in cumulative
    }

    rows = []
    for fast_decay in DECAYS:
        for pressure_decay in PRESSURE_DECAYS:
            for suspicion_ratio in SUSPICION_RATIOS:
                for confirmation_steps in CONFIRMATION_STEPS:
                    rows.append(
                        _evaluate(
                            RegimePlasticityPolicy(
                                fast_decay=fast_decay,
                                pressure_decay=pressure_decay,
                                suspicion_ratio=suspicion_ratio,
                                confirmation_steps=confirmation_steps,
                                maturity_cap=MATURITY_CAP,
                            ),
                            baseline_detection=baseline_detection,
                            noise_seeds=noise_seeds,
                            noise_steps=noise_steps,
                        )
                    )

    feasible = [row for row in rows if _feasible(row)]
    frontier = _pareto(feasible)
    selected = _knee(frontier) if frontier else None

    decay_baseline = [
        _decay_only_summary(decay, preserve_history=False)
        for decay in DECAYS
    ]
    dual_no_hysteresis = [
        _decay_only_summary(decay, preserve_history=True)
        for decay in DECAYS
    ]

    gates = {
        "feasible_candidates_exist": bool(feasible),
        "pareto_frontier_exists": bool(frontier),
        "selected_candidate_exists": selected is not None,
        "selected_preserves_v0_2_detection_contract": (
            selected is not None and selected["detection_excess"] == 0
        ),
        "selected_resists_declared_noise": (
            selected is not None
            and selected["burst_false_switches"] == 0
            and selected["stationary_false_switch_steps"] == 0
        ),
        "selected_reverses": (
            selected is not None
            and selected["reversal"]["reverse_lag"] is not None
        ),
        "selected_preserves_history_locality_memory": (
            selected is not None
            and selected["history_retained"]
            and selected["locality_preserved"]
            and selected["memory_bounded"]
        ),
        "selected_bounded_shift_lag": (
            selected is not None
            and selected["max_shift_lag"] <= 16
            and selected["shift_lag_spread"] <= 4
        ),
        "selected_improves_cumulative_rearticulation": (
            selected is not None
            and selected["max_shift_lag"]
            < max(int(row["new_execute"]) for row in cumulative)
        ),
    }

    return {
        "protocol": PROTOCOL,
        "noise_seeds": noise_seeds,
        "noise_steps": noise_steps,
        "parameter_count": len(rows),
        "cumulative_baseline": cumulative,
        "naive_exponential_decay": decay_baseline,
        "dual_timescale_without_hysteresis": dual_no_hysteresis,
        "feasible_count": len(feasible),
        "pareto_count": len(frontier),
        "pareto_frontier": sorted(
            frontier,
            key=lambda row: (
                row["max_shift_lag"],
                row["noisy_shift_p95_80"],
                row["mean_recovery_support"],
                row["stationary_unresolved_rate"],
            ),
        ),
        "selected": selected,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-seeds", type=int, default=8)
    parser.add_argument("--noise-steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.noise_seeds < 1 or args.noise_steps < 1:
        raise SystemExit("noise seeds and steps must be positive")

    report = campaign(args.noise_seeds, args.noise_steps)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
