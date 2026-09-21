#!/usr/bin/env python3
"""Relational Regime Plasticity v0.1 campaign.

Compare cumulative evidence with a dual-timescale regime articulation that
preserves historical evidence while bounding adaptation to persistent change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import statistics

from cortex.articulation_controller import ArticulationState
from cortex.controller_calibration import (
    CalibratedArticulationController,
    frozen_calibration_v0_2,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep
from cortex.regime_plasticity import (
    RegimePhase,
    RelationalRegimeMemory,
    decide_with_regime,
)


PROTOCOL = "relational-regime-plasticity-v0.1"
PRIOR_MASSES = (4, 8, 16, 32, 64, 128, 256)
SIGNATURE = (
    RelationStep("R0", "forward"),
    RelationStep("R1", "forward"),
)


def _geometry(seed: int) -> HypergraphGeometry:
    rng = Random(710003 + seed)
    suffix = f"{rng.getrandbits(40):010x}"
    return HypergraphGeometry(
        (
            RelationalHyperedge(
                f"a-{suffix}",
                0,
                10,
                1,
                f"A-{suffix}",
                "R0",
                f"B-{suffix}",
                2.0 / 3.0,
            ),
            RelationalHyperedge(
                f"b-{suffix}",
                1,
                11,
                2,
                f"B-{suffix}",
                "R1",
                f"C-{suffix}",
                2.0 / 3.0,
            ),
        )
    )


def _source(geometry: HypergraphGeometry) -> str:
    return geometry.hyperedges[0].subject


def _controller(geometry: HypergraphGeometry):
    return CalibratedArticulationController(
        geometry,
        frozen_calibration_v0_2(),
    )


def _cumulative_rules(old_count: int, new_count: int):
    total = old_count + new_count
    rows = [
        CompositionRule(
            signature=SIGNATURE,
            target_relation="OLD",
            examples=old_count,
            confidence=old_count / total,
            mean_support=2.0 / 3.0,
        )
    ]
    if new_count:
        rows.append(
            CompositionRule(
                signature=SIGNATURE,
                target_relation="NEW",
                examples=new_count,
                confidence=new_count / total,
                mean_support=2.0 / 3.0,
            )
        )
    return tuple(rows)


def _cumulative_switch_lag(
    geometry: HypergraphGeometry,
    prior_mass: int,
) -> int:
    controller = _controller(geometry)
    source = _source(geometry)
    for new_count in range(1, 32 * prior_mass + 1):
        decision = controller.decide(
            source,
            "NEW",
            _cumulative_rules(prior_mass, new_count),
        )
        if decision.state is ArticulationState.EXECUTE:
            return new_count
    raise RuntimeError("cumulative replacement did not execute")


def _regime_switch(seed: int, prior_mass: int) -> dict:
    geometry = _geometry(seed)
    controller = _controller(geometry)
    source = _source(geometry)
    regime = RelationalRegimeMemory(SIGNATURE)

    for _ in range(prior_mass):
        regime.observe("OLD")
    initial = decide_with_regime(
        controller,
        regime,
        source,
        "OLD",
    )

    old_exit = None
    unresolved = None
    new_execute = None
    trace = []
    for index in range(1, 80):
        read = regime.observe("NEW")
        old_decision = decide_with_regime(
            controller,
            regime,
            source,
            "OLD",
        )
        new_decision = decide_with_regime(
            controller,
            regime,
            source,
            "NEW",
        )
        trace.append(
            {
                "new_examples": index,
                "phase": read.phase.value,
                "accepted": read.accepted_relation,
                "candidate": read.candidate_relation,
                "recent_top": (
                    list(read.recent_top)
                    if read.recent_top is not None
                    else None
                ),
                "old_state": old_decision.state.value,
                "new_state": new_decision.state.value,
            }
        )
        if (
            old_exit is None
            and old_decision.state is not ArticulationState.EXECUTE
        ):
            old_exit = index
        if (
            unresolved is None
            and read.phase is RegimePhase.UNRESOLVED
        ):
            unresolved = index
        if new_decision.state is ArticulationState.EXECUTE:
            new_execute = index
            break

    return {
        "seed": seed,
        "prior_mass": prior_mass,
        "initial_executes": initial.state is ArticulationState.EXECUTE,
        "old_exit_execute": old_exit,
        "unresolved": unresolved,
        "new_execute": new_execute,
        "historical_old_preserved": (
            regime.historical_counts.get("OLD") == prior_mass
        ),
        "old_archived": (
            bool(regime.archives)
            and regime.archives[0].relation == "OLD"
        ),
        "trace": trace,
    }


def _stationary_noise(
    seed: int,
    noise_rate: float,
    steps: int = 1200,
) -> dict:
    rng = Random(810013 + seed * 1009 + int(noise_rate * 1000))
    geometry = _geometry(seed + 10000)
    controller = _controller(geometry)
    source = _source(geometry)
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(32):
        regime.observe("OLD")

    phase_counts = {phase.value: 0 for phase in RegimePhase}
    false_switches = 0
    false_new_executes = 0
    for _ in range(steps):
        relation = "NEW" if rng.random() < noise_rate else "OLD"
        before = regime.read().accepted_relation
        read = regime.observe(relation)
        phase_counts[read.phase.value] += 1
        if before == "OLD" and read.accepted_relation == "NEW":
            false_switches += 1
        decision = decide_with_regime(
            controller,
            regime,
            source,
            "NEW",
        )
        false_new_executes += int(
            decision.state is ArticulationState.EXECUTE
        )

    return {
        "seed": seed,
        "noise_rate": noise_rate,
        "false_switches": false_switches,
        "false_new_execute_rate": false_new_executes / steps,
        "stable_fraction": phase_counts["stable"] / steps,
        "challenged_fraction": phase_counts["challenged"] / steps,
        "unresolved_fraction": phase_counts["unresolved"] / steps,
        "accepted_relation": regime.read().accepted_relation,
    }


def _burst(seed: int, burst: int) -> dict:
    geometry = _geometry(seed + 20000)
    controller = _controller(geometry)
    source = _source(geometry)
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(32):
        regime.observe("OLD")

    deepest = RegimePhase.STABLE
    order = {
        RegimePhase.BOOTSTRAP: 0,
        RegimePhase.STABLE: 1,
        RegimePhase.CHALLENGED: 2,
        RegimePhase.UNRESOLVED: 3,
    }
    switched = False
    for _ in range(burst):
        read = regime.observe("NEW")
        if order[read.phase] > order[deepest]:
            deepest = read.phase
        switched |= read.accepted_relation == "NEW"

    recovery = 0
    while (
        regime.read().accepted_relation != "OLD"
        or regime.read().phase is not RegimePhase.STABLE
    ):
        regime.observe("OLD")
        recovery += 1
        if recovery > 80:
            break

    old_decision = decide_with_regime(
        controller,
        regime,
        source,
        "OLD",
    )
    return {
        "seed": seed,
        "burst": burst,
        "deepest_phase": deepest.value,
        "switched_to_new": switched,
        "recovery_observations": recovery,
        "old_executes_after_recovery": (
            old_decision.state is ArticulationState.EXECUTE
        ),
    }


def _recurrence(seed: int) -> dict:
    geometry = _geometry(seed + 30000)
    controller = _controller(geometry)
    source = _source(geometry)
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(64):
        regime.observe("OLD")

    to_new = None
    for index in range(1, 80):
        regime.observe("NEW")
        if (
            decide_with_regime(
                controller,
                regime,
                source,
                "NEW",
            ).state
            is ArticulationState.EXECUTE
        ):
            to_new = index
            break

    to_old = None
    for index in range(1, 80):
        regime.observe("OLD")
        if (
            decide_with_regime(
                controller,
                regime,
                source,
                "OLD",
            ).state
            is ArticulationState.EXECUTE
        ):
            to_old = index
            break

    return {
        "seed": seed,
        "to_new": to_new,
        "to_old": to_old,
        "archives": [row.relation for row in regime.archives],
        "historical_counts": dict(regime.historical_counts),
    }


def campaign(seeds: int) -> dict:
    switches = [
        _regime_switch(seed, prior)
        for seed in range(seeds)
        for prior in PRIOR_MASSES
    ]
    grouped = {}
    for prior in PRIOR_MASSES:
        rows = [row for row in switches if row["prior_mass"] == prior]
        geometry = _geometry(90000 + prior)
        baseline_lag = _cumulative_switch_lag(geometry, prior)
        candidate_lag = statistics.median(
            row["new_execute"] for row in rows
        )
        grouped[str(prior)] = {
            "old_exit_execute_median": statistics.median(
                row["old_exit_execute"] for row in rows
            ),
            "unresolved_median": statistics.median(
                row["unresolved"] for row in rows
            ),
            "new_execute_median": candidate_lag,
            "cumulative_new_execute": baseline_lag,
            "adaptation_speedup": baseline_lag / candidate_lag,
        }

    candidate_lags = [
        grouped[str(prior)]["new_execute_median"]
        for prior in PRIOR_MASSES
    ]

    noise = [
        _stationary_noise(seed, rate)
        for seed in range(seeds * 3)
        for rate in (0.05, 0.10, 0.20, 0.30)
    ]
    noise_summary = {}
    for rate in (0.05, 0.10, 0.20, 0.30):
        rows = [row for row in noise if row["noise_rate"] == rate]
        noise_summary[f"{rate:.2f}"] = {
            "false_switches": sum(row["false_switches"] for row in rows),
            "mean_false_new_execute_rate": statistics.mean(
                row["false_new_execute_rate"] for row in rows
            ),
            "mean_stable_fraction": statistics.mean(
                row["stable_fraction"] for row in rows
            ),
            "mean_challenged_fraction": statistics.mean(
                row["challenged_fraction"] for row in rows
            ),
            "mean_unresolved_fraction": statistics.mean(
                row["unresolved_fraction"] for row in rows
            ),
            "all_end_old": all(
                row["accepted_relation"] == "OLD" for row in rows
            ),
        }

    bursts = [
        _burst(seed, burst)
        for seed in range(seeds)
        for burst in (1, 2, 4, 8, 16, 24)
    ]
    burst_summary = {}
    for burst in (1, 2, 4, 8, 16, 24):
        rows = [row for row in bursts if row["burst"] == burst]
        burst_summary[str(burst)] = {
            "deepest_phases": sorted(
                {row["deepest_phase"] for row in rows}
            ),
            "switch_fraction": statistics.mean(
                int(row["switched_to_new"]) for row in rows
            ),
            "median_recovery_observations": statistics.median(
                row["recovery_observations"] for row in rows
            ),
            "all_recover_and_execute": all(
                row["old_executes_after_recovery"] for row in rows
            ),
        }

    recurrence = [_recurrence(seed) for seed in range(seeds)]

    gates = {
        "initial_stability": all(
            row["initial_executes"] for row in switches
        ),
        "bounded_switch_lag": max(candidate_lags) <= 24,
        "history_invariant_switch_lag": (
            max(candidate_lags) - min(candidate_lags) <= 10
        ),
        "history_preserved": all(
            row["historical_old_preserved"] and row["old_archived"]
            for row in switches
        ),
        "stationary_10pct_no_false_switch": (
            noise_summary["0.10"]["false_switches"] == 0
            and noise_summary["0.10"]["all_end_old"]
        ),
        "stationary_20pct_no_false_switch": (
            noise_summary["0.20"]["false_switches"] == 0
            and noise_summary["0.20"]["all_end_old"]
        ),
        "short_burst_stable": all(
            not row["switched_to_new"]
            for row in bursts
            if row["burst"] <= 8
        ),
        "burst_recovery": all(
            row["old_executes_after_recovery"] for row in bursts
        ),
        "recurrence_bounded": all(
            row["to_new"] is not None
            and row["to_old"] is not None
            and row["to_new"] <= 24
            and row["to_old"] <= 24
            for row in recurrence
        ),
        "recurrence_archives_history": all(
            row["archives"][:2] == ["OLD", "NEW"]
            for row in recurrence
        ),
    }

    return {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "grouped": grouped,
        "candidate_lag_range": [
            min(candidate_lags),
            max(candidate_lags),
        ],
        "noise": noise_summary,
        "bursts": burst_summary,
        "recurrence": recurrence,
        "gates": gates,
        "passed": all(gates.values()),
        "switches": switches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")

    report = campaign(args.seeds)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
