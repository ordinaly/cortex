#!/usr/bin/env python3
"""Characterize the stability-plasticity frontier of Cortex compositions.

This campaign freezes the current cumulative evidence semantics and measures
how quickly a previously stable relational composition can be revised when a
new regime supplies incompatible outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
import statistics

from cortex.articulation_controller import ArticulationController
from cortex.controller_calibration import (
    CalibratedArticulationController,
    frozen_calibration_v0_2,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import HypergraphCompositionLearner


PROTOCOL = "plasticity-frontier-v1"
RESOLUTION = 0.5
PRIOR_MASSES = (4, 8, 16, 32, 64)


def _token(rng: Random, prefix: str, index: int) -> str:
    return f"{prefix}-{index}-{rng.getrandbits(48):012x}"


def _triangle(
    rng: Random,
    *,
    serial: int,
    r0: str,
    r1: str,
    outcome_relation: str | None,
) -> tuple[RelationalHyperedge, ...]:
    source = _token(rng, "s", serial)
    middle = _token(rng, "m", serial)
    target = _token(rng, "t", serial)
    rows = [
        RelationalHyperedge(
            f"p0-{serial}",
            serial * 10,
            serial * 10 + 1,
            serial * 10 + 2,
            source,
            r0,
            middle,
            2.0 / 3.0,
        ),
        RelationalHyperedge(
            f"p1-{serial}",
            serial * 10 + 2,
            serial * 10 + 3,
            serial * 10 + 4,
            middle,
            r1,
            target,
            2.0 / 3.0,
        ),
    ]
    if outcome_relation is not None:
        rows.append(
            RelationalHyperedge(
                f"out-{serial}",
                serial * 10,
                serial * 10 + 5,
                serial * 10 + 4,
                source,
                outcome_relation,
                target,
                2.0 / 3.0,
            )
        )
    return tuple(rows)


def _query_triangle(
    rng: Random,
    *,
    serial: int,
    r0: str,
    r1: str,
) -> tuple[tuple[RelationalHyperedge, ...], str]:
    rows = _triangle(
        rng,
        serial=serial,
        r0=r0,
        r1=r1,
        outcome_relation=None,
    )
    return rows, rows[0].subject


def _fit(edges):
    geometry = HypergraphGeometry(edges)
    rules = HypergraphCompositionLearner(
        geometry,
        path_length=2,
        minimum_examples=1,
        confidence_threshold=0.0,
        resolution=RESOLUTION,
    ).fit()
    return geometry, rules


def _rule_row(rules, relation):
    candidates = [row for row in rules if row.target_relation == relation]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row.confidence, row.examples))


def _state(geometry, rules, source, relation, *, calibrated):
    if calibrated:
        controller = CalibratedArticulationController(
            geometry,
            frozen_calibration_v0_2(),
        )
    else:
        controller = ArticulationController(geometry)
    decision = controller.decide(source, relation, rules)
    return decision.state.value


def _first(rows, predicate):
    for row in rows:
        if predicate(row):
            return row["new_examples"]
    return None


def _shift_curve(seed: int, prior_mass: int) -> dict:
    rng = Random(410003 + seed * 997 + prior_mass * 31)
    r0 = _token(rng, "R0", 0)
    r1 = _token(rng, "R1", 0)
    old_relation = _token(rng, "OLD", 0)
    new_relation = _token(rng, "NEW", 0)
    c0 = _token(rng, "C0", 0)
    c1 = _token(rng, "C1", 0)
    control_relation = _token(rng, "CONTROL", 0)

    edges = []
    serial = 0
    for _ in range(prior_mass):
        edges.extend(
            _triangle(
                rng,
                serial=serial,
                r0=r0,
                r1=r1,
                outcome_relation=old_relation,
            )
        )
        serial += 1

    # A disjoint control law must not move while the target regime changes.
    for _ in range(8):
        edges.extend(
            _triangle(
                rng,
                serial=serial,
                r0=c0,
                r1=c1,
                outcome_relation=control_relation,
            )
        )
        serial += 1

    query_edges, query_source = _query_triangle(
        rng,
        serial=serial,
        r0=r0,
        r1=r1,
    )
    edges.extend(query_edges)
    serial += 1

    control_edges, control_source = _query_triangle(
        rng,
        serial=serial,
        r0=c0,
        r1=c1,
    )
    edges.extend(control_edges)
    serial += 1

    # Continue until the new regime executes, or until 4x prior evidence has
    # arrived. This makes linear-in-history adaptation visible.
    trace = []
    max_new = max(24, 4 * prior_mass)
    for new_examples in range(max_new + 1):
        geometry, rules = _fit(tuple(edges))
        old_rule = _rule_row(rules, old_relation)
        new_rule = _rule_row(rules, new_relation)
        control_rule = _rule_row(rules, control_relation)
        row = {
            "new_examples": new_examples,
            "old_confidence": (
                old_rule.confidence if old_rule is not None else 0.0
            ),
            "new_confidence": (
                new_rule.confidence if new_rule is not None else 0.0
            ),
            "old_state_calibrated": _state(
                geometry,
                rules,
                query_source,
                old_relation,
                calibrated=True,
            ),
            "new_state_calibrated": _state(
                geometry,
                rules,
                query_source,
                new_relation,
                calibrated=True,
            ),
            "old_state_fixed": _state(
                geometry,
                rules,
                query_source,
                old_relation,
                calibrated=False,
            ),
            "new_state_fixed": _state(
                geometry,
                rules,
                query_source,
                new_relation,
                calibrated=False,
            ),
            "control_confidence": (
                control_rule.confidence if control_rule is not None else 0.0
            ),
            "control_state": _state(
                geometry,
                rules,
                control_source,
                control_relation,
                calibrated=True,
            ),
        }
        trace.append(row)

        if (
            new_examples > 0
            and row["new_state_calibrated"] == "execute"
            and row["old_state_calibrated"] != "execute"
        ):
            # Keep one extra point after crossover for inspectability.
            if new_examples < max_new:
                edges.extend(
                    _triangle(
                        rng,
                        serial=serial,
                        r0=r0,
                        r1=r1,
                        outcome_relation=new_relation,
                    )
                )
                serial += 1
                geometry, rules = _fit(tuple(edges))
                old_rule = _rule_row(rules, old_relation)
                new_rule = _rule_row(rules, new_relation)
                trace.append(
                    {
                        "new_examples": new_examples + 1,
                        "old_confidence": old_rule.confidence,
                        "new_confidence": new_rule.confidence,
                        "old_state_calibrated": _state(
                            geometry, rules, query_source, old_relation,
                            calibrated=True,
                        ),
                        "new_state_calibrated": _state(
                            geometry, rules, query_source, new_relation,
                            calibrated=True,
                        ),
                        "old_state_fixed": _state(
                            geometry, rules, query_source, old_relation,
                            calibrated=False,
                        ),
                        "new_state_fixed": _state(
                            geometry, rules, query_source, new_relation,
                            calibrated=False,
                        ),
                        "control_confidence": _rule_row(
                            rules, control_relation
                        ).confidence,
                        "control_state": _state(
                            geometry, rules, control_source, control_relation,
                            calibrated=True,
                        ),
                    }
                )
            break

        if new_examples < max_new:
            edges.extend(
                _triangle(
                    rng,
                    serial=serial,
                    r0=r0,
                    r1=r1,
                    outcome_relation=new_relation,
                )
            )
            serial += 1

    old_exit = _first(
        trace,
        lambda row: row["old_state_calibrated"] != "execute",
    )
    old_unresolved = _first(
        trace,
        lambda row: row["old_state_calibrated"] == "unresolved",
    )
    new_execute = _first(
        trace,
        lambda row: row["new_state_calibrated"] == "execute",
    )
    return {
        "seed": seed,
        "prior_mass": prior_mass,
        "old_exit_execute": old_exit,
        "old_unresolved": old_unresolved,
        "new_execute": new_execute,
        "adaptation_ratio": (
            new_execute / prior_mass
            if new_execute is not None
            else None
        ),
        "control_preserved": all(
            row["control_state"] == "execute"
            and math.isclose(row["control_confidence"], 1.0)
            for row in trace
        ),
        "trace": trace,
    }


def _noise_burst(seed: int, prior_mass: int, burst: int) -> dict:
    rng = Random(510007 + seed * 1013 + prior_mass * 17 + burst)
    r0 = _token(rng, "R0", 0)
    r1 = _token(rng, "R1", 0)
    old_relation = _token(rng, "OLD", 0)
    new_relation = _token(rng, "NOISE", 0)
    edges = []
    serial = 0
    for _ in range(prior_mass):
        edges.extend(
            _triangle(
                rng,
                serial=serial,
                r0=r0,
                r1=r1,
                outcome_relation=old_relation,
            )
        )
        serial += 1
    query_edges, query_source = _query_triangle(
        rng,
        serial=serial,
        r0=r0,
        r1=r1,
    )
    edges.extend(query_edges)
    serial += 1

    states = []
    for _ in range(burst):
        edges.extend(
            _triangle(
                rng,
                serial=serial,
                r0=r0,
                r1=r1,
                outcome_relation=new_relation,
            )
        )
        serial += 1
        geometry, rules = _fit(tuple(edges))
        states.append(
            _state(
                geometry,
                rules,
                query_source,
                old_relation,
                calibrated=True,
            )
        )

    support_needed = 0
    geometry, rules = _fit(tuple(edges))
    current = _state(
        geometry, rules, query_source, old_relation, calibrated=True
    )
    while current != "execute" and support_needed <= 4 * max(1, burst):
        edges.extend(
            _triangle(
                rng,
                serial=serial,
                r0=r0,
                r1=r1,
                outcome_relation=old_relation,
            )
        )
        serial += 1
        support_needed += 1
        geometry, rules = _fit(tuple(edges))
        current = _state(
            geometry, rules, query_source, old_relation, calibrated=True
        )

    return {
        "seed": seed,
        "prior_mass": prior_mass,
        "burst": burst,
        "minimum_state_during_burst": (
            "unresolved"
            if "unresolved" in states
            else "explore"
            if "explore" in states
            else "execute"
        ),
        "recovery_support_needed": support_needed,
        "recovered": current == "execute",
    }


def campaign(seeds: int) -> dict:
    curves = [
        _shift_curve(seed, prior_mass)
        for seed in range(seeds)
        for prior_mass in PRIOR_MASSES
    ]
    noise = [
        _noise_burst(seed, 32, burst)
        for seed in range(seeds)
        for burst in (1, 2, 4, 8)
    ]

    grouped = {}
    for prior in PRIOR_MASSES:
        rows = [row for row in curves if row["prior_mass"] == prior]
        grouped[str(prior)] = {
            "old_exit_execute_median": statistics.median(
                row["old_exit_execute"] for row in rows
            ),
            "old_unresolved_median": statistics.median(
                row["old_unresolved"] for row in rows
            ),
            "new_execute_median": statistics.median(
                row["new_execute"] for row in rows
            ),
            "adaptation_ratio_median": statistics.median(
                row["adaptation_ratio"] for row in rows
            ),
            "control_preserved": all(
                row["control_preserved"] for row in rows
            ),
        }

    new_executes = [
        grouped[str(prior)]["new_execute_median"]
        for prior in PRIOR_MASSES
    ]
    normalized = [
        grouped[str(prior)]["adaptation_ratio_median"]
        for prior in PRIOR_MASSES
    ]

    # Linear historical inertia is diagnosed when the number of new outcomes
    # required for the replacement composition grows monotonically with prior
    # evidence and its ratio to prior evidence remains approximately stable.
    ratio_spread = max(normalized) - min(normalized)
    historical_inertia = (
        all(a < b for a, b in zip(new_executes, new_executes[1:]))
        and ratio_spread <= 0.35
    )

    noise_summary = {}
    for burst in (1, 2, 4, 8):
        rows = [row for row in noise if row["burst"] == burst]
        noise_summary[str(burst)] = {
            "states": sorted(
                {row["minimum_state_during_burst"] for row in rows}
            ),
            "median_recovery_support": statistics.median(
                row["recovery_support_needed"] for row in rows
            ),
            "all_recovered": all(row["recovered"] for row in rows),
        }

    gates = {
        "baseline_executes": all(
            row["trace"][0]["old_state_calibrated"] == "execute"
            for row in curves
        ),
        "replacement_eventually_executes": all(
            row["new_execute"] is not None for row in curves
        ),
        "locality_preserved": all(
            row["control_preserved"] for row in curves
        ),
        "noise_recovery": all(row["recovered"] for row in noise),
    }

    return {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "prior_masses": list(PRIOR_MASSES),
        "grouped": grouped,
        "historical_inertia_detected": historical_inertia,
        "adaptation_ratio_spread": ratio_spread,
        "noise": noise_summary,
        "gates": gates,
        "passed": all(gates.values()),
        "curves": curves,
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
