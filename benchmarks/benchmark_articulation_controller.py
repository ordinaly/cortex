#!/usr/bin/env python3
"""Validation campaign for the Cortex articulation controller v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cortex.articulation_controller import (
    ArticulationController,
    ArticulationState,
    ExecutionBackend,
)
from cortex.hypergraph_inference import HypergraphCompositionLearner

import benchmark_hypergraph_lpu_stress as stress


PROTOCOL = "articulation-controller-v0.1"


def _rules(training, depth):
    # HypergraphCompositionLearner currently has no separate margin gate; a
    # zero confidence threshold preserves competing relation hypotheses so the
    # controller, not the learner, owns the epistemic decision.
    return HypergraphCompositionLearner(
        training,
        path_length=depth,
        minimum_examples=1,
        confidence_threshold=0.0,
        resolution=0.5,
    ).fit()


def run_seed(seed: int, width: int) -> dict:
    clean = []
    for depth in range(2, 6):
        world = stress.make_world(
            seed=seed,
            depth=depth,
            width=width,
            contradiction_fraction=0.0,
        )
        runtime, report, training, full = stress._build(world)
        rules = _rules(training, depth)
        local = stress.KnowledgeGraphView(
            report.snapshot_json,
            report.symbol_bindings,
            runtime.config_json(),
            report.statement_incidence,
        ).local_articulation(world.source, resolution=0.5)

        controller = ArticulationController(full)
        before = runtime.snapshot_json()
        controlled = controller.execute(
            world.source,
            world.target_relation,
            rules,
            source_support=local.topology.top_entity.support,
            source_entropy=local.topology.entity_entropy,
        )
        after = runtime.snapshot_json()

        expected_backend = (
            ExecutionBackend.LPU
            if depth >= 4
            else ExecutionBackend.CORTEX
        )
        targets = (
            {row.target for row in controlled.execution.candidates}
            if controlled.execution is not None
            else set()
        )
        clean.append(
            {
                "depth": depth,
                "state": controlled.decision.state.value,
                "backend": (
                    controlled.decision.backend.value
                    if controlled.decision.backend is not None
                    else None
                ),
                "expected_backend": expected_backend.value,
                "resolution": controlled.decision.resolution,
                "path_multiplicity": controlled.decision.path_multiplicity,
                "materialized_edges": controlled.decision.materialized_edges,
                "compression_ratio": controlled.decision.compression_ratio,
                "correct": targets == {world.target},
                "backend_expected": (
                    controlled.decision.backend is expected_backend
                ),
                "memory_unchanged": before == after,
            }
        )

    # Moderate conflict should remain executable.
    world20 = stress.make_world(
        seed=seed,
        depth=3,
        width=width,
        contradiction_fraction=0.2,
    )
    runtime20, report20, training20, full20 = stress._build(world20)
    rules20 = _rules(training20, 3)
    decision20 = ArticulationController(full20).decide(
        world20.source,
        world20.target_relation,
        rules20,
    )

    # Stronger conflict is evidence-backed disagreement, not mere ignorance.
    world40 = stress.make_world(
        seed=seed,
        depth=3,
        width=width,
        contradiction_fraction=0.4,
    )
    runtime40, report40, training40, full40 = stress._build(world40)
    rules40 = _rules(training40, 3)
    decision40 = ArticulationController(full40).decide(
        world40.source,
        world40.target_relation,
        rules40,
    )

    # Same clean depth-2 problem, but an exact witness request should route to
    # the LPU regardless of path-compression economics.
    witness_world = stress.make_world(
        seed=seed,
        depth=2,
        width=width,
        contradiction_fraction=0.0,
    )
    runtime_w, report_w, training_w, full_w = stress._build(witness_world)
    rules_w = _rules(training_w, 2)
    witness = ArticulationController(full_w).execute(
        witness_world.source,
        witness_world.target_relation,
        rules_w,
        exact_witness=True,
    )

    # Simulate an ambiguous perceptual source. Geometry/rules are unchanged;
    # only the local articulation confidence is unstable.
    ambiguous = ArticulationController(full_w).decide(
        witness_world.source,
        witness_world.target_relation,
        rules_w,
        source_support=0.55,
        source_entropy=0.70,
    )

    return {
        "seed": seed,
        "clean": clean,
        "noise20_state": decision20.state.value,
        "noise20_best_confidence": (
            decision20.assessments[0].rule.confidence
            if decision20.assessments
            else 0.0
        ),
        "noise40_state": decision40.state.value,
        "noise40_reason": list(decision40.reasons),
        "witness_backend": (
            witness.decision.backend.value
            if witness.decision.backend is not None
            else None
        ),
        "witness_count": (
            len(witness.execution.witnesses)
            if witness.execution is not None
            else 0
        ),
        "ambiguous_source_state": ambiguous.state.value,
        "binding_conflicts": (
            report20.binding_conflicts
            + report40.binding_conflicts
            + report_w.binding_conflicts
        ),
    }


def campaign(seeds: int, width: int) -> dict:
    rows = [run_seed(seed, width) for seed in range(seeds)]
    clean = [case for row in rows for case in row["clean"]]

    backend_counts = {
        "cortex": sum(case["backend"] == "cortex" for case in clean),
        "lpu": sum(case["backend"] == "lpu" for case in clean),
    }
    summary = {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "width": width,
        "clean_cases": len(clean),
        "clean_execute_rate": sum(
            case["state"] == "execute" for case in clean
        ) / len(clean),
        "clean_accuracy": sum(case["correct"] for case in clean) / len(clean),
        "backend_policy_accuracy": sum(
            case["backend_expected"] for case in clean
        ) / len(clean),
        "memory_isolation_rate": sum(
            case["memory_unchanged"] for case in clean
        ) / len(clean),
        "backend_counts": backend_counts,
        "noise20_execute_rate": sum(
            row["noise20_state"] == "execute" for row in rows
        ) / len(rows),
        "noise20_mean_confidence": sum(
            row["noise20_best_confidence"] for row in rows
        ) / len(rows),
        "noise40_unresolved_rate": sum(
            row["noise40_state"] == "unresolved" for row in rows
        ) / len(rows),
        "ambiguous_source_explore_rate": sum(
            row["ambiguous_source_state"] == "explore" for row in rows
        ) / len(rows),
        "exact_witness_routes_lpu_rate": sum(
            row["witness_backend"] == "lpu" and row["witness_count"] >= 1
            for row in rows
        ) / len(rows),
        "all_zero_binding_conflicts": all(
            row["binding_conflicts"] == 0 for row in rows
        ),
        "depth_profiles": {
            str(depth): {
                "mean_paths": sum(
                    case["path_multiplicity"]
                    for case in clean
                    if case["depth"] == depth
                ) / seeds,
                "mean_edges": sum(
                    case["materialized_edges"]
                    for case in clean
                    if case["depth"] == depth
                ) / seeds,
                "backend": next(
                    case["backend"]
                    for case in clean
                    if case["depth"] == depth
                ),
            }
            for depth in range(2, 6)
        },
    }
    gates = {
        "clean_executes": summary["clean_execute_rate"] == 1.0,
        "clean_correct": summary["clean_accuracy"] == 1.0,
        "backend_policy": summary["backend_policy_accuracy"] == 1.0,
        "memory_isolated": summary["memory_isolation_rate"] == 1.0,
        "moderate_conflict_executes": summary["noise20_execute_rate"] == 1.0,
        "strong_conflict_abstains": (
            summary["noise40_unresolved_rate"] == 1.0
        ),
        "ambiguous_source_explores": (
            summary["ambiguous_source_explore_rate"] == 1.0
        ),
        "exact_witness_uses_lpu": (
            summary["exact_witness_routes_lpu_rate"] == 1.0
        ),
        "binding_stable": summary["all_zero_binding_conflicts"],
    }
    return {
        **summary,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = campaign(args.seeds, args.width)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
