#!/usr/bin/env python3
"""Development probe for Closed Articulation Loop v0.1 on OWR-A through D.

The same default ClosedLoopPolicy is used for every stage. This script does not
modify the OWR generators, does not consume hidden oracle fields during learner
execution, and does not generate official seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cortex.articulation_controller import ArticulationState
from cortex.closed_articulation_loop import (
    ClosedArticulationLoop,
    ClosedLoopPolicy,
    EvidenceAtom,
)
from open_world_robustness_world import PROTOCOL, HiddenWorld, build_suite

PROBE = "closed-articulation-loop-v0.1-owr-development-probe"


def _feed_query_stream(world: HiddenWorld, loop: ClosedArticulationLoop):
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    queries = sorted(world.queries, key=lambda row: (row.query_time, row.query_id))
    cursor = 0
    last = {}
    bubble_candidates = []
    rows = []

    for query in queries:
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            read = loop.observe(
                EvidenceAtom(
                    context=observation.context,
                    relation=observation.observed_relation,
                    event_time=observation.event_time,
                    arrival_time=observation.arrival_time,
                    provenance=observation.source_id,
                    bootstrap=observation.bootstrap,
                )
            )
            last[observation.context] = read
            if (
                read.background_candidate is not None
                and read.background_candidate.bubble_eligible
            ):
                bubble_candidates.append(read.background_candidate)
            cursor += 1

        read = last.get(query.context)
        executable = (
            read is not None
            and read.state is ArticulationState.EXECUTE
        )
        answer = read.accepted_relation if executable else None
        rows.append(
            {
                "query_id": query.query_id,
                "query_time": query.query_time,
                "context": query.context,
                "answerable": query.answerable,
                "expected": query.expected_relation,
                "answer": answer,
                "resolved": answer is not None,
                "correct": answer == query.expected_relation,
            }
        )

    return rows, bubble_candidates


def _score(rows):
    answerable = [row for row in rows if row["answerable"]]
    resolved = [row for row in answerable if row["resolved"]]
    correct = [row for row in resolved if row["correct"]]
    wrong = len(resolved) - len(correct)
    return {
        "queries": len(answerable),
        "resolved": len(resolved),
        "correct": len(correct),
        "wrong": wrong,
        "coverage": len(resolved) / max(1, len(answerable)),
        "resolved_accuracy": len(correct) / max(1, len(resolved)),
        "accuracy": len(correct) / max(1, len(answerable)),
        "selective_risk": wrong / max(1, len(resolved)),
    }


def _a_partitions(world, rows):
    traps = {tuple(value) for value in world.oracle["trap_contexts"]}
    return {
        "all": rows,
        "calibration": [
            row for row in rows if row["query_id"].startswith("AQC")
        ],
        "target": [
            row for row in rows if row["query_id"].startswith("AQT")
        ],
        "target_trap": [
            row
            for row in rows
            if row["query_id"].startswith("AQT")
            and row["context"] in traps
        ],
        "target_control": [
            row
            for row in rows
            if row["query_id"].startswith("AQT")
            and row["context"] not in traps
        ],
    }


def _b_partitions(rows):
    return {
        "all": rows,
        "pre_shift": [row for row in rows if row["query_time"] == 35],
        "near_b": [row for row in rows if row["query_time"] in {47, 52}],
        "settled_b": [row for row in rows if row["query_time"] == 70],
        "near_c": [row for row in rows if row["query_time"] in {87, 92}],
        "settled_c": [row for row in rows if row["query_time"] == 110],
        "near_recurrence": [
            row for row in rows if row["query_time"] in {127, 132}
        ],
        "settled_recurrence": [
            row for row in rows if row["query_time"] == 160
        ],
    }


def _run_a(world: HiddenWorld):
    loop = ClosedArticulationLoop()
    rows, candidates = _feed_query_stream(world, loop)
    return {
        "metrics": {
            name: _score(selected)
            for name, selected in _a_partitions(world, rows).items()
        },
        "provenance_groups": [
            sorted(group) for group in loop.provenance_groups()
        ],
        "memory": loop.snapshot.__dict__ if hasattr(loop.snapshot, "__dict__") else {
            "stored_basins": loop.snapshot.stored_basins,
            "active_basins": loop.snapshot.active_basins,
            "latent_basins": loop.snapshot.latent_basins,
            "coarsened_basins": loop.snapshot.coarsened_basins,
            "coarsened_mass": loop.snapshot.coarsened_mass,
        },
        "bubble_candidates": len(candidates),
    }


def _run_b(world: HiddenWorld):
    loop = ClosedArticulationLoop()
    rows, candidates = _feed_query_stream(world, loop)
    historical_only = 0

    # Re-run once as a pure stream to count late evidence explicitly; this is
    # learner-visible diagnostic state, not an oracle signal.
    diagnostic = ClosedArticulationLoop()
    reactivations = 0
    for observation in sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    ):
        read = diagnostic.observe(
            EvidenceAtom(
                context=observation.context,
                relation=observation.observed_relation,
                event_time=observation.event_time,
                arrival_time=observation.arrival_time,
                provenance=observation.source_id,
                bootstrap=observation.bootstrap,
            )
        )
        historical_only += int(read.historical_only)
        reactivations += int(read.reactivated)

    return {
        "metrics": {
            name: _score(selected)
            for name, selected in _b_partitions(rows).items()
        },
        "historical_only_events": historical_only,
        "memory_reactivations": reactivations,
        "bubble_candidates": len(candidates),
    }


def _run_c(world: HiddenWorld):
    loop = ClosedArticulationLoop()
    rows, candidates = _feed_query_stream(world, loop)
    answerable = [row for row in rows if row["answerable"]]
    compound = [
        candidate
        for candidate in candidates
        if len(candidate.changes) >= 2
    ]
    multi_subject = [
        candidate
        for candidate in compound
        if len({change.subject for change in candidate.changes}) >= 2
    ]
    return {
        "local_relation": _score(answerable),
        "bubble_candidates": len(candidates),
        "compound_candidates": len(compound),
        "multi_subject_candidates": len(multi_subject),
        "ambiguous_query_count": sum(not row["answerable"] for row in rows),
        "component_labels_emitted": False,
    }


def _run_d(world: HiddenWorld):
    policy = ClosedLoopPolicy(active_budget=64, latent_budget=192)
    loop = ClosedArticulationLoop(policy)
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    last = {}
    for observation in observations:
        read = loop.observe(
            EvidenceAtom(
                context=observation.context,
                relation=observation.observed_relation,
                event_time=observation.event_time,
                arrival_time=observation.arrival_time,
                provenance=observation.source_id,
                bootstrap=observation.bootstrap,
            )
        )
        last[observation.context] = read

    rows = []
    for query in world.queries:
        read = last.get(query.context)
        executable = (
            read is not None
            and read.state is ArticulationState.EXECUTE
        )
        answer = read.accepted_relation if executable else None
        rows.append(
            {
                "query_id": query.query_id,
                "answerable": query.answerable,
                "expected": query.expected_relation,
                "answer": answer,
                "resolved": answer is not None,
                "correct": answer == query.expected_relation,
            }
        )

    snap = loop.snapshot
    return {
        "metrics": _score(rows),
        "stored_basins": snap.stored_basins,
        "active_basins": snap.active_basins,
        "latent_basins": snap.latent_basins,
        "coarsened_basins": snap.coarsened_basins,
        "coarsened_mass": snap.coarsened_mass,
        "budget": policy.total_budget,
        "hard_failure": False,
    }


def run_world(world: HiddenWorld):
    if world.stage == "A":
        result = _run_a(world)
    elif world.stage == "B":
        result = _run_b(world)
    elif world.stage == "C":
        result = _run_c(world)
    elif world.stage == "D":
        result = _run_d(world)
    else:
        raise ValueError("probe is restricted to OWR-A through OWR-D")
    return {
        "stage": world.stage,
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "result": result,
    }


def _pool_metric(rows, path):
    selected = []
    for row in rows:
        value = row["result"]
        for key in path:
            value = value[key]
        selected.append(value)
    queries = sum(item["queries"] for item in selected)
    resolved = sum(item["resolved"] for item in selected)
    correct = sum(item["correct"] for item in selected)
    wrong = sum(item["wrong"] for item in selected)
    return {
        "queries": queries,
        "resolved": resolved,
        "correct": correct,
        "wrong": wrong,
        "coverage": resolved / max(1, queries),
        "resolved_accuracy": correct / max(1, resolved),
        "accuracy": correct / max(1, queries),
        "selective_risk": wrong / max(1, resolved),
    }


def campaign(worlds_per_stage: int, material: str):
    worlds = build_suite(
        material=material,
        count=worlds_per_stage,
        stages=("A", "B", "C", "D"),
        scale="development",
    )
    rows = [run_world(world) for world in worlds]
    by_stage = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage in ("A", "B", "C", "D")
    }

    aggregate = {
        "A": {
            part: _pool_metric(by_stage["A"], ("metrics", part))
            for part in (
                "all",
                "calibration",
                "target",
                "target_trap",
                "target_control",
            )
        },
        "B": {
            part: _pool_metric(by_stage["B"], ("metrics", part))
            for part in (
                "all",
                "pre_shift",
                "near_b",
                "settled_b",
                "near_c",
                "settled_c",
                "near_recurrence",
                "settled_recurrence",
            )
        },
        "C": {
            "local_relation": _pool_metric(
                by_stage["C"],
                ("local_relation",),
            ),
            "worlds_with_compound_candidate": sum(
                row["result"]["compound_candidates"] > 0
                for row in by_stage["C"]
            ),
            "worlds_with_multi_subject_candidate": sum(
                row["result"]["multi_subject_candidates"] > 0
                for row in by_stage["C"]
            ),
        },
        "D": {
            "query": _pool_metric(by_stage["D"], ("metrics",)),
            "hard_failure_rate": sum(
                row["result"]["hard_failure"] for row in by_stage["D"]
            ) / max(1, len(by_stage["D"])),
            "mean_stored_basins": sum(
                row["result"]["stored_basins"] for row in by_stage["D"]
            ) / max(1, len(by_stage["D"])),
            "mean_coarsened_basins": sum(
                row["result"]["coarsened_basins"] for row in by_stage["D"]
            ) / max(1, len(by_stage["D"])),
        },
    }
    return {
        "protocol": PROBE,
        "parent_protocol": PROTOCOL,
        "mode": "development-probe",
        "policy": ClosedLoopPolicy().__dict__ if hasattr(ClosedLoopPolicy(), "__dict__") else {
            "active_budget": ClosedLoopPolicy().active_budget,
            "latent_budget": ClosedLoopPolicy().latent_budget,
            "utility_decay": ClosedLoopPolicy().utility_decay,
            "dependence_min_overlap": ClosedLoopPolicy().dependence_min_overlap,
            "dependence_agreement": ClosedLoopPolicy().dependence_agreement,
            "episode_min_completed": ClosedLoopPolicy().episode_min_completed,
            "episode_separation_ratio": ClosedLoopPolicy().episode_separation_ratio,
            "recurrence_min_history": ClosedLoopPolicy().recurrence_min_history,
            "recurrence_confirmations": ClosedLoopPolicy().recurrence_confirmations,
            "residual_window": ClosedLoopPolicy().residual_window,
            "compound_min_events": ClosedLoopPolicy().compound_min_events,
            "late_event_tolerance": ClosedLoopPolicy().late_event_tolerance,
        },
        "worlds_per_stage": worlds_per_stage,
        "aggregate": aggregate,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument(
        "--material",
        default="closed-articulation-loop-v0.1-development-probe",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = campaign(args.worlds, args.material)
    rendered = json.dumps(report, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
