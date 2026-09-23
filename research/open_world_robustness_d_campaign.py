#!/usr/bin/env python3
"""OWR-D development characterization: memory pressure and interference.

The campaign keeps Cortex behavior frozen. It combines the current temporal
regime surface with the actual bounded native knowledge runtime so capacity
exhaustion is measured where it really occurs rather than in an unbounded
Python-side dictionary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cortex import CortexRuntime
from cortex.articulation_controller import ArticulationState
from cortex.autonomy_campaign import KnowledgeTimeline
from cortex.autonomy_robustness import TemporalObservation
from cortex.autonomy_temporal_phase_b import (
    PhaseBTemporalMemory,
    TemporalRobustnessPolicy,
)
from cortex.knowledge import KnowledgeStatement, make_knowledge_runtime
from open_world_robustness_world import PROTOCOL, HiddenWorld, build_suite

PROTOCOL_D = "open-world-robustness-v1-owr-d-development"
PROPERTY = "owr-d-observed-relation"


def _feed_temporal(memory, observation):
    memory.observe(
        TemporalObservation(
            context=observation.context,
            relation=observation.observed_relation,
            source=observation.source_id,
            source_reliability=1.0,
            bootstrap=observation.bootstrap,
        )
    )


def _temporal_metrics(memory, world: HiddenWorld):
    rows = []
    for query in world.queries:
        snap = memory.snapshot(query.context)
        executable = snap is not None and snap.state is ArticulationState.EXECUTE
        answer = snap.accepted_relation if executable else None
        rows.append({
            "query_id": query.query_id,
            "expected": query.expected_relation,
            "answer": answer,
            "resolved": answer is not None,
            "correct": answer == query.expected_relation,
            "recurrence": query.query_id == "DQREC",
        })
    resolved = [row for row in rows if row["resolved"]]
    correct = [row for row in resolved if row["correct"]]
    recurrence = next(row for row in rows if row["recurrence"])
    return {
        "queries": len(rows),
        "coverage": len(resolved) / max(1, len(rows)),
        "resolved_accuracy": len(correct) / max(1, len(resolved)),
        "exact_accuracy": len(correct) / max(1, len(rows)),
        "recurrence_resolved": recurrence["resolved"],
        "recurrence_correct": recurrence["correct"],
        "rows": rows,
    }


def _unbounded_temporal(world: HiddenWorld):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    for observation in sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    ):
        _feed_temporal(memory, observation)
    return {
        **_temporal_metrics(memory, world),
        "context_count": memory.context_count,
        "active_cells": memory.active_cells,
    }


def _bounded_native(world: HiddenWorld, max_entities: int, *, batch_size: int = 12):
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=max_entities,
        budget=64,
    )
    timeline = KnowledgeTimeline(runtime)
    temporal = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    processed = 0
    error = None

    for offset in range(0, len(observations), batch_size):
        batch = observations[offset:offset + batch_size]
        statements = tuple(
            KnowledgeStatement(
                statement_id=f"owr-d-{row.event_id}",
                subject=row.context[0],
                property=PROPERTY,
                value=str(row.observed_relation),
                provenance=PROTOCOL_D,
            )
            for row in batch
            if row.observed_relation is not None
        )
        try:
            if statements:
                timeline.feed(f"stream-{offset // batch_size}", statements)
        except Exception as exc:
            error = type(exc).__name__
            break
        for observation in batch:
            _feed_temporal(temporal, observation)
        processed += len(batch)

    try:
        active_entities = timeline.cortex_entities
    except Exception:
        active_entities = None

    metrics = _temporal_metrics(temporal, world)
    native_direct = {}
    if error is None:
        view = timeline.view()
        for query in world.queries:
            native_direct[query.query_id] = list(
                view.objects(query.context[0], PROPERTY)
            )

    return {
        "max_entities": max_entities,
        "processed_events": processed,
        "stream_events": len(observations),
        "processed_fraction": processed / max(1, len(observations)),
        "error": error,
        "graceful_capacity_signal": error is None,
        "active_entities": active_entities,
        "binding_conflicts": timeline.binding_conflicts,
        "temporal": metrics,
        "native_direct_objects": native_direct,
    }


def run_world(world: HiddenWorld, capacities):
    if world.stage != "D":
        raise ValueError("OWR-D campaign requires stage D")
    return {
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "unbounded_temporal_reference": _unbounded_temporal(world),
        "bounded_native": [
            _bounded_native(world, capacity)
            for capacity in capacities
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--material", default="owr-d-development-characterization-v1")
    parser.add_argument(
        "--capacities",
        default="512,2048,8192",
        help="comma-separated native max_entities sweep",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capacities = tuple(int(value) for value in args.capacities.split(",") if value)
    if not capacities or any(value < 64 for value in capacities):
        raise SystemExit("capacities must contain positive practical limits")

    worlds = build_suite(
        material=args.material,
        count=args.worlds,
        stages=("D",),
        scale="development",
    )
    results = [run_world(world, capacities) for world in worlds]

    capacity_summary = {}
    for capacity in capacities:
        rows = [
            next(item for item in result["bounded_native"] if item["max_entities"] == capacity)
            for result in results
        ]
        capacity_summary[str(capacity)] = {
            "worlds": len(rows),
            "hard_failure_rate": sum(row["error"] is not None for row in rows) / len(rows),
            "mean_processed_fraction": sum(row["processed_fraction"] for row in rows) / len(rows),
            "mean_query_coverage": sum(row["temporal"]["coverage"] for row in rows) / len(rows),
            "mean_resolved_accuracy": sum(
                row["temporal"]["resolved_accuracy"] for row in rows
            ) / len(rows),
            "recurrence_correct_rate": sum(
                row["temporal"]["recurrence_correct"] for row in rows
            ) / len(rows),
        }

    unbounded = [row["unbounded_temporal_reference"] for row in results]
    payload = {
        "protocol": PROTOCOL_D,
        "parent_protocol": PROTOCOL,
        "mode": "development-characterization",
        "worlds": len(worlds),
        "capacities": list(capacities),
        "unbounded_temporal_reference": {
            "mean_coverage": sum(row["coverage"] for row in unbounded) / len(unbounded),
            "mean_resolved_accuracy": sum(
                row["resolved_accuracy"] for row in unbounded
            ) / len(unbounded),
            "recurrence_correct_rate": sum(
                row["recurrence_correct"] for row in unbounded
            ) / len(unbounded),
            "mean_context_count": sum(row["context_count"] for row in unbounded) / len(unbounded),
            "mean_active_cells": sum(row["active_cells"] for row in unbounded) / len(unbounded),
        },
        "bounded_native": capacity_summary,
        "graceful_degradation_supported": all(
            row["error"] is None
            for result in results
            for row in result["bounded_native"]
        ),
        "world_results": results,
    }
    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
