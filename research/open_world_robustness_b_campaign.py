#!/usr/bin/env python3
"""OWR-B development characterization: temporal disorder and recurrence.

No Cortex mechanism is changed. The campaign measures the frozen Phase-B
temporal surface in arrival order, then compares it with public-metadata
baselines that deduplicate and/or replay arrived evidence by event time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cortex.articulation_controller import ArticulationState
from cortex.autonomy_robustness import TemporalObservation
from cortex.autonomy_temporal_phase_b import (
    PhaseBTemporalMemory,
    TemporalRobustnessPolicy,
)
from open_world_robustness_world import PROTOCOL, HiddenWorld, build_suite

PROTOCOL_B = "open-world-robustness-v1-owr-b-development"


def _truth_at(world: HiddenWorld, t: int) -> str:
    for start, end, relation in world.oracle["regime_schedule"]:
        if int(start) <= t <= int(end):
            return str(relation)
    raise ValueError(f"no regime at time {t}")


def _score(rows):
    resolved = [row for row in rows if row["answer"] is not None]
    correct = [row for row in resolved if row["answer"] == row["expected"]]
    wrong = len(resolved) - len(correct)
    return {
        "queries": len(rows),
        "resolved": len(resolved),
        "correct": len(correct),
        "wrong": wrong,
        "coverage": len(resolved) / max(1, len(rows)),
        "resolved_accuracy": len(correct) / max(1, len(resolved)),
        "accuracy": len(correct) / max(1, len(rows)),
        "selective_risk": wrong / max(1, len(resolved)),
    }


def _partitions(rows):
    return {
        "all": list(rows),
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


def _metrics(rows):
    return {name: _score(selected) for name, selected in _partitions(rows).items()}


def _answer(memory: PhaseBTemporalMemory, context):
    snap = memory.snapshot(context)
    if snap is None:
        return None, ArticulationState.EXPLORE.value
    suspended = (
        snap.state is ArticulationState.EXECUTE
        and memory.missing_streak(context) > memory.policy.freshness_missing_grace
    )
    state = ArticulationState.EXPLORE if suspended else snap.state
    return (
        snap.accepted_relation if state is ArticulationState.EXECUTE else None,
        state.value,
    )


def _feed(memory, observation):
    memory.observe(
        TemporalObservation(
            context=observation.context,
            relation=observation.observed_relation,
            source=observation.source_id,
            source_reliability=1.0,
            bootstrap=observation.bootstrap,
        )
    )


def _current_cortex_rows(world: HiddenWorld):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    cursor = 0
    rows = []
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        while cursor < len(observations) and observations[cursor].arrival_time <= query.query_time:
            _feed(memory, observations[cursor])
            cursor += 1
        answer, state = _answer(memory, query.context)
        rows.append({
            "query_id": query.query_id,
            "query_time": query.query_time,
            "context": query.context,
            "expected": query.expected_relation,
            "answer": answer,
            "state": state,
        })
    return rows


def _event_time_replay_rows(world: HiddenWorld, *, deduplicate: bool):
    """Public-data reference: replay only arrived evidence by event time."""
    rows = []
    arrivals = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        available = [row for row in arrivals if row.arrival_time <= query.query_time]
        if deduplicate:
            roots = set()
            selected = []
            for row in available:
                root = row.duplicate_of or row.event_id
                if root in roots:
                    continue
                roots.add(root)
                selected.append(row)
            available = selected
        available.sort(key=lambda row: (row.event_time, row.arrival_time, row.event_id))
        memory = PhaseBTemporalMemory(
            TemporalRobustnessPolicy(source_weighting="unweighted")
        )
        for observation in available:
            _feed(memory, observation)
        answer, state = _answer(memory, query.context)
        rows.append({
            "query_id": query.query_id,
            "query_time": query.query_time,
            "context": query.context,
            "expected": query.expected_relation,
            "answer": answer,
            "state": state,
        })
    return rows


def _recent_vote_rows(world: HiddenWorld, *, deduplicate: bool, window: int = 7):
    by_context = defaultdict(list)
    rows = []
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    cursor = 0
    seen = set()
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        while cursor < len(observations) and observations[cursor].arrival_time <= query.query_time:
            observation = observations[cursor]
            cursor += 1
            root = observation.duplicate_of or observation.event_id
            if deduplicate and root in seen:
                continue
            seen.add(root)
            if observation.observed_relation is not None:
                by_context[observation.context].append(observation.observed_relation)
        values = by_context[query.context][-window:]
        counts = Counter(values)
        answer = None
        if counts:
            top = max(counts.values())
            winners = sorted(k for k, v in counts.items() if v == top)
            if len(winners) == 1:
                answer = winners[0]
        rows.append({
            "query_id": query.query_id,
            "query_time": query.query_time,
            "context": query.context,
            "expected": query.expected_relation,
            "answer": answer,
            "state": "vote",
        })
    return rows


def _switch_latencies(world: HiddenWorld):
    """Measure current Cortex adaptation in arrival time for each noninitial regime."""
    schedule = tuple(world.oracle["regime_schedule"])
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    unresolved = {
        (context, int(start)): {
            "relation": str(relation),
            "start": int(start),
            "latency": None,
        }
        for context in {row.context for row in observations}
        for start, _end, relation in schedule[1:]
    }
    for observation in observations:
        _feed(memory, observation)
        answer, _state = _answer(memory, observation.context)
        for (context, start), record in unresolved.items():
            if context != observation.context or record["latency"] is not None:
                continue
            if observation.arrival_time < start:
                continue
            if answer == record["relation"]:
                record["latency"] = observation.arrival_time - start
    by_start = defaultdict(list)
    for (_context, start), record in unresolved.items():
        if record["latency"] is not None:
            by_start[start].append(record["latency"])
    return {
        str(start): {
            "resolved_contexts": len(values),
            "mean_arrival_latency": sum(values) / len(values) if values else None,
            "max_arrival_latency": max(values) if values else None,
        }
        for start, values in sorted(by_start.items())
    }


def run_world(world: HiddenWorld):
    if world.stage != "B":
        raise ValueError("OWR-B campaign requires stage B")
    methods = {
        "current_cortex_arrival_order": _current_cortex_rows(world),
        "event_time_replay": _event_time_replay_rows(world, deduplicate=False),
        "event_time_replay_deduplicated": _event_time_replay_rows(world, deduplicate=True),
        "recent7_arrival_vote": _recent_vote_rows(world, deduplicate=False),
        "recent7_deduplicated_vote": _recent_vote_rows(world, deduplicate=True),
    }
    return {
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "metrics": {name: _metrics(rows) for name, rows in methods.items()},
        "switch_latencies": _switch_latencies(world),
    }


def _pool(results, method, partition):
    rows = [r["metrics"][method][partition] for r in results]
    counts = {
        key: sum(row[key] for row in rows)
        for key in ("queries", "resolved", "correct", "wrong")
    }
    return {
        **counts,
        "coverage": counts["resolved"] / max(1, counts["queries"]),
        "resolved_accuracy": counts["correct"] / max(1, counts["resolved"]),
        "accuracy": counts["correct"] / max(1, counts["queries"]),
        "selective_risk": counts["wrong"] / max(1, counts["resolved"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--material", default="owr-b-development-characterization-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    worlds = build_suite(
        material=args.material,
        count=args.worlds,
        stages=("B",),
        scale="development",
    )
    results = [run_world(world) for world in worlds]
    methods = tuple(results[0]["metrics"])
    partitions = (
        "all",
        "pre_shift",
        "near_b",
        "settled_b",
        "near_c",
        "settled_c",
        "near_recurrence",
        "settled_recurrence",
    )
    aggregate = {
        method: {
            partition: _pool(results, method, partition)
            for partition in partitions
        }
        for method in methods
    }
    latency_rows = defaultdict(list)
    for result in results:
        for start, values in result["switch_latencies"].items():
            if values["mean_arrival_latency"] is not None:
                latency_rows[start].append(values["mean_arrival_latency"])
    aggregate_latencies = {
        start: sum(values) / len(values)
        for start, values in sorted(latency_rows.items())
    }
    payload = {
        "protocol": PROTOCOL_B,
        "parent_protocol": PROTOCOL,
        "mode": "development-characterization",
        "fixture_revision": "boundary-collision-v2",
        "worlds": len(worlds),
        "cortex_uses_event_time": False,
        "cortex_deduplicates_public_duplicates": False,
        "aggregate": aggregate,
        "mean_switch_latency_by_regime_start": aggregate_latencies,
        "recurrence_comparison": {
            "novel_b_start": 45,
            "novel_c_start": 85,
            "recurrence_a_start": 125,
            "note": (
                "Recurrence is compared with prior regime switches, not first-ever "
                "learning, because the frozen tracker executes its first observation."
            ),
        },
        "world_results": results,
    }
    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
