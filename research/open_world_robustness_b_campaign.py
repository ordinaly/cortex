#!/usr/bin/env python3
"""OWR-B development characterization: temporal disorder and recurrence.

The campaign does not change Cortex. It compares the current arrival-order
Phase-B memory with public-data temporal baselines that use event timestamps and
duplicate provenance already present in the learner-facing stream.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
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


def _score_rows(rows):
    resolved = [row for row in rows if row["answer"] is not None]
    correct = [row for row in resolved if row["answer"] == row["expected"]]
    wrong = len(resolved) - len(correct)
    return {
        "queries": len(rows),
        "resolved": len(resolved),
        "correct": len(correct),
        "wrong": wrong,
        "coverage": len(resolved) / max(1, len(rows)),
        "accuracy": len(correct) / max(1, len(rows)),
        "resolved_accuracy": len(correct) / max(1, len(resolved)),
        "selective_risk": wrong / max(1, len(resolved)),
    }


def _query_phase(world, query):
    for qtime, phase in world.oracle["query_checkpoints"]:
        if int(qtime) == int(query.query_time):
            return str(phase)
    raise KeyError(query.query_time)


def _row(world, query, answer, state):
    return {
        "query_id": query.query_id,
        "query_time": query.query_time,
        "expected": query.expected_relation,
        "answer": answer,
        "state": state,
        "phase": _query_phase(world, query),
        "recurrence": query.query_time >= 125,
    }


def _partition_rows(rows):
    return {
        "all": list(rows),
        "boundary": [row for row in rows if row["phase"] == "boundary"],
        "early": [row for row in rows if row["phase"] == "early"],
        "settled": [row for row in rows if row["phase"] == "settled"],
        "stable": [row for row in rows if row["phase"] == "stable"],
        "recurrence": [row for row in rows if row["recurrence"]],
        "novel_regime": [row for row in rows if not row["recurrence"]],
    }


def _partition_metrics(rows):
    return {
        name: _score_rows(selected)
        for name, selected in _partition_rows(rows).items()
    }


def _cortex_answer(memory, context):
    snapshot = memory.snapshot(context)
    if snapshot is None:
        return None, ArticulationState.EXPLORE
    state = snapshot.state
    if (
        state is ArticulationState.EXECUTE
        and memory.missing_streak(context) > memory.policy.freshness_missing_grace
    ):
        state = ArticulationState.EXPLORE
    answer = snapshot.accepted_relation if state is ArticulationState.EXECUTE else None
    return answer, state


def _current_cortex_rows(world: HiddenWorld, *, dedupe: bool):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(
            source_weighting="unweighted",
            freshness_missing_grace=1,
        )
    )
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    queries = sorted(world.queries, key=lambda value: (value.query_time, value.query_id))
    cursor = 0
    rows = []

    for query in queries:
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            cursor += 1
            if dedupe and observation.duplicate_of is not None:
                continue
            memory.observe(
                TemporalObservation(
                    context=observation.context,
                    relation=observation.observed_relation,
                    source=observation.source_id,
                    source_reliability=1.0,
                )
            )

        answer, state = _cortex_answer(memory, query.context)
        rows.append(_row(world, query, answer, state.value))
    return rows


def _vote_answer(values):
    counts = Counter(values)
    if not counts:
        return None
    best = max(counts.values())
    winners = sorted(
        relation for relation, count in counts.items() if count == best
    )
    return winners[0] if len(winners) == 1 else None


def _temporal_vote_rows(
    world: HiddenWorld,
    *,
    use_event_time: bool,
    dedupe: bool,
    window: int = 5,
):
    arrived = defaultdict(list)
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    queries = sorted(world.queries, key=lambda value: (value.query_time, value.query_id))
    cursor = 0
    rows = []

    for query in queries:
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            cursor += 1
            if dedupe and observation.duplicate_of is not None:
                continue
            if observation.observed_relation is not None:
                arrived[observation.context].append(observation)

        values = list(arrived[query.context])
        if use_event_time:
            values.sort(
                key=lambda value: (
                    value.event_time,
                    value.arrival_time,
                    value.event_id,
                )
            )
        else:
            values.sort(
                key=lambda value: (
                    value.arrival_time,
                    value.event_time,
                    value.event_id,
                )
            )
        selected = [
            observation.observed_relation
            for observation in values[-window:]
            if observation.observed_relation is not None
        ]
        answer = _vote_answer(selected)
        rows.append(
            _row(
                world,
                query,
                answer,
                "event-time-vote" if use_event_time else "arrival-vote",
            )
        )
    return rows


def _oracle_rows(world: HiddenWorld):
    return [
        _row(world, query, query.expected_relation, "oracle-current-truth")
        for query in sorted(
            world.queries,
            key=lambda value: (value.query_time, value.query_id),
        )
    ]


def _transition_latencies(world: HiddenWorld, *, dedupe: bool):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(
            source_weighting="unweighted",
            freshness_missing_grace=1,
        )
    )
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    contexts = sorted({observation.context for observation in observations})
    schedule = tuple(world.oracle["regime_schedule"])
    transitions = [
        {
            "start": int(schedule[index][0]),
            "previous": schedule[index - 1][2],
            "target": schedule[index][2],
            "kind": "recurrence" if index == len(schedule) - 1 else "novel",
        }
        for index in range(1, len(schedule))
    ]
    states = {
        (transition["start"], context): {
            "precondition": None,
            "latency": None,
        }
        for transition in transitions
        for context in contexts
    }
    crossed = set()

    for observation in observations:
        for transition in transitions:
            start = transition["start"]
            if start not in crossed and observation.arrival_time >= start:
                for context in contexts:
                    answer, state = _cortex_answer(memory, context)
                    states[(start, context)]["precondition"] = (
                        state is ArticulationState.EXECUTE
                        and answer == transition["previous"]
                    )
                crossed.add(start)

        if dedupe and observation.duplicate_of is not None:
            continue
        memory.observe(
            TemporalObservation(
                context=observation.context,
                relation=observation.observed_relation,
                source=observation.source_id,
                source_reliability=1.0,
            )
        )

        for transition in transitions:
            start = transition["start"]
            record = states[(start, observation.context)]
            if (
                observation.arrival_time >= start
                and record["precondition"] is True
                and record["latency"] is None
            ):
                answer, state = _cortex_answer(memory, observation.context)
                if (
                    state is ArticulationState.EXECUTE
                    and answer == transition["target"]
                ):
                    record["latency"] = observation.arrival_time - start

    rows = []
    for transition in transitions:
        for context in contexts:
            record = states[(transition["start"], context)]
            rows.append(
                {
                    "start": transition["start"],
                    "kind": transition["kind"],
                    "context": context,
                    **record,
                }
            )
    return rows


def _latency_summary(rows):
    eligible = [row for row in rows if row["precondition"] is True]
    values = sorted(
        int(row["latency"])
        for row in eligible
        if row["latency"] is not None
    )
    return {
        "cases": len(rows),
        "eligible": len(eligible),
        "resolved": len(values),
        "precondition_rate": len(eligible) / max(1, len(rows)),
        "success_rate": len(values) / max(1, len(eligible)),
        "median": statistics.median(values) if values else None,
        "p95": (
            values[min(len(values) - 1, max(0, int(0.95 * len(values)) - 1))]
            if values
            else None
        ),
        "mean": statistics.mean(values) if values else None,
    }


def run_world(world: HiddenWorld):
    if world.stage != "B":
        raise ValueError("OWR-B campaign requires a stage-B world")

    methods = {
        "current_cortex_arrival_order": _current_cortex_rows(
            world, dedupe=False
        ),
        "current_cortex_public_dedupe": _current_cortex_rows(
            world, dedupe=True
        ),
        "arrival_recent_five_vote": _temporal_vote_rows(
            world,
            use_event_time=False,
            dedupe=False,
        ),
        "event_time_recent_five_vote": _temporal_vote_rows(
            world,
            use_event_time=True,
            dedupe=True,
        ),
        "oracle_current_truth": _oracle_rows(world),
    }
    latency_raw = {
        "current_cortex_arrival_order": _transition_latencies(
            world, dedupe=False
        ),
        "current_cortex_public_dedupe": _transition_latencies(
            world, dedupe=True
        ),
    }
    return {
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "metrics": {
            name: _partition_metrics(rows)
            for name, rows in methods.items()
        },
        "latency": {
            name: {
                "all": _latency_summary(rows),
                "novel": _latency_summary(
                    [row for row in rows if row["kind"] == "novel"]
                ),
                "recurrence": _latency_summary(
                    [row for row in rows if row["kind"] == "recurrence"]
                ),
            }
            for name, rows in latency_raw.items()
        },
    }


def _pooled_metric(results, method, partition):
    counts = {
        key: sum(
            row["metrics"][method][partition][key] for row in results
        )
        for key in ("queries", "resolved", "correct", "wrong")
    }
    return {
        **counts,
        "coverage": counts["resolved"] / max(1, counts["queries"]),
        "accuracy": counts["correct"] / max(1, counts["queries"]),
        "resolved_accuracy": counts["correct"] / max(1, counts["resolved"]),
        "selective_risk": counts["wrong"] / max(1, counts["resolved"]),
    }


def _pooled_latencies(results, method, partition):
    summaries = [row["latency"][method][partition] for row in results]
    # Per-world summaries retain enough aggregate information for rate/mean.
    cases = sum(row["cases"] for row in summaries)
    eligible = sum(row["eligible"] for row in summaries)
    resolved = sum(row["resolved"] for row in summaries)
    weighted_mean_numerator = sum(
        (row["mean"] or 0.0) * row["resolved"] for row in summaries
    )
    medians = [row["median"] for row in summaries if row["median"] is not None]
    p95s = [row["p95"] for row in summaries if row["p95"] is not None]
    return {
        "cases": cases,
        "eligible": eligible,
        "resolved": resolved,
        "precondition_rate": eligible / max(1, cases),
        "success_rate": resolved / max(1, eligible),
        "mean": weighted_mean_numerator / max(1, resolved),
        "median_of_world_medians": statistics.median(medians) if medians else None,
        "max_world_p95": max(p95s) if p95s else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument(
        "--material",
        default="owr-b-development-characterization-v1",
    )
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
        "boundary",
        "early",
        "settled",
        "stable",
        "recurrence",
        "novel_regime",
    )
    aggregate = {
        method: {
            partition: _pooled_metric(results, method, partition)
            for partition in partitions
        }
        for method in methods
    }
    latency = {
        method: {
            partition: _pooled_latencies(results, method, partition)
            for partition in ("all", "novel", "recurrence")
        }
        for method in (
            "current_cortex_arrival_order",
            "current_cortex_public_dedupe",
        )
    }

    payload = {
        "protocol": PROTOCOL_B,
        "parent_protocol": PROTOCOL,
        "mode": "development-characterization",
        "worlds": len(worlds),
        "event_time_supplied_in_public_stream": True,
        "duplicate_provenance_supplied_in_public_stream": True,
        "current_temporal_adapter_consumes_event_time": False,
        "current_temporal_adapter_consumes_duplicate_provenance": False,
        "aggregate": aggregate,
        "transition_latency": latency,
        "world_results": results,
    }
    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
