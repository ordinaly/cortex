#!/usr/bin/env python3
"""OWR-A development characterization: hidden source reliability.

This campaign does not add source learning to Cortex. It measures the current
source-blind temporal surface against simple baselines and an evaluator-only
oracle reference on development worlds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import log
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
from open_world_robustness_world import (
    PROTOCOL,
    HiddenWorld,
    Query,
    build_suite,
)

PROTOCOL_A = "open-world-robustness-v1-owr-a-development"


def _score_rows(rows):
    answerable = [row for row in rows if row["expected"] is not None]
    resolved = [row for row in answerable if row["answer"] is not None]
    correct = [row for row in resolved if row["answer"] == row["expected"]]
    wrong = len(resolved) - len(correct)
    return {
        "queries": len(answerable),
        "coverage": len(resolved) / max(1, len(answerable)),
        "resolved_accuracy": len(correct) / max(1, len(resolved)),
        "accuracy": len(correct) / max(1, len(answerable)),
        "selective_risk": wrong / max(1, len(resolved)),
    }


def _partition_metrics(rows):
    result = {"all": _score_rows(rows)}
    partitions = {
        "pre": [row for row in rows if row["query_id"].startswith("AQP")],
        "final": [row for row in rows if row["query_id"].startswith("AQF")],
        "final_trap": [
            row
            for row in rows
            if row["query_id"].startswith("AQF") and row["trap"]
        ],
        "final_control": [
            row
            for row in rows
            if row["query_id"].startswith("AQF") and not row["trap"]
        ],
    }
    for name, selected in partitions.items():
        result[name] = _score_rows(selected)
    return result


def _current_cortex_rows(world: HiddenWorld):
    policy = TemporalRobustnessPolicy(source_weighting="unweighted")
    memory = PhaseBTemporalMemory(policy)
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    queries = sorted(world.queries, key=lambda row: (row.query_time, row.query_id))
    trap_contexts = {tuple(row) for row in world.oracle["trap_contexts"]}
    cursor = 0
    rows = []

    for query in queries:
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            memory.observe(
                TemporalObservation(
                    context=observation.context,
                    relation=observation.observed_relation,
                    source=observation.source_id,
                    # OWR-A hides reliability. Unit reliability intentionally
                    # characterizes the current source-blind policy.
                    source_reliability=1.0,
                    bootstrap=observation.bootstrap,
                )
            )
            cursor += 1

        snapshot = memory.snapshot(query.context)
        executable = (
            snapshot is not None
            and snapshot.state is ArticulationState.EXECUTE
        )
        answer = (
            snapshot.accepted_relation
            if executable and snapshot is not None
            else None
        )
        rows.append(
            {
                "query_id": query.query_id,
                "expected": query.expected_relation,
                "answer": answer,
                "state": (
                    snapshot.state.value
                    if snapshot is not None
                    else ArticulationState.EXPLORE.value
                ),
                "trap": query.context in trap_contexts,
            }
        )
    return rows


def _vote_rows(world: HiddenWorld, *, recent: int | None):
    trap_contexts = {tuple(row) for row in world.oracle["trap_contexts"]}
    by_context = defaultdict(list)
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    rows = []
    cursor = 0
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            if observation.observed_relation is not None:
                by_context[observation.context].append(observation.observed_relation)
            cursor += 1
        values = by_context[query.context]
        selected = values if recent is None else values[-recent:]
        counts = Counter(selected)
        answer = None
        if counts:
            top = counts.most_common()
            best = top[0][1]
            winners = sorted(value for value, count in top if count == best)
            answer = winners[0] if len(winners) == 1 else None
        rows.append(
            {
                "query_id": query.query_id,
                "expected": query.expected_relation,
                "answer": answer,
                "state": "vote",
                "trap": query.context in trap_contexts,
            }
        )
    return rows


def _oracle_reliability(world: HiddenWorld, observation) -> float:
    source_profiles = world.oracle["source_profiles"]
    context_key = "|".join(observation.context)
    family = world.oracle["context_family"][context_key]
    events_per_round = 12 * 5
    round_index = observation.event_time // events_per_round
    reliability = float(source_profiles[observation.source_id][family])

    drift = world.oracle["drift"]
    if (
        observation.source_id == drift["source"]
        and family == drift["family"]
        and round_index >= int(drift["after_round"])
    ):
        reliability = float(drift["reliability"])

    trap = world.oracle["copied_majority_trap"]
    trap_contexts = {tuple(row) for row in world.oracle["trap_contexts"]}
    if (
        observation.source_id == trap["root_source"]
        and observation.context in trap_contexts
        and round_index >= int(trap["after_round"])
    ):
        reliability = float(trap["root_reliability"])
    return reliability


def _oracle_independent_rows(world: HiddenWorld):
    """Evaluator reference using hidden reliabilities and copy dependence.

    Copy children are omitted; their root claim already represents that
    dependency cluster. This is intentionally unavailable to Cortex.
    """
    vocab = tuple(world.oracle["relation_vocab"])
    copy_parent = dict(world.oracle["copy_parent"])
    roots = {source for source, parent in copy_parent.items() if source == parent}
    copied_children = {
        source
        for source, parent in copy_parent.items()
        if source != parent
    }
    trap_contexts = {tuple(row) for row in world.oracle["trap_contexts"]}
    scores = defaultdict(lambda: defaultdict(float))
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    cursor = 0
    rows = []

    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            cursor += 1
            if (
                observation.observed_relation is None
                or observation.source_id in copied_children
            ):
                continue
            reliability = _oracle_reliability(world, observation)
            incorrect = (1.0 - reliability) / max(1, len(vocab) - 1)
            for candidate in vocab:
                likelihood = (
                    reliability
                    if observation.observed_relation == candidate
                    else incorrect
                )
                scores[observation.context][candidate] += log(max(likelihood, 1e-12))

        context_scores = scores[query.context]
        answer = (
            max(context_scores, key=context_scores.get)
            if context_scores
            else None
        )
        rows.append(
            {
                "query_id": query.query_id,
                "expected": query.expected_relation,
                "answer": answer,
                "state": "oracle-reference",
                "trap": query.context in trap_contexts,
            }
        )
    return rows


def run_world(world: HiddenWorld):
    if world.stage != "A":
        raise ValueError("OWR-A campaign requires a stage-A world")
    methods = {
        "current_cortex_source_blind": _current_cortex_rows(world),
        "global_majority": _vote_rows(world, recent=None),
        "recent_five_vote": _vote_rows(world, recent=5),
        "oracle_independent_source_reference": _oracle_independent_rows(world),
    }
    return {
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "metrics": {
            name: _partition_metrics(rows)
            for name, rows in methods.items()
        },
    }


def _mean_world_metric(results, method, partition, metric):
    values = [
        row["metrics"][method][partition][metric]
        for row in results
    ]
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument(
        "--material",
        default="owr-a-development-characterization-v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    worlds = build_suite(
        material=args.material,
        count=args.worlds,
        stages=("A",),
        scale="development",
    )
    results = [run_world(world) for world in worlds]
    methods = tuple(results[0]["metrics"])
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for partition in ("all", "pre", "final", "final_trap", "final_control"):
            aggregate[method][partition] = {
                metric: _mean_world_metric(
                    results, method, partition, metric
                )
                for metric in (
                    "coverage",
                    "resolved_accuracy",
                    "accuracy",
                    "selective_risk",
                )
            }

    payload = {
        "protocol": PROTOCOL_A,
        "parent_protocol": PROTOCOL,
        "mode": "development-characterization",
        "worlds": len(worlds),
        "source_reliability_supplied_to_cortex": False,
        "source_dependence_supplied_to_cortex": False,
        "source_reliability_estimation_exposed_by_current_cortex": False,
        "aggregate": aggregate,
        "world_results": results,
    }
    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
