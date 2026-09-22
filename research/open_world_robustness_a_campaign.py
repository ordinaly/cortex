#!/usr/bin/env python3
"""OWR-A development characterization: hidden source reliability/dependence.

No Cortex mechanism is added here. The campaign measures today's source-blind
temporal surface against context-local baselines, a public-data dependence
baseline, and an evaluator-only oracle reference.
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
from open_world_robustness_world import PROTOCOL, HiddenWorld, build_suite

PROTOCOL_A = "open-world-robustness-v1-owr-a-development"


def _score_rows(rows):
    answerable = [row for row in rows if row["expected"] is not None]
    resolved = [row for row in answerable if row["answer"] is not None]
    correct = [row for row in resolved if row["answer"] == row["expected"]]
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


def _partition_rows(rows):
    return {
        "all": list(rows),
        "calibration": [
            row for row in rows if row["query_id"].startswith("AQC")
        ],
        "target": [
            row for row in rows if row["query_id"].startswith("AQT")
        ],
        "target_trap": [
            row
            for row in rows
            if row["query_id"].startswith("AQT") and row["trap"]
        ],
        "target_control": [
            row
            for row in rows
            if row["query_id"].startswith("AQT") and not row["trap"]
        ],
    }


def _partition_metrics(rows):
    return {
        name: _score_rows(selected)
        for name, selected in _partition_rows(rows).items()
    }


def _row(world, query, answer, state):
    trap_contexts = {tuple(value) for value in world.oracle["trap_contexts"]}
    return {
        "query_id": query.query_id,
        "expected": query.expected_relation,
        "answer": answer,
        "state": state,
        "trap": query.context in trap_contexts,
    }


def _current_cortex_rows(world: HiddenWorld):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
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
            memory.observe(
                TemporalObservation(
                    context=observation.context,
                    relation=observation.observed_relation,
                    source=observation.source_id,
                    # Hidden reliability must not leak into current Cortex.
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
        answer = snapshot.accepted_relation if executable else None
        state = (
            snapshot.state.value
            if snapshot is not None
            else ArticulationState.EXPLORE.value
        )
        rows.append(_row(world, query, answer, state))
    return rows


def _vote_rows(world: HiddenWorld, *, recent: int | None):
    by_context = defaultdict(list)
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    rows = []
    cursor = 0
    for query in sorted(world.queries, key=lambda value: (value.query_time, value.query_id)):
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
            top_count = max(counts.values())
            winners = sorted(
                relation for relation, count in counts.items() if count == top_count
            )
            if len(winners) == 1:
                answer = winners[0]
        rows.append(_row(world, query, answer, "vote"))
    return rows


def _source_claim_sequences(world: HiddenWorld, query_time: int):
    """Return chronological per-source claim sequences on calibration contexts."""
    calibration = {
        tuple(value) for value in world.oracle["calibration_contexts"]
    }
    sequences = defaultdict(list)
    for observation in sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    ):
        if observation.arrival_time > query_time:
            break
        if (
            observation.context in calibration
            and observation.observed_relation is not None
        ):
            sequences[observation.source_id].append(observation.observed_relation)
    return sequences


def _dependency_groups(world: HiddenWorld, query_time: int, threshold: float = 0.91):
    """Infer highly dependent source groups using public calibration agreement."""
    sequences = _source_claim_sequences(world, query_time)
    sources = sorted(sequences)
    parent = {source: source for source in sources}

    def find(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            pairs = list(zip(sequences[left], sequences[right]))
            if not pairs:
                continue
            agreement = sum(a == b for a, b in pairs) / len(pairs)
            if agreement >= threshold:
                union(left, right)

    groups = defaultdict(list)
    for source in sources:
        groups[find(source)].append(source)
    return tuple(tuple(sorted(group)) for group in groups.values())


def _dependence_collapsed_rows(world: HiddenWorld):
    """Public-data baseline: learn source dependence, then collapse copied votes.

    This baseline receives no hidden reliability, truth labels, or copy graph.
    It is intentionally simple: calibration agreement above a frozen development
    threshold creates one vote group; each group contributes at most one target
    vote after internally aggregating its member claims.
    """
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    rows = []
    cursor = 0
    claims = defaultdict(lambda: defaultdict(list))
    groups = None

    for query in sorted(world.queries, key=lambda value: (value.query_time, value.query_id)):
        while cursor < len(observations):
            observation = observations[cursor]
            if observation.arrival_time > query.query_time:
                break
            if observation.observed_relation is not None:
                claims[observation.context][observation.source_id].append(
                    observation.observed_relation
                )
            cursor += 1

        if groups is None:
            groups = _dependency_groups(world, query.query_time)

        group_votes = []
        for group in groups:
            member_votes = []
            for source in group:
                values = claims[query.context][source]
                if not values:
                    continue
                counts = Counter(values)
                best = max(counts.values())
                winners = sorted(
                    relation
                    for relation, count in counts.items()
                    if count == best
                )
                if len(winners) == 1:
                    member_votes.append(winners[0])
            if not member_votes:
                continue
            counts = Counter(member_votes)
            best = max(counts.values())
            winners = sorted(
                relation for relation, count in counts.items() if count == best
            )
            if len(winners) == 1:
                group_votes.append(winners[0])

        answer = None
        if group_votes:
            counts = Counter(group_votes)
            best = max(counts.values())
            winners = sorted(
                relation for relation, count in counts.items() if count == best
            )
            if len(winners) == 1:
                answer = winners[0]
        rows.append(_row(world, query, answer, "dependency-collapsed"))
    return rows


def _oracle_reliability(world: HiddenWorld, observation) -> float:
    sources = world.oracle["source_profiles"]
    drift = world.oracle["drift"]
    transfer = world.oracle["transfer_trap"]
    root = transfer["root_source"]

    if "transfer-trap" in observation.tags:
        if observation.source_id == root:
            return 0.02
        # Copy children are omitted by the oracle dependence collapse.
        profile = {
            source: (0.96 if index == 0 else 0.94 if index == 4 else 0.55)
            for index, source in enumerate(sorted(sources))
        }
        return float(profile.get(observation.source_id, 0.55))

    if "transfer-control" in observation.tags:
        # Exact values need not match the generator; these are evaluator-only
        # reliability references sufficient to rank independent evidence.
        if observation.source_id == root:
            return 0.86
        return 0.93

    family = next(
        tag for tag in observation.tags if tag in {"A", "B"}
    )
    reliability = float(sources[observation.source_id][family])
    if "source-drift" in observation.tags and observation.source_id == drift["source"]:
        reliability = float(drift["reliability"])
    return reliability


def _oracle_independent_rows(world: HiddenWorld):
    """Evaluator-only ceiling using hidden reliability and copy dependence."""
    vocab = tuple(world.oracle["relation_vocab"])
    copy_parent = dict(world.oracle["copy_parent"])
    copied_children = {
        source
        for source, parent in copy_parent.items()
        if source != parent
    }
    scores = defaultdict(lambda: defaultdict(float))
    observations = sorted(
        world.observations,
        key=lambda value: (value.arrival_time, value.event_time, value.event_id),
    )
    rows = []
    cursor = 0

    for query in sorted(world.queries, key=lambda value: (value.query_time, value.query_id)):
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
                scores[observation.context][candidate] += log(
                    max(likelihood, 1.0e-12)
                )

        context_scores = scores[query.context]
        answer = max(context_scores, key=context_scores.get) if context_scores else None
        rows.append(_row(world, query, answer, "oracle-reference"))
    return rows


def run_world(world: HiddenWorld):
    if world.stage != "A":
        raise ValueError("OWR-A campaign requires a stage-A world")
    methods = {
        "current_cortex_source_blind": _current_cortex_rows(world),
        "global_majority": _vote_rows(world, recent=None),
        "recent_five_vote": _vote_rows(world, recent=5),
        "public_dependency_collapsed_vote": _dependence_collapsed_rows(world),
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
        "resolved_accuracy": counts["correct"] / max(1, counts["resolved"]),
        "accuracy": counts["correct"] / max(1, counts["queries"]),
        "selective_risk": counts["wrong"] / max(1, counts["resolved"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument(
        "--material",
        default="owr-a-development-characterization-v2",
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
    partitions = ("all", "calibration", "target", "target_trap", "target_control")
    aggregate = {
        method: {
            partition: _pooled_metric(results, method, partition)
            for partition in partitions
        }
        for method in methods
    }

    payload = {
        "protocol": PROTOCOL_A,
        "parent_protocol": PROTOCOL,
        "fixture_revision": "fresh-transfer-targets-v2",
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
