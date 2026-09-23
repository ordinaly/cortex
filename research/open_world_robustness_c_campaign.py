#!/usr/bin/env python3
"""OWR-C development characterization: compound novelty.

The frozen temporal Cortex is evaluated for its local relation state only; it
has no current API for emitting a set of simultaneous structural causes.
A public-data heuristic is included only to verify that the revised fixture
contains enough learner-visible signal to separate the declared cases.
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

PROTOCOL_C = "open-world-robustness-v1-owr-c-development"


def _dominant(values):
    counts = Counter(values)
    if not counts:
        return None
    best = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best)
    return winners[0] if len(winners) == 1 else None


def _current_cortex(world: HiddenWorld):
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    observations = sorted(
        world.observations,
        key=lambda row: (row.arrival_time, row.event_time, row.event_id),
    )
    rows = []
    cursor = 0
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        while cursor < len(observations) and observations[cursor].arrival_time <= query.query_time:
            observation = observations[cursor]
            memory.observe(
                TemporalObservation(
                    context=observation.context,
                    relation=observation.observed_relation,
                    source=observation.source_id,
                    source_reliability=1.0,
                    bootstrap=observation.bootstrap,
                )
            )
            cursor += 1
        snap = memory.snapshot(query.context)
        executable = snap is not None and snap.state is ArticulationState.EXECUTE
        answer = snap.accepted_relation if executable else None
        rows.append({
            "query_id": query.query_id,
            "answerable": query.answerable,
            "expected_relation": query.expected_relation,
            "local_answer": answer,
            "local_state": (
                snap.state.value if snap is not None else ArticulationState.EXPLORE.value
            ),
            # Frozen Cortex currently exposes no compound novelty component set
            # from this temporal surface.
            "structural_components": None,
        })
    return rows


def _public_component_baseline(world: HiddenWorld):
    observations = sorted(
        world.observations,
        key=lambda row: (row.event_time, row.event_id),
    )
    prelude = [row for row in observations if row.event_time < 40]
    known_entities = {row.context[0] for row in prelude}
    known_predicates = {row.context[1] for row in prelude}
    prelude_values = defaultdict(list)
    for row in prelude:
        if row.observed_relation is not None:
            prelude_values[row.context].append(row.observed_relation)
    baseline_relation = {
        context: _dominant(values) for context, values in prelude_values.items()
    }

    rows = []
    for query in sorted(world.queries, key=lambda row: (row.query_time, row.query_id)):
        available = [
            row for row in observations
            if row.event_time <= query.query_time and row.observed_relation is not None
        ]
        by_context = defaultdict(list)
        for row in available:
            by_context[row.context].append(row.observed_relation)
        current = {
            context: _dominant(values[-12:])
            for context, values in by_context.items()
        }
        entity, predicate = query.context
        relation = current.get(query.context)
        components = set()
        unresolved = False

        if predicate not in known_predicates:
            components.add("new_relation")

        if entity not in known_entities:
            peers = [
                context
                for context, candidate in current.items()
                if context != query.context
                and context[0] not in known_entities
                and context[1] == predicate
                and candidate == relation
            ]
            if peers:
                components.add("new_entity_type")

            if predicate in known_predicates and relation is not None:
                coherent_changed = [
                    context
                    for context, candidate in current.items()
                    if context[0] in known_entities
                    and context[1] == predicate
                    and baseline_relation.get(context) not in (None, candidate)
                    and candidate == relation
                ]
                if len(coherent_changed) >= 2:
                    unresolved = True
                    components.clear()
        else:
            prior = baseline_relation.get(query.context)
            if prior is not None and relation is not None and prior != relation:
                coherent_changed = [
                    context
                    for context, candidate in current.items()
                    if context[0] in known_entities
                    and context[1] == predicate
                    and baseline_relation.get(context) not in (None, candidate)
                    and candidate == relation
                ]
                if len(coherent_changed) >= 2:
                    components.add("new_regime")

        rows.append({
            "query_id": query.query_id,
            "components": tuple(sorted(components)),
            "unresolved": unresolved,
        })
    return rows


def run_world(world: HiddenWorld):
    if world.stage != "C":
        raise ValueError("OWR-C campaign requires stage C")
    cortex_rows = _current_cortex(world)
    baseline_rows = _public_component_baseline(world)
    expected = world.oracle["expected_components"]
    ambiguous = set(world.oracle["ambiguous_queries"])

    local_answerable = [row for row in cortex_rows if row["answerable"]]
    local_correct = sum(
        row["local_answer"] == row["expected_relation"]
        for row in local_answerable
    )
    local_resolved = sum(row["local_answer"] is not None for row in local_answerable)

    baseline_exact = 0
    baseline_ambiguous = 0
    for row in baseline_rows:
        query_id = row["query_id"]
        if query_id in ambiguous:
            baseline_ambiguous += int(row["unresolved"])
        else:
            baseline_exact += int(
                tuple(row["components"]) == tuple(sorted(expected[query_id]))
                and not row["unresolved"]
            )

    return {
        "seed": world.seed,
        "fingerprint": world.fingerprint(),
        "current_cortex": {
            "local_relation_queries": len(local_answerable),
            "local_relation_resolved": local_resolved,
            "local_relation_correct": local_correct,
            "local_relation_coverage": local_resolved / max(1, len(local_answerable)),
            "local_relation_accuracy": local_correct / max(1, len(local_answerable)),
            "compound_component_output_supported": False,
            "compound_component_coverage": 0.0,
            "rows": cortex_rows,
        },
        "public_component_baseline": {
            "resolved_cases": len(world.queries) - len(ambiguous),
            "resolved_exact": baseline_exact,
            "ambiguous_cases": len(ambiguous),
            "ambiguous_unresolved": baseline_ambiguous,
            "rows": baseline_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--material", default="owr-c-development-characterization-v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    worlds = build_suite(
        material=args.material,
        count=args.worlds,
        stages=("C",),
        scale="development",
    )
    results = [run_world(world) for world in worlds]
    current_answerable = sum(
        row["current_cortex"]["local_relation_queries"] for row in results
    )
    current_resolved = sum(
        row["current_cortex"]["local_relation_resolved"] for row in results
    )
    current_correct = sum(
        row["current_cortex"]["local_relation_correct"] for row in results
    )
    baseline_cases = sum(
        row["public_component_baseline"]["resolved_cases"] for row in results
    )
    baseline_exact = sum(
        row["public_component_baseline"]["resolved_exact"] for row in results
    )
    ambiguous_cases = sum(
        row["public_component_baseline"]["ambiguous_cases"] for row in results
    )
    ambiguous_unresolved = sum(
        row["public_component_baseline"]["ambiguous_unresolved"] for row in results
    )
    payload = {
        "protocol": PROTOCOL_C,
        "parent_protocol": PROTOCOL,
        "fixture_revision": "schema-outcome-separated-v2",
        "mode": "development-characterization",
        "worlds": len(worlds),
        "current_cortex": {
            "local_relation_coverage": current_resolved / max(1, current_answerable),
            "local_relation_accuracy": current_correct / max(1, current_answerable),
            "compound_component_output_supported": False,
            "compound_component_coverage": 0.0,
        },
        "public_component_baseline": {
            "resolved_exact_rate": baseline_exact / max(1, baseline_cases),
            "ambiguous_unresolved_rate": ambiguous_unresolved / max(1, ambiguous_cases),
        },
        "world_results": results,
    }
    rendered = json.dumps(payload, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
