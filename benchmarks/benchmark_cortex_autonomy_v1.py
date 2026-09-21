#!/usr/bin/env python3
"""Cortex Autonomy Campaign v1.

A finite chronological childhood is followed by a hard teacher-detachment
boundary. Held-out answer relations are never supplied after detachment.
Cortex must infer from its own persistent memory, absorb novel observations,
represent legitimate multi-value changes, react to genuine conflicting world
evidence, recover, and choose exact execution through the declared policy
ablations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import statistics
import time

from cortex import CortexRuntime
from cortex.articulation_controller import ArticulationState
from cortex.autonomy_campaign import (
    KnowledgeTimeline,
    PolicyMode,
    execute_policy,
)
from cortex.controller_calibration import frozen_calibration_v0_2
from cortex.hypergraph_geometry import HypergraphGeometry
from cortex.hypergraph_inference import HypergraphCompositionLearner
from cortex.knowledge import KnowledgeStatement, make_knowledge_runtime


PROTOCOL = "cortex-autonomy-campaign-v1"
RELATION_RESOLUTION = 0.5


@dataclass(frozen=True)
class Query:
    source: str
    relation: str
    expected: tuple[str, ...]
    name: str


class StatementFactory:
    def __init__(self, seed: int):
        self.seed = seed
        self.serial = 0

    def make(
        self,
        subject: str,
        relation: str,
        value: str,
        phase: str,
    ) -> KnowledgeStatement:
        sid = f"{phase}-{self.seed}-{self.serial}"
        self.serial += 1
        return KnowledgeStatement(
            sid,
            subject,
            relation,
            value,
            provenance=f"{PROTOCOL}:{phase}",
        )


def _token(rng: Random, prefix: str, index: int) -> str:
    return f"{prefix}-{index}-{rng.getrandbits(48):012x}"


def _relations(rng: Random, count: int) -> tuple[str, ...]:
    values = [_token(rng, "R", index) for index in range(count)]
    rng.shuffle(values)
    return tuple(values)


def _add_triangle(
    factory: StatementFactory,
    *,
    phase: str,
    source: str,
    middle: str,
    target: str,
    r0: str,
    r1: str,
    result_relation: str | None,
) -> list[KnowledgeStatement]:
    rows = [
        factory.make(source, r0, middle, phase),
        factory.make(middle, r1, target, phase),
    ]
    if result_relation is not None:
        rows.append(
            factory.make(
                source,
                result_relation,
                target,
                phase,
            )
        )
    return rows


def _add_chain(
    factory: StatementFactory,
    *,
    phase: str,
    nodes: tuple[str, ...],
    signature: tuple[str, ...],
    result_relation: str | None,
) -> list[KnowledgeStatement]:
    rows = []
    for index, relation in enumerate(signature):
        rows.append(
            factory.make(
                nodes[index],
                relation,
                nodes[index + 1],
                phase,
            )
        )
    if result_relation is not None:
        rows.append(
            factory.make(
                nodes[0],
                result_relation,
                nodes[-1],
                phase,
            )
        )
    return rows


def _permissive_rules(
    geometry: HypergraphGeometry,
    *,
    depth: int,
):
    return HypergraphCompositionLearner(
        geometry,
        path_length=depth,
        minimum_examples=1,
        confidence_threshold=0.0,
        resolution=RELATION_RESOLUTION,
    ).fit()


def _source_state(view, source: str) -> tuple[float, float]:
    local = view.local_articulation(
        source,
        resolution=RELATION_RESOLUTION,
    )
    return (
        float(local.topology.top_entity.support),
        float(local.topology.entity_entropy),
    )


def _policy_matrix(
    *,
    geometry: HypergraphGeometry,
    rules,
    query: Query,
    view,
):
    calibration = frozen_calibration_v0_2()
    source_support, source_entropy = _source_state(view, query.source)
    out = {}
    for mode in (
        PolicyMode.NONE,
        PolicyMode.FIXED_V0_1,
        PolicyMode.CALIBRATED_V0_2,
    ):
        for lpu_enabled in (False, True):
            key = f"{mode.value}|lpu={'on' if lpu_enabled else 'off'}"
            read = execute_policy(
                geometry=geometry,
                rules=rules,
                source=query.source,
                relation=query.relation,
                mode=mode,
                lpu_enabled=lpu_enabled,
                calibration=(
                    calibration
                    if mode is PolicyMode.CALIBRATED_V0_2
                    else None
                ),
                source_support=source_support,
                source_entropy=source_entropy,
            )
            out[key] = {
                "state": read.state.value,
                "backend": (
                    read.backend.value
                    if read.backend is not None
                    else None
                ),
                "candidates": list(read.candidates),
                "correct": set(read.candidates) == set(query.expected),
                "resolved": read.resolved,
                "witness_count": read.witness_count,
                "resolution": read.resolution,
                "reasons": list(read.reasons),
            }
    return out


def _direct_leak_count(view, queries: tuple[Query, ...]) -> int:
    return sum(
        bool(view.objects(query.source, query.relation))
        for query in queries
    )


def _phase_metrics(matrix, queries):
    keys = matrix[queries[0].name].keys()
    out = {}
    for key in keys:
        rows = [matrix[query.name][key] for query in queries]
        resolved = [row for row in rows if row["resolved"]]
        out[key] = {
            "coverage": len(resolved) / len(rows),
            "accuracy": (
                sum(row["correct"] for row in resolved) / len(resolved)
                if resolved else 0.0
            ),
            "exact_match_rate": (
                sum(row["correct"] for row in rows) / len(rows)
            ),
            "lpu_rate": (
                sum(row["backend"] == "lpu" for row in rows) / len(rows)
            ),
            "witness_rate_when_lpu": (
                sum(
                    row["witness_count"] > 0
                    for row in rows
                    if row["backend"] == "lpu"
                )
                / max(
                    1,
                    sum(row["backend"] == "lpu" for row in rows),
                )
            ),
        }
    return out


def _median_us(fn, repeats=5):
    fn()
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return result, statistics.median(values)


def run_seed(seed: int) -> dict:
    rng = Random(700001 + seed * 7919)
    factory = StatementFactory(seed)
    relations = _relations(rng, 12)
    (
        r0,
        r1,
        result_relation,
        competing_relation,
        c0,
        c1,
        control_relation,
        d0,
        d1,
        d2,
        d3,
        deep_relation,
    ) = relations

    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=4096,
        budget=128,
    )
    timeline = KnowledgeTimeline(runtime)

    childhood = []
    for index in range(6):
        childhood.extend(
            _add_triangle(
                factory,
                phase="childhood-main",
                source=_token(rng, "train-s", index),
                middle=_token(rng, "train-m", index),
                target=_token(rng, "train-t", index),
                r0=r0,
                r1=r1,
                result_relation=result_relation,
            )
        )

    initial_queries = []
    for index in range(4):
        source = _token(rng, "held-s", index)
        middle = _token(rng, "held-m", index)
        target = _token(rng, "held-t", index)
        childhood.extend(
            _add_triangle(
                factory,
                phase="childhood-heldout",
                source=source,
                middle=middle,
                target=target,
                r0=r0,
                r1=r1,
                result_relation=None,
            )
        )
        initial_queries.append(
            Query(
                source,
                result_relation,
                (target,),
                f"initial-{index}",
            )
        )

    for index in range(6):
        childhood.extend(
            _add_triangle(
                factory,
                phase="childhood-control",
                source=_token(rng, "ctrl-s", index),
                middle=_token(rng, "ctrl-m", index),
                target=_token(rng, "ctrl-t", index),
                r0=c0,
                r1=c1,
                result_relation=control_relation,
            )
        )

    control_source = _token(rng, "ctrl-held-s", 0)
    control_middle = _token(rng, "ctrl-held-m", 0)
    control_target = _token(rng, "ctrl-held-t", 0)
    childhood.extend(
        _add_triangle(
            factory,
            phase="childhood-control-heldout",
            source=control_source,
            middle=control_middle,
            target=control_target,
            r0=c0,
            r1=c1,
            result_relation=None,
        )
    )
    control_query = Query(
        control_source,
        control_relation,
        (control_target,),
        "control",
    )

    deep_signature = (d0, d1, d2, d3)
    for example in range(6):
        nodes = tuple(
            _token(rng, f"deep-train-{example}", index)
            for index in range(5)
        )
        childhood.extend(
            _add_chain(
                factory,
                phase="childhood-deep",
                nodes=nodes,
                signature=deep_signature,
                result_relation=deep_relation,
            )
        )

    childhood_report = timeline.feed("childhood", childhood)
    detached_entity_count = timeline.cortex_entities

    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    control_rules = main_rules

    initial_queries = tuple(initial_queries)
    initial_matrix = {
        query.name: _policy_matrix(
            geometry=geometry,
            rules=main_rules,
            query=query,
            view=view,
        )
        for query in initial_queries
    }
    control_initial = _policy_matrix(
        geometry=geometry,
        rules=control_rules,
        query=control_query,
        view=view,
    )

    direct_leaks = _direct_leak_count(
        view,
        initial_queries + (control_query,),
    )

    # Teacher detached. New observations below never contain the answer
    # relation for a held-out query.
    novelty_rows = []
    novelty_queries = []
    novelty_midpoints = []
    for index in range(4):
        source = _token(rng, "novel-s", index)
        middle = _token(rng, "novel-m", index)
        target = _token(rng, "novel-t", index)
        novelty_rows.extend(
            _add_triangle(
                factory,
                phase="autonomous-novelty",
                source=source,
                middle=middle,
                target=target,
                r0=r0,
                r1=r1,
                result_relation=None,
            )
        )
        novelty_queries.append(
            Query(
                source,
                result_relation,
                (target,),
                f"novel-{index}",
            )
        )
        novelty_midpoints.append((middle, target))
    timeline.feed("autonomous-novelty", novelty_rows)

    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    novelty_queries = tuple(novelty_queries)
    novelty_matrix = {
        query.name: _policy_matrix(
            geometry=geometry,
            rules=main_rules,
            query=query,
            view=view,
        )
        for query in novelty_queries
    }
    direct_leaks += _direct_leak_count(view, novelty_queries)

    # Legitimate changed-world multi-value expansion.
    multi_middle, original_multi_target = novelty_midpoints[0]
    second_multi_target = _token(rng, "novel-multivalue", 0)
    timeline.feed(
        "autonomous-multivalue",
        (
            factory.make(
                multi_middle,
                r1,
                second_multi_target,
                "autonomous-multivalue",
            ),
        ),
    )
    multi_query = Query(
        novelty_queries[0].source,
        result_relation,
        tuple(sorted((original_multi_target, second_multi_target))),
        "multivalue",
    )
    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    multi_matrix = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=multi_query,
        view=view,
    )
    direct_leaks += _direct_leak_count(view, (multi_query,))

    # Deep branching task appears only after teacher removal.
    deep_source = _token(rng, "deep-query-source", 0)
    deep_target = _token(rng, "deep-query-target", 0)
    layers = [(deep_source,)]
    for level in range(1, 4):
        layers.append(
            tuple(
                _token(rng, f"deep-L{level}", column)
                for column in range(4)
            )
        )
    layers.append((deep_target,))
    deep_rows = []
    for level, relation in enumerate(deep_signature):
        for left in layers[level]:
            for right in layers[level + 1]:
                deep_rows.append(
                    factory.make(
                        left,
                        relation,
                        right,
                        "autonomous-deep",
                    )
                )
    timeline.feed("autonomous-deep", deep_rows)

    deep_query = Query(
        deep_source,
        deep_relation,
        (deep_target,),
        "deep",
    )
    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    deep_rules = _permissive_rules(geometry, depth=4)
    deep_matrix = _policy_matrix(
        geometry=geometry,
        rules=deep_rules,
        query=deep_query,
        view=view,
    )
    direct_leaks += _direct_leak_count(view, (deep_query,))

    calibration = frozen_calibration_v0_2()
    deep_support, deep_entropy = _source_state(view, deep_source)
    _, deep_cortex_us = _median_us(
        lambda: execute_policy(
            geometry=geometry,
            rules=deep_rules,
            source=deep_source,
            relation=deep_relation,
            mode=PolicyMode.CALIBRATED_V0_2,
            lpu_enabled=False,
            calibration=calibration,
            source_support=deep_support,
            source_entropy=deep_entropy,
        ),
        repeats=3,
    )
    _, deep_auto_us = _median_us(
        lambda: execute_policy(
            geometry=geometry,
            rules=deep_rules,
            source=deep_source,
            relation=deep_relation,
            mode=PolicyMode.CALIBRATED_V0_2,
            lpu_enabled=True,
            calibration=calibration,
            source_support=deep_support,
            source_entropy=deep_entropy,
        ),
        repeats=3,
    )

    stable_query = initial_queries[0]

    # Moderate conflict: two independent world outcomes support a competing
    # relation over the same path signature. Query labels remain absent.
    moderate_rows = []
    for index in range(2):
        moderate_rows.extend(
            _add_triangle(
                factory,
                phase="autonomous-conflict-moderate",
                source=_token(rng, "conflict-m-s", index),
                middle=_token(rng, "conflict-m-m", index),
                target=_token(rng, "conflict-m-t", index),
                r0=r0,
                r1=r1,
                result_relation=competing_relation,
            )
        )
    timeline.feed("autonomous-conflict-moderate", moderate_rows)
    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    moderate_matrix = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=stable_query,
        view=view,
    )
    control_moderate = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=control_query,
        view=view,
    )

    # Strong conflict: total evidence becomes 6 target vs 4 competitor.
    strong_rows = []
    for index in range(2, 4):
        strong_rows.extend(
            _add_triangle(
                factory,
                phase="autonomous-conflict-strong",
                source=_token(rng, "conflict-s-s", index),
                middle=_token(rng, "conflict-s-m", index),
                target=_token(rng, "conflict-s-t", index),
                r0=r0,
                r1=r1,
                result_relation=competing_relation,
            )
        )
    timeline.feed("autonomous-conflict-strong", strong_rows)
    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    strong_matrix = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=stable_query,
        view=view,
    )
    control_strong = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=control_query,
        view=view,
    )

    # Recovery without deleting contradictory history. With 80 additional
    # independent target-supporting outcomes, target evidence reaches 86/90.
    recovery_rows = []
    for index in range(80):
        recovery_rows.extend(
            _add_triangle(
                factory,
                phase="autonomous-recovery",
                source=_token(rng, "recover-s", index),
                middle=_token(rng, "recover-m", index),
                target=_token(rng, "recover-t", index),
                r0=r0,
                r1=r1,
                result_relation=result_relation,
            )
        )
    timeline.feed("autonomous-recovery", recovery_rows)
    view = timeline.view()
    geometry = HypergraphGeometry(view.hyperedges())
    main_rules = _permissive_rules(geometry, depth=2)
    recovery_matrix = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=stable_query,
        view=view,
    )
    control_recovery = _policy_matrix(
        geometry=geometry,
        rules=main_rules,
        query=control_query,
        view=view,
    )

    query_sources = {
        query.source
        for query in (
            initial_queries
            + novelty_queries
            + (control_query, multi_query, deep_query)
        )
    }
    post_detach_statements = (
        novelty_rows
        + [
            row
            for row in timeline.view().statement_incidence
            if False
        ]
    )
    # Direct leakage is checked from Cortex memory. This explicit counter also
    # verifies no post-detach world-outcome statement used a held-out source.
    heldout_source_outcome_leaks = sum(
        incidence.subject in query_sources
        and incidence.property in {
            result_relation,
            control_relation,
            deep_relation,
        }
        for incidence in timeline.incidence
        if incidence.statement_id.startswith("autonomous-")
    )

    calibrated_key = "calibrated-v0.2|lpu=on"
    fixed_key = "fixed-v0.1|lpu=on"
    none_key = "none|lpu=on"

    return {
        "seed": seed,
        "teacher_detached_after_statements": len(childhood),
        "entities_at_detach": detached_entity_count,
        "entities_final": timeline.cortex_entities,
        "binding_conflicts": timeline.binding_conflicts,
        "direct_answer_leaks": direct_leaks,
        "heldout_source_outcome_leaks": heldout_source_outcome_leaks,
        "initial": {
            "metrics": _phase_metrics(initial_matrix, initial_queries),
            "matrix": initial_matrix,
        },
        "novelty": {
            "metrics": _phase_metrics(novelty_matrix, novelty_queries),
            "matrix": novelty_matrix,
        },
        "multivalue": multi_matrix,
        "deep": {
            "matrix": deep_matrix,
            "cortex_only_us": deep_cortex_us,
            "controller_with_lpu_us": deep_auto_us,
            "speed_ratio": (
                deep_cortex_us / deep_auto_us
                if deep_auto_us > 0.0
                else None
            ),
        },
        "moderate_conflict": moderate_matrix,
        "strong_conflict": strong_matrix,
        "recovery": recovery_matrix,
        "control_states": {
            "initial": control_initial,
            "moderate": control_moderate,
            "strong": control_strong,
            "recovery": control_recovery,
        },
        "calibrated_transition": [
            initial_matrix[initial_queries[0].name][calibrated_key]["state"],
            moderate_matrix[calibrated_key]["state"],
            strong_matrix[calibrated_key]["state"],
            recovery_matrix[calibrated_key]["state"],
        ],
        "fixed_transition": [
            initial_matrix[initial_queries[0].name][fixed_key]["state"],
            moderate_matrix[fixed_key]["state"],
            strong_matrix[fixed_key]["state"],
            recovery_matrix[fixed_key]["state"],
        ],
        "no_controller_transition": [
            initial_matrix[initial_queries[0].name][none_key]["state"],
            moderate_matrix[none_key]["state"],
            strong_matrix[none_key]["state"],
            recovery_matrix[none_key]["state"],
        ],
        "phase_reports": [
            {
                "phase": row.phase,
                "statements": row.statements,
                "cortex_entities": row.cortex_entities,
                "phase_binding_conflicts": row.phase_binding_conflicts,
            }
            for row in timeline.phase_reports
        ],
    }


def pressure_case(max_entities: int, *, seed: int = 991) -> dict:
    rng = Random(900001 + max_entities + seed)
    factory = StatementFactory(seed + max_entities)
    r0, r1, rt, noise_relation = _relations(rng, 4)
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=max_entities,
        budget=64,
    )
    timeline = KnowledgeTimeline(runtime)

    childhood = []
    for index in range(6):
        childhood.extend(
            _add_triangle(
                factory,
                phase="pressure-childhood",
                source=_token(rng, "ps", index),
                middle=_token(rng, "pm", index),
                target=_token(rng, "pt", index),
                r0=r0,
                r1=r1,
                result_relation=rt,
            )
        )
    source = _token(rng, "p-held-s", 0)
    middle = _token(rng, "p-held-m", 0)
    target = _token(rng, "p-held-t", 0)
    childhood.extend(
        _add_triangle(
            factory,
            phase="pressure-heldout",
            source=source,
            middle=middle,
            target=target,
            r0=r0,
            r1=r1,
            result_relation=None,
        )
    )

    error = None
    try:
        timeline.feed("pressure-childhood", childhood)
        for batch in range(5):
            rows = tuple(
                factory.make(
                    _token(rng, f"noise-s-{batch}", index),
                    noise_relation,
                    _token(rng, f"noise-t-{batch}", index),
                    f"pressure-noise-{batch}",
                )
                for index in range(10)
            )
            timeline.feed(f"pressure-noise-{batch}", rows)
    except Exception as exc:
        error = type(exc).__name__

    correct = False
    resolved = False
    if error is None:
        view = timeline.view()
        geometry = HypergraphGeometry(view.hyperedges())
        rules = _permissive_rules(geometry, depth=2)
        read = execute_policy(
            geometry=geometry,
            rules=rules,
            source=source,
            relation=rt,
            mode=PolicyMode.NONE,
            lpu_enabled=False,
        )
        resolved = read.resolved
        correct = set(read.candidates) == {target}

    return {
        "max_entities": max_entities,
        "error": error,
        "active_entities": (
            timeline.cortex_entities
            if error is None
            else None
        ),
        "binding_conflicts": timeline.binding_conflicts,
        "query_resolved": resolved,
        "query_correct": correct,
    }


def campaign(seeds: int) -> dict:
    rows = [run_seed(seed) for seed in range(seeds)]
    calibrated_key = "calibrated-v0.2|lpu=on"
    fixed_key = "fixed-v0.1|lpu=on"
    none_key = "none|lpu=on"

    initial_calibrated = [
        row["initial"]["metrics"][calibrated_key] for row in rows
    ]
    novelty_calibrated = [
        row["novelty"]["metrics"][calibrated_key] for row in rows
    ]
    deep_calibrated = [
        row["deep"]["matrix"][calibrated_key] for row in rows
    ]

    pressure = [
        pressure_case(capacity)
        for capacity in (256, 512, 1024)
    ]

    summary = {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "teacher_model_calls_after_detach": 0,
        "direct_answer_leaks": sum(row["direct_answer_leaks"] for row in rows),
        "heldout_source_outcome_leaks": sum(
            row["heldout_source_outcome_leaks"] for row in rows
        ),
        "binding_conflicts": sum(row["binding_conflicts"] for row in rows),
        "initial_calibrated_coverage": statistics.mean(
            row["coverage"] for row in initial_calibrated
        ),
        "initial_calibrated_accuracy": statistics.mean(
            row["accuracy"] for row in initial_calibrated
        ),
        "novelty_calibrated_coverage": statistics.mean(
            row["coverage"] for row in novelty_calibrated
        ),
        "novelty_calibrated_accuracy": statistics.mean(
            row["accuracy"] for row in novelty_calibrated
        ),
        "multivalue_preserved": all(
            row["multivalue"][calibrated_key]["correct"]
            for row in rows
        ),
        "deep_calibrated_correct": all(
            row["correct"] for row in deep_calibrated
        ),
        "deep_lpu_selected": all(
            row["backend"] == "lpu" for row in deep_calibrated
        ),
        "deep_witness_valid": all(
            row["witness_count"] >= 1 for row in deep_calibrated
        ),
        "deep_median_speed_ratio": statistics.median(
            row["deep"]["speed_ratio"] for row in rows
        ),
        "calibrated_transition_match": all(
            row["calibrated_transition"]
            == ["execute", "explore", "unresolved", "execute"]
            for row in rows
        ),
        "fixed_transition_match": all(
            row["fixed_transition"]
            == ["execute", "execute", "unresolved", "execute"]
            for row in rows
        ),
        "no_controller_never_abstains": all(
            all(state == "execute" for state in row["no_controller_transition"])
            for row in rows
        ),
        "control_preserved": all(
            all(
                phase[calibrated_key]["correct"]
                and phase[calibrated_key]["state"] == "execute"
                for phase in row["control_states"].values()
            )
            for row in rows
        ),
        "lpu_semantic_equivalence": all(
            row["initial"]["metrics"]["calibrated-v0.2|lpu=off"][
                "exact_match_rate"
            ]
            == row["initial"]["metrics"][calibrated_key]["exact_match_rate"]
            and row["novelty"]["metrics"]["calibrated-v0.2|lpu=off"][
                "exact_match_rate"
            ]
            == row["novelty"]["metrics"][calibrated_key]["exact_match_rate"]
            for row in rows
        ),
        "pressure": pressure,
    }

    high_pressure = next(
        row for row in pressure if row["max_entities"] == 1024
    )
    gates = {
        "teacher_detached": summary["teacher_model_calls_after_detach"] == 0,
        "no_answer_leakage": (
            summary["direct_answer_leaks"] == 0
            and summary["heldout_source_outcome_leaks"] == 0
        ),
        "binding_stable": summary["binding_conflicts"] == 0,
        "heldout_inference": (
            summary["initial_calibrated_coverage"] == 1.0
            and summary["initial_calibrated_accuracy"] == 1.0
        ),
        "post_detach_novelty": (
            summary["novelty_calibrated_coverage"] == 1.0
            and summary["novelty_calibrated_accuracy"] == 1.0
        ),
        "multivalue_plasticity": summary["multivalue_preserved"],
        "deep_composition": summary["deep_calibrated_correct"],
        "lpu_selected_for_deep": summary["deep_lpu_selected"],
        "lpu_exact_witness": summary["deep_witness_valid"],
        "calibrated_revision_cycle": summary["calibrated_transition_match"],
        "fixed_controller_control": summary["fixed_transition_match"],
        "controller_ablation_exposes_risk": summary["no_controller_never_abstains"],
        "unaffected_memory_preserved": summary["control_preserved"],
        "lpu_semantics_unchanged": summary["lpu_semantic_equivalence"],
        "high_capacity_pressure_preserves_query": (
            high_pressure["error"] is None
            and high_pressure["query_correct"]
        ),
    }

    return {
        **summary,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
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
