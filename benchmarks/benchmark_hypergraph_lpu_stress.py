#!/usr/bin/env python3
"""Stress-test Cortex relational geometry and its LPU execution boundary."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import statistics
import time

from cortex import CortexRuntime
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import HypergraphCompositionLearner
from cortex.hypergraph_lpu import (
    EnumeratingRuleExecutor,
    LPURuleExecutor,
    raw_lpu_reachability,
)
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    feed_knowledge,
    make_knowledge_runtime,
)


PROTOCOL = "hypergeometry-lpu-stress-v0.2"


@dataclass(frozen=True)
class World:
    statements: tuple[KnowledgeStatement, ...]
    target_relation: str
    source: str
    target: str
    decoys: tuple[str, ...]
    training_prefix: str = "train-"


def _token_map(rng: Random, count: int) -> tuple[str, ...]:
    values = [f"R{i}" for i in range(count)]
    rng.shuffle(values)
    return tuple(values)


def make_world(
    *,
    seed: int,
    depth: int,
    width: int,
    contradiction_fraction: float,
) -> World:
    if depth < 2:
        raise ValueError("depth must be at least two")
    if width < 2:
        raise ValueError("width must be at least two")

    rng = Random((seed + 1) * 1009 + depth * 97 + width)
    tokens = _token_map(rng, depth + 3)
    signature = tokens[:depth]
    target_relation = tokens[depth]
    wrong_target_relation = tokens[depth + 1]
    decoy_relation = tokens[depth + 2]

    statements: list[KnowledgeStatement] = []
    serial = 0

    # Ten solved examples induce the composition. Contradictory examples use
    # the same path signature but a different direct target relation.
    training_examples = 10
    contradiction_count = round(training_examples * contradiction_fraction)
    for example in range(training_examples):
        nodes = [
            f"train-{seed}-{depth}-{example}-n{i}"
            for i in range(depth + 1)
        ]
        for step, relation in enumerate(signature):
            statements.append(
                KnowledgeStatement(
                    f"train-{serial}",
                    nodes[step],
                    relation,
                    nodes[step + 1],
                    provenance=PROTOCOL,
                )
            )
            serial += 1

        direct_relation = (
            wrong_target_relation
            if example < contradiction_count
            else target_relation
        )
        statements.append(
            KnowledgeStatement(
                f"train-{serial}",
                nodes[0],
                direct_relation,
                nodes[-1],
                provenance=PROTOCOL,
            )
        )
        serial += 1

    source = f"query-{seed}-{depth}-source"
    target = f"query-{seed}-{depth}-target"

    # Correct query geometry is a layered, fully-connected lattice. It has
    # width**(depth-1) valid paths but only O(depth*width) states.
    layers: list[tuple[str, ...]] = [(source,)]
    for layer in range(1, depth):
        layers.append(
            tuple(
                f"query-{seed}-{depth}-L{layer}-{column}"
                for column in range(width)
            )
        )
    layers.append((target,))

    for step, relation in enumerate(signature):
        for left in layers[step]:
            for right in layers[step + 1]:
                statements.append(
                    KnowledgeStatement(
                        f"query-{serial}",
                        left,
                        relation,
                        right,
                        provenance=PROTOCOL,
                    )
                )
                serial += 1

    # Three relation-agnostic decoy paths are genuinely reachable, but each
    # violates the learned relation signature at one position. This makes raw
    # LPU reachability unable to select the semantic target.
    decoys = []
    for decoy_index in range(3):
        nodes = [source]
        nodes.extend(
            f"decoy-{seed}-{depth}-{decoy_index}-n{i}"
            for i in range(1, depth)
        )
        decoy = f"decoy-{seed}-{depth}-{decoy_index}-target"
        nodes.append(decoy)
        decoys.append(decoy)
        bad_step = decoy_index % depth
        for step in range(depth):
            relation = (
                decoy_relation
                if step == bad_step
                else signature[step]
            )
            statements.append(
                KnowledgeStatement(
                    f"decoy-{serial}",
                    nodes[step],
                    relation,
                    nodes[step + 1],
                    provenance=PROTOCOL,
                )
            )
            serial += 1

    # Irrelevant noise uses the same relation vocabulary but disconnected
    # entities. This should not alter the held-out inference.
    all_relations = signature + (target_relation, wrong_target_relation)
    for noise in range(20):
        statements.append(
            KnowledgeStatement(
                f"noise-{serial}",
                f"noise-{seed}-{depth}-a{noise}",
                rng.choice(all_relations),
                f"noise-{seed}-{depth}-b{noise}",
                provenance=PROTOCOL,
            )
        )
        serial += 1

    rng.shuffle(statements)
    return World(
        statements=tuple(statements),
        target_relation=target_relation,
        source=source,
        target=target,
        decoys=tuple(decoys),
    )


def _build(world: World):
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=8192,
        budget=128,
    )
    report = feed_knowledge(runtime, world.statements)
    view = KnowledgeGraphView(
        report.snapshot_json,
        report.symbol_bindings,
        runtime.config_json(),
        report.statement_incidence,
    )
    all_edges = view.hyperedges()
    training = HypergraphGeometry(
        edge
        for edge in all_edges
        if edge.edge_id.startswith(world.training_prefix)
    )
    full = HypergraphGeometry(all_edges)
    return runtime, report, training, full


def _rules(
    training: HypergraphGeometry,
    *,
    depth: int,
):
    learner = HypergraphCompositionLearner(
        training,
        path_length=depth,
        minimum_examples=5,
        confidence_threshold=0.75,
        resolution=0.5,
    )
    return learner.fit()


def _median_us(fn, repeats: int = 5):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return result, statistics.median(samples)


def run_case(
    *,
    seed: int,
    depth: int,
    width: int,
    contradiction_fraction: float,
) -> dict:
    world = make_world(
        seed=seed,
        depth=depth,
        width=width,
        contradiction_fraction=contradiction_fraction,
    )
    runtime, report, training, full = _build(world)
    rules = _rules(training, depth=depth)
    target_rules = tuple(
        rule
        for rule in rules
        if rule.target_relation == world.target_relation
    )

    before_lpu = runtime.snapshot_json()
    enumerator = EnumeratingRuleExecutor(full, resolution=0.5)
    lpu = LPURuleExecutor(full, resolution=0.5)

    cortex_read, cortex_us = _median_us(
        lambda: enumerator.infer(
            world.source,
            world.target_relation,
            target_rules,
        )
    )
    lpu_read, lpu_us = _median_us(
        lambda: lpu.infer(
            world.source,
            world.target_relation,
            target_rules,
        )
    )
    after_lpu = runtime.snapshot_json()

    candidates = (world.target,) + world.decoys
    raw = raw_lpu_reachability(
        full,
        world.source,
        candidates,
        resolution=0.5,
    )

    expected_paths = width ** (depth - 1)
    cortex_targets = {row.target for row in cortex_read.candidates}
    lpu_targets = {row.target for row in lpu_read.candidates}

    witness_valid = all(
        witness.exact
        and len(witness.layered_path) == depth + 1
        and len(witness.hyperedge_ids) == depth
        and witness.layered_path[0] == (0, world.source)
        and witness.layered_path[-1] == (depth, witness.target)
        for witness in lpu_read.witnesses
    )

    return {
        "seed": seed,
        "depth": depth,
        "width": width,
        "contradiction_fraction": contradiction_fraction,
        "binding_conflicts": report.binding_conflicts,
        "target_rule_count": len(target_rules),
        "target_rule_confidence": (
            max(rule.confidence for rule in target_rules)
            if target_rules
            else 0.0
        ),
        "cortex_targets": sorted(cortex_targets),
        "lpu_targets": sorted(lpu_targets),
        "expected_target": world.target,
        "correct": (
            cortex_targets == {world.target}
            and lpu_targets == {world.target}
        ),
        "agreement": cortex_targets == lpu_targets,
        "raw_lpu_reachable": raw,
        "raw_lpu_ambiguous": all(raw.values()),
        "witness_valid": witness_valid,
        "memory_unchanged": before_lpu == after_lpu,
        "expected_matching_paths": expected_paths,
        "enumerated_completed_paths": cortex_read.completed_paths,
        "enumerating_expanded_states": cortex_read.expanded_states,
        "lpu_materialized_edges": lpu_read.materialized_edges,
        "lpu_expanded_states": lpu_read.expanded_states,
        "lpu_calls": lpu_read.lpu_calls,
        "cortex_median_us": cortex_us,
        "lpu_median_us": lpu_us,
        "time_ratio_cortex_over_lpu": (
            cortex_us / lpu_us if lpu_us > 0.0 else None
        ),
    }


def multivalue_case() -> dict:
    from cortex.hypergraph_inference import CompositionRule, RelationStep

    rule = CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=0.8,
    )
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge("m0", 0, 10, 1, "A", "R0", "B", 0.8),
            RelationalHyperedge("m1", 1, 11, 2, "B", "R1", "C", 0.8),
            RelationalHyperedge("m2", 1, 11, 3, "B", "R1", "D", 0.8),
        )
    )
    cortex = EnumeratingRuleExecutor(geometry, resolution=0.5).infer(
        "A", "RT", (rule,)
    )
    lpu = LPURuleExecutor(geometry, resolution=0.5).infer(
        "A", "RT", (rule,)
    )
    return {
        "cortex_targets": sorted(row.target for row in cortex.candidates),
        "lpu_targets": sorted(row.target for row in lpu.candidates),
        "witnesses": sorted(row.target for row in lpu.witnesses),
        "passed": (
            {row.target for row in cortex.candidates} == {"C", "D"}
            and {row.target for row in lpu.candidates} == {"C", "D"}
        ),
    }


def revision_case() -> dict:
    from cortex.hypergraph_inference import CompositionRule, RelationStep

    rule = CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=0.8,
    )
    original = (
        RelationalHyperedge("r0", 0, 10, 1, "A", "R0", "B", 0.8),
        RelationalHyperedge("r1", 1, 11, 2, "B", "R1", "C", 0.8),
    )
    before = LPURuleExecutor(
        HypergraphGeometry(original),
        resolution=0.5,
    ).infer("A", "RT", (rule,))
    revised = tuple(
        RelationalHyperedge(
            edge.edge_id,
            edge.subject_id,
            edge.property_id,
            edge.object_id,
            edge.subject,
            edge.property,
            edge.object,
            0.4 if edge.edge_id == "r1" else edge.support,
        )
        for edge in original
    )
    after = LPURuleExecutor(
        HypergraphGeometry(revised),
        resolution=0.5,
    ).infer("A", "RT", (rule,))
    return {
        "before": sorted(row.target for row in before.candidates),
        "after": sorted(row.target for row in after.candidates),
        "after_lpu_calls": after.lpu_calls,
        "passed": (
            {row.target for row in before.candidates} == {"C"}
            and not after.candidates
            and after.lpu_calls == 0
        ),
    }


def campaign(seeds: int, width: int) -> dict:
    clean = []
    noisy = []
    unresolved = []

    for seed in range(seeds):
        for depth in range(2, 6):
            clean.append(
                run_case(
                    seed=seed,
                    depth=depth,
                    width=width,
                    contradiction_fraction=0.0,
                )
            )
        noisy.append(
            run_case(
                seed=seed,
                depth=3,
                width=width,
                contradiction_fraction=0.2,
            )
        )
        unresolved.append(
            run_case(
                seed=seed,
                depth=3,
                width=width,
                contradiction_fraction=0.4,
            )
        )

    multi = multivalue_case()
    revision = revision_case()
    deepest = [row for row in clean if row["depth"] == 5]

    summary = {
        "protocol": PROTOCOL,
        "seeds": seeds,
        "width": width,
        "clean_cases": len(clean),
        "clean_accuracy": sum(row["correct"] for row in clean) / len(clean),
        "clean_agreement": sum(row["agreement"] for row in clean) / len(clean),
        "exact_witness_rate": (
            sum(row["witness_valid"] for row in clean) / len(clean)
        ),
        "memory_isolation_rate": (
            sum(row["memory_unchanged"] for row in clean) / len(clean)
        ),
        "raw_lpu_ambiguity_rate": (
            sum(row["raw_lpu_ambiguous"] for row in clean) / len(clean)
        ),
        "noise20_accuracy": (
            sum(row["correct"] for row in noisy) / len(noisy)
        ),
        "noise20_mean_rule_confidence": statistics.mean(
            row["target_rule_confidence"] for row in noisy
        ),
        "noise40_unresolved_rate": (
            sum(row["target_rule_count"] == 0 for row in unresolved)
            / len(unresolved)
        ),
        "depth5_mean_completed_paths": statistics.mean(
            row["enumerated_completed_paths"] for row in deepest
        ),
        "depth5_mean_lpu_edges": statistics.mean(
            row["lpu_materialized_edges"] for row in deepest
        ),
        "depth5_median_cortex_us": statistics.median(
            row["cortex_median_us"] for row in deepest
        ),
        "depth5_median_lpu_us": statistics.median(
            row["lpu_median_us"] for row in deepest
        ),
        "depth5_median_speed_ratio": statistics.median(
            row["time_ratio_cortex_over_lpu"] for row in deepest
        ),
        "multivalue": multi,
        "revision": revision,
        "all_zero_binding_conflicts": all(
            row["binding_conflicts"] == 0
            for row in clean + noisy + unresolved
        ),
    }

    gates = {
        "clean_correct": summary["clean_accuracy"] == 1.0,
        "executors_agree": summary["clean_agreement"] == 1.0,
        "exact_witnesses": summary["exact_witness_rate"] == 1.0,
        "lpu_isolation": summary["memory_isolation_rate"] == 1.0,
        "lpu_only_ambiguous": summary["raw_lpu_ambiguity_rate"] == 1.0,
        "moderate_contradiction_tolerated": summary["noise20_accuracy"] == 1.0,
        "high_contradiction_becomes_unresolved": (
            summary["noise40_unresolved_rate"] == 1.0
        ),
        "multivalue_preserved": multi["passed"],
        "revision_respected": revision["passed"],
        "path_multiplicity_compressed": (
            summary["depth5_mean_completed_paths"]
            >= width ** 4
            and summary["depth5_mean_lpu_edges"]
            < summary["depth5_mean_completed_paths"]
        ),
        "binding_stable": summary["all_zero_binding_conflicts"],
    }

    return {
        **summary,
        "gates": gates,
        "passed": all(gates.values()),
        "clean": clean,
        "noise20": noisy,
        "noise40": unresolved,
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
