"""Official campaign generator for counterfactual-reasoning-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from counterfactual_reasoning import (
    CorrelationBaseline,
    CortexStructuralCounterfactual,
    DirectEdgeBaseline,
    Edge,
    Node,
    ground_truth_reachable,
    path_is_valid,
    shortest_path_length,
)

FAMILIES = (
    "chain",
    "fork",
    "collider",
    "diamond",
    "bridge",
    "disconnected",
    "random-dag",
)
OFFICIAL_SEEDS = tuple(range(100, 120))


@dataclass(frozen=True)
class World:
    family: str
    seed: int
    nodes: tuple[Node, ...]
    edges: frozenset[Edge]
    traps: frozenset[Edge]
    integer_to_label: dict[int, Node]
    special_bridge: Edge | None


def integer_edges(family: str, seed: int) -> set[tuple[int, int]]:
    if family == "chain":
        return {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        }
    if family == "fork":
        return {
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
        }
    if family == "collider":
        return {
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 4),
            (4, 6),
        }
    if family == "diamond":
        return {
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
        }
    if family == "bridge":
        return {
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
        }
    if family == "disconnected":
        return {
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (5, 6),
        }
    if family == "random-dag":
        rng = Random(seed * 1009 + 73)
        edges = {
            (0, 1),
            (1, 2),
        }
        for left in range(8):
            for right in range(left + 1, 8):
                if (left, right) in edges:
                    continue
                if rng.random() < 0.22:
                    edges.add((left, right))
        return edges
    raise ValueError(f"unknown family: {family}")


def build_world(family: str, seed: int) -> World:
    raw_edges = integer_edges(family, seed)
    rng = Random(seed * 10_007 + sum(ord(ch) for ch in family))
    labels = [f"N{index}" for index in range(8)]
    rng.shuffle(labels)
    mapping = {
        index: labels[index]
        for index in range(8)
    }
    edges = frozenset(
        (mapping[left], mapping[right])
        for left, right in raw_edges
    )

    non_edges = [
        (mapping[left], mapping[right])
        for left in range(8)
        for right in range(8)
        if left != right
        and (left, right) not in raw_edges
    ]
    rng.shuffle(non_edges)
    traps = frozenset(non_edges[:2])

    special_bridge = None
    if family == "bridge":
        special_bridge = (
            mapping[2],
            mapping[3],
        )

    return World(
        family=family,
        seed=seed,
        nodes=tuple(sorted(mapping.values())),
        edges=edges,
        traps=traps,
        integer_to_label=mapping,
        special_bridge=special_bridge,
    )


def populate_model(
    world: World,
    *,
    unresolved_edge: Edge | None = None,
) -> CortexStructuralCounterfactual:
    model = CortexStructuralCounterfactual(world.nodes)

    for source in world.nodes:
        for target in world.nodes:
            if source == target:
                continue
            edge = (source, target)

            if edge == unresolved_edge:
                model.observe_counts(
                    source,
                    target,
                    do_positive=4,
                    do_negative=0,
                    ctrl_positive=0,
                    ctrl_negative=4,
                )
            elif edge in world.edges:
                model.observe_counts(
                    source,
                    target,
                    do_positive=18,
                    do_negative=2,
                    ctrl_positive=2,
                    ctrl_negative=18,
                )
            elif edge in world.traps:
                model.observe_counts(
                    source,
                    target,
                    do_positive=16,
                    do_negative=4,
                    ctrl_positive=16,
                    ctrl_negative=4,
                )
            else:
                model.observe_counts(
                    source,
                    target,
                    do_positive=2,
                    do_negative=18,
                    ctrl_positive=2,
                    ctrl_negative=18,
                )

    return model


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_direct(world: World, model: CortexStructuralCounterfactual) -> dict:
    total = 0
    resolved = 0
    correct = 0
    causal_total = 0
    causal_correct = 0
    null_total = 0
    null_correct = 0
    trap_correct = 0

    correlation = CorrelationBaseline(model)
    correlation_trap_correct = 0

    for source in world.nodes:
        for target in world.nodes:
            if source == target:
                continue
            total += 1
            expected = 1 if (source, target) in world.edges else -1
            if expected == 1:
                causal_total += 1
            else:
                null_total += 1

            state = model.direct_state(source, target)
            if state != 0:
                resolved += 1
                correct += int(state == expected)

            if expected == 1:
                causal_correct += int(state == 1)
            else:
                null_correct += int(state == -1)

            if (source, target) in world.traps:
                trap_correct += int(state == -1)
                correlation_trap_correct += int(
                    correlation.direct_state(source, target) == -1
                )

    return {
        "resolved_precision": ratio(correct, resolved),
        "resolved_coverage": ratio(resolved, total),
        "causal_recall": ratio(causal_correct, causal_total),
        "null_recall": ratio(null_correct, null_total),
        "trap_rejection": ratio(trap_correct, len(world.traps)),
        "correlation_baseline_trap_accuracy": ratio(
            correlation_trap_correct,
            len(world.traps),
        ),
        "counts": {
            "total": total,
            "resolved": resolved,
            "correct": correct,
            "causal_total": causal_total,
            "causal_correct": causal_correct,
            "null_total": null_total,
            "null_correct": null_correct,
            "traps": len(world.traps),
            "trap_correct": trap_correct,
        },
    }


def score_multi_hop(
    world: World,
    model: CortexStructuralCounterfactual,
) -> dict:
    total = 0
    resolved = 0
    correct = 0
    proof_total = 0
    proof_valid = 0
    baseline_correct = 0
    baseline = DirectEdgeBaseline(model)
    confirmed = model.confirmed_edges()

    for source in world.nodes:
        for target in world.nodes:
            if source == target:
                continue
            length = shortest_path_length(
                source,
                target,
                world.edges,
            )
            if length is None or length < 2:
                continue
            if (source, target) in world.edges:
                continue

            total += 1
            result = model.query(source, target)
            if result.state != 0:
                resolved += 1
                correct += int(result.state == 1)
            baseline_correct += int(
                baseline.query(source, target) == 1
            )

            if result.state == 1:
                proof_total += 1
                proof_valid += int(
                    path_is_valid(
                        result.path,
                        source=source,
                        target=target,
                        active_edges=confirmed,
                    )
                )

    return {
        "accuracy": ratio(correct, total),
        "coverage": ratio(resolved, total),
        "resolved_accuracy": ratio(correct, resolved),
        "direct_edge_baseline_accuracy": ratio(
            baseline_correct,
            total,
        ),
        "proof_valid": proof_valid,
        "proof_total": proof_total,
        "counts": {
            "total": total,
            "resolved": resolved,
            "correct": correct,
            "baseline_correct": baseline_correct,
        },
    }


def score_negative(
    world: World,
    model: CortexStructuralCounterfactual,
) -> dict:
    total = 0
    correct = 0

    for source in world.nodes:
        for target in world.nodes:
            if source == target:
                continue
            if ground_truth_reachable(
                source,
                target,
                world.edges,
            ):
                continue
            total += 1
            result = model.query(source, target)
            correct += int(result.state == -1)

    return {
        "accuracy": ratio(correct, total),
        "counts": {
            "total": total,
            "correct": correct,
        },
    }


def score_counterfactuals(
    world: World,
    model: CortexStructuralCounterfactual,
) -> dict:
    total = 0
    resolved = 0
    correct = 0
    proof_total = 0
    proof_valid = 0
    restoration_total = 0
    restoration_correct = 0
    redundant_total = 0
    redundant_correct = 0
    bridge_cut_total = 0
    bridge_cut_correct = 0

    confirmed = model.confirmed_edges()

    for removed in sorted(world.edges):
        source = removed[0]
        for target in world.nodes:
            if target == source:
                continue
            if not ground_truth_reachable(
                source,
                target,
                world.edges,
            ):
                continue

            expected = 1 if ground_truth_reachable(
                source,
                target,
                world.edges,
                remove_edges=(removed,),
            ) else -1

            before = model.fingerprint()
            result = model.query(
                source,
                target,
                remove_edges=(removed,),
            )
            after = model.fingerprint()

            total += 1
            restoration_total += 1
            restoration_correct += int(before == after)
            if result.state != 0:
                resolved += 1
                correct += int(result.state == expected)

            if expected == 1:
                redundant_total += 1
                redundant_correct += int(result.state == 1)

            if result.state == 1:
                proof_total += 1
                proof_valid += int(
                    path_is_valid(
                        result.path,
                        source=source,
                        target=target,
                        active_edges=confirmed,
                        removed_edges=(removed,),
                    )
                )

    if world.family == "diamond":
        source = world.integer_to_label[0]
        targets = (
            world.integer_to_label[3],
            world.integer_to_label[4],
            world.integer_to_label[5],
        )
        removed_edges = (
            (
                world.integer_to_label[0],
                world.integer_to_label[1],
            ),
            (
                world.integer_to_label[0],
                world.integer_to_label[2],
            ),
        )
        for removed in removed_edges:
            for target in targets:
                before = model.fingerprint()
                result = model.query(
                    source,
                    target,
                    remove_edges=(removed,),
                )
                after = model.fingerprint()
                redundant_total += 1
                redundant_correct += int(result.state == 1)
                restoration_total += 1
                restoration_correct += int(before == after)
                if result.state == 1:
                    proof_total += 1
                    proof_valid += int(
                        path_is_valid(
                            result.path,
                            source=source,
                            target=target,
                            active_edges=confirmed,
                            removed_edges=(removed,),
                        )
                    )

    if world.family == "bridge":
        assert world.special_bridge is not None
        source = world.integer_to_label[0]
        targets = (
            world.integer_to_label[3],
            world.integer_to_label[4],
            world.integer_to_label[5],
            world.integer_to_label[6],
        )
        for target in targets:
            before = model.fingerprint()
            result = model.query(
                source,
                target,
                remove_edges=(world.special_bridge,),
            )
            after = model.fingerprint()
            bridge_cut_total += 1
            bridge_cut_correct += int(result.state == -1)
            restoration_total += 1
            restoration_correct += int(before == after)

    return {
        "accuracy": ratio(correct, total),
        "coverage": ratio(resolved, total),
        "resolved_accuracy": ratio(correct, resolved),
        "redundant_path_rate": (
            None
            if redundant_total == 0
            else ratio(redundant_correct, redundant_total)
        ),
        "bridge_cut_rate": (
            None
            if bridge_cut_total == 0
            else ratio(bridge_cut_correct, bridge_cut_total)
        ),
        "proof_valid": proof_valid,
        "proof_total": proof_total,
        "restoration_rate": ratio(
            restoration_correct,
            restoration_total,
        ),
        "counts": {
            "total": total,
            "resolved": resolved,
            "correct": correct,
            "redundant_total": redundant_total,
            "redundant_correct": redundant_correct,
            "bridge_cut_total": bridge_cut_total,
            "bridge_cut_correct": bridge_cut_correct,
            "proof_total": proof_total,
            "proof_valid": proof_valid,
            "restoration_total": restoration_total,
            "restoration_correct": restoration_correct,
        },
    }


def score_missing_bridge(world: World) -> dict | None:
    if world.family != "bridge":
        return None
    assert world.special_bridge is not None

    model = populate_model(
        world,
        unresolved_edge=world.special_bridge,
    )
    source = world.integer_to_label[0]
    targets = (
        world.integer_to_label[3],
        world.integer_to_label[4],
        world.integer_to_label[5],
        world.integer_to_label[6],
    )
    unresolved = 0
    for target in targets:
        result = model.query(source, target)
        unresolved += int(result.state == 0)

    return {
        "unresolved_rate": ratio(unresolved, len(targets)),
        "total": len(targets),
        "unresolved": unresolved,
    }


def run_world(family: str, seed: int) -> dict:
    world = build_world(family, seed)
    model = populate_model(world)

    direct = score_direct(world, model)
    multi_hop = score_multi_hop(world, model)
    negative = score_negative(world, model)
    counterfactual = score_counterfactuals(world, model)
    missing_bridge = score_missing_bridge(world)

    return {
        "protocol": "counterfactual-reasoning-v1",
        "family": family,
        "seed": seed,
        "nodes": len(world.nodes),
        "ground_truth_edges": len(world.edges),
        "correlation_traps": len(world.traps),
        "direct": direct,
        "multi_hop": multi_hop,
        "negative": negative,
        "counterfactual": counterfactual,
        "missing_bridge": missing_bridge,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def campaign(
    output: Path,
    *,
    seeds: Iterable[int] = OFFICIAL_SEEDS,
) -> dict:
    seeds = tuple(seeds)
    rows = [
        run_world(family, seed)
        for family in FAMILIES
        for seed in seeds
    ]

    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    special_redundant = [
        row["counterfactual"]["redundant_path_rate"]
        for row in rows
        if row["counterfactual"]["redundant_path_rate"]
        is not None
    ]
    special_bridge = [
        row["counterfactual"]["bridge_cut_rate"]
        for row in rows
        if row["counterfactual"]["bridge_cut_rate"]
        is not None
    ]
    missing = [
        row["missing_bridge"]["unresolved_rate"]
        for row in rows
        if row["missing_bridge"] is not None
    ]

    proof_valid = sum(
        row["multi_hop"]["proof_valid"]
        + row["counterfactual"]["proof_valid"]
        for row in rows
    )
    proof_total = sum(
        row["multi_hop"]["proof_total"]
        + row["counterfactual"]["proof_total"]
        for row in rows
    )

    summary = {
        "protocol": "counterfactual-reasoning-v1",
        "rows": len(rows),
        "families": list(FAMILIES),
        "seeds": list(seeds),
        "metrics": {
            "direct_resolved_precision": _mean(
                row["direct"]["resolved_precision"]
                for row in rows
            ),
            "direct_resolved_coverage": _mean(
                row["direct"]["resolved_coverage"]
                for row in rows
            ),
            "causal_edge_recall": _mean(
                row["direct"]["causal_recall"]
                for row in rows
            ),
            "null_edge_recall": _mean(
                row["direct"]["null_recall"]
                for row in rows
            ),
            "correlation_trap_rejection": _mean(
                row["direct"]["trap_rejection"]
                for row in rows
            ),
            "multi_hop_accuracy": _mean(
                row["multi_hop"]["accuracy"]
                for row in rows
            ),
            "multi_hop_coverage": _mean(
                row["multi_hop"]["coverage"]
                for row in rows
            ),
            "multi_hop_resolved_accuracy": _mean(
                row["multi_hop"]["resolved_accuracy"]
                for row in rows
                if row["multi_hop"]["counts"]["total"] > 0
            ),
            "direct_baseline_multi_hop_accuracy": _mean(
                row["multi_hop"]["direct_edge_baseline_accuracy"]
                for row in rows
                if row["multi_hop"]["counts"]["total"] > 0
            ),
            "negative_query_accuracy": _mean(
                row["negative"]["accuracy"]
                for row in rows
            ),
            "counterfactual_accuracy": _mean(
                row["counterfactual"]["accuracy"]
                for row in rows
            ),
            "counterfactual_coverage": _mean(
                row["counterfactual"]["coverage"]
                for row in rows
            ),
            "counterfactual_resolved_accuracy": _mean(
                row["counterfactual"]["resolved_accuracy"]
                for row in rows
                if row["counterfactual"]["counts"]["total"] > 0
            ),
            "redundant_path_preservation": _mean(
                special_redundant
            ),
            "bridge_cut_success": _mean(
                special_bridge
            ),
            "missing_bridge_unresolved": _mean(missing),
            "proof_valid_rate": ratio(
                proof_valid,
                proof_total,
            ),
            "state_restoration_rate": _mean(
                row["counterfactual"]["restoration_rate"]
                for row in rows
            ),
            "correlation_baseline_trap_accuracy": _mean(
                row["direct"]["correlation_baseline_trap_accuracy"]
                for row in rows
            ),
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("counterfactual-reasoning-v1.jsonl"),
    )
    parser.add_argument(
        "--development-seeds",
        type=int,
        default=0,
        help=(
            "Use development seeds 0..N-1 instead of the frozen "
            "official seed range. Official CI must leave this at 0."
        ),
    )
    args = parser.parse_args()
    seeds = (
        tuple(range(args.development_seeds))
        if args.development_seeds > 0
        else OFFICIAL_SEEDS
    )
    campaign(args.output, seeds=seeds)


if __name__ == "__main__":
    main()
