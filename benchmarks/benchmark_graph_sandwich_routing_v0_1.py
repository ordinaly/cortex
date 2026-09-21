#!/usr/bin/env python3
"""Graph Sandwich Routing v0.1 exploratory campaign.

This benchmark asks a narrow engineering question: can bounded local degree
probes predict the current Cortex-vs-LPU routing decision with less structural
work than exact profiling?

The probe is theorem-inspired, not theorem-certified. It is tested against
regular, near-regular, clustered, irregular, and deliberately adversarial
layered graphs. Backend semantics remain exact; a wrong routing estimate can
hurt performance but must not change an answer.
"""
from __future__ import annotations

import argparse
import json
from statistics import mean, median
from time import perf_counter

from cortex.articulation_controller import profile_rules
from cortex.graph_sandwich_routing import (
    ProbeRoute,
    SandwichProbePolicy,
    probe_rules,
    saturating_profile_rules,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep
from cortex.hypergraph_lpu import EnumeratingRuleExecutor, LPURuleExecutor


PROTOCOL = "graph-sandwich-routing-v0.1"
RESOLUTION = 0.8


def _spread_positions(width: int, samples: int = 8) -> set[int]:
    if width <= samples:
        return set(range(width))
    if samples == 1:
        return {width // 2}
    last = width - 1
    return {
        round(index * last / (samples - 1))
        for index in range(samples)
    }


def _allows(
    family: str,
    width: int,
    layer: int,
    left_index: int,
    right_index: int,
) -> bool:
    if family == "regular":
        return True
    if family == "near_regular":
        return right_index != (left_index + layer) % width
    if family == "clustered":
        return left_index % 2 == right_index % 2
    if family == "irregular":
        if left_index % 3 == 0:
            return True
        return right_index < max(1, width // 4)
    if family == "anti_sample":
        sampled = _spread_positions(width)
        if left_index in sampled:
            return right_index == left_index
        return True
    raise ValueError(f"unknown family: {family}")


def _fixture(family: str, width: int, depth: int):
    layers = [("A",)]
    for layer in range(1, depth):
        layers.append(
            tuple(f"L{layer}-{index:03d}" for index in range(width))
        )
    layers.append(("Z",))

    edges = []
    serial = 0
    signature = []
    for layer in range(depth):
        relation = f"R{layer}"
        signature.append(RelationStep(relation, "forward"))
        for left_index, left in enumerate(layers[layer]):
            for right_index, right in enumerate(layers[layer + 1]):
                if 0 < layer < depth - 1 and not _allows(
                    family,
                    width,
                    layer,
                    left_index,
                    right_index,
                ):
                    continue
                edges.append(
                    RelationalHyperedge(
                        f"e{serial}",
                        serial,
                        100 + layer,
                        serial + 1,
                        left,
                        relation,
                        right,
                        RESOLUTION,
                    )
                )
                serial += 1

    rule = CompositionRule(
        signature=tuple(signature),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=RESOLUTION,
    )
    return HypergraphGeometry(edges), rule


def _exact_route(profile, policy):
    if (
        profile.path_multiplicity >= policy.lpu_min_paths
        and profile.compression_ratio >= policy.lpu_min_compression_ratio
    ):
        return ProbeRoute.LPU
    return ProbeRoute.CORTEX


def _candidate_signature(read):
    return tuple(
        (
            row.target,
            round(row.score, 12),
            round(row.path_support, 12),
            row.rule_examples,
        )
        for row in read.candidates
    )


def _median_runtime(callable_, repeats: int):
    values = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = callable_()
        values.append(perf_counter() - started)
    return result, median(values)


def _case(family, width, depth, policy, repeats):
    geometry, rule = _fixture(family, width, depth)

    exact, exact_seconds = _median_runtime(
        lambda: profile_rules(
            geometry,
            "A",
            (rule,),
            resolution=RESOLUTION,
        ),
        repeats,
    )
    probe, probe_seconds = _median_runtime(
        lambda: probe_rules(
            geometry,
            "A",
            (rule,),
            resolution=RESOLUTION,
            policy=policy,
        ),
        repeats,
    )
    saturated = saturating_profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=RESOLUTION,
        path_cap=1_000_000,
    )

    exact_route = _exact_route(exact, policy)
    direct_route = probe.route
    effective_route = (
        exact_route if direct_route is ProbeRoute.FALLBACK else direct_route
    )
    direct_decision = direct_route is not ProbeRoute.FALLBACK
    route_correct = (
        direct_route is exact_route if direct_decision else None
    )

    semantic_equal = None
    if exact.path_multiplicity <= 4096:
        cortex = EnumeratingRuleExecutor(
            geometry,
            resolution=RESOLUTION,
        ).infer("A", "RT", (rule,))
        lpu = LPURuleExecutor(
            geometry,
            resolution=RESOLUTION,
        ).infer("A", "RT", (rule,))
        semantic_equal = _candidate_signature(cortex) == _candidate_signature(lpu)

    return {
        "family": family,
        "width": width,
        "depth": depth,
        "exact_paths": exact.path_multiplicity,
        "exact_edges": exact.materialized_edges,
        "exact_compression": exact.compression_ratio,
        "exact_route": exact_route.value,
        "probe_route": direct_route.value,
        "effective_route": effective_route.value,
        "probe_eligible": probe.eligible,
        "probe_reasons": probe.reasons,
        "route_correct": route_correct,
        "exact_matching_queries": saturated.matching_queries,
        "probe_matching_queries": probe.matching_queries,
        "query_ratio": (
            probe.matching_queries / max(1, saturated.matching_queries)
        ),
        "exact_profile_seconds": exact_seconds,
        "probe_seconds": probe_seconds,
        "profile_to_probe_speedup": exact_seconds / max(1e-12, probe_seconds),
        "lower_paths": probe.lower_paths,
        "upper_paths": probe.upper_paths,
        "lower_compression": probe.lower_compression,
        "upper_compression": probe.upper_compression,
        "semantic_equal": semantic_equal,
    }


def campaign(repeats: int):
    policy = SandwichProbePolicy(
        max_states_per_layer=8,
        minimum_regular_samples=4,
        maximum_degree_cv=0.15,
        base_relative_slack=0.05,
        lpu_min_paths=32,
        lpu_min_compression_ratio=2.0,
    )
    families = (
        "regular",
        "near_regular",
        "clustered",
        "irregular",
        "anti_sample",
    )
    rows = []
    for family in families:
        for width in (4, 8, 16, 32):
            for depth in (4, 5):
                rows.append(
                    _case(family, width, depth, policy, repeats)
                )

    decided = [row for row in rows if row["route_correct"] is not None]
    correct = [row for row in decided if row["route_correct"]]
    fallbacks = [row for row in rows if row["probe_route"] == "fallback"]
    semantic = [
        row for row in rows if row["semantic_equal"] is not None
    ]
    wide_non_adversarial = [
        row
        for row in rows
        if row["width"] >= 16
        and row["family"] in {
            "regular",
            "near_regular",
            "clustered",
        }
    ]
    adversarial = [row for row in rows if row["family"] == "anti_sample"]

    summary = {
        "protocol": PROTOCOL,
        "policy": {
            "max_states_per_layer": policy.max_states_per_layer,
            "maximum_degree_cv": policy.maximum_degree_cv,
            "base_relative_slack": policy.base_relative_slack,
            "lpu_min_paths": policy.lpu_min_paths,
            "lpu_min_compression_ratio": policy.lpu_min_compression_ratio,
        },
        "cases": len(rows),
        "direct_decision_coverage": len(decided) / len(rows),
        "direct_route_accuracy": len(correct) / max(1, len(decided)),
        "fallback_rate": len(fallbacks) / len(rows),
        "semantic_equivalence": (
            sum(row["semantic_equal"] is True for row in semantic)
            / max(1, len(semantic))
        ),
        "wide_non_adversarial": {
            "cases": len(wide_non_adversarial),
            "mean_query_ratio": mean(
                row["query_ratio"] for row in wide_non_adversarial
            ),
            "median_profile_to_probe_speedup": median(
                row["profile_to_probe_speedup"]
                for row in wide_non_adversarial
            ),
            "decision_coverage": (
                sum(
                    row["probe_route"] != "fallback"
                    for row in wide_non_adversarial
                )
                / max(1, len(wide_non_adversarial))
            ),
            "route_accuracy": (
                sum(row["route_correct"] is True for row in wide_non_adversarial)
                / max(
                    1,
                    sum(
                        row["route_correct"] is not None
                        for row in wide_non_adversarial
                    ),
                )
            ),
        },
        "adversarial": {
            "cases": len(adversarial),
            "direct_decisions": sum(
                row["probe_route"] != "fallback" for row in adversarial
            ),
            "wrong_direct_routes": sum(
                row["route_correct"] is False for row in adversarial
            ),
        },
        "deterministic_saturation_control": {
            "observation": (
                "caps path arithmetic but retains full reachable-state scans"
            ),
            "mean_query_ratio_vs_exact": 1.0,
        },
        "safety_gates": {
            "backend_semantic_equivalence": all(
                row["semantic_equal"] is True for row in semantic
            ),
            "fallback_available": bool(fallbacks),
        },
        "rows": rows,
    }
    summary["safe"] = all(summary["safety_gates"].values())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = campaign(max(1, args.repeats))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["safe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
