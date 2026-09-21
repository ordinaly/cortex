#!/usr/bin/env python3
"""Relational Hypergeometry v0.1 validation campaign."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

from cortex import CortexRuntime
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    feed_knowledge,
    make_knowledge_runtime,
)


PROTOCOL = "relational-hypergeometry-v0.1"
SEED = 7429


def fixture() -> tuple[KnowledgeStatement, ...]:
    rows = (
        ("France", "capital", "Paris"),
        ("France", "continent", "Europe"),
        ("Germany", "capital", "Berlin"),
        ("Germany", "continent", "Europe"),
        ("Japan", "capital", "Tokyo"),
        ("Japan", "continent", "Asia"),
        ("Brazil", "capital", "Brasilia"),
        ("Brazil", "continent", "South America"),
        ("Canada", "capital", "Ottawa"),
        ("Canada", "official language", "English"),
        ("Canada", "official language", "French"),
        ("Earth", "orbits", "Sun"),
        ("Moon", "orbits", "Earth"),
        ("Mars", "orbits", "Sun"),
    )
    return tuple(
        KnowledgeStatement(
            statement_id=f"hg-{index:03d}",
            subject=subject,
            property=relation,
            value=value,
            provenance=PROTOCOL,
        )
        for index, (subject, relation, value) in enumerate(rows)
    )


def build(statements):
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=1024,
        budget=64,
    )
    report = feed_knowledge(runtime, statements)
    view = KnowledgeGraphView(
        report.snapshot_json,
        report.symbol_bindings,
        runtime.config_json(),
        report.statement_incidence,
    )
    return runtime, report, view, HypergraphGeometry(view.hyperedges())


def ultrametric_max_violation(geometry: HypergraphGeometry) -> float:
    concepts = geometry.concepts
    distances = {
        (a, b): geometry.resolution_distance(a, b)
        for a in concepts
        for b in concepts
    }
    violation = 0.0
    for a, b, c in itertools.product(concepts, repeat=3):
        lhs = distances[(a, c)]
        rhs = max(distances[(a, b)], distances[(b, c)])
        violation = max(violation, lhs - rhs)
    return violation


def synthetic_filtration():
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge(
                "strong", 0, 1, 2, "A", "R", "B", 0.9
            ),
            RelationalHyperedge(
                "weak", 2, 3, 4, "B", "S", "C", 0.4
            ),
        )
    )
    high = geometry.articulated_component("A", resolution=0.5)
    low = geometry.articulated_component("A", resolution=0.3)
    high_again = geometry.articulated_component("A", resolution=0.5)
    return {
        "high_resolution_component": list(high),
        "low_resolution_component": list(low),
        "high_resolution_repeat": list(high_again),
        "a_b_affinity": geometry.bottleneck_affinity("A", "B"),
        "a_c_affinity": geometry.bottleneck_affinity("A", "C"),
        "a_c_distance": geometry.resolution_distance("A", "C"),
        "edge_count_after_queries": len(geometry.hyperedges),
    }


def run_campaign():
    statements = fixture()
    _runtime, report, view, geometry = build(statements)

    capital_neighbors = {
        row.basin: row.direct_support
        for row in geometry.direct_neighbors("capital", resolution=0.5)
    }
    capital_examples = {
        (row.source, row.target): row.support
        for row in geometry.directions("capital", resolution=0.5)
        if row.orientation == "relation-example"
    }
    languages = geometry.transport(
        "official language",
        "Canada",
        resolution=0.5,
    )

    local = view.local_articulation("capital", resolution=0.5)
    synthetic = synthetic_filtration()
    ultrametric_violation = ultrametric_max_violation(geometry)

    shuffled = list(statements)
    random.Random(SEED).shuffle(shuffled)
    _runtime2, _report2, view2, geometry2 = build(tuple(shuffled))

    signature = tuple(
        (
            edge.edge_id,
            edge.subject,
            edge.property,
            edge.object,
            edge.support,
        )
        for edge in geometry.hyperedges
    )
    signature2 = tuple(
        (
            edge.edge_id,
            edge.subject,
            edge.property,
            edge.object,
            edge.support,
        )
        for edge in geometry2.hyperedges
    )

    capital_neighbors2 = geometry2.direct_neighbors(
        "capital",
        resolution=0.5,
    )

    gates = {
        "one_hyperedge_per_statement": len(geometry.hyperedges) == len(statements),
        "capital_is_relation_basin": (
            "France" in capital_neighbors
            and "Paris" in capital_neighbors
            and ("France", "Paris") in capital_examples
            and ("Germany", "Berlin") in capital_examples
        ),
        "multi_value_field": (
            {row.target for row in languages} == {"English", "French"}
            and all(
                abs(row.normalized_support - 0.5) < 1.0e-12
                for row in languages
            )
        ),
        "resolution_reversible": (
            set(synthetic["high_resolution_component"]) == {"A", "R", "B"}
            and set(synthetic["low_resolution_component"])
            == {"A", "R", "B", "S", "C"}
            and synthetic["high_resolution_component"]
            == synthetic["high_resolution_repeat"]
            and synthetic["edge_count_after_queries"] == 2
        ),
        "bottleneck_geometry": (
            abs(synthetic["a_b_affinity"] - 0.9) < 1.0e-12
            and abs(synthetic["a_c_affinity"] - 0.4) < 1.0e-12
            and abs(synthetic["a_c_distance"] - 0.6) < 1.0e-12
        ),
        "ultrametric_consistency": ultrametric_violation <= 1.0e-12,
        "presentation_order_invariant": (
            signature == signature2
            and geometry.direct_neighbors("capital", resolution=0.5)
            == capital_neighbors2
        ),
        "local_articulation_contains_geometry": (
            local.hypergeometry.anchor == "capital"
            and bool(local.hypergeometry.incident_hyperedges)
            and bool(local.hypergeometry.adjacent_basins)
        ),
    }

    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "statements": report.statements,
        "hyperedges": len(geometry.hyperedges),
        "concept_basins": len(geometry.concepts),
        "capital_neighbors": capital_neighbors,
        "capital_examples": {
            f"{source}->{target}": support
            for (source, target), support in capital_examples.items()
        },
        "canada_official_language": [
            {
                "target": row.target,
                "support": row.support,
                "normalized_support": row.normalized_support,
            }
            for row in languages
        ],
        "synthetic_filtration": synthetic,
        "ultrametric_max_violation": ultrametric_violation,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_campaign()
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
