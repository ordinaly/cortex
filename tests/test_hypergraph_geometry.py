from __future__ import annotations

from cortex import CortexRuntime
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeStatement,
    KnowledgeStatementIncidence,
    feed_knowledge,
    make_knowledge_runtime,
)


def _knowledge_geometry():
    rows = (
        KnowledgeStatement("f1", "France", "capital", "Paris"),
        KnowledgeStatement("f2", "Germany", "capital", "Berlin"),
        KnowledgeStatement("f3", "Japan", "capital", "Tokyo"),
        KnowledgeStatement("f4", "Canada", "official language", "English"),
        KnowledgeStatement("f5", "Canada", "official language", "French"),
    )
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=256,
        budget=32,
    )
    report = feed_knowledge(runtime, rows)
    view = KnowledgeGraphView(
        report.snapshot_json,
        report.symbol_bindings,
        runtime.config_json(),
        report.statement_incidence,
    )
    return view, HypergraphGeometry(view.hyperedges())


def test_statement_is_one_higher_order_hyperedge():
    view, geometry = _knowledge_geometry()
    edges = geometry.hyperedges
    assert len(edges) == 5

    france = next(edge for edge in edges if edge.edge_id == "f1")
    assert france.participants == ("France", "capital", "Paris")
    assert france.support == 2.0 / 3.0

    # The relation itself is a basin participating in the same hyperedge.
    capital_neighbors = {
        row.basin
        for row in geometry.direct_neighbors("capital", resolution=0.5)
    }
    assert {"France", "Paris", "Germany", "Berlin"} <= capital_neighbors


def test_relation_basin_exposes_examples_and_transport_field():
    _view, geometry = _knowledge_geometry()

    directions = geometry.directions("capital", resolution=0.5)
    examples = {
        (row.source, row.target)
        for row in directions
        if row.orientation == "relation-example"
    }
    assert ("France", "Paris") in examples
    assert ("Germany", "Berlin") in examples
    assert ("Japan", "Tokyo") in examples

    france = geometry.transport("capital", "France", resolution=0.5)
    assert len(france) == 1
    assert france[0].target == "Paris"
    assert france[0].normalized_support == 1.0


def test_multi_value_relation_remains_a_field_not_a_function():
    _view, geometry = _knowledge_geometry()
    rows = geometry.transport(
        "official language",
        "Canada",
        resolution=0.5,
    )
    assert {row.target for row in rows} == {"English", "French"}
    assert all(abs(row.normalized_support - 0.5) < 1.0e-12 for row in rows)


def test_resolution_filtration_is_reversible_and_does_not_delete_edges():
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge(
                "strong",
                0,
                1,
                2,
                "A",
                "R",
                "B",
                0.9,
            ),
            RelationalHyperedge(
                "weak",
                2,
                3,
                4,
                "B",
                "S",
                "C",
                0.4,
            ),
        )
    )

    coarse = geometry.articulated_component("A", resolution=0.5)
    fine = geometry.articulated_component("A", resolution=0.3)
    coarse_again = geometry.articulated_component("A", resolution=0.5)

    assert set(coarse) == {"A", "R", "B"}
    assert set(fine) == {"A", "R", "B", "S", "C"}
    assert coarse_again == coarse
    assert len(geometry.hyperedges) == 2


def test_bottleneck_affinity_induces_resolution_distance():
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge(
                "strong",
                0,
                1,
                2,
                "A",
                "R",
                "B",
                0.9,
            ),
            RelationalHyperedge(
                "weak",
                2,
                3,
                4,
                "B",
                "S",
                "C",
                0.4,
            ),
        )
    )
    assert geometry.bottleneck_affinity("A", "B") == 0.9
    assert geometry.bottleneck_affinity("A", "C") == 0.4
    assert geometry.resolution_distance("A", "C") == 0.6

    path = geometry.strongest_path("A", "C")
    assert path is not None
    assert path.concepts == ("A", "B", "C")
    assert path.hyperedges == ("strong", "weak")
    assert path.bottleneck_affinity == 0.4


def test_resolution_connectivity_obeys_ultrametric_inequality():
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge("ab", 0, 1, 2, "A", "R", "B", 0.8),
            RelationalHyperedge("bc", 2, 3, 4, "B", "S", "C", 0.6),
            RelationalHyperedge("ac", 0, 5, 4, "A", "T", "C", 0.2),
        )
    )
    d_ab = geometry.resolution_distance("A", "B")
    d_bc = geometry.resolution_distance("B", "C")
    d_ac = geometry.resolution_distance("A", "C")
    assert d_ac <= max(d_ab, d_bc) + 1.0e-12


def test_local_geometry_is_order_invariant():
    view, geometry = _knowledge_geometry()
    reversed_geometry = HypergraphGeometry(reversed(view.hyperedges()))

    left = geometry.local_geometry("capital", resolution=0.5)
    right = reversed_geometry.local_geometry("capital", resolution=0.5)

    assert left.adjacent_basins == right.adjacent_basins
    assert left.directions == right.directions
    assert left.articulated_component == right.articulated_component


def test_hyperedge_identity_survives_support_revision():
    import json

    motif = (
        (10, 11),
        (10, 12),
        (10, 13),
        (11, 20),
        (12, 21),
        (13, 22),
        (11, 1),
        (12, 2),
        (13, 3),
    )
    incidence = (
        KnowledgeStatementIncidence(
            statement_id="stable-id",
            subject="France",
            property="capital",
            value="Paris",
            subject_id=1,
            property_id=2,
            object_id=3,
            motif_edges=motif,
        ),
    )

    def snapshot(weaken=False):
        rows = []
        for index, (a, b) in enumerate(motif):
            alpha, beta = (2.0, 1.0)
            if weaken and index == 0:
                alpha, beta = (1.0, 4.0)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "evidence": {
                        "a": alpha,
                        "b": beta,
                        "exposure": 1,
                        "state": 1 if alpha > beta else 0,
                        "last_updated": 1,
                    },
                }
            )
        return json.dumps({"graph": {"relations": rows}})

    before = KnowledgeGraphView(
        snapshot(False),
        {},
        statement_incidence=incidence,
    ).hyperedges()
    after = KnowledgeGraphView(
        snapshot(True),
        {},
        statement_incidence=incidence,
    ).hyperedges()

    assert len(before) == len(after) == 1
    assert before[0].edge_id == after[0].edge_id == "stable-id"
    assert before[0].participants == after[0].participants
    assert before[0].support == 2.0 / 3.0
    assert after[0].support == 0.2
