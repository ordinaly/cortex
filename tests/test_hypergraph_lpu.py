from __future__ import annotations

from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep
from cortex.hypergraph_lpu import (
    EnumeratingRuleExecutor,
    LPURuleExecutor,
    raw_lpu_reachability,
)


def _rule():
    return CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
            RelationStep("R2", "forward"),
        ),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=0.8,
    )


def _geometry():
    edges = (
        RelationalHyperedge("a0", 0, 10, 1, "A", "R0", "B", 0.8),
        RelationalHyperedge("a1", 1, 11, 2, "B", "R1", "C", 0.8),
        RelationalHyperedge("a2", 2, 12, 3, "C", "R2", "D", 0.8),
        # Relation-agnostic decoy path A -> X -> Y -> Z.
        RelationalHyperedge("d0", 0, 20, 4, "A", "Q0", "X", 0.8),
        RelationalHyperedge("d1", 4, 21, 5, "X", "Q1", "Y", 0.8),
        RelationalHyperedge("d2", 5, 22, 6, "Y", "Q2", "Z", 0.8),
    )
    return HypergraphGeometry(edges)


def test_lpu_executor_matches_enumerating_cortex_executor():
    geometry = _geometry()
    rule = _rule()

    cortex = EnumeratingRuleExecutor(geometry, resolution=0.5).infer(
        "A", "RT", (rule,)
    )
    lpu = LPURuleExecutor(geometry, resolution=0.5).infer(
        "A", "RT", (rule,)
    )

    assert cortex.top is not None
    assert lpu.top is not None
    assert cortex.top.target == lpu.top.target == "D"
    assert cortex.top.score == lpu.top.score


def test_lpu_returns_exact_layered_witness():
    read = LPURuleExecutor(_geometry(), resolution=0.5).infer(
        "A", "RT", (_rule(),)
    )
    assert len(read.witnesses) == 1
    witness = read.witnesses[0]
    assert witness.target == "D"
    assert witness.layered_path == (
        (0, "A"),
        (1, "B"),
        (2, "C"),
        (3, "D"),
    )
    assert witness.hyperedge_ids == ("a0", "a1", "a2")
    assert witness.exact


def test_lpu_only_raw_reachability_cannot_disambiguate_relation():
    reached = raw_lpu_reachability(
        _geometry(),
        "A",
        ("D", "Z"),
        resolution=0.5,
    )
    assert reached == {"D": True, "Z": True}


def test_resolution_is_decided_by_cortex_before_lpu_execution():
    rule = _rule()
    weakened = HypergraphGeometry(
        tuple(
            RelationalHyperedge(
                edge.edge_id,
                edge.subject_id,
                edge.property_id,
                edge.object_id,
                edge.subject,
                edge.property,
                edge.object,
                0.4 if edge.edge_id == "a1" else edge.support,
            )
            for edge in _geometry().hyperedges
        )
    )
    read = LPURuleExecutor(weakened, resolution=0.5).infer(
        "A", "RT", (rule,)
    )
    assert not read.resolved
    assert read.lpu_calls == 0


def test_multivalued_geometry_is_preserved():
    rule = CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation="RT",
        examples=6,
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
    read = LPURuleExecutor(geometry, resolution=0.5).infer(
        "A", "RT", (rule,)
    )
    assert {row.target for row in read.candidates} == {"C", "D"}
    assert {row.target for row in read.witnesses} == {"C", "D"}
