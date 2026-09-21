from __future__ import annotations

from cortex import CortexRuntime
from cortex.knowledge import (
    KnowledgeStatement,
    KnowledgeSymbolEncoder,
    feed_knowledge,
    make_knowledge_runtime,
)
from cortex.topology import TopologyView


def _runtime_and_report():
    statements = (
        KnowledgeStatement("s1", "France", "capital", "Paris"),
        KnowledgeStatement("s2", "France", "continent", "Europe"),
        KnowledgeStatement("s3", "Canada", "official language", "English"),
        KnowledgeStatement("s4", "Canada", "official language", "French"),
    )
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=256,
        budget=32,
    )
    report = feed_knowledge(runtime, statements)
    labels = {value: key for key, value in report.symbol_bindings.items()}
    return runtime, report, labels


def test_local_articulation_returns_normalized_entity_support():
    runtime, report, labels = _runtime_and_report()
    view = TopologyView.from_runtime(runtime, labels)
    france_id = report.symbol_bindings["France"]
    local = view.at_entity(france_id, top_k=6)

    assert local.top_entity is not None
    assert local.top_entity.entity_id == france_id
    assert local.top_entity.label == "France"
    assert local.top_entity.support > 0.99
    assert 0.0 <= local.entity_entropy <= 1.0
    assert 0.0 <= local.residual_entity_support <= 1.0
    assert len(local.dimensions) == 64
    assert all(0.0 <= row.reliability <= 1.0 for row in local.dimensions)


def test_arbitrary_point_can_have_multiple_entity_memberships():
    runtime, report, labels = _runtime_and_report()
    view = TopologyView.from_runtime(runtime, labels)
    encoder = KnowledgeSymbolEncoder(feature_dim=64)
    france = encoder.vector("France")
    paris = encoder.vector("Paris")
    midpoint = tuple((a + b) / 2.0 for a, b in zip(france, paris))

    local = view.at_vector(midpoint, top_k=8)
    support = {row.label: row.support for row in local.entities}

    assert "France" in support
    assert "Paris" in support
    assert abs(support["France"] - support["Paris"]) < 1.0e-12
    assert support["France"] > 0.1
    assert local.entity_entropy > 0.0


def test_local_articulation_exposes_adjacent_relation_evidence():
    runtime, report, labels = _runtime_and_report()
    view = TopologyView.from_runtime(runtime, labels)
    france_id = report.symbol_bindings["France"]
    local = view.at_entity(france_id, top_k=1)

    assert local.relations
    assert all(row.state in (0, 1) for row in local.relations)
    assert all(0.0 <= row.posterior_mean <= 1.0 for row in local.relations)
    assert any(
        row.source_id == france_id or row.target_id == france_id
        for row in local.relations
    )


def test_local_articulation_is_read_only():
    runtime, report, labels = _runtime_and_report()
    before = runtime.snapshot_json()
    view = TopologyView.from_runtime(runtime, labels)
    france_id = report.symbol_bindings["France"]

    view.at_entity(france_id)
    view.at_vector(KnowledgeSymbolEncoder(64).vector("Paris"))

    assert runtime.snapshot_json() == before
