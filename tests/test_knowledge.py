from __future__ import annotations

import json

from cortex import CortexRuntime
from cortex.knowledge import (
    KnowledgeGraphView,
    KnowledgeQualifier,
    KnowledgeStatement,
    KnowledgeSymbolEncoder,
    feed_knowledge,
    make_knowledge_runtime,
)


def _statements():
    return (
        KnowledgeStatement("s1", "France", "capital", "Paris"),
        KnowledgeStatement("s2", "France", "continent", "Europe"),
        KnowledgeStatement("s3", "Germany", "capital", "Berlin"),
        KnowledgeStatement("s4", "Earth", "orbits", "Sun"),
        KnowledgeStatement("s5", "Moon", "orbits", "Earth"),
        KnowledgeStatement("s6", "South Africa", "capital", "Pretoria"),
        KnowledgeStatement("s7", "South Africa", "capital", "Cape Town"),
        KnowledgeStatement("s8", "South Africa", "capital", "Bloemfontein"),
    )


def test_symbol_encoder_is_stable_and_separates_sample_identities():
    encoder = KnowledgeSymbolEncoder(feature_dim=64)
    a = encoder.vector("France")
    b = encoder.vector("France")
    c = encoder.vector("Paris")
    assert a == b
    assert a != c
    mse = sum((x - y) ** 2 for x, y in zip(a, c)) / len(a)
    assert mse > 0.7


def test_statement_grammar_preserves_roles_and_qualifiers():
    encoder = KnowledgeSymbolEncoder()
    statement = KnowledgeStatement(
        "qualified",
        "Country-X",
        "capital",
        "City-A",
        qualifiers=(
            KnowledgeQualifier("start-time", "2020"),
            KnowledgeQualifier("end-time", "2025"),
        ),
    )
    encoded = encoder.encode(statement)
    assert "@syntax:subject" in encoded.symbols
    assert "@syntax:property" in encoded.symbols
    assert "@syntax:object" in encoded.symbols
    assert "@syntax:qualifier" in encoded.symbols
    assert len(encoded.positive_pairs) == 25


def test_feed_binds_repeated_symbols_consistently_and_queries_memory():
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=512,
        budget=32,
    )
    report = feed_knowledge(runtime, _statements())
    assert report.binding_conflicts == 0
    assert report.statements == len(_statements())
    assert report.cortex_entities == report.symbols

    view = KnowledgeGraphView(report.snapshot_json, report.symbol_bindings)
    assert view.objects("France", "capital") == ("Paris",)
    assert view.objects("France", "continent") == ("Europe",)
    assert view.objects("Earth", "orbits") == ("Sun",)
    assert view.objects("South Africa", "capital") == (
        "Bloemfontein",
        "Cape Town",
        "Pretoria",
    )
    assert view.objects("France", "orbits") == ()


def test_every_encoded_positive_edge_is_active_in_cortex_snapshot():
    encoder = KnowledgeSymbolEncoder()
    statements = _statements()
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=512,
        budget=32,
    )
    report = feed_knowledge(runtime, statements, encoder=encoder)
    snapshot = json.loads(report.snapshot_json)
    active = {
        tuple(sorted((int(entry["a"]), int(entry["b"]))))
        for entry in snapshot["graph"]["relations"]
        if int(entry["evidence"]["state"]) == 1
    }

    for statement in statements:
        encoded = encoder.encode(statement)
        for left, right in encoded.positive_pairs:
            pair = tuple(
                sorted(
                    (
                        report.symbol_bindings[left],
                        report.symbol_bindings[right],
                    )
                )
            )
            assert pair in active


def test_external_symbols_cannot_impersonate_syntax_nodes():
    import pytest

    with pytest.raises(ValueError, match="reserved"):
        KnowledgeStatement("bad", "@syntax:subject", "relation", "value")
    with pytest.raises(ValueError, match="reserved"):
        KnowledgeQualifier("relation", "@syntax:object")


def test_local_knowledge_articulation_decodes_sparse_properties():
    runtime = make_knowledge_runtime(
        CortexRuntime,
        feature_dim=64,
        max_entities=512,
        budget=32,
    )
    report = feed_knowledge(runtime, _statements())
    view = KnowledgeGraphView(
        report.snapshot_json,
        report.symbol_bindings,
        runtime.config_json(),
    )

    local = view.local_articulation("France", top_k=6)
    assert local.topology.top_entity is not None
    assert local.topology.top_entity.label == "France"

    coordinates = {
        (row.property, row.value): row.structural_support
        for row in local.properties
    }
    assert ("capital", "Paris") in coordinates
    assert ("continent", "Europe") in coordinates
    assert coordinates[("capital", "Paris")] > 0.5

    south_africa = view.local_articulation("South Africa")
    capitals = {
        row.value
        for row in south_africa.properties
        if row.property == "capital"
    }
    assert capitals == {"Pretoria", "Cape Town", "Bloemfontein"}
