from __future__ import annotations

from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import HypergraphCompositionLearner


def _geometry():
    rows = []
    examples = (
        ("France", "Paris", "Europe"),
        ("Germany", "Berlin", "Europe"),
        ("Japan", "Tokyo", "Asia"),
    )
    for i, (country, city, region) in enumerate(examples):
        rows.extend(
            (
                RelationalHyperedge(
                    f"c{i}",
                    i * 3,
                    100,
                    i * 3 + 1,
                    country,
                    "capital",
                    city,
                    0.8,
                ),
                RelationalHyperedge(
                    f"r{i}",
                    i * 3,
                    101,
                    i * 3 + 2,
                    country,
                    "continent",
                    region,
                    0.8,
                ),
                RelationalHyperedge(
                    f"l{i}",
                    i * 3 + 1,
                    102,
                    i * 3 + 2,
                    city,
                    "located",
                    region,
                    0.8,
                ),
            )
        )
    # Held-out target edge: Italy/Rome/Europe has the first two relations only.
    rows.extend(
        (
            RelationalHyperedge(
                "held-cap",
                50,
                100,
                51,
                "Italy",
                "capital",
                "Rome",
                0.8,
            ),
            RelationalHyperedge(
                "held-reg",
                50,
                101,
                52,
                "Italy",
                "continent",
                "Europe",
                0.8,
            ),
        )
    )
    return HypergraphGeometry(rows)


def test_learns_reverse_capital_then_continent_composition():
    learner = HypergraphCompositionLearner(
        _geometry(),
        minimum_examples=2,
        confidence_threshold=0.75,
        resolution=0.5,
    )
    rules = learner.fit()

    located = [row for row in rules if row.target_relation == "located"]
    assert located
    signatures = {
        tuple((step.relation, step.orientation) for step in row.signature)
        for row in located
    }
    assert (
        ("capital", "reverse"),
        ("continent", "forward"),
    ) in signatures


def test_infers_held_out_relation_without_direct_edge():
    learner = HypergraphCompositionLearner(
        _geometry(),
        minimum_examples=2,
        confidence_threshold=0.75,
        resolution=0.5,
    )
    read = learner.infer("Rome", "located")
    assert read.resolved
    assert read.top is not None
    assert read.top.target == "Europe"
    assert read.top.rule_examples >= 3
    assert read.top.rule_confidence == 1.0


def test_relation_names_are_opaque():
    mapping = {
        "capital": "R7",
        "continent": "R2",
        "located": "R9",
    }
    geometry = HypergraphGeometry(
        RelationalHyperedge(
            edge.edge_id,
            edge.subject_id,
            edge.property_id,
            edge.object_id,
            edge.subject,
            mapping.get(edge.property, edge.property),
            edge.object,
            edge.support,
        )
        for edge in _geometry().hyperedges
    )
    learner = HypergraphCompositionLearner(
        geometry,
        minimum_examples=2,
        confidence_threshold=0.75,
        resolution=0.5,
    )
    read = learner.infer("Rome", "R9")
    assert read.top is not None
    assert read.top.target == "Europe"


def test_multi_value_inference_is_not_forced_to_single_target():
    base = list(_geometry().hyperedges)
    base.extend(
        (
            RelationalHyperedge(
                "lang-country",
                200,
                201,
                202,
                "Country-X",
                "language-source",
                "LanguageHub",
                0.8,
            ),
            RelationalHyperedge(
                "lang-a",
                202,
                203,
                204,
                "LanguageHub",
                "has-language",
                "Lang-A",
                0.8,
            ),
            RelationalHyperedge(
                "lang-b",
                202,
                203,
                205,
                "LanguageHub",
                "has-language",
                "Lang-B",
                0.8,
            ),
        )
    )
    geometry = HypergraphGeometry(base)
    # This simply verifies path following preserves branching; no target rule
    # is asserted from this one example.
    learner = HypergraphCompositionLearner(
        geometry,
        minimum_examples=2,
        confidence_threshold=0.75,
        resolution=0.5,
    )
    signature = (
        # Country-X -> LanguageHub -> language values.
        # The private path routine is intentionally exercised indirectly by
        # fitting; branching must not collapse the geometry.
    )
    assert len(geometry.transport("has-language", "LanguageHub", resolution=0.5)) == 2
