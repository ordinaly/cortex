from cortex.articulation_controller import profile_rules
from cortex.graph_sandwich_routing import (
    ProbeRoute,
    SandwichProbePolicy,
    probe_rules,
    saturating_profile_rules,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep


def _lattice(width: int, depth: int, *, irregular: bool = False):
    layers = [("A",)]
    for layer in range(1, depth):
        layers.append(tuple(f"L{layer}-{i}" for i in range(width)))
    layers.append(("Z",))

    edges = []
    serial = 0
    signature = []
    for layer in range(depth):
        relation = f"R{layer}"
        signature.append(RelationStep(relation, "forward"))
        for left_index, left in enumerate(layers[layer]):
            for right_index, right in enumerate(layers[layer + 1]):
                if (
                    irregular
                    and 0 < layer < depth - 1
                    and left_index % 2
                    and right_index >= max(1, width // 4)
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
                        0.8,
                    )
                )
                serial += 1

    rule = CompositionRule(
        signature=tuple(signature),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=0.8,
    )
    return HypergraphGeometry(edges), rule


def test_regular_deep_lattice_selects_lpu_without_exact_profile():
    geometry, rule = _lattice(width=8, depth=4)
    policy = SandwichProbePolicy(base_relative_slack=0.05)

    estimate = probe_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
        policy=policy,
    )
    exact = profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
    )

    assert estimate.eligible
    assert estimate.route is ProbeRoute.LPU
    assert exact.path_multiplicity == 512
    assert exact.compression_ratio > policy.lpu_min_compression_ratio
    # The probe queries one local neighborhood per unique layer state rather
    # than enumerating hundreds of complete paths.
    assert estimate.matching_queries < exact.path_multiplicity


def test_small_regular_lattice_selects_cortex():
    geometry, rule = _lattice(width=2, depth=3)
    policy = SandwichProbePolicy(base_relative_slack=0.05)

    estimate = probe_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
        policy=policy,
    )
    exact = profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
    )

    assert estimate.eligible
    assert estimate.route is ProbeRoute.CORTEX
    assert exact.path_multiplicity < policy.lpu_min_paths


def test_irregular_lattice_abstains_to_exact_fallback():
    geometry, rule = _lattice(width=8, depth=4, irregular=True)
    estimate = probe_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
        policy=SandwichProbePolicy(base_relative_slack=0.05),
    )

    assert estimate.route is ProbeRoute.FALLBACK
    assert not estimate.eligible
    assert any("degree-irregular" in reason for reason in estimate.reasons)


def test_boundary_case_falls_back_instead_of_forcing_a_route():
    geometry, rule = _lattice(width=4, depth=4)
    estimate = probe_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
        policy=SandwichProbePolicy(base_relative_slack=0.05),
    )

    assert estimate.route is ProbeRoute.FALLBACK
    assert estimate.eligible
    assert "routing-boundary-overlap" in estimate.reasons


def test_saturating_control_preserves_structure_but_scans_all_layer_states():
    geometry, rule = _lattice(width=8, depth=5)
    exact = profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
    )
    saturated = saturating_profile_rules(
        geometry,
        "A",
        (rule,),
        resolution=0.8,
        path_cap=128,
    )

    assert saturated.saturated
    assert saturated.path_multiplicity_lower_bound == 128
    assert saturated.materialized_edges == exact.materialized_edges
    assert saturated.layered_states == exact.layered_states
    assert saturated.matching_queries == 1 + 8 * 4
