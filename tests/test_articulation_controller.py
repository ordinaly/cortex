from __future__ import annotations

from cortex.articulation_controller import (
    ArticulationController,
    ArticulationState,
    ControllerPolicy,
    ExecutionBackend,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep


def _rule(
    confidence=1.0,
    examples=8,
    mean_support=0.8,
    target="RT",
):
    return CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation=target,
        examples=examples,
        confidence=confidence,
        mean_support=mean_support,
    )


def _chain(width=1, support=0.8):
    edges = []
    for i in range(width):
        middle = f"B{i}"
        edges.append(
            RelationalHyperedge(
                f"a{i}",
                0,
                10,
                i + 1,
                "A",
                "R0",
                middle,
                support,
            )
        )
        edges.append(
            RelationalHyperedge(
                f"b{i}",
                i + 1,
                11,
                100,
                middle,
                "R1",
                "Z",
                support,
            )
        )
    return HypergraphGeometry(edges)


def test_stable_small_articulation_uses_native_cortex():
    controller = ArticulationController(_chain(width=2))
    read = controller.execute("A", "RT", (_rule(),))
    assert read.decision.state is ArticulationState.EXECUTE
    assert read.decision.backend is ExecutionBackend.CORTEX
    assert read.execution is not None
    assert read.execution.top is not None
    assert read.execution.top.target == "Z"


def test_exact_witness_request_routes_to_lpu():
    controller = ArticulationController(_chain(width=1))
    read = controller.execute(
        "A",
        "RT",
        (_rule(),),
        exact_witness=True,
    )
    assert read.decision.backend is ExecutionBackend.LPU
    assert read.execution is not None
    assert len(read.execution.witnesses) == 1


def test_high_input_entropy_stays_in_exploration():
    controller = ArticulationController(_chain())
    decision = controller.decide(
        "A",
        "RT",
        (_rule(),),
        source_entropy=0.8,
    )
    assert decision.state is ArticulationState.EXPLORE
    assert decision.backend is None


def test_low_example_count_stays_in_exploration():
    controller = ArticulationController(_chain())
    decision = controller.decide(
        "A",
        "RT",
        (_rule(examples=2),),
    )
    assert decision.state is ArticulationState.EXPLORE


def test_substantial_competing_evidence_is_unresolved():
    target = _rule(confidence=0.6, examples=6, target="RT")
    competitor = _rule(confidence=0.4, examples=4, target="RX")
    controller = ArticulationController(_chain())
    decision = controller.decide(
        "A",
        "RT",
        (target, competitor),
    )
    assert decision.state is ArticulationState.UNRESOLVED
    assert decision.backend is None


def test_twenty_percent_contradiction_remains_executable():
    target = _rule(confidence=0.8, examples=8, target="RT")
    competitor = _rule(confidence=0.2, examples=2, target="RX")
    controller = ArticulationController(_chain())
    decision = controller.decide(
        "A",
        "RT",
        (target, competitor),
    )
    assert decision.state is ArticulationState.EXECUTE


def test_controller_selects_highest_viable_resolution():
    geometry = _chain(width=1, support=2.0 / 3.0)
    controller = ArticulationController(geometry)
    decision = controller.decide(
        "A",
        "RT",
        (_rule(mean_support=2.0 / 3.0),),
    )
    assert decision.state is ArticulationState.EXECUTE
    assert decision.resolution == 0.65


def test_revision_below_resolution_blocks_execution():
    geometry = _chain(width=1, support=0.4)
    controller = ArticulationController(geometry)
    decision = controller.decide(
        "A",
        "RT",
        (_rule(mean_support=0.8),),
    )
    assert decision.state is ArticulationState.UNRESOLVED


def test_path_multiplicity_can_route_to_lpu():
    # Depth 4, width 4 fully reconverging lattice -> 64 paths.
    edges = []
    layers = [("A",)]
    layers.extend(
        tuple(f"L{depth}-{i}" for i in range(4))
        for depth in range(1, 4)
    )
    layers.append(("Z",))
    signature = []
    serial = 0
    for depth in range(4):
        relation = f"R{depth}"
        signature.append(RelationStep(relation, "forward"))
        for left in layers[depth]:
            for right in layers[depth + 1]:
                edges.append(
                    RelationalHyperedge(
                        f"e{serial}",
                        serial,
                        100 + depth,
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
    policy = ControllerPolicy(
        lpu_min_paths=32,
        lpu_min_compression_ratio=1.0,
    )
    decision = ArticulationController(
        HypergraphGeometry(edges),
        policy=policy,
    ).decide("A", "RT", (rule,))
    assert decision.state is ArticulationState.EXECUTE
    assert decision.backend is ExecutionBackend.LPU
    assert decision.path_multiplicity == 64
    assert decision.compression_ratio > 1.0


def test_multivalued_targets_are_not_treated_as_ambiguity():
    geometry = HypergraphGeometry(
        (
            RelationalHyperedge("a", 0, 10, 1, "A", "R0", "B", 0.8),
            RelationalHyperedge("b", 1, 11, 2, "B", "R1", "C", 0.8),
            RelationalHyperedge("c", 1, 11, 3, "B", "R1", "D", 0.8),
        )
    )
    read = ArticulationController(geometry).execute(
        "A",
        "RT",
        (_rule(),),
    )
    assert read.decision.state is ArticulationState.EXECUTE
    assert read.execution is not None
    assert {row.target for row in read.execution.candidates} == {"C", "D"}
