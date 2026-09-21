from __future__ import annotations

from cortex.articulation_controller import ArticulationState, ExecutionBackend
from cortex.autonomy_campaign import PolicyMode, execute_policy
from cortex.controller_calibration import frozen_calibration_v0_2
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep


def _rule(confidence=1.0, examples=6, target="RT"):
    return CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation=target,
        examples=examples,
        confidence=confidence,
        mean_support=2.0 / 3.0,
    )


def _geometry():
    return HypergraphGeometry(
        (
            RelationalHyperedge(
                "a", 0, 10, 1, "A", "R0", "B", 2.0 / 3.0
            ),
            RelationalHyperedge(
                "b", 1, 11, 2, "B", "R1", "C", 2.0 / 3.0
            ),
        )
    )


def test_no_controller_executes_when_rule_exists():
    read = execute_policy(
        geometry=_geometry(),
        rules=(_rule(),),
        source="A",
        relation="RT",
        mode=PolicyMode.NONE,
        lpu_enabled=False,
    )
    assert read.state is ArticulationState.EXECUTE
    assert read.candidates == ("C",)
    assert read.backend is ExecutionBackend.CORTEX


def test_lpu_ablation_preserves_semantic_answer():
    cortex = execute_policy(
        geometry=_geometry(),
        rules=(_rule(),),
        source="A",
        relation="RT",
        mode=PolicyMode.NONE,
        lpu_enabled=False,
    )
    lpu = execute_policy(
        geometry=_geometry(),
        rules=(_rule(),),
        source="A",
        relation="RT",
        mode=PolicyMode.NONE,
        lpu_enabled=True,
    )
    assert cortex.candidates == lpu.candidates == ("C",)
    assert lpu.witness_count == 1


def test_calibrated_policy_explores_moderate_conflict():
    target = _rule(confidence=0.75, examples=6, target="RT")
    competitor = _rule(confidence=0.25, examples=2, target="RX")
    read = execute_policy(
        geometry=_geometry(),
        rules=(target, competitor),
        source="A",
        relation="RT",
        mode=PolicyMode.CALIBRATED_V0_2,
        lpu_enabled=True,
        calibration=frozen_calibration_v0_2(),
    )
    assert read.state is ArticulationState.EXPLORE
    assert not read.candidates


def test_calibrated_policy_abstains_on_strong_conflict():
    target = _rule(confidence=0.6, examples=6, target="RT")
    competitor = _rule(confidence=0.4, examples=4, target="RX")
    read = execute_policy(
        geometry=_geometry(),
        rules=(target, competitor),
        source="A",
        relation="RT",
        mode=PolicyMode.CALIBRATED_V0_2,
        lpu_enabled=True,
        calibration=frozen_calibration_v0_2(),
    )
    assert read.state is ArticulationState.UNRESOLVED
    assert not read.candidates
