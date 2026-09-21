from __future__ import annotations

from cortex.articulation_controller import ArticulationState
from cortex.controller_calibration import (
    CalibratedArticulationController,
    frozen_calibration_v0_2,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import RelationStep
from cortex.regime_plasticity import (
    RegimePhase,
    RelationalRegimeMemory,
    decide_with_regime,
)


SIGNATURE = (
    RelationStep("R0", "forward"),
    RelationStep("R1", "forward"),
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


def _decision(regime, relation):
    controller = CalibratedArticulationController(
        _geometry(),
        frozen_calibration_v0_2(),
    )
    return decide_with_regime(controller, regime, "A", relation)


def test_initial_regime_requires_sustained_evidence():
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(3):
        read = regime.observe("OLD")
        assert read.phase is RegimePhase.BOOTSTRAP
    read = regime.observe("OLD")
    assert read.phase is RegimePhase.STABLE
    assert read.accepted_relation == "OLD"
    assert _decision(regime, "OLD").state is ArticulationState.EXECUTE


def test_one_off_contradiction_does_not_destabilize_mature_regime():
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(32):
        regime.observe("OLD")
    read = regime.observe("NEW")
    assert read.phase is RegimePhase.STABLE
    assert read.accepted_relation == "OLD"
    assert _decision(regime, "OLD").state is ArticulationState.EXECUTE


def test_persistent_shift_is_bounded_and_archives_history():
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(64):
        regime.observe("OLD")

    phases = []
    switched_at = None
    for index in range(1, 40):
        read = regime.observe("NEW")
        phases.append(read.phase)
        if read.accepted_relation == "NEW":
            switched_at = index
            break

    assert RegimePhase.CHALLENGED in phases
    assert RegimePhase.UNRESOLVED in phases
    assert switched_at is not None
    assert switched_at <= 24
    assert _decision(regime, "NEW").state is ArticulationState.EXECUTE
    assert regime.historical_counts["OLD"] == 64
    assert regime.archives[0].relation == "OLD"


def test_recurrent_regime_can_return_without_erasing_first_regime():
    regime = RelationalRegimeMemory(SIGNATURE)
    for _ in range(32):
        regime.observe("OLD")
    for _ in range(30):
        regime.observe("NEW")
    assert regime.read().accepted_relation == "NEW"

    returned = None
    for index in range(1, 40):
        read = regime.observe("OLD")
        if read.accepted_relation == "OLD":
            returned = index
            break

    assert returned is not None
    assert returned <= 24
    assert [row.relation for row in regime.archives] == ["OLD", "NEW"]
    assert regime.historical_counts["OLD"] > 32
    assert regime.historical_counts["NEW"] == 30
