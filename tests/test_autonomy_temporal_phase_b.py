from cortex.articulation_controller import ArticulationState
from cortex.autonomy_robustness import TemporalObservation
from cortex.autonomy_temporal_phase_b import (
    PhaseBTemporalMemory,
    TemporalRobustnessPolicy,
)
from cortex.regime_plasticity import RelationalRegimeTracker


CTX = ("R0:forward", "R1:forward")


def _bootstrap(memory, count=32):
    for _ in range(count):
        memory.observe(
            TemporalObservation(
                CTX,
                "OLD",
                source="childhood",
                source_reliability=1.0,
                true_relation="OLD",
                bootstrap=True,
            )
        )


def test_unit_current_weight_preserves_frozen_tracker_semantics():
    left = RelationalRegimeTracker()
    right = RelationalRegimeTracker()

    sequence = ["OLD"] * 32 + ["NEW"] * 12 + ["OLD"] * 12
    for index, relation in enumerate(sequence):
        bootstrap = index < 32
        left_snap = left.observe(relation, bootstrap=bootstrap)
        right_snap = right.observe(
            relation,
            bootstrap=bootstrap,
            current_weight=1.0,
        )
        assert left_snap == right_snap


def test_weighted_current_evidence_keeps_raw_history_exact():
    tracker = RelationalRegimeTracker()
    for _ in range(4):
        tracker.observe("OLD", bootstrap=True)

    tracker.observe("NEW", current_weight=0.25)
    tracker.observe("OLD", current_weight=0.10)

    history = tracker.historical_evidence
    assert history["OLD"] == 5
    assert history["NEW"] == 1
    assert sum(history.values()) == 6


def test_reliability_weight_candidates_are_bounded_and_ordered():
    unweighted = TemporalRobustnessPolicy(source_weighting="unweighted")
    linear = TemporalRobustnessPolicy(source_weighting="linear")
    odds = TemporalRobustnessPolicy(source_weighting="odds")

    assert unweighted.current_weight(0.20) == 1.0
    assert linear.current_weight(0.95) == 1.0
    assert odds.current_weight(0.95) == 1.0
    assert 0.0 < odds.current_weight(0.20) < linear.current_weight(0.20) < 1.0
    assert odds.current_weight(0.0) == 0.0
    assert odds.current_weight(1.0) == 1.0


def test_odds_weighting_resolves_trusted_minority_over_weak_majority():
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="odds")
    )
    _bootstrap(memory)

    pattern = (
        ("NEW", "trusted", 0.95),
        ("OLD", "weak-a", 0.20),
        ("NEW", "trusted", 0.95),
        ("OLD", "weak-b", 0.20),
        ("OLD", "weak-c", 0.20),
    )
    switched = False
    for _ in range(12):
        for relation, source, reliability in pattern:
            read = memory.observe(
                TemporalObservation(
                    CTX,
                    relation,
                    source=source,
                    source_reliability=reliability,
                    true_relation="NEW",
                )
            )
            switched |= (
                read.accepted_relation == "NEW"
                and read.state is ArticulationState.EXECUTE
            )

    assert switched
    snap = memory.snapshot(CTX)
    assert snap is not None
    assert snap.accepted_relation == "NEW"
    # 32 raw OLD bootstrap + 12 * (3 OLD + 2 NEW).
    assert snap.historical_evidence["OLD"] == 68
    assert snap.historical_evidence["NEW"] == 24


def test_unweighted_control_retains_phase_a_source_blindness():
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(source_weighting="unweighted")
    )
    _bootstrap(memory)

    pattern = (
        ("NEW", 0.95),
        ("OLD", 0.20),
        ("NEW", 0.95),
        ("OLD", 0.20),
        ("OLD", 0.20),
    )
    for _ in range(12):
        for relation, reliability in pattern:
            memory.observe(
                TemporalObservation(
                    CTX,
                    relation,
                    source_reliability=reliability,
                    true_relation="NEW",
                )
            )

    snap = memory.snapshot(CTX)
    assert snap is not None
    assert snap.accepted_relation == "OLD"


def test_freshness_suspends_execution_without_mutating_regime_memory():
    memory = PhaseBTemporalMemory(
        TemporalRobustnessPolicy(freshness_missing_grace=1)
    )
    _bootstrap(memory, count=8)
    before = memory.snapshot(CTX)
    assert before is not None

    first = memory.observe(
        TemporalObservation(CTX, None, true_relation="OLD")
    )
    second = memory.observe(
        TemporalObservation(CTX, None, true_relation="OLD")
    )
    after = memory.snapshot(CTX)

    assert first.state is ArticulationState.EXECUTE
    assert not first.freshness_suspended
    assert second.state is ArticulationState.EXPLORE
    assert second.freshness_suspended
    assert second.memory_state is ArticulationState.EXECUTE
    assert after == before

    refreshed = memory.observe(
        TemporalObservation(CTX, "OLD", true_relation="OLD")
    )
    assert refreshed.state is ArticulationState.EXECUTE
    assert refreshed.execution_fresh
    assert refreshed.missing_streak == 0
