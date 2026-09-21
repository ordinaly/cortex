from cortex.articulation_controller import ArticulationState
from cortex.regime_plasticity import (
    RegimePlasticityPolicy,
    RelationalRegimeTracker,
    assess_fast_evidence,
)


def _bootstrap(prior: int, policy: RegimePlasticityPolicy | None = None):
    tracker = RelationalRegimeTracker(policy)
    for _ in range(prior):
        tracker.observe("OLD", bootstrap=True)
    return tracker



def test_default_policy_is_frozen_to_pareto_knee():
    policy = RegimePlasticityPolicy()

    assert policy.fast_decay == 0.50
    assert policy.pressure_decay == 0.90
    assert policy.suspicion_ratio == 0.05
    assert policy.confirmation_steps == 5
    assert policy.maturity_cap == 64


def test_fast_assessment_uses_frozen_execution_surface():
    strong = assess_fast_evidence("OLD", {"OLD": 32.0})
    conflicted = assess_fast_evidence("OLD", {"OLD": 16.0, "NEW": 16.0})

    assert strong.state is ArticulationState.EXECUTE
    assert conflicted.state is not ArticulationState.EXECUTE
    assert strong.margin > conflicted.margin


def test_default_policy_preserves_v02_change_detection_shape():
    expected = {4: 1, 8: 1, 16: 1, 32: 2, 64: 4}
    for prior, target_lag in expected.items():
        tracker = _bootstrap(prior)
        detected = None
        switched = None
        for step in range(1, 32):
            snap = tracker.observe("NEW")
            if detected is None and snap.state is not ArticulationState.EXECUTE:
                detected = step
            if (
                snap.state is ArticulationState.EXECUTE
                and snap.accepted_relation == "NEW"
            ):
                switched = step
                break

        assert detected == target_lag
        assert switched is not None
        assert switched <= 16


def test_transient_eight_event_burst_does_not_switch_default_policy():
    tracker = _bootstrap(32)

    for _ in range(8):
        snap = tracker.observe("NEW")
        assert snap.accepted_relation == "OLD"

    recovered = None
    for step in range(1, 32):
        snap = tracker.observe("OLD")
        if (
            snap.state is ArticulationState.EXECUTE
            and snap.accepted_relation == "OLD"
        ):
            recovered = step
            break

    assert recovered is not None


def test_regime_switch_is_reversible_without_erasing_history():
    tracker = _bootstrap(32)

    for _ in range(32):
        snap = tracker.observe("NEW")
        if snap.accepted_relation == "NEW":
            break
    assert tracker.accepted_relation == "NEW"

    for _ in range(32):
        tracker.observe("NEW")

    for _ in range(32):
        snap = tracker.observe("OLD")
        if snap.accepted_relation == "OLD":
            break
    assert tracker.accepted_relation == "OLD"

    history = tracker.historical_evidence
    assert history["OLD"] >= 32
    assert history["NEW"] >= 1


def test_fixed_vocabulary_memory_does_not_grow_with_observation_count():
    tracker = _bootstrap(64)
    cells = None
    for index in range(2000):
        snap = tracker.observe("NEW" if index % 7 == 0 else "OLD")
        if index == 100:
            cells = snap.active_cells

    assert snap.active_cells == cells
    assert sum(snap.historical_evidence.values()) == 2064
