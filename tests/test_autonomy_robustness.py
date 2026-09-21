from cortex.articulation_controller import ArticulationState
from cortex.autonomy_robustness import (
    AutonomyTemporalMemory,
    TemporalObservation,
)


CTX = ("R0:forward", "R1:forward")


def _bootstrap(memory: AutonomyTemporalMemory, count: int = 32) -> None:
    for _ in range(count):
        memory.observe(
            TemporalObservation(
                context=CTX,
                relation="OLD",
                true_relation="OLD",
                bootstrap=True,
            )
        )


def test_missing_observation_is_a_true_noop_on_regime_evidence():
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)
    before = memory.snapshot(CTX)
    assert before is not None

    read = memory.observe(
        TemporalObservation(
            context=CTX,
            relation=None,
            true_relation="OLD",
        )
    )
    after = memory.snapshot(CTX)

    assert after == before
    assert read.missing
    assert read.state is ArticulationState.EXECUTE
    assert read.accepted_relation == "OLD"
    assert read.safe_execute is True


def test_contexts_are_locally_independent():
    memory = AutonomyTemporalMemory()
    other = ("Q0:forward", "Q1:forward")
    for _ in range(32):
        memory.observe(TemporalObservation(CTX, "OLD", bootstrap=True))
        memory.observe(TemporalObservation(other, "KEEP", bootstrap=True))

    switched = False
    for _ in range(32):
        read = memory.observe(
            TemporalObservation(CTX, "NEW", true_relation="NEW")
        )
        if read.accepted_relation == "NEW":
            switched = True
            break

    assert switched
    stable = memory.snapshot(other)
    assert stable is not None
    assert stable.accepted_relation == "KEEP"
    assert stable.state is ArticulationState.EXECUTE


def test_downstream_safety_marks_stale_exact_execution():
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)

    read = memory.observe(
        TemporalObservation(
            context=CTX,
            relation="NEW",
            true_relation="NEW",
        )
    )
    assert read.accepted_relation == "OLD"
    assert read.unsafe_execute is True


def test_fixed_context_vocabulary_state_is_bounded_in_stream_length():
    memory = AutonomyTemporalMemory()
    _bootstrap(memory, 64)

    # Seed the complete fixed vocabulary for the context.
    for _ in range(8):
        memory.observe(TemporalObservation(CTX, "NEW"))
    for _ in range(8):
        memory.observe(TemporalObservation(CTX, "OLD"))
    cells = memory.active_cells

    for index in range(4000):
        relation = "NEW" if index % 9 == 0 else "OLD"
        memory.observe(TemporalObservation(CTX, relation))

    assert memory.active_cells == cells
    assert memory.context_count == 1


def test_source_reliability_is_metadata_only_in_phase_a():
    memory = AutonomyTemporalMemory()
    _bootstrap(memory)

    memory.observe(
        TemporalObservation(
            CTX,
            "NEW",
            source="trusted",
            source_reliability=0.99,
        )
    )
    history_after_trusted = memory.snapshot(CTX).historical_evidence
    memory.observe(
        TemporalObservation(
            CTX,
            "OLD",
            source="weak",
            source_reliability=0.05,
        )
    )
    history_after_weak = memory.snapshot(CTX).historical_evidence

    assert history_after_trusted["NEW"] == 1
    assert history_after_weak["OLD"] == history_after_trusted["OLD"] + 1
