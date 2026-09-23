from cortex.articulation_controller import ArticulationState
from cortex.closed_articulation_loop import (
    ClosedArticulationLoop,
    ClosedLoopPolicy,
    EvidenceAtom,
)


def _atom(
    context,
    relation,
    event_time,
    *,
    arrival_time=None,
    provenance="s0",
    bootstrap=False,
):
    return EvidenceAtom(
        context=tuple(context),
        relation=relation,
        event_time=event_time,
        arrival_time=event_time if arrival_time is None else arrival_time,
        provenance=provenance,
        bootstrap=bootstrap,
    )


def test_downward_memory_compresses_repeated_articulation_into_one_basin():
    loop = ClosedArticulationLoop()

    for step in range(20):
        read = loop.observe(_atom(("ctx",), "A", step))

    assert read.accepted_relation == "A"
    assert read.state is ArticulationState.EXECUTE
    assert loop.snapshot.stored_basins == 1
    active = loop.condition(("ctx",))
    assert len(active) == 1
    assert active[0].relation == "A"
    assert active[0].effective_mass == 20.0


def test_late_evidence_updates_history_without_rewriting_foreground():
    loop = ClosedArticulationLoop()

    first = loop.observe(_atom(("ctx",), "NEW", 10, arrival_time=10))
    late = loop.observe(_atom(("ctx",), "OLD", 5, arrival_time=11))

    assert first.accepted_relation == "NEW"
    assert late.historical_only
    assert late.accepted_relation == "NEW"
    assert late.state is ArticulationState.EXECUTE
    assert loop.snapshot.stored_basins == 2
    assert {row.relation for row in loop.condition(("ctx",))} == {"NEW", "OLD"}


def test_provenance_dependence_is_inferred_from_repeated_public_agreement():
    policy = ClosedLoopPolicy(
        dependence_min_overlap=6,
        dependence_agreement=0.95,
    )
    loop = ClosedArticulationLoop(policy)

    for step in range(6):
        context = (f"ctx-{step}",)
        loop.observe(_atom(context, "A", step, provenance="copy-root"))
        loop.observe(_atom(context, "A", step, provenance="copy-child"))
        independent = "B" if step == 0 else "A"
        loop.observe(_atom(context, independent, step, provenance="independent"))

    groups = {frozenset(group) for group in loop.provenance_groups()}
    assert frozenset({"copy-root", "copy-child"}) in groups
    assert frozenset({"independent"}) in groups

    weighted = loop.observe(
        _atom(("transfer",), "A", 10, provenance="copy-child")
    )
    assert weighted.source_weight == 0.5


def test_memory_recurrence_reactivates_without_fabricating_history():
    policy = ClosedLoopPolicy(
        recurrence_min_history=5.0,
        recurrence_confirmations=3,
    )
    loop = ClosedArticulationLoop(policy)

    for step in range(6):
        loop.observe(_atom(("ctx",), "A", step))

    switched = None
    for offset in range(1, 20):
        read = loop.observe(_atom(("ctx",), "B", 10 + offset))
        if read.accepted_relation == "B":
            switched = offset
            break
    assert switched is not None

    recurrence_step = None
    recurrence_read = None
    for offset in range(1, 8):
        read = loop.observe(_atom(("ctx",), "A", 40 + offset))
        if read.reactivated:
            recurrence_step = offset
            recurrence_read = read
            break

    assert recurrence_step == 3
    assert recurrence_read is not None
    assert recurrence_read.accepted_relation == "A"
    assert recurrence_read.state is ArticulationState.EXECUTE


def test_downward_residuals_form_joint_background_candidate():
    policy = ClosedLoopPolicy(
        residual_window=4,
        compound_min_events=2,
    )
    loop = ClosedArticulationLoop(policy)

    for step in range(3):
        loop.observe(_atom(("left",), "OLD", step))
        loop.observe(_atom(("right",), "OLD", step))

    left = loop.observe(_atom(("left",), "NEW", 10))
    right = loop.observe(_atom(("right",), "NEW", 10))

    assert left.background_candidate is not None
    candidate = right.background_candidate
    assert candidate is not None
    assert candidate.bubble_eligible
    assert {change.subject for change in candidate.changes} >= {
        ("left",),
        ("right",),
    }
    assert any(change.axis == "current-relation" for change in candidate.changes)


def test_new_coordinates_are_exposed_as_background_structure_not_hard_labels():
    loop = ClosedArticulationLoop()

    first = loop.observe(_atom(("entity-a", "schema-a"), "X", 0))
    second = loop.observe(_atom(("entity-b", "schema-a"), "X", 1))
    third = loop.observe(_atom(("entity-c", "schema-b"), "Y", 2))

    assert set(first.novel_coordinates) == {
        "context:0",
        "context:1",
        "relation",
    }
    assert second.novel_coordinates == ("context:0",)
    assert set(third.novel_coordinates) == {
        "context:0",
        "context:1",
        "relation",
    }


def test_memory_pressure_coarsens_instead_of_raising():
    policy = ClosedLoopPolicy(
        active_budget=2,
        latent_budget=2,
    )
    loop = ClosedArticulationLoop(policy)

    for step in range(20):
        read = loop.observe(
            _atom((f"distractor-{step}",), f"R-{step}", step)
        )

    snap = loop.snapshot
    assert snap.stored_basins <= policy.total_budget
    assert snap.active_basins <= policy.active_budget
    assert snap.coarsened_basins > 0
    assert snap.coarsened_mass > 0.0
    assert read.state is ArticulationState.EXECUTE


def test_memory_conditioning_is_context_local():
    loop = ClosedArticulationLoop()

    for step in range(5):
        loop.observe(_atom(("a",), "A", step))
        loop.observe(_atom(("b",), "B", step))

    a = loop.condition(("a",))
    b = loop.condition(("b",))

    assert {row.relation for row in a} == {"A"}
    assert {row.relation for row in b} == {"B"}


def test_dependent_provenance_is_compressed_before_foreground_update():
    policy = ClosedLoopPolicy(
        dependence_min_overlap=4,
        dependence_agreement=1.0,
    )
    loop = ClosedArticulationLoop(policy)

    # Learn that copy-a and copy-b are one redundant stream.
    copied = ("A", "B", "A", "B")
    independent_a = ("A", "A", "B", "B")
    independent_b = ("B", "A", "A", "B")
    context = ("cal",)
    for step in range(4):
        base = step * 16
        loop.observe(_atom(context, copied[step], base, provenance="copy-a"))
        loop.observe(
            _atom(context, copied[step], base + 1, provenance="copy-b")
        )
        loop.observe(
            _atom(
                context,
                independent_a[step],
                base + 2,
                provenance="independent-a",
            )
        )
        loop.observe(
            _atom(
                context,
                independent_b[step],
                base + 3,
                provenance="independent-b",
            )
        )

    assert any(
        group == frozenset({"copy-a", "copy-b"})
        for group in loop.provenance_groups()
    )

    # On a fresh context the copied pair contributes one WRONG event while the
    # two independent streams each contribute TRUE. The result should converge
    # to TRUE without counting the copied pair twice.
    last = None
    for round_index in range(6):
        base = 100 + round_index * 16
        last = loop.observe(
            _atom(("target",), "WRONG", base, provenance="copy-a")
        )
        last = loop.observe(
            _atom(("target",), "WRONG", base + 1, provenance="copy-b")
        )
        last = loop.observe(
            _atom(
                ("target",),
                "TRUE",
                base + 2,
                provenance="independent-a",
            )
        )
        last = loop.observe(
            _atom(
                ("target",),
                "TRUE",
                base + 3,
                provenance="independent-b",
            )
        )

    assert last is not None
    assert last.accepted_relation == "TRUE"
    assert not (
        last.state is ArticulationState.EXECUTE
        and last.accepted_relation == "WRONG"
    )


def test_alternating_reporters_are_not_mistaken_for_one_episode():
    policy = ClosedLoopPolicy(
        dependence_min_overlap=4,
        dependence_agreement=0.875,
        episode_min_completed=3,
        episode_separation_ratio=2.0,
    )
    loop = ClosedArticulationLoop(policy)

    # Two sources alternate continuously with no burst/separation structure.
    for ordinal in range(6):
        loop.observe(
            _atom(
                ("ctx",),
                "A",
                2 * ordinal,
                provenance="source-a",
            )
        )
        loop.observe(
            _atom(
                ("ctx",),
                "A",
                2 * ordinal + 1,
                provenance="source-b",
            )
        )

    groups = loop.provenance_groups()
    assert len(groups) >= 1
    assert not loop._episodic_coreporting(groups)
