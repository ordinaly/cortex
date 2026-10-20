from cortex.five_operator_loop import (
    ArticulationInput,
    CausalModel,
    CausalObservation,
    DistinctionBank,
    DistinctionNeuron,
    DynamicComplexPolicy,
    DynamicDistinguishabilityComplex,
    FiveOperatorArticulationLoop,
    Integrator,
    LocalView,
    OrientationField,
    TemporalTransport,
)


def test_distinction_bank_aggregates_typed_channels():
    bank = DistinctionBank(
        (
            DistinctionNeuron("space-x", "space", lambda a, b: abs(a[0] - b[0])),
            DistinctionNeuron("space-y", "space", lambda a, b: abs(a[1] - b[1])),
            DistinctionNeuron("tone", "semantic", lambda a, b: abs(a[2] - b[2])),
        )
    )
    profile = bank.compare((0.0, 0.0, 0.2), (1.0, 0.0, 0.6))
    assert profile.channel("space") == 0.5
    assert abs(profile.channel("semantic") - 0.4) < 1e-12


def test_temporal_transport_keeps_identity_as_transport_not_similarity():
    transport = TemporalTransport(prior=0.0)
    transport.observe("kitten", "young-cat", weight=3.0)
    transport.observe("kitten", "other", weight=1.0)
    assert transport.probability("kitten", "young-cat") == 0.75
    assert transport.best_successor("kitten") == ("young-cat", 0.75)


def test_causal_influence_requires_do_and_control_evidence():
    model = CausalModel()
    model.observe(CausalObservation("push", "move", 1.0, True))
    assert model.influence("push", "move") == 0.0
    model.observe(CausalObservation("push", "move", 0.0, False))
    assert model.signed_effect("push", "move") == 1.0


def test_dynamic_complex_hysteresis_preserves_faces():
    policy = DynamicComplexPolicy(
        activation_threshold=0.6,
        deactivation_threshold=0.4,
        synergy_gain=0.0,
    )
    complex_ = DynamicDistinguishabilityComplex(
        ("a", "b"),
        potential_simplices=(("a", "b"),),
        policy=policy,
    )
    high = complex_.update({"a": 0.8, "b": 0.8})
    assert frozenset(("a", "b")) in high.active
    assert frozenset(("a",)) in high.active
    assert frozenset(("b",)) in high.active

    middle = complex_.update({"a": 0.5, "b": 0.5})
    assert frozenset(("a", "b")) in middle.active

    low = complex_.update({"a": 0.2, "b": 0.2})
    assert not low.active


def test_integrator_exposes_incompatible_local_views():
    integrator = Integrator(conflict_tolerance=0.2)
    simplex = frozenset(("a",))
    read = integrator.integrate(
        (
            LocalView(simplex, {"position": 0.0}),
            LocalView(simplex, {"position": 1.0}),
        ),
        active_complex=(simplex,),
    )
    assert read.articulation == (("position", 0.5),)
    assert read.residual > 0.0
    assert read.conflicts == ("position",)


def test_five_operator_loop_couples_shared_topology():
    neurons = (
        DistinctionNeuron(
            "shape",
            "structural",
            lambda a, b: abs(a[0] - b[0]),
            value_axes=("goal",),
        ),
        DistinctionNeuron(
            "motion",
            "predictive",
            lambda a, b: abs(a[1] - b[1]),
            value_axes=("goal",),
        ),
    )
    bank = DistinctionBank(neurons)
    policy = DynamicComplexPolicy(
        activation_threshold=0.5,
        deactivation_threshold=0.3,
        novelty_weight=0.4,
        temporal_weight=0.1,
        causal_weight=0.2,
        value_weight=0.2,
        base_weight=0.1,
        synergy_gain=0.4,
    )
    complex_ = DynamicDistinguishabilityComplex(
        bank.neuron_ids,
        potential_simplices=(("shape", "motion"),),
        policy=policy,
    )
    loop = FiveOperatorArticulationLoop(
        bank,
        TemporalTransport(),
        CausalModel(),
        OrientationField({"goal": 1.0}),
        Integrator(conflict_tolerance=0.25),
        complex_,
    )

    first = loop.step(ArticulationInput("s0", (0.0, 0.0), orientation={"goal": 0.0}))
    assert first.distinction.aggregate == 0.0

    edge = frozenset(("shape", "motion"))
    second = loop.step(
        ArticulationInput(
            "s1",
            (1.0, 1.0),
            orientation={"goal": 1.0},
            simplex_synergy={edge: 1.0},
            local_views=(
                LocalView(frozenset(("shape",)), {"entity": 0.9}),
                LocalView(frozenset(("motion",)), {"entity": 0.8}),
                LocalView(edge, {"entity": 0.85}),
            ),
        )
    )
    assert second.distinction.aggregate == 1.0
    assert edge in second.complex.active
    assert second.integration.used_views == 3
    assert second.integration.conflicts == ()
    assert second.orientation_utility == 1.0
