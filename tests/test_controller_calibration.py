from __future__ import annotations

from cortex.articulation_controller import ArticulationState, ExecutionBackend
from cortex.controller_calibration import (
    BACKEND_FEATURE_NAMES,
    CONFLICT_FEATURE_NAMES,
    EXECUTE_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    CalibrationBundle,
    CalibratedArticulationController,
    LinearProbabilitySurface,
    SurfaceObservation,
    fit_logistic_surface,
    select_precision_threshold,
)
from cortex.hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from cortex.hypergraph_inference import CompositionRule, RelationStep


def _surface(names, weights, intercept, threshold=0.5):
    return LinearProbabilitySurface(
        feature_names=tuple(names),
        weights=tuple(weights),
        intercept=intercept,
        threshold=threshold,
    )


def _bundle():
    return CalibrationBundle(
        version="test",
        execute_surface=_surface(
            EXECUTE_FEATURE_NAMES,
            (0.5, 1.5, 1.0, 0.5, 4.0, 4.0),
            -10.0,
            0.5,
        ),
        conflict_surface=_surface(
            CONFLICT_FEATURE_NAMES,
            (2.0, 4.0, 3.0, 3.0, 1.0),
            -6.0,
            0.5,
        ),
        structural_surface=_surface(
            STRUCTURAL_FEATURE_NAMES,
            (2.0, 5.0, 1.0),
            -5.0,
            0.5,
        ),
        backend_surface=_surface(
            BACKEND_FEATURE_NAMES,
            (2.0, 5.0),
            -3.0,
            0.5,
        ),
    )


def _rule(confidence=1.0, examples=8, target="RT"):
    return CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation=target,
        examples=examples,
        confidence=confidence,
        mean_support=0.8,
    )


def _geometry(width=1):
    edges = []
    for index in range(width):
        middle = f"B{index}"
        edges.extend(
            (
                RelationalHyperedge(
                    f"a{index}",
                    0,
                    10,
                    index + 1,
                    "A",
                    "R0",
                    middle,
                    0.8,
                ),
                RelationalHyperedge(
                    f"b{index}",
                    index + 1,
                    11,
                    100,
                    middle,
                    "R1",
                    "Z",
                    0.8,
                ),
            )
        )
    return HypergraphGeometry(edges)


def test_logistic_fit_learns_positive_monotone_surface():
    rows = (
        SurfaceObservation((0.1, 0.2), False),
        SurfaceObservation((0.2, 0.3), False),
        SurfaceObservation((0.8, 0.7), True),
        SurfaceObservation((0.9, 0.9), True),
    )
    surface = fit_logistic_surface(rows, ("a", "b"), epochs=800)
    assert all(weight >= 0.0 for weight in surface.weights)
    assert surface.probability((0.9, 0.9)) > surface.probability((0.1, 0.2))


def test_precision_threshold_is_fitted_on_validation_rows():
    rows = tuple(
        SurfaceObservation((value,), value >= 0.6)
        for value in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
    )
    surface = fit_logistic_surface(rows, ("x",), epochs=1000)
    calibrated = select_precision_threshold(
        surface,
        rows,
        minimum_precision=1.0,
    )
    predicted = [
        row
        for row in rows
        if calibrated.decide(row.features)
    ]
    assert predicted
    assert all(row.label for row in predicted)


def test_bundle_round_trip():
    bundle = _bundle()
    restored = CalibrationBundle.from_json(bundle.to_json())
    assert restored == bundle


def test_calibrated_controller_executes_stable_rule():
    controller = CalibratedArticulationController(_geometry(), _bundle())
    read = controller.execute("A", "RT", (_rule(),))
    assert read.decision.state is ArticulationState.EXECUTE
    assert read.execution is not None
    assert read.execution.top is not None
    assert read.execution.top.target == "Z"


def test_calibrated_controller_explores_ambiguous_source():
    controller = CalibratedArticulationController(_geometry(), _bundle())
    decision = controller.decide(
        "A",
        "RT",
        (_rule(),),
        source_support=0.5,
        source_entropy=0.8,
    )
    assert decision.state is not ArticulationState.EXECUTE


def test_competing_rule_can_produce_unresolved():
    bundle = _bundle()
    controller = CalibratedArticulationController(_geometry(), bundle)
    target = _rule(confidence=0.55, examples=6, target="RT")
    competitor = _rule(confidence=0.45, examples=5, target="RX")
    decision = controller.decide("A", "RT", (target, competitor))
    assert decision.state is ArticulationState.UNRESOLVED


def test_exact_witness_still_overrides_backend_surface():
    bundle = CalibrationBundle(
        version="test",
        execute_surface=_bundle().execute_surface,
        conflict_surface=_bundle().conflict_surface,
        structural_surface=_bundle().structural_surface,
        backend_surface=_surface(
            BACKEND_FEATURE_NAMES,
            (0.0, 0.0),
            -20.0,
            0.99,
        ),
    )
    controller = CalibratedArticulationController(_geometry(), bundle)
    read = controller.execute(
        "A",
        "RT",
        (_rule(),),
        exact_witness=True,
    )
    assert read.decision.backend is ExecutionBackend.LPU
    assert read.execution is not None
    assert read.execution.witnesses


def test_structural_weakness_has_distinct_unresolved_reason():
    rule = CompositionRule(
        signature=(
            RelationStep("R0", "forward"),
            RelationStep("R1", "forward"),
        ),
        target_relation="RT",
        examples=8,
        confidence=1.0,
        mean_support=0.2,
    )
    controller = CalibratedArticulationController(_geometry(), _bundle())
    decision = controller.decide(
        "A",
        "RT",
        (rule,),
        source_support=0.5,
        source_entropy=0.8,
    )
    assert decision.state is ArticulationState.UNRESOLVED
    assert decision.reasons == ("learned-structural-weakness",)
