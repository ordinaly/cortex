from research.five_operator_bench.run import (
    causal_case,
    delta_oracle_case,
    dynamic_topology_case,
    integration_conflict_case,
    run_benchmark,
    temporal_identity_case,
)


def test_delta_recovers_exact_predictive_quotient():
    result = delta_oracle_case()
    assert result.passed, result.details
    assert tuple(result.details["expected"]) == ((0, 2), (1,), (3,))


def test_temporal_identity_is_transport_not_static_similarity():
    result = temporal_identity_case()
    assert result.passed, result.details
    assert result.details["best_successor"][0] == "young-cat"


def test_causal_influence_requires_intervention_and_control():
    result = causal_case()
    assert result.passed, result.details
    assert result.details["before_control"] == 0.0
    assert result.details["signed_effect"] == 1.0


def test_same_substrate_selects_different_context_topologies():
    result = dynamic_topology_case()
    assert result.passed, result.details
    assert ("motion", "shape") in result.details["motion_active"]
    assert ("cause", "motion") in result.details["causal_active"]
    assert ("motion", "shape") not in result.details["causal_active"]


def test_integration_preserves_visible_contradiction():
    result = integration_conflict_case()
    assert result.passed, result.details
    assert result.details["conflicts"] == ("position",)


def test_five_operator_bench_v0_1_passes_all_oracle_assisted_cases():
    report = run_benchmark()
    assert report["passed"], report
    assert report["passed_cases"] == report["total_cases"] == 5
