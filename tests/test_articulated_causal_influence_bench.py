from research.five_operator_bench.articulated_causal_influence import run_articulated_causal_influence_experiment


def test_articulated_causal_influence_experiment_passes() -> None:
    report = run_articulated_causal_influence_experiment()
    assert report["passed"] is True
