from research.five_operator_bench.simplicial_integration import run_simplicial_integration_experiment


def test_simplicial_integration_experiment_passes() -> None:
    report = run_simplicial_integration_experiment()
    assert report["passed"] is True
