from research.five_operator_bench.geometric_temporal_transport import run_geometric_temporal_transport_experiment


def test_geometric_temporal_transport_experiment_passes() -> None:
    report = run_geometric_temporal_transport_experiment()
    assert report["passed"] is True
