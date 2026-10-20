from research.five_operator_bench.distinction_connectivity import (
    run_distinction_connectivity_experiment,
)


def test_noisy_delayed_minimum_basis_forms_one_complementary_edge():
    report = run_distinction_connectivity_experiment()
    assert report["passed"], report
    assert report["declared_probes"] == 14
    assert report["active_probes"] == 2
    assert report["graph_edges"] == 1
    assert report["left_unique_context_edges"] > 0
    assert report["right_unique_context_edges"] > 0
    assert report["complementarity"] > 0.0
