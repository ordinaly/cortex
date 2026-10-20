from research.five_operator_bench.reactivation_pareto import run_reactivation_pareto


def test_noisy_reactivation_calibration_has_a_safe_pareto_candidate():
    report = run_reactivation_pareto(seeds=tuple(range(8)))
    assert report["passed"], report
    assert report["frontier"], report
    assert report["safe_candidates"], report
    recommended = report["recommended"]
    assert recommended is not None
    assert recommended["false_trigger_rate"] <= 0.10
    assert recommended["false_activation_rate"] <= 0.10
    assert recommended["reactivation_rate"] >= 0.90
    assert recommended["mean_speedup_vs_sparse"] > 1.0
    assert recommended["mean_query_saving"] > 0.0
