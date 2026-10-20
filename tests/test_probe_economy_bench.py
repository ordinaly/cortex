from research.five_operator_bench.probe_economy import run_probe_economy_experiment


def test_probe_economy_preserves_oracle_quotients_with_minimum_empirical_bases():
    report = run_probe_economy_experiment()
    assert report["passed"], report
    for row in report["cases"]:
        assert row["active_probes"] == row["minimum_probes"]
        assert row["active_probes"] < row["total_probes"]
        assert row["coverage"] == 1.0
        assert row["reduced_partition"] == row["exact_partition"]
