from research.five_operator_bench.probe_sampling import run_probe_sampling_experiment


def test_probe_sampling_reduces_query_load_without_starving_latent_questions():
    report = run_probe_sampling_experiment(ticks=120)
    assert report["passed"], report
    assert report["declared_probes"] == 14
    assert report["active_probes"] == 2
    assert report["latent_probes"] == 12
    assert report["active_continuous"]
    assert report["latent_seen"]
    assert report["latent_gap_ok"]
    assert report["snapshot"]["relative_query_saving"] >= 0.75
