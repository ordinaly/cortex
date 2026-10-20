from research.five_operator_bench.latent_reactivation import (
    run_latent_reactivation_experiment,
)


def test_sentinel_escalation_materially_accelerates_reactivation():
    report = run_latent_reactivation_experiment()
    assert report["passed"], report
    sparse = report["sparse_only"]
    escalated = report["sentinel_escalation"]
    assert not escalated["false_trigger_during_stable"]
    assert escalated["trigger_tick_after_change"] is not None
    assert escalated["reactivation_tick_after_change"] < sparse["reactivation_tick_after_change"]
    assert report["reactivation_speedup"] >= 1.5
    assert escalated["final_active"] == ((1,),)
