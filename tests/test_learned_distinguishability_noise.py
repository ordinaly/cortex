from research.five_operator_bench.learn_delta_noise import (
    evaluate_controlled,
    run_noise_experiment,
)


def test_noisy_immediate_world_does_not_split_before_temporal_evidence():
    h1 = evaluate_controlled(delay=False, horizon=1, seed=31)
    assert h1["passed"], h1
    assert h1["proposed"] == ((0, 1, 2, 3),)


def test_noisy_immediate_world_learns_only_the_two_supported_classes():
    h2 = evaluate_controlled(delay=False, horizon=2, seed=32)
    assert h2["passed"], h2
    assert h2["proposed"] == ((0, 2), (1, 3))
    assert h2["oracle_audit"]["minimum_feasible_cells"] == 2


def test_noisy_delayed_world_stays_merged_through_horizon_two():
    h1 = evaluate_controlled(delay=True, horizon=1, seed=41)
    h2 = evaluate_controlled(delay=True, horizon=2, seed=42)
    assert h1["passed"], h1
    assert h2["passed"], h2
    assert h1["proposed"] == ((0, 1, 2, 3),)
    assert h2["proposed"] == ((0, 1, 2, 3),)


def test_noisy_delayed_world_splits_only_when_horizon_three_is_accessible():
    h3 = evaluate_controlled(delay=True, horizon=3, seed=43)
    assert h3["passed"], h3
    assert h3["proposed"] == ((0,), (1,), (2,), (3,))
    assert h3["oracle_audit"]["minimum_feasible_cells"] == 4


def test_noisy_refinement_ladder_matches_the_exact_oracle():
    report = run_noise_experiment()
    assert report["passed"], report
    assert report["expected_refinement_ladder"]
    assert len(report["cases"]) == 5
